"""
14_kfold_global.py -- Global XGBoost via temporal stratified K-fold CV.

Methodology:
  - Time-based 80/20 train/test split (anchor_date cutoff)
  - StratifiedKFold(5, shuffle=False) on time-sorted training pool
  - Grid search within training folds (mean val AUC-PR across all folds)
  - Retrain best hyperparams on full training pool
  - Report AUC-PR, AUC-ROC, F1 on held-out test set

Leakage audit:
  - META_COLS excluded: business_id, closed_within_6m, anchor_date, city, state, metro
  - stars_yelp_global flagged as mild leakage (Yelp aggregate includes post-anchor reviews)
  - All other features computed strictly from date < anchor_date

Run after compute_embeddings.py:
    python 14_kfold_global.py

Output:
    models/xgboost_kfold_global.pkl
    models/kfold_global_results.json
    figures/34_kfold_global_results.png
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, precision_recall_curve,
)
from xgboost import XGBClassifier

from config_00 import TARGET_COL, RANDOM_SEED, MODEL_DIR, FIG_DIR
from app_helpers import add_null_flags

plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LATEST_ANCHOR = pd.Timestamp("2020-06-01")   # per CLAUDE.md (not config_00.py)
N_FOLDS       = 5
TEST_FRAC     = 0.20   # time-based tail for held-out test

META_COLS = {"business_id", TARGET_COL, "anchor_date", "city", "state", "metro"}

METROS = {
    "tampa":         "data/processed",
    "philadelphia":  "data/processed_philly",
    "indianapolis":  "data/processed_indianapolis",
    "tucson":        "data/processed_tucson",
    "nashville":     "data/processed_nashville",
    "new_orleans":   "data/processed_new_orleans",
    "saint_louis":   "data/processed_saint_louis",
    "reno":          "data/processed_reno",
    "boise":         "data/processed_boise",
}

# Grid: 36 combos -- same as 10_lomo_cv.py so results are comparable
PARAM_GRID = [
    {
        "n_estimators": n, "max_depth": d, "learning_rate": lr,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": w,
        "scale_pos_weight": 10, "eval_metric": "aucpr",
        "random_state": RANDOM_SEED, "verbosity": 0,
    }
    for n in [300, 500]
    for d in [3, 4, 6]
    for lr in [0.05, 0.1]
    for w in [3, 5, 10]
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_all() -> pd.DataFrame:
    """Load all 9 metros, add null flags, join embeddings, filter to LATEST_ANCHOR."""
    frames = []
    for metro, ddir in METROS.items():
        p = Path(ddir) / "features.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p} -- run 09_multi_metro.py first")
        df = pd.read_parquet(p)
        df = add_null_flags(df)

        emb_path = Path(ddir) / "review_embeddings.parquet"
        if emb_path.exists():
            emb = pd.read_parquet(emb_path)
            df  = df.merge(emb, on="business_id", how="left")

        df["metro"]       = metro
        df["anchor_date"] = pd.to_datetime(df["anchor_date"])
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    return all_df[all_df["anchor_date"] <= LATEST_ANCHOR].copy()


def get_feat_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in META_COLS]


def print_leakage_audit(df: pd.DataFrame, feat_cols: list) -> None:
    print("\n[LEAKAGE AUDIT]")
    print(f"  Total columns in data:    {len(df.columns)}")
    print(f"  META_COLS excluded:       {sorted(META_COLS)}")
    print(f"  Feature columns used:     {len(feat_cols)}")

    leakage_candidates = [c for c in feat_cols if any(
        kw in c.lower() for kw in ["closed", "is_open", "status"]
    )]
    if leakage_candidates:
        print(f"  WARNING -- potential leakers in features: {leakage_candidates}")
    else:
        print("  No direct leakage keywords found in feature names.")

    if "stars_yelp_global" in feat_cols:
        print("  NOTE: stars_yelp_global is from business.json snapshot (2022) --")
        print("        includes post-anchor reviews for older cohorts. Mild leakage.")
        print("        Kept for parity with LOMO model.")
    print()


# ── Split + metrics ──────────────────────────────────────────────────────────

def time_split(df: pd.DataFrame, test_frac: float = 0.20):
    """Sort by anchor_date, hold out the latest test_frac% as test set."""
    df_s   = df.sort_values("anchor_date").reset_index(drop=True)
    n_test = max(1, int(len(df_s) * test_frac))
    return df_s.iloc[:-n_test].copy(), df_s.iloc[-n_test:].copy()


def eval_metrics(model, X_test, y_test, X_val=None, y_val=None) -> dict:
    """AUC-PR, AUC-ROC, F1. Threshold F1-optimized on X_val if given, else on X_test."""
    prob    = model.predict_proba(X_test)[:, 1]
    auc_pr  = float(average_precision_score(y_test, prob))
    auc_roc = float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) > 1 else float("nan")

    thr_prob = model.predict_proba(X_val)[:, 1] if X_val is not None else prob
    thr_y    = y_val if X_val is not None else y_test
    prec, rec, thr = precision_recall_curve(thr_y, thr_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    opt = float(thr[np.argmax(f1s[:-1])])
    f1  = float(f1_score(y_test, (prob >= opt).astype(int)))

    return {"AUC_PR": round(auc_pr, 4), "AUC_ROC": round(auc_roc, 4),
            "F1": round(f1, 4), "threshold": round(opt, 4)}


# ── K-fold tuning ────────────────────────────────────────────────────────────

def kfold_tune(train_df: pd.DataFrame, feat_cols: list) -> tuple:
    """
    StratifiedKFold(N_FOLDS, shuffle=False) on time-sorted training data.
    Selects best hyperparams by mean val AUC-PR across all folds.
    Returns (best_params, fold_results_list).
    """
    train_s = train_df.sort_values("anchor_date").reset_index(drop=True)
    X_all   = train_s[feat_cols]
    y_all   = train_s[TARGET_COL].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=False)

    param_auc_prs = np.zeros(len(PARAM_GRID))
    fold_results  = []

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        X_tr_fold  = X_all.iloc[tr_idx]
        y_tr_fold  = y_all[tr_idx]
        X_val_fold = X_all.iloc[val_idx]
        y_val_fold = y_all[val_idx]

        # Medians fit on training split only (anti-leakage)
        medians    = X_tr_fold.median()
        X_tr_fold  = X_tr_fold.fillna(medians)
        X_val_fold = X_val_fold.fillna(medians)

        fold_best_pr = -1.0
        fold_best_p  = None
        for pi, params in enumerate(PARAM_GRID):
            m = XGBClassifier(**params)
            m.fit(X_tr_fold, y_tr_fold, verbose=False)
            pr = float(average_precision_score(y_val_fold, m.predict_proba(X_val_fold)[:, 1]))
            param_auc_prs[pi] += pr
            if pr > fold_best_pr:
                fold_best_pr = pr
                fold_best_p  = params.copy()

        fold_results.append({
            "fold":        fold_i + 1,
            "n_train":     int(len(tr_idx)),
            "n_val":       int(len(val_idx)),
            "best_val_pr": round(fold_best_pr, 4),
            "best_params": fold_best_p,
        })
        print(f"  Fold {fold_i+1}/{N_FOLDS}  best val AUC-PR={fold_best_pr:.4f}"
              f"  n={fold_best_p['n_estimators']} d={fold_best_p['max_depth']}"
              f" lr={fold_best_p['learning_rate']} mcw={fold_best_p['min_child_weight']}")

    best_idx    = int(np.argmax(param_auc_prs))
    best_params = PARAM_GRID[best_idx].copy()
    mean_pr     = param_auc_prs[best_idx] / N_FOLDS
    print(f"\n  Best params across all folds (mean val AUC-PR={mean_pr:.4f}):")
    print(f"    n_estimators={best_params['n_estimators']}  max_depth={best_params['max_depth']}"
          f"  learning_rate={best_params['learning_rate']}  min_child_weight={best_params['min_child_weight']}")

    return best_params, fold_results


# ── Final training + test eval ────────────────────────────────────────────────

def train_and_evaluate(train_df, test_df, feat_cols, best_params):
    """Retrain on full training pool with best_params. Evaluate on held-out test."""
    medians = train_df[feat_cols].median()
    X_train = train_df[feat_cols].fillna(medians)
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[feat_cols].fillna(medians)
    y_test  = test_df[TARGET_COL].values

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)

    metrics = eval_metrics(model, X_test, y_test)
    return model, metrics, medians


# ── Figure ───────────────────────────────────────────────────────────────────

def plot_results(fold_results: list, test_metrics: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: per-fold val AUC-PR
    ax = axes[0]
    fold_nums = [f["fold"] for f in fold_results]
    val_prs   = [f["best_val_pr"] for f in fold_results]
    bars = ax.bar(fold_nums, val_prs, color="#2E86AB", alpha=0.85, width=0.6)
    for bar, v in zip(bars, val_prs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    mean_val = float(np.mean(val_prs))
    ax.axhline(mean_val, color="#E84855", ls="--", lw=1.5,
               label=f"Mean = {mean_val:.3f}")
    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("Best Val AUC-PR", fontsize=11)
    ax.set_title(f"K-Fold Validation AUC-PR ({N_FOLDS} folds, temporal)", fontweight="bold")
    ax.set_ylim(0, max(val_prs) * 1.3)
    ax.legend(fontsize=9)

    # Panel 2: test set metrics
    ax2 = axes[1]
    metric_names  = ["AUC-PR", "AUC-ROC", "F1"]
    metric_values = [test_metrics["AUC_PR"], test_metrics["AUC_ROC"], test_metrics["F1"]]
    colors_m      = ["#2E86AB", "#1DB954", "#F4A261"]
    bars2 = ax2.bar(metric_names, metric_values, color=colors_m, alpha=0.85, width=0.5)
    for bar, v in zip(bars2, metric_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("Held-Out Test Set Performance\n(time-based 20% tail)", fontweight="bold")
    ax2.set_ylim(0, 1.1)

    plt.suptitle(
        "Global XGBoost: Temporal K-Fold CV Tuning + Test Evaluation\n"
        "All 9 metros, 81 features (null flags + embeddings)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = Path(FIG_DIR) / "34_kfold_global_results.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 14 -- Global Model: Temporal K-Fold CV + Test Eval")
    print("=" * 60)

    print("\n[1] Loading all 9 metros (LATEST_ANCHOR = 2020-06-01)...")
    all_df    = load_all()
    feat_cols = get_feat_cols(all_df)
    n_closed  = int(all_df[TARGET_COL].sum())
    print(f"  Restaurants: {len(all_df):,}  Closed: {n_closed:,} ({all_df[TARGET_COL].mean():.1%})"
          f"  Features: {len(feat_cols)}")
    print_leakage_audit(all_df, feat_cols)

    print("[2] Time-based 80/20 train/test split...")
    train_df, test_df = time_split(all_df, test_frac=TEST_FRAC)
    print(f"  Train: {len(train_df):,}  "
          f"({train_df['anchor_date'].min().date()} to {train_df['anchor_date'].max().date()})"
          f"  closure={train_df[TARGET_COL].mean():.1%}")
    print(f"  Test:  {len(test_df):,}  "
          f"({test_df['anchor_date'].min().date()} to {test_df['anchor_date'].max().date()})"
          f"  closure={test_df[TARGET_COL].mean():.1%}")

    print(f"\n[3] Hyperparameter tuning: StratifiedKFold({N_FOLDS}) on training data...")
    print(f"  Grid: {len(PARAM_GRID)} param combos x {N_FOLDS} folds = "
          f"{len(PARAM_GRID) * N_FOLDS} model fits")
    best_params, fold_results = kfold_tune(train_df, feat_cols)

    print("\n[4] Retraining on full training pool with best params...")
    model, test_metrics, medians = train_and_evaluate(train_df, test_df, feat_cols, best_params)
    print(f"  TEST SET: AUC-PR={test_metrics['AUC_PR']:.4f}  "
          f"AUC-ROC={test_metrics['AUC_ROC']:.4f}  "
          f"F1={test_metrics['F1']:.4f}  "
          f"threshold={test_metrics['threshold']:.4f}")

    print("\n[5] Saving model...")
    out_model = Path(MODEL_DIR) / "xgboost_kfold_global.pkl"
    joblib.dump(model, out_model)
    print(f"  Saved -> {out_model}")

    print("\n[6] Saving results JSON...")
    results = {
        "method":              "temporal_stratified_kfold",
        "n_folds":             N_FOLDS,
        "latest_anchor":       str(LATEST_ANCHOR.date()),
        "n_features":          len(feat_cols),
        "train_n":             int(len(train_df)),
        "test_n":              int(len(test_df)),
        "train_closure_rate":  round(float(train_df[TARGET_COL].mean()), 4),
        "test_closure_rate":   round(float(test_df[TARGET_COL].mean()),  4),
        "best_params":         {k: v for k, v in best_params.items()
                                if k not in ("eval_metric", "random_state", "verbosity")},
        "fold_results":        fold_results,
        "test_metrics":        test_metrics,
        "leakage_notes": {
            "excluded_meta_cols": sorted(META_COLS),
            "stars_yelp_global":  "mild leakage: Yelp aggregate includes post-anchor reviews",
        },
    }
    out_json = Path(MODEL_DIR) / "kfold_global_results.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Saved -> {out_json}")

    print("\n[7] Generating figure...")
    plot_results(fold_results, test_metrics)

    mean_val_pr = float(np.mean([f["best_val_pr"] for f in fold_results]))
    print("\n=== SUMMARY ===")
    print(f"  Mean K-fold val AUC-PR:   {mean_val_pr:.4f}")
    print(f"  Test set AUC-PR:          {test_metrics['AUC_PR']:.4f}")
    print(f"  Test set AUC-ROC:         {test_metrics['AUC_ROC']:.4f}")
    print(f"  Test set F1:              {test_metrics['F1']:.4f}")
    print(f"  Model saved:              {out_model}")


if __name__ == "__main__":
    main()
