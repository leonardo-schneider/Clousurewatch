"""
10_lomo_cv.py -- Leave-One-Metro-Out CV + global model.

For each of the 9 US metros, holds it out as the test set and trains
XGBoost on the remaining 8. Hyperparameters are tuned on a time-based
val split within the train pool (no data from the held-out metro is used
for tuning). After all folds, a global model trained on all 9 metros is
evaluated on Edmonton as an out-of-distribution test.

Run after all metros have features.parquet:
    python 10_lomo_cv.py

Output:
    models/lomo_results.json
    models/xgboost_global.pkl
    figures/19_lomo_cv_results.png
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
from typing import Dict, List, Tuple
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier

from config_00 import TARGET_COL, RANDOM_SEED, MODEL_DIR, FIG_DIR

plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

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
EDMONTON_DIR = Path("data/processed_edmonton")

# XGBoost hyperparameter grid (16 combinations)
PARAM_GRID = [
    {
        "n_estimators": n, "max_depth": d, "learning_rate": lr,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": w,
        "scale_pos_weight": 10, "eval_metric": "aucpr",
        "random_state": RANDOM_SEED, "verbosity": 0,
    }
    for n in [300, 500]
    for d in [4, 6]
    for lr in [0.05, 0.1]
    for w in [3, 5]
]


def load_all_metros() -> Dict[str, pd.DataFrame]:
    """Load features.parquet for each metro; return dict keyed by metro name."""
    dfs = {}
    for metro, directory in METROS.items():
        p = Path(directory) / "features.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}  — run 09_multi_metro.py for {metro} first")
        df = pd.read_parquet(p)
        df["metro"] = metro
        df["anchor_date"] = pd.to_datetime(df["anchor_date"])
        dfs[metro] = df
        n_closed = int(df[TARGET_COL].sum())
        print(f"  {metro:15s}: {len(df):5,} restaurants, {n_closed} closed ({n_closed/len(df):.1%})")
    return dfs


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS]


def time_val_split(df: pd.DataFrame, val_frac: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Latest val_frac% by anchor_date → val; rest → train."""
    df_s = df.sort_values("anchor_date")
    n_val = max(1, int(len(df_s) * val_frac))
    return df_s.iloc[:-n_val].copy(), df_s.iloc[-n_val:].copy()


def tune_xgb(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val:   pd.DataFrame, y_val:   pd.Series,
) -> Tuple[dict, float]:
    """
    Grid search over PARAM_GRID; select params that maximize val AUC-PR.
    Imputation medians are already applied to X_train/X_val before this call.
    """
    best_auc_pr = -1.0
    best_params = PARAM_GRID[0]

    for params in PARAM_GRID:
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc_pr = average_precision_score(y_val, y_prob)
        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            best_params = params.copy()

    return best_params, best_auc_pr


def run_lomo_fold(
    held_out_metro: str,
    all_dfs: Dict[str, pd.DataFrame],
) -> dict:
    """
    Train on 8 metros, tune hyperparams on a time-based val split within
    those 8, retrain on full 8-metro pool, evaluate on held-out metro.
    """
    print(f"\n  ── Fold: held-out = {held_out_metro} ──")

    # Build train pool from the 8 non-held-out metros
    train_pool = pd.concat(
        [df for name, df in all_dfs.items() if name != held_out_metro],
        ignore_index=True,
    )
    held_df = all_dfs[held_out_metro].copy()

    feat_cols = get_feature_cols(train_pool)

    # Time-based 80/20 split within train pool
    train_df, val_df = time_val_split(train_pool, val_frac=0.20)

    X_train = train_df[feat_cols].copy()
    y_train = train_df[TARGET_COL]
    X_val   = val_df[feat_cols].copy()
    y_val   = val_df[TARGET_COL]

    # Fit imputation medians on train only (anti-leakage rule 3)
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_val   = X_val.fillna(train_medians)

    # Hyperparameter tuning on val
    print(f"    Tuning XGBoost ({len(PARAM_GRID)} param combos)...")
    best_params, val_auc_pr = tune_xgb(X_train, y_train, X_val, y_val)
    print(f"    Best val AUC-PR: {val_auc_pr:.4f}  params: n={best_params['n_estimators']} d={best_params['max_depth']} lr={best_params['learning_rate']}")

    # Retrain on full train pool (train + val) with best params
    X_full = train_pool[feat_cols].fillna(train_medians)
    y_full = train_pool[TARGET_COL]
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_full, y_full, verbose=False)

    # Evaluate on held-out metro
    # Reuses 80%-split medians for held-out imputation (conservative: no held-out
    # data contaminates medians, at the cost of slight miscalibration on X_full).
    X_held = held_df.reindex(columns=feat_cols).fillna(train_medians)
    y_held = held_df[TARGET_COL]

    y_prob = final_model.predict_proba(X_held)[:, 1]
    auc_pr  = float(average_precision_score(y_held, y_prob))
    auc_roc = float(roc_auc_score(y_held, y_prob)) if len(y_held.unique()) > 1 else float("nan")

    # Threshold: F1-optimize on val
    prec, rec, thr = precision_recall_curve(y_val, final_model.predict_proba(X_val)[:, 1])
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    opt_thr = float(thr[np.argmax(f1s[:-1])])
    f1 = float(f1_score(y_held, (y_prob >= opt_thr).astype(int)))

    result = {
        "metro":        held_out_metro,
        "AUC_PR":       round(auc_pr,  4),
        "AUC_ROC":      round(auc_roc, 4),
        "F1":           round(f1,      4),
        "n":            int(len(held_df)),
        "closure_rate": round(float(y_held.mean()), 4),
        "val_AUC_PR":   round(val_auc_pr, 4),
        "best_params":  best_params,
    }
    print(f"    Test → AUC-PR={auc_pr:.4f}  AUC-ROC={auc_roc:.4f}  F1={f1:.4f}  n={len(held_df)}")
    return result


def train_global_model(
    all_dfs: Dict[str, pd.DataFrame],
    best_params: dict,
) -> Tuple[object, List[str], pd.Series]:
    """
    Train XGBoost on all 9 metros concatenated.
    Returns (model, feat_cols, train_medians).
    """
    full_df = pd.concat(all_dfs.values(), ignore_index=True)
    feat_cols = get_feature_cols(full_df)

    X = full_df[feat_cols].copy()
    y = full_df[TARGET_COL]

    train_medians = X.median()
    X = X.fillna(train_medians)

    model = XGBClassifier(**best_params)
    model.fit(X, y, verbose=False)
    return model, feat_cols, train_medians


def evaluate_edmonton(
    model,
    feat_cols: List[str],
    train_medians: pd.Series,
) -> dict:
    """Load Edmonton features.parquet and evaluate the global model."""
    edm_path = EDMONTON_DIR / "features.parquet"
    if not edm_path.exists():
        print(f"  WARNING: Edmonton features not found at {edm_path}. Skipping OOD eval.")
        return {}

    edm = pd.read_parquet(edm_path)
    edm["anchor_date"] = pd.to_datetime(edm["anchor_date"])

    X_edm  = edm.reindex(columns=feat_cols).fillna(train_medians)
    y_edm  = edm[TARGET_COL]

    y_prob  = model.predict_proba(X_edm)[:, 1]
    auc_pr  = float(average_precision_score(y_edm, y_prob))
    auc_roc = float(roc_auc_score(y_edm, y_prob)) if len(y_edm.unique()) > 1 else float("nan")

    n_closed = int(y_edm.sum())
    print(f"\n  === EDMONTON OOD ===")
    print(f"  n={len(edm):,}  closed={n_closed} ({y_edm.mean():.1%})")
    print(f"  AUC-PR={auc_pr:.4f}  AUC-ROC={auc_roc:.4f}")

    return {
        "AUC_PR":       round(auc_pr,  4),
        "AUC_ROC":      round(auc_roc, 4),
        "n":            int(len(edm)),
        "closure_rate": round(float(y_edm.mean()), 4),
    }


def plot_lomo_results(fold_results: List[dict]) -> None:
    """
    Grouped horizontal bar chart — one row per metro sorted by AUC-PR descending.
    Two bars per metro: AUC-PR (blue) and AUC-ROC (green).
    Tampa single-city reference lines: AUC-PR=0.203, AUC-ROC=0.694.
    Saved to figures/19_lomo_cv_results.png.
    """
    TAMPA_AUC_PR  = 0.203
    TAMPA_AUC_ROC = 0.694

    df = pd.DataFrame(fold_results).sort_values("AUC_PR", ascending=True)
    metros = df["metro"].tolist()
    n = len(metros)

    y = np.arange(n)
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.7 + 1.5)))
    bars_pr  = ax.barh(y + height/2, df["AUC_PR"],  height, label="AUC-PR",  color="#2E86AB", alpha=0.85)
    bars_roc = ax.barh(y - height/2, df["AUC_ROC"], height, label="AUC-ROC", color="#3BB273", alpha=0.85)

    ax.axvline(TAMPA_AUC_PR,  color="#2E86AB", ls="--", lw=1.2, alpha=0.6,
               label=f"Tampa AUC-PR={TAMPA_AUC_PR:.3f}")
    ax.axvline(TAMPA_AUC_ROC, color="#3BB273", ls="--", lw=1.2, alpha=0.6,
               label=f"Tampa AUC-ROC={TAMPA_AUC_ROC:.3f}")

    for bar in bars_pr:
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.3f}", va="center", fontsize=8)
    for bar in bars_roc:
        v = bar.get_width()
        label = f"{v:.3f}" if not np.isnan(v) else "n/a"
        ax.text(v + 0.005 if not np.isnan(v) else 0.005,
                bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([m.replace("_", " ").title() for m in metros], fontsize=10)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("LOMO CV — Generalization Across 9 US Metros", fontweight="bold", fontsize=12)
    plt.tight_layout()

    out = FIG_DIR / "19_lomo_cv_results.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def main():
    print("=" * 60)
    print("STEP 10 -- Leave-One-Metro-Out CV + Global Model")
    print("=" * 60)

    # ── 1. Load all metros ──────────────────────────────────────────────────
    print("\n[1] Loading metro features")
    all_dfs = load_all_metros()
    total = sum(len(df) for df in all_dfs.values())
    print(f"  Total: {total:,} restaurants across {len(all_dfs)} metros")

    # ── 2. LOMO CV (9 folds) ────────────────────────────────────────────────
    print("\n[2] LOMO CV (9 folds)")
    fold_results = []
    for metro in METROS:
        result = run_lomo_fold(metro, all_dfs)
        fold_results.append(result)

    # ── 3. Aggregate metrics ────────────────────────────────────────────────
    auc_prs  = [r["AUC_PR"]  for r in fold_results]
    auc_rocs = [r["AUC_ROC"] for r in fold_results if not np.isnan(r["AUC_ROC"])]  # NaN if held-out is single-class
    aggregate = {
        "mean_AUC_PR":  round(float(np.mean(auc_prs)),  4),
        "std_AUC_PR":   round(float(np.std(auc_prs)),   4),
        "mean_AUC_ROC": round(float(np.mean(auc_rocs)), 4),
        "std_AUC_ROC":  round(float(np.std(auc_rocs)),  4),
    }
    print(f"\n  Aggregate: AUC-PR={aggregate['mean_AUC_PR']:.4f}±{aggregate['std_AUC_PR']:.4f}  "
          f"AUC-ROC={aggregate['mean_AUC_ROC']:.4f}±{aggregate['std_AUC_ROC']:.4f}")

    # ── 4. Global model — use best fold's hyperparams ───────────────────────
    print("\n[3] Training global model on all 9 metros")
    best_fold = max(fold_results, key=lambda r: r["val_AUC_PR"])
    print(f"  Using hyperparams from best fold: {best_fold['metro']} (val AUC-PR={best_fold['val_AUC_PR']:.4f})")
    global_params = best_fold["best_params"]

    global_model, feat_cols, train_medians = train_global_model(all_dfs, global_params)
    joblib.dump(global_model, MODEL_DIR / "xgboost_global.pkl")
    print(f"  Global model saved -> {MODEL_DIR}/xgboost_global.pkl")

    # ── 5. Edmonton OOD evaluation ──────────────────────────────────────────
    print("\n[4] Edmonton OOD evaluation")
    edmonton_results = evaluate_edmonton(global_model, feat_cols, train_medians)

    # ── 6. Save results JSON ────────────────────────────────────────────────
    def _nan_to_none(v):
        return None if isinstance(v, float) and np.isnan(v) else v

    results = {
        "folds": [
            {k: _nan_to_none(v) for k, v in r.items() if k not in ("best_params", "val_AUC_PR")}
            for r in fold_results
        ],
        "aggregate": aggregate,
        "global_model": {
            "train_metros": len(METROS),
            "test_metro":   "edmonton",
            **edmonton_results,
        },
    }
    out_json = MODEL_DIR / "lomo_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {out_json}")

    # ── 7. Figure ───────────────────────────────────────────────────────────
    print("\n[5] Generating figure")
    plot_lomo_results(fold_results)

    # ── 8. Print summary table ──────────────────────────────────────────────
    print("\n  === LOMO CV SUMMARY ===")
    print(f"  {'Metro':15s} {'AUC-PR':>8} {'AUC-ROC':>8} {'F1':>7} {'N':>6} {'Rate':>6}")
    print("  " + "-" * 58)
    for r in sorted(fold_results, key=lambda x: x["AUC_PR"], reverse=True):
        roc_str = f"{r['AUC_ROC']:8.4f}" if not np.isnan(r["AUC_ROC"]) else "     n/a"
        print(f"  {r['metro']:15s} {r['AUC_PR']:8.4f} {roc_str} "
              f"{r['F1']:7.4f} {r['n']:6,} {r['closure_rate']:6.1%}")
    print("  " + "-" * 58)
    print(f"  {'MEAN':15s} {aggregate['mean_AUC_PR']:8.4f} {aggregate['mean_AUC_ROC']:8.4f}")

    if edmonton_results:
        print(f"\n  Edmonton OOD: AUC-PR={edmonton_results['AUC_PR']:.4f}  "
              f"AUC-ROC={edmonton_results.get('AUC_ROC', float('nan')):.4f}")


if __name__ == "__main__":
    main()
