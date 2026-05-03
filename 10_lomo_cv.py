"""
10_lomo_cv.py -- Leave-One-Metro-Out CV + global model.

Three models per fold:
  - Logistic Regression (LR) -- baseline, tuned C on val AUC-PR, scaled features
  - XGBoost (XGB) -- expanded grid search (36 combos), retrained on full pool
  - XGBoost + isotonic calibration (XGB+cal) -- base trained on 80%, calibrated on 20%

Hyperparameters tuned on time-based 20% val split within train pool.
Imputation medians and feature scalers fit on 80% train split only (no leakage).

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

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_curve,
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

# XGBoost grid: 36 combinations (2x3x2x3).
# max_depth=3 added: shallower trees generalize better cross-city.
# min_child_weight=10 added: stronger leaf regularization for sparse metros.
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

LR_C_GRID = [0.01, 0.1, 1.0, 10.0]


# ── Data loading ────────────────────────────────────────────────────────────

def load_all_metros() -> Dict[str, pd.DataFrame]:
    dfs = {}
    for metro, directory in METROS.items():
        p = Path(directory) / "features.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}  -- run 09_multi_metro.py for {metro} first"
            )
        df = pd.read_parquet(p)
        df = add_null_flags(df)

        # Join sentence-embedding features if available
        emb_path = Path(directory) / "review_embeddings.parquet"
        if emb_path.exists():
            emb = pd.read_parquet(emb_path)
            df  = df.merge(emb, on="business_id", how="left")

        df["metro"] = metro
        df["anchor_date"] = pd.to_datetime(df["anchor_date"])
        dfs[metro] = df
        n_closed = int(df[TARGET_COL].sum())
        print(f"  {metro:15s}: {len(df):5,} restaurants, {n_closed} closed ({n_closed/len(df):.1%})")
    return dfs


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS]


def time_val_split(
    df: pd.DataFrame, val_frac: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Latest val_frac% by anchor_date -> val; rest -> train."""
    df_s  = df.sort_values("anchor_date")
    n_val = max(1, int(len(df_s) * val_frac))
    return df_s.iloc[:-n_val].copy(), df_s.iloc[-n_val:].copy()


# ── Shared evaluation ────────────────────────────────────────────────────────

def _eval_model(model, X_held, y_held, X_val, y_val) -> dict:
    """
    AUC-PR, AUC-ROC, F1 on held-out data.
    Threshold is F1-optimized on val, then applied to held-out.
    Accepts DataFrames or numpy arrays for X inputs.
    """
    y_prob_held = model.predict_proba(X_held)[:, 1]
    y_prob_val  = model.predict_proba(X_val)[:, 1]

    auc_pr  = float(average_precision_score(y_held, y_prob_held))
    auc_roc = (float(roc_auc_score(y_held, y_prob_held))
               if len(np.unique(y_held)) > 1 else float("nan"))

    prec, rec, thr = precision_recall_curve(y_val, y_prob_val)
    f1s     = 2 * prec * rec / (prec + rec + 1e-9)
    opt_thr = float(thr[np.argmax(f1s[:-1])])
    f1      = float(f1_score(y_held, (y_prob_held >= opt_thr).astype(int)))

    return {"AUC_PR": round(auc_pr, 4), "AUC_ROC": round(auc_roc, 4), "F1": round(f1, 4)}


def _eval_auc(model, X, y) -> dict:
    """AUC-PR and AUC-ROC only (used for global model / Edmonton where no separate val exists)."""
    y_prob  = model.predict_proba(X)[:, 1]
    auc_pr  = float(average_precision_score(y, y_prob))
    auc_roc = float(roc_auc_score(y, y_prob)) if len(np.unique(y)) > 1 else float("nan")
    return {"AUC_PR": round(auc_pr, 4), "AUC_ROC": round(auc_roc, 4)}


# ── Tuning ───────────────────────────────────────────────────────────────────

def tune_xgb(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val:   pd.DataFrame, y_val:   pd.Series,
) -> Tuple[dict, float]:
    """Grid search over PARAM_GRID; returns (best_params copy, best_val_AUC_PR)."""
    best_auc_pr = -1.0
    best_params = PARAM_GRID[0]
    for params in PARAM_GRID:
        model  = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        auc_pr = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            best_params = params.copy()
    return best_params, best_auc_pr


def tune_lr(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val:   pd.DataFrame, y_val:   pd.Series,
) -> Tuple[float, float, StandardScaler]:
    """
    Fit StandardScaler on X_train, sweep LR_C_GRID on val AUC-PR.
    Returns (best_C, best_val_AUC_PR, fitted_scaler).
    """
    scaler = StandardScaler()
    Xtr    = scaler.fit_transform(X_train)
    Xvl    = scaler.transform(X_val)

    best_auc_pr = -1.0
    best_c      = 1.0
    for c in LR_C_GRID:
        model = LogisticRegression(
            C=c, class_weight="balanced", solver="lbfgs",
            max_iter=1000, random_state=RANDOM_SEED,
        )
        model.fit(Xtr, y_train)
        auc_pr = average_precision_score(y_val, model.predict_proba(Xvl)[:, 1])
        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            best_c      = c

    return best_c, best_auc_pr, scaler


# ── LOMO fold ────────────────────────────────────────────────────────────────

def run_lomo_fold(
    held_out_metro: str,
    all_dfs: Dict[str, pd.DataFrame],
) -> dict:
    """
    Train on 8 metros, evaluate on 1 held-out metro.
    Three models: LR, XGB, XGB+isotonic calibration.
    """
    print(f"\n  -- Fold: held-out = {held_out_metro} --")

    train_pool = pd.concat(
        [df for name, df in all_dfs.items() if name != held_out_metro],
        ignore_index=True,
    )
    held_df   = all_dfs[held_out_metro].copy()
    feat_cols = get_feature_cols(train_pool)

    train_df, val_df = time_val_split(train_pool, val_frac=0.20)

    X_train = train_df[feat_cols].copy()
    y_train = train_df[TARGET_COL]
    X_val   = val_df[feat_cols].copy()
    y_val   = val_df[TARGET_COL]

    # Imputation medians fit on train only (anti-leakage rule 3)
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_val   = X_val.fillna(train_medians)
    X_full  = train_pool[feat_cols].fillna(train_medians)
    y_full  = train_pool[TARGET_COL]
    X_held  = held_df.reindex(columns=feat_cols).fillna(train_medians)
    y_held  = held_df[TARGET_COL]

    # ---- XGBoost (tuned, retrained on full train pool) ----------------------
    print(f"    [XGB] Tuning ({len(PARAM_GRID)} combos)...")
    best_params, xgb_val_pr = tune_xgb(X_train, y_train, X_val, y_val)
    print(f"    [XGB] val AUC-PR={xgb_val_pr:.4f}  "
          f"n={best_params['n_estimators']} d={best_params['max_depth']} "
          f"lr={best_params['learning_rate']} mcw={best_params['min_child_weight']}")
    xgb_full = XGBClassifier(**best_params)
    xgb_full.fit(X_full, y_full, verbose=False)
    xgb_m = _eval_model(xgb_full, X_held, y_held, X_val, y_val)
    print(f"    [XGB] AUC-PR={xgb_m['AUC_PR']:.4f}  "
          f"AUC-ROC={xgb_m['AUC_ROC']:.4f}  F1={xgb_m['F1']:.4f}")

    # ---- XGBoost + isotonic calibration -------------------------------------
    # Base trained on 80% split; calibration fit on 20% val.
    # No held-out data used at any point.
    print(f"    [XGB+cal] Calibrating (isotonic on val)...")
    xgb_base = XGBClassifier(**best_params)
    xgb_base.fit(X_train, y_train, verbose=False)
    xgb_cal = CalibratedClassifierCV(xgb_base, cv="prefit", method="isotonic")
    xgb_cal.fit(X_val, y_val)
    xgb_cal_m = _eval_model(xgb_cal, X_held, y_held, X_val, y_val)
    print(f"    [XGB+cal] AUC-PR={xgb_cal_m['AUC_PR']:.4f}  "
          f"AUC-ROC={xgb_cal_m['AUC_ROC']:.4f}  F1={xgb_cal_m['F1']:.4f}")

    # ---- Logistic Regression ------------------------------------------------
    print(f"    [LR] Tuning ({len(LR_C_GRID)} C values)...")
    best_c, lr_val_pr, scaler = tune_lr(X_train, y_train, X_val, y_val)
    print(f"    [LR] val AUC-PR={lr_val_pr:.4f}  C={best_c}")
    lr_full = LogisticRegression(
        C=best_c, class_weight="balanced", solver="lbfgs",
        max_iter=1000, random_state=RANDOM_SEED,
    )
    lr_full.fit(scaler.transform(X_full), y_full)
    lr_m = _eval_model(
        lr_full,
        scaler.transform(X_held), y_held,
        scaler.transform(X_val),  y_val,
    )
    print(f"    [LR] AUC-PR={lr_m['AUC_PR']:.4f}  "
          f"AUC-ROC={lr_m['AUC_ROC']:.4f}  F1={lr_m['F1']:.4f}")

    return {
        "metro":        held_out_metro,
        "n":            int(len(held_df)),
        "closure_rate": round(float(y_held.mean()), 4),
        "xgb":     {**xgb_m,     "val_AUC_PR": round(xgb_val_pr, 4), "best_params": best_params},
        "xgb_cal": xgb_cal_m,
        "lr":      {**lr_m,      "val_AUC_PR": round(lr_val_pr,  4), "best_C": best_c},
    }


# ── Global model ─────────────────────────────────────────────────────────────

def train_global_models(
    all_dfs: Dict[str, pd.DataFrame],
    xgb_params: dict,
    lr_c: float,
) -> Tuple[object, object, object, List[str], pd.Series, StandardScaler]:
    """
    Train LR, XGB, and XGB+cal on all 9 metros combined.
    XGB+cal: base on 80%, calibration on 20% (time-based split).
    Returns (lr, xgb, xgb_cal, feat_cols, medians, scaler).
    """
    full_df   = pd.concat(all_dfs.values(), ignore_index=True)
    feat_cols = get_feature_cols(full_df)

    X = full_df[feat_cols].copy()
    y = full_df[TARGET_COL]

    medians = X.median()
    X       = X.fillna(medians)

    # XGBoost (uncalibrated) — full 9-metro pool
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X, y, verbose=False)

    # XGBoost + calibration — 80/20 time split for calibration data
    train_g, cal_g = time_val_split(full_df, val_frac=0.20)
    X_tr_g  = train_g[feat_cols].fillna(medians)
    y_tr_g  = train_g[TARGET_COL]
    X_cal_g = cal_g[feat_cols].fillna(medians)
    y_cal_g = cal_g[TARGET_COL]
    xgb_base_g = XGBClassifier(**xgb_params)
    xgb_base_g.fit(X_tr_g, y_tr_g, verbose=False)
    xgb_cal_g = CalibratedClassifierCV(xgb_base_g, cv="prefit", method="isotonic")
    xgb_cal_g.fit(X_cal_g, y_cal_g)

    # Logistic Regression — full 9-metro pool, scaled
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    lr     = LogisticRegression(
        C=lr_c, class_weight="balanced", solver="lbfgs",
        max_iter=1000, random_state=RANDOM_SEED,
    )
    lr.fit(X_sc, y)

    return lr, xgb, xgb_cal_g, feat_cols, medians, scaler


def evaluate_edmonton(
    lr, xgb, xgb_cal,
    feat_cols:  List[str],
    medians:    pd.Series,
    scaler:     StandardScaler,
) -> dict:
    """Evaluate all three global models on Edmonton (OOD). Returns {} if not found."""
    edm_path = EDMONTON_DIR / "features.parquet"
    if not edm_path.exists():
        print(f"  WARNING: Edmonton features not found at {edm_path}. Skipping.")
        return {}

    edm   = pd.read_parquet(edm_path)
    edm["anchor_date"] = pd.to_datetime(edm["anchor_date"])
    X_edm = edm.reindex(columns=feat_cols).fillna(medians)
    y_edm = edm[TARGET_COL]

    xgb_e     = _eval_auc(xgb,     X_edm,                y_edm)
    xgb_cal_e = _eval_auc(xgb_cal, X_edm,                y_edm)
    lr_e      = _eval_auc(lr,      scaler.transform(X_edm), y_edm)

    n_closed = int(y_edm.sum())
    print(f"\n  === EDMONTON OOD ===")
    print(f"  n={len(edm):,}  closed={n_closed} ({y_edm.mean():.1%})")
    print(f"  [XGB]     AUC-PR={xgb_e['AUC_PR']:.4f}  AUC-ROC={xgb_e['AUC_ROC']:.4f}")
    print(f"  [XGB+cal] AUC-PR={xgb_cal_e['AUC_PR']:.4f}  AUC-ROC={xgb_cal_e['AUC_ROC']:.4f}")
    print(f"  [LR]      AUC-PR={lr_e['AUC_PR']:.4f}  AUC-ROC={lr_e['AUC_ROC']:.4f}")

    return {
        "n":            int(len(edm)),
        "closure_rate": round(float(y_edm.mean()), 4),
        "xgb":          xgb_e,
        "xgb_cal":      xgb_cal_e,
        "lr":           lr_e,
    }


# ── Aggregate helpers ────────────────────────────────────────────────────────

def _agg(fold_results: List[dict], model_key: str) -> dict:
    prs  = [r[model_key]["AUC_PR"]  for r in fold_results]
    rocs = [r[model_key]["AUC_ROC"] for r in fold_results
            if not np.isnan(r[model_key]["AUC_ROC"])]
    return {
        "mean_AUC_PR":  round(float(np.mean(prs)),  4),
        "std_AUC_PR":   round(float(np.std(prs)),   4),
        "mean_AUC_ROC": round(float(np.mean(rocs)), 4),
        "std_AUC_ROC":  round(float(np.std(rocs)),  4),
    }


def _clean(obj):
    """Recursively replace float NaN with None for JSON-safe output."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


# ── Figure ───────────────────────────────────────────────────────────────────

def plot_lomo_results(fold_results: List[dict]) -> None:
    """
    Two-panel horizontal bar chart.
    Left: AUC-PR, Right: AUC-ROC.
    Three bars per metro: LR (orange), XGB (blue), XGB+cal (purple).
    Sorted by XGB AUC-PR descending. Tampa single-city reference lines.
    """
    TAMPA_AUC_PR  = 0.203
    TAMPA_AUC_ROC = 0.694

    df = (pd.DataFrame(fold_results)
            .assign(xgb_pr=lambda d: d["xgb"].apply(lambda x: x["AUC_PR"]))
            .sort_values("xgb_pr", ascending=True))
    metros = df["metro"].tolist()
    n      = len(metros)

    lr_prs      = [r["lr"]["AUC_PR"]      for r in df.to_dict("records")]
    xgb_prs     = [r["xgb"]["AUC_PR"]     for r in df.to_dict("records")]
    xgb_cal_prs = [r["xgb_cal"]["AUC_PR"] for r in df.to_dict("records")]

    lr_rocs      = [r["lr"]["AUC_ROC"]      for r in df.to_dict("records")]
    xgb_rocs     = [r["xgb"]["AUC_ROC"]     for r in df.to_dict("records")]
    xgb_cal_rocs = [r["xgb_cal"]["AUC_ROC"] for r in df.to_dict("records")]

    y      = np.arange(n)
    h      = 0.22
    labels = [m.replace("_", " ").title() for m in metros]
    colors = {"lr": "#F4A261", "xgb": "#2E86AB", "xgb_cal": "#7B2D8B"}

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n * 0.8 + 2)), sharey=True)

    for ax, (vals_lr, vals_xgb, vals_cal, ref, title) in zip(axes, [
        (lr_prs,  xgb_prs,  xgb_cal_prs,  TAMPA_AUC_PR,  "AUC-PR"),
        (lr_rocs, xgb_rocs, xgb_cal_rocs, TAMPA_AUC_ROC, "AUC-ROC"),
    ]):
        ax.barh(y + h,   vals_lr,  h, color=colors["lr"],      alpha=0.85, label="LR")
        ax.barh(y,       vals_xgb, h, color=colors["xgb"],     alpha=0.85, label="XGB")
        ax.barh(y - h,   vals_cal, h, color=colors["xgb_cal"], alpha=0.85, label="XGB+cal")

        ax.axvline(ref, color="gray", ls="--", lw=1.2, alpha=0.6,
                   label=f"Tampa single-city {ref:.3f}")

        for i, (a, b, c) in enumerate(zip(vals_lr, vals_xgb, vals_cal)):
            for val, offset in [(a, h), (b, 0), (c, -h)]:
                if not np.isnan(val):
                    ax.text(val + 0.005, y[i] + offset, f"{val:.3f}",
                            va="center", fontsize=7)

        ax.set_xlabel(title, fontsize=11)
        ax.set_xlim(0, 1.05)
        ax.set_title(f"LOMO CV -- {title}", fontweight="bold", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=10)
    axes[1].legend(fontsize=9, loc="lower right")

    fig.suptitle("Leave-One-Metro-Out CV: 3-Model Comparison", fontweight="bold", fontsize=13)
    plt.tight_layout()

    out = FIG_DIR / "19_lomo_cv_results.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 10 -- Leave-One-Metro-Out CV + Global Model")
    print("=" * 60)

    print("\n[1] Loading metro features")
    all_dfs = load_all_metros()
    total   = sum(len(df) for df in all_dfs.values())
    print(f"  Total: {total:,} restaurants across {len(all_dfs)} metros")

    print("\n[2] LOMO CV (9 folds x 3 models)")
    fold_results = []
    for metro in METROS:
        result = run_lomo_fold(metro, all_dfs)
        fold_results.append(result)

    print("\n[3] Aggregate metrics")
    aggregate = {
        "xgb":     _agg(fold_results, "xgb"),
        "xgb_cal": _agg(fold_results, "xgb_cal"),
        "lr":      _agg(fold_results, "lr"),
    }
    for key, agg in aggregate.items():
        print(f"  [{key:8s}] AUC-PR={agg['mean_AUC_PR']:.4f}+/-{agg['std_AUC_PR']:.4f}  "
              f"AUC-ROC={agg['mean_AUC_ROC']:.4f}+/-{agg['std_AUC_ROC']:.4f}")

    print("\n[4] Training global models on all 9 metros")
    best_xgb_fold = max(fold_results, key=lambda r: r["xgb"]["val_AUC_PR"])
    best_lr_fold  = max(fold_results, key=lambda r: r["lr"]["val_AUC_PR"])
    global_xgb_params = best_xgb_fold["xgb"]["best_params"]
    global_lr_c       = best_lr_fold["lr"]["best_C"]
    print(f"  XGB params from best fold: {best_xgb_fold['metro']} "
          f"(val AUC-PR={best_xgb_fold['xgb']['val_AUC_PR']:.4f})")
    print(f"  LR C from best fold: {best_lr_fold['metro']} "
          f"(val AUC-PR={best_lr_fold['lr']['val_AUC_PR']:.4f}), C={global_lr_c}")

    lr_g, xgb_g, xgb_cal_g, feat_cols, medians, scaler = train_global_models(
        all_dfs, global_xgb_params, global_lr_c
    )
    joblib.dump(xgb_g, MODEL_DIR / "xgboost_global.pkl")
    print(f"  Global XGB saved -> {MODEL_DIR}/xgboost_global.pkl")

    print("\n[5] Edmonton OOD evaluation")
    edmonton_results = evaluate_edmonton(lr_g, xgb_g, xgb_cal_g, feat_cols, medians, scaler)

    print("\n[6] Saving results JSON")
    results = _clean({
        "folds": [
            {
                "metro":        r["metro"],
                "n":            r["n"],
                "closure_rate": r["closure_rate"],
                "xgb":          {k: v for k, v in r["xgb"].items()
                                 if k not in ("best_params", "val_AUC_PR")},
                "xgb_cal":      r["xgb_cal"],
                "lr":           {k: v for k, v in r["lr"].items()
                                 if k not in ("val_AUC_PR", "best_C")},
            }
            for r in fold_results
        ],
        "aggregate": aggregate,
        "global_model": {
            "train_metros": len(METROS),
            "test_metro":   "edmonton",
            **edmonton_results,
        },
    })
    out_json = MODEL_DIR / "lomo_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved -> {out_json}")

    print("\n[7] Generating figure")
    plot_lomo_results(fold_results)

    print("\n  === LOMO CV SUMMARY ===")
    print(f"  {'Metro':15s} {'XGB-PR':>8} {'CAL-PR':>8} {'LR-PR':>8} "
          f"{'XGB-ROC':>9} {'CAL-ROC':>9} {'LR-ROC':>8}")
    print("  " + "-" * 72)
    for r in sorted(fold_results, key=lambda x: x["xgb"]["AUC_PR"], reverse=True):
        def roc(key):
            v = r[key]["AUC_ROC"]
            return f"{v:9.4f}" if not np.isnan(v) else "      n/a"
        print(f"  {r['metro']:15s} "
              f"{r['xgb']['AUC_PR']:8.4f} {r['xgb_cal']['AUC_PR']:8.4f} {r['lr']['AUC_PR']:8.4f} "
              f"{roc('xgb')} {roc('xgb_cal')} {roc('lr')}")
    print("  " + "-" * 72)
    for key, label in [("xgb", "XGB"), ("xgb_cal", "XGB+cal"), ("lr", "LR")]:
        agg = aggregate[key]
        print(f"  {'MEAN ' + label:15s} {agg['mean_AUC_PR']:8.4f} {'':8s} {'':8s} "
              f"{agg['mean_AUC_ROC']:9.4f}")

    if edmonton_results:
        print(f"\n  Edmonton OOD:")
        for key in ("xgb", "xgb_cal", "lr"):
            e = edmonton_results[key]
            print(f"    [{key:8s}] AUC-PR={e['AUC_PR']:.4f}  AUC-ROC={e['AUC_ROC']:.4f}")


if __name__ == "__main__":
    main()
