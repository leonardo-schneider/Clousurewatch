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
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_prob = model.predict_proba(X_val)[:, 1]
        auc_pr = average_precision_score(y_val, y_prob)
        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            best_params = params

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
    # Use train_pool medians for imputation (no data from held-out metro)
    X_held = held_df.reindex(columns=feat_cols).fillna(train_medians)
    y_held = held_df[TARGET_COL]

    y_prob = final_model.predict_proba(X_held)[:, 1]
    auc_pr  = float(average_precision_score(y_held, y_prob))
    auc_roc = float(roc_auc_score(y_held, y_prob))

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
