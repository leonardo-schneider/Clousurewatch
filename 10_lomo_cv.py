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
