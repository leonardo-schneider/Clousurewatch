"""
07_model_analysis.py — Deep EDA and model analysis.
Reads features.parquet, ensemble_predictions.parquet, models/xgboost.pkl.
Saves 9 figures to figures/.

Run:
    python 07_model_analysis.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import shap
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

from config_00 import PROC_DIR, MODEL_DIR, FIG_DIR, TARGET_COL

plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
PALETTE = {0: "#2E86AB", 1: "#E84855"}

COVID_START = pd.Timestamp("2020-03-01")
COVID_END   = pd.Timestamp("2021-06-01")
OPT_THRESHOLD = 0.2704   # from ensemble_results.json

_META = {"business_id", "closed_within_6m", "anchor_date", "city", "state", "covid_flag"}


def save(name: str):
    p = FIG_DIR / name
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p}")


def load_data():
    feat = pd.read_parquet(PROC_DIR / "features.parquet")
    feat["anchor_date"] = pd.to_datetime(feat["anchor_date"])
    feat["covid_flag"] = (
        (feat["anchor_date"] >= COVID_START) & (feat["anchor_date"] <= COVID_END)
    ).astype(int)

    preds = pd.read_parquet(PROC_DIR / "ensemble_predictions.parquet")
    model = joblib.load(MODEL_DIR / "xgboost.pkl")

    # Merge test predictions with full feature set
    test = preds.merge(
        feat.drop(columns=["closed_within_6m"], errors="ignore"),
        on="business_id", how="left",
    )
    test["predicted"] = (test["risk_score"] >= OPT_THRESHOLD).astype(int)

    return feat, test, model


def plot_feature_distributions(feat: pd.DataFrame, model):
    """2a — Violin plots for top 10 features by XGBoost importance."""
    feat_cols = [c for c in feat.columns if c not in _META]
    importance = dict(zip(model.feature_names_in_, model.feature_importances_))
    top10 = sorted(
        [c for c in feat_cols if c in importance],
        key=lambda c: importance[c], reverse=True
    )[:10]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()

    for i, col in enumerate(top10):
        ax = axes[i]
        data_open   = feat.loc[feat[TARGET_COL] == 0, col].dropna()
        data_closed = feat.loc[feat[TARGET_COL] == 1, col].dropna()
        ax.violinplot([data_open, data_closed], positions=[0, 1],
                      showmedians=True, showextrema=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Open", "Closed"], fontsize=8)
        ax.set_title(col.replace("_", "\n"), fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7)

    plt.suptitle("Top 10 Feature Distributions by XGBoost Importance",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    save("11_feature_distributions_violin.png")
    print(f"    Top 10 features: {top10}")


def plot_outlier_profiles(feat: pd.DataFrame):
    """2b — Table of most extreme restaurants per feature."""
    COLS = ["n_reviews_obs", "n_checkins_obs", "days_since_last_review",
            "review_velocity", "checkin_velocity"]
    rows = []
    for col in COLS:
        top3 = feat.nlargest(3, col)[["business_id", col, TARGET_COL]]
        for _, r in top3.iterrows():
            rows.append({
                "feature": col,
                "business_id": r["business_id"][:12] + "...",
                "value": round(float(r[col]), 1),
                "closed": int(r[TARGET_COL]),
            })
    df_out = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, len(rows) * 0.4 + 1))
    ax.axis("off")
    tbl = ax.table(
        cellText=df_out.values,
        colLabels=df_out.columns,
        loc="center", cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.4)
    plt.title("Most Extreme Restaurants by Feature (top 3 each)", fontweight="bold", pad=10)
    save("12_outlier_profiles.png")


def main():
    print("=" * 60)
    print("STEP 7 -- Model Analysis & Deep EDA")
    print("=" * 60)

    feat, test, model = load_data()
    print(f"  Features: {feat.shape[0]:,} rows | Test set: {test.shape[0]:,} rows\n")

    print("[2a] Feature distributions (violin)...")
    plot_feature_distributions(feat, model)

    print("[2b] Outlier profiles...")
    plot_outlier_profiles(feat)

    # Remaining plots added in Tasks 4-8


if __name__ == "__main__":
    main()
