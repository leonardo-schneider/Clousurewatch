"""Temporary: comparison curves — both models on same split."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve,
)
from sklearn.preprocessing import StandardScaler
from config_00 import TARGET_COL, MODEL_DIR, FIG_DIR
from app_helpers import add_null_flags

plt.rcParams.update({"font.family": "serif", "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})

LATEST_ANCHOR = pd.Timestamp("2020-06-01")
TEST_FRAC     = 0.20
META_COLS     = {"business_id", TARGET_COL, "anchor_date", "city", "state", "metro"}
METRO_DIRS    = {
    "tampa": Path("data/processed"), "philadelphia": Path("data/processed_philly"),
    "indianapolis": Path("data/processed_indianapolis"), "tucson": Path("data/processed_tucson"),
    "nashville": Path("data/processed_nashville"), "new_orleans": Path("data/processed_new_orleans"),
    "saint_louis": Path("data/processed_saint_louis"), "reno": Path("data/processed_reno"),
    "boise": Path("data/processed_boise"),
}

# ── Load data ─────────────────────────────────────────────────────────────────
frames = []
for metro, d in METRO_DIRS.items():
    df = pd.read_parquet(d / "features.parquet")
    df = add_null_flags(df)
    emb = d / "review_embeddings.parquet"
    if emb.exists():
        df = df.merge(pd.read_parquet(emb), on="business_id", how="left")
    df["metro"] = metro
    df["anchor_date"] = pd.to_datetime(df["anchor_date"])
    frames.append(df)
all_df = pd.concat(frames, ignore_index=True)
all_df = all_df[all_df["anchor_date"] <= LATEST_ANCHOR].copy()
all_df_s = all_df.sort_values("anchor_date").reset_index(drop=True)
n_test = max(1, int(len(all_df_s) * TEST_FRAC))
train_df = all_df_s.iloc[:-n_test].copy()
test_df  = all_df_s.iloc[-n_test:].copy()
feat_cols = [c for c in train_df.columns if c not in META_COLS]
medians   = train_df[feat_cols].median()
X_train   = train_df[feat_cols].fillna(medians).values
y_train   = train_df[TARGET_COL].values
X_test    = test_df[feat_cols].fillna(medians).values
y_test    = test_df[TARGET_COL].values

# ── Load models ───────────────────────────────────────────────────────────────
xgb    = joblib.load(Path(MODEL_DIR) / "xgboost_final.pkl")
lr     = joblib.load(Path(MODEL_DIR) / "logistic_regression_final.pkl")
scaler = joblib.load(Path(MODEL_DIR) / "lr_scaler_final.pkl")

prob_xgb_train = xgb.predict_proba(X_train)[:, 1]
prob_xgb_test  = xgb.predict_proba(X_test)[:, 1]
prob_lr_train  = lr.predict_proba(scaler.transform(X_train))[:, 1]
prob_lr_test   = lr.predict_proba(scaler.transform(X_test))[:, 1]


# ── Plot helper ───────────────────────────────────────────────────────────────
def save_pr(y, p_xgb, p_lr, split_label, path):
    baseline = y.mean()
    px, rx, _ = precision_recall_curve(y, p_xgb)
    pl, rl, _ = precision_recall_curve(y, p_lr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rx, px, color="#2E86AB", lw=2,
            label=f"XGBoost  AUC-PR = {average_precision_score(y, p_xgb):.3f}")
    ax.plot(rl, pl, color="#E84855", lw=2, ls="--",
            label=f"Logistic Regression  AUC-PR = {average_precision_score(y, p_lr):.3f}")
    ax.axhline(baseline, color="gray", ls=":", lw=1.2, alpha=0.6,
               label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Precision-Recall Curve -- {split_label} Set", fontweight="bold")
    ax.legend(fontsize=9)
    fig.suptitle(f"n={len(y):,}  closure rate={baseline:.1%}", fontsize=9, y=0.01)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"Saved -> {path}")


def save_roc(y, p_xgb, p_lr, split_label, path):
    fpr_x, tpr_x, _ = roc_curve(y, p_xgb)
    fpr_l, tpr_l, _ = roc_curve(y, p_lr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_x, tpr_x, color="#2E86AB", lw=2,
            label=f"XGBoost  AUC-ROC = {roc_auc_score(y, p_xgb):.3f}")
    ax.plot(fpr_l, tpr_l, color="#E84855", lw=2, ls="--",
            label=f"Logistic Regression  AUC-ROC = {roc_auc_score(y, p_lr):.3f}")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1.2, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"ROC Curve -- {split_label} Set", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    fig.suptitle(f"n={len(y):,}  closure rate={y.mean():.1%}", fontsize=9, y=0.01)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"Saved -> {path}")


save_pr( y_train, prob_xgb_train, prob_lr_train, "Training", Path(FIG_DIR) / "37_pr_train_comparison.png")
save_pr( y_test,  prob_xgb_test,  prob_lr_test,  "Test",     Path(FIG_DIR) / "37b_pr_test_comparison.png")
save_roc(y_train, prob_xgb_train, prob_lr_train, "Training", Path(FIG_DIR) / "38_roc_train_comparison.png")
save_roc(y_test,  prob_xgb_test,  prob_lr_test,  "Test",     Path(FIG_DIR) / "38b_roc_test_comparison.png")
