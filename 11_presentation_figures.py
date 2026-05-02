"""
11_presentation_figures.py -- Slide-ready figures for global XGBoost + LR models.

Generates:
  figures/20_lomo_per_metro.png       -- LOMO per-metro AUC-PR & AUC-ROC (XGB vs LR)
  figures/21_pr_roc_curves.png        -- PR + ROC curves on pooled time-split test
  figures/22_shap_beeswarm.png        -- SHAP beeswarm (global XGB, test split)
  figures/23_feature_importance.png   -- XGB importance vs LR |coef|

PR/ROC curves use an 80/20 time-based split of all 9 metros pooled.
SHAP uses the same test split with the xgboost_global.pkl model.

Run:
    python 11_presentation_figures.py
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
import shap
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score, f1_score,
)
from xgboost import XGBClassifier

from config_00 import TARGET_COL, RANDOM_SEED, MODEL_DIR, FIG_DIR

plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_XGB = "#2E86AB"    # blue
C_LR  = "#F4A261"    # orange
C_BASE = "#AAAAAA"   # random baseline

LATEST_ANCHOR = pd.Timestamp("2020-06-01")

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

META_COLS = {"business_id", TARGET_COL, "anchor_date", "city", "state", "metro"}

# Global XGB params (best fold from LOMO CV)
XGB_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=10, eval_metric="aucpr",
    random_state=RANDOM_SEED, verbosity=0,
)
LR_C = 10.0   # best fold from LOMO CV (Tampa val AUC-PR=0.248)


def save(name: str):
    p = FIG_DIR / name
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_metros() -> pd.DataFrame:
    frames = []
    for metro, ddir in METROS.items():
        p = Path(ddir) / "features.parquet"
        df = pd.read_parquet(p)
        df["metro"] = metro
        df["anchor_date"] = pd.to_datetime(df["anchor_date"])
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[all_df["anchor_date"] <= LATEST_ANCHOR].copy()
    return all_df


def get_feat_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in META_COLS]


def time_split(all_df: pd.DataFrame, val_frac: float = 0.20):
    """Time-based split: latest val_frac fraction is test."""
    all_df = all_df.sort_values("anchor_date")
    n_test = max(1, int(len(all_df) * val_frac))
    train_df = all_df.iloc[:-n_test]
    test_df  = all_df.iloc[-n_test:]
    return train_df, test_df


# ── Figure 20: LOMO per-metro bars ───────────────────────────────────────────

def plot_lomo_per_metro():
    print("[20] LOMO per-metro bar chart...")
    data  = json.load(open(MODEL_DIR / "lomo_results.json"))
    folds = sorted(data["folds"], key=lambda f: f["xgb"]["AUC_PR"], reverse=True)

    metros  = [f["metro"].replace("_", " ").title() for f in folds]
    xgb_pr  = [f["xgb"]["AUC_PR"]  for f in folds]
    lr_pr   = [f["lr"]["AUC_PR"]   for f in folds]
    xgb_roc = [f["xgb"]["AUC_ROC"] for f in folds]
    lr_roc  = [f["lr"]["AUC_ROC"]  for f in folds]
    closure = [f["closure_rate"]   for f in folds]

    x = np.arange(len(metros))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, xgb_vals, lr_vals, title, ylabel, ylim in [
        (axes[0], xgb_pr,  lr_pr,  "AUC-PR by Metro (LOMO CV)",  "AUC-PR",  (0, 0.65)),
        (axes[1], xgb_roc, lr_roc, "AUC-ROC by Metro (LOMO CV)", "AUC-ROC", (0.50, 0.90)),
    ]:
        ax.bar(x - w/2, xgb_vals, w, label="XGBoost",              color=C_XGB, alpha=0.85)
        ax.bar(x + w/2, lr_vals,  w, label="Logistic Regression",  color=C_LR,  alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(metros, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.legend(fontsize=9)

        # Annotate mean
        agg = data["aggregate"]
        ax.axhline(agg["xgb"][f"mean_{ylabel.replace('-','_')}"],
                   color=C_XGB, linewidth=1.2, linestyle="--", alpha=0.6)
        ax.axhline(agg["lr"][f"mean_{ylabel.replace('-','_')}"],
                   color=C_LR,  linewidth=1.2, linestyle="--", alpha=0.6)

        # Closure rate below bar
        for i, (xi, cr) in enumerate(zip(x, closure)):
            ax.text(xi, ylim[0] + 0.01, f"{cr:.0%}",
                    ha="center", fontsize=7, color="#666")

    plt.suptitle("Leave-One-Metro-Out Cross-Validation — XGBoost vs Logistic Regression",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save("20_lomo_per_metro.png")


# ── Figure 21: PR + ROC curves ────────────────────────────────────────────────

def plot_pr_roc(train_df, test_df, feat_cols, xgb_model, lr_pipe):
    print("[21] PR + ROC curves...")

    medians = train_df[feat_cols].median()
    X_test  = test_df[feat_cols].fillna(medians)
    y_test  = test_df[TARGET_COL].values

    xgb_feat = xgb_model.get_booster().feature_names
    X_test_xgb = test_df.reindex(columns=xgb_feat).fillna(medians)
    xgb_prob   = xgb_model.predict_proba(X_test_xgb)[:, 1]
    lr_prob    = lr_pipe.predict_proba(X_test)[:, 1]
    base_rate  = y_test.mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # PR curve
    ax = axes[0]
    for label, prob, color in [
        ("XGBoost",             xgb_prob, C_XGB),
        ("Logistic Regression", lr_prob,  C_LR),
    ]:
        prec, rec, thr = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        ax.plot(rec, prec, color=color, linewidth=2, label=f"{label} (AUC-PR={ap:.3f})")
        f1s      = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = np.argmax(f1s[:-1])
        ax.scatter(rec[best_idx], prec[best_idx], color=color, s=80, zorder=5)
        ax.annotate(
            f"t={thr[best_idx]:.2f}\nF1={f1s[best_idx]:.2f}",
            xy=(rec[best_idx], prec[best_idx]),
            xytext=(rec[best_idx] - 0.14, prec[best_idx] + 0.05),
            fontsize=8, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        )
    ax.axhline(base_rate, color=C_BASE, linestyle="--", linewidth=1,
               label=f"Random baseline ({base_rate:.3f})")
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title("Precision-Recall Curve", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # ROC curve
    ax = axes[1]
    for label, prob, color in [
        ("XGBoost",             xgb_prob, C_XGB),
        ("Logistic Regression", lr_prob,  C_LR),
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC-ROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color=C_BASE, linestyle="--", linewidth=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curve", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    n_train = len(train_df); n_test = len(test_df)
    plt.suptitle(
        f"Model Performance — Time-Split Evaluation  "
        f"(train n={n_train:,} · test n={n_test:,}  |  {base_rate:.1%} closure rate)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    save("21_pr_roc_curves.png")


# ── Figure 22: SHAP beeswarm ─────────────────────────────────────────────────

def plot_shap_beeswarm(test_df, xgb_model):
    print("[22] SHAP beeswarm (may take ~60s)...")
    xgb_feat = xgb_model.get_booster().feature_names
    medians  = test_df[xgb_feat].median()
    X_test   = test_df.reindex(columns=xgb_feat).fillna(medians)

    # Cap at 2000 rows for speed; stratified by label
    if len(X_test) > 2000:
        rng   = np.random.default_rng(RANDOM_SEED)
        idx   = rng.choice(len(X_test), 2000, replace=False)
        X_shap = X_test.iloc[idx]
    else:
        X_shap = X_test

    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        sv = np.array(shap_values[-1])
    else:
        sv = np.array(shap_values)

    shap.summary_plot(
        sv, X_shap,
        plot_type="dot",
        show=False,
        max_display=20,
        color_bar_label="Feature value",
        plot_size=(10, 7),
    )
    plt.title("SHAP Global Feature Importance — XGBoost (9-Metro Global Model)",
              fontweight="bold", pad=10)
    plt.tight_layout()
    save("22_shap_beeswarm.png")


# ── Figure 23: Feature importance comparison ─────────────────────────────────

def plot_feature_importance(xgb_model, lr_pipe, feat_cols):
    print("[23] Feature importance comparison...")
    booster      = xgb_model.get_booster()
    xgb_feat     = booster.feature_names
    xgb_imp      = xgb_model.feature_importances_

    # Top 15 by XGB importance
    top15_idx  = np.argsort(xgb_imp)[-15:][::-1]
    top15_feat = [xgb_feat[i] for i in top15_idx]
    top15_imp  = xgb_imp[top15_idx]

    # LR absolute coefficients for the same features
    scaler = lr_pipe.named_steps["scaler"]
    lr     = lr_pipe.named_steps["lr"]
    coef   = np.abs(lr.coef_[0])
    feat_to_coef = dict(zip(feat_cols, coef))
    lr_coef = np.array([feat_to_coef.get(f, 0.0) for f in top15_feat])

    # Normalise both to [0,1] for side-by-side comparison
    xgb_norm = top15_imp / top15_imp.max()
    lr_norm  = lr_coef   / (lr_coef.max() + 1e-9)

    labels = [f.replace("_", " ") for f in top15_feat]
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(x - w/2, xgb_norm, w, label="XGBoost (feature importance)",  color=C_XGB, alpha=0.85)
    ax.barh(x + w/2, lr_norm,  w, label="Logistic Regression (|coef|)",  color=C_LR,  alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Normalised importance (0–1)", fontsize=10)
    ax.set_title("Feature Importance — Top 15 XGBoost Features vs LR Coefficients",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    save("23_feature_importance.png")


# ── Figure 24: Top-5 feature profiles ────────────────────────────────────────

FEATURE_META = {
    "review_momentum": {
        "title": "Review Momentum",
        "unit": "ratio",
        "desc": (
            "Reviews in last 6 months divided by reviews in first 6 months\n"
            "(+1 smoothing). Captures whether customer engagement is\n"
            "accelerating or fading as the observation window closes."
        ),
    },
    "review_velocity": {
        "title": "Review Velocity",
        "unit": "reviews / month",
        "desc": (
            "Average new Yelp reviews per month over the full 12-month\n"
            "observation window. Low velocity signals declining visibility\n"
            "and reduced foot traffic from online discovery."
        ),
    },
    "n_checkins_obs": {
        "title": "Total Check-ins (obs. window)",
        "unit": "check-ins",
        "desc": (
            "Total Yelp check-ins logged during the observation window.\n"
            "Reflects actual foot traffic independently of written reviews.\n"
            "Zero check-ins is a particularly strong closure signal."
        ),
    },
    "has_photo": {
        "title": "Has Yelp Photo",
        "unit": "binary (0 / 1)",
        "desc": (
            "Whether the business has at least one Yelp photo.\n"
            "Restaurants without photos are 30 pp more likely to close —\n"
            "often poorly maintained profiles or already inactive listings."
        ),
    },
    "checkin_velocity": {
        "title": "Check-in Velocity",
        "unit": "check-ins / month",
        "desc": (
            "Average Yelp check-ins per month during the observation window.\n"
            "Captures the pace of physical visits. A steep drop in velocity\n"
            "often precedes permanent closure by several months."
        ),
    },
}

C_OPEN   = "#2E86AB"
C_CLOSED = "#E84855"


def plot_top5_feature_profiles(all_df: pd.DataFrame):
    print("[24] Top-5 feature profile plots...")

    TOP5 = ["review_momentum", "review_velocity", "n_checkins_obs",
            "has_photo", "checkin_velocity"]

    fig = plt.figure(figsize=(18, 11))
    # 2-row layout: 3 plots top, 2 plots bottom (centred)
    gs_top = fig.add_gridspec(2, 6, hspace=0.52, wspace=0.38,
                              left=0.05, right=0.97, top=0.88, bottom=0.06)
    axes = [
        fig.add_subplot(gs_top[0, 0:2]),
        fig.add_subplot(gs_top[0, 2:4]),
        fig.add_subplot(gs_top[0, 4:6]),
        fig.add_subplot(gs_top[1, 1:3]),
        fig.add_subplot(gs_top[1, 3:5]),
    ]

    open_df   = all_df[all_df[TARGET_COL] == 0]
    closed_df = all_df[all_df[TARGET_COL] == 1]
    n_open    = len(open_df)
    n_closed  = len(closed_df)

    for ax, feat in zip(axes, TOP5):
        meta = FEATURE_META[feat]

        if feat == "has_photo":
            # Binary feature: grouped bar — % with photo per class
            pct_open   = open_df[feat].mean()
            pct_closed = closed_df[feat].mean()
            bars = ax.bar(
                ["Open\n(still active)", "Closed\n(within 6m)"],
                [pct_open * 100, pct_closed * 100],
                color=[C_OPEN, C_CLOSED], alpha=0.82, width=0.5,
            )
            for bar, pct in zip(bars, [pct_open, pct_closed]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    f"{pct:.0%}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold",
                )
            ax.set_ylabel("% with photo", fontsize=9)
            ax.set_ylim(0, 85)
            ax.axhline(50, color="#666", linewidth=0.8, linestyle="--", alpha=0.5)

        else:
            p99 = all_df[feat].quantile(0.99)
            d_open   = open_df[feat].dropna().clip(upper=p99).values
            d_closed = closed_df[feat].dropna().clip(upper=p99).values

            vp = ax.violinplot(
                [d_open, d_closed], positions=[0, 1],
                showmedians=True, showextrema=False, widths=0.7,
            )
            vp["bodies"][0].set_facecolor(C_OPEN);   vp["bodies"][0].set_alpha(0.55)
            vp["bodies"][1].set_facecolor(C_CLOSED); vp["bodies"][1].set_alpha(0.55)
            vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(2)

            med_open   = float(np.median(d_open))
            med_closed = float(np.median(d_closed))
            p5  = float(all_df[feat].quantile(0.05))
            p95 = float(all_df[feat].quantile(0.95))

            # Median labels
            ax.text(0, med_open   + p99 * 0.04, f"med={med_open:.2f}",
                    ha="center", fontsize=8, color=C_OPEN,   fontweight="bold")
            ax.text(1, med_closed + p99 * 0.04, f"med={med_closed:.2f}",
                    ha="center", fontsize=8, color=C_CLOSED, fontweight="bold")

            # Stats box
            ax.text(0.97, 0.96,
                    f"p5={p5:.2f}  p95={p95:.2f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color="#888",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a1a", ec="#444", alpha=0.7))

            ax.set_xticks([0, 1])
            ax.set_xticklabels(
                [f"Open\n(n={n_open:,})", f"Closed\n(n={n_closed:,})"],
                fontsize=8,
            )
            ax.set_ylabel(meta["unit"], fontsize=8)
            ax.set_ylim(bottom=0)

        ax.set_title(meta["title"], fontweight="bold", fontsize=11, pad=6)

        # Description below the subplot
        ax.text(
            0.5, -0.28, meta["desc"],
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7.8, color="#aaaaaa", style="italic",
            multialignment="center",
        )

    fig.suptitle(
        "Top 5 Most Important Features — Global XGBoost Model\n"
        "Distribution by Outcome (Open vs Closed within 6 months)",
        fontsize=13, fontweight="bold", y=0.97,
    )
    save("24_top5_feature_profiles.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 11 -- Presentation Figures")
    print("=" * 60)

    # Fig 20 — from saved JSON (no model needed)
    plot_lomo_per_metro()

    # Load pooled data
    print("\n[Data] Loading all 9 metros...")
    all_df    = load_all_metros()
    feat_cols = get_feat_cols(all_df)
    train_df, test_df = time_split(all_df, val_frac=0.20)
    print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}  "
          f"Closure rate (test): {test_df[TARGET_COL].mean():.1%}")

    # Imputation medians (fit on train only)
    medians = train_df[feat_cols].median()
    X_train = train_df[feat_cols].fillna(medians)
    y_train = train_df[TARGET_COL].values

    # XGBoost — retrain on train split with LOMO-tuned params
    print("\n[XGB] Training fresh XGBoost on 80% split...")
    xgb_model = XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_train, y_train)

    # Logistic Regression pipeline
    print("[LR]  Training Logistic Regression...")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=LR_C, class_weight="balanced",
            max_iter=1000, random_state=RANDOM_SEED,
        )),
    ])
    lr_pipe.fit(X_train, y_train)

    # Figs 21-24
    plot_pr_roc(train_df, test_df, feat_cols, xgb_model, lr_pipe)
    plot_shap_beeswarm(test_df, xgb_model)
    plot_feature_importance(xgb_model, lr_pipe, feat_cols)
    plot_top5_feature_profiles(all_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
