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


# ── Figure 25: Confusion matrices ────────────────────────────────────────────

def _opt_threshold(y_true, y_prob):
    """F1-optimal threshold."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[np.argmax(f1s[:-1])])


def plot_confusion_matrices(y_test, xgb_prob, lr_prob):
    from sklearn.metrics import confusion_matrix, classification_report
    print("[25] Confusion matrices...")

    thr_xgb = _opt_threshold(y_test, xgb_prob)
    thr_lr  = _opt_threshold(y_test, lr_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, prob, thr, label, color in [
        (axes[0], xgb_prob, thr_xgb, "XGBoost",             C_XGB),
        (axes[1], lr_prob,  thr_lr,  "Logistic Regression",  C_LR),
    ]:
        y_pred = (prob >= thr).astype(int)
        cm     = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        n = len(y_test)

        # Heatmap — row = actual, col = predicted
        im = ax.imshow(cm, cmap="Blues", aspect="auto")

        labels_rc = [["TN", "FP"], ["FN", "TP"]]
        for r in range(2):
            for c in range(2):
                val  = cm[r, c]
                pct  = val / n * 100
                tag  = labels_rc[r][c]
                dark = val > cm.max() / 2
                ax.text(c, r,
                        f"{tag}\n{val:,}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=12,
                        color="white" if dark else "#222",
                        fontweight="bold")

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nOpen", "Predicted\nClosed"], fontsize=10)
        ax.set_yticklabels(["Actual\nOpen", "Actual\nClosed"], fontsize=10)

        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        ap   = average_precision_score(y_test, prob)
        ax.set_title(
            f"{label}\n"
            f"threshold={thr:.2f}  |  Precision={prec:.2f}  Recall={rec:.2f}  F1={f1:.2f}  AUC-PR={ap:.3f}",
            fontweight="bold", fontsize=10,
        )

    plt.suptitle("Confusion Matrices at F1-Optimal Threshold — Time-Split Test Set",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save("25_confusion_matrices.png")


# ── Figure 26: Standalone ROC-AUC curve ──────────────────────────────────────

def plot_roc_standalone(y_test, xgb_prob, lr_prob):
    from sklearn.metrics import roc_curve, roc_auc_score
    print("[26] Standalone ROC-AUC curve...")

    fig, ax = plt.subplots(figsize=(7, 6))

    for label, prob, color in [
        ("XGBoost",             xgb_prob, C_XGB),
        ("Logistic Regression", lr_prob,  C_LR),
    ]:
        fpr, tpr, thr = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f"{label}  (AUC = {auc:.3f})")

        # Mark the point closest to (0,1)
        dist      = np.sqrt(fpr**2 + (1 - tpr)**2)
        best_idx  = np.argmin(dist)
        ax.scatter(fpr[best_idx], tpr[best_idx], color=color, s=90, zorder=5)
        ax.annotate(
            f"t={thr[best_idx]:.2f}",
            xy=(fpr[best_idx], tpr[best_idx]),
            xytext=(fpr[best_idx] + 0.05, tpr[best_idx] - 0.06),
            fontsize=8.5, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
        )

    ax.plot([0, 1], [0, 1], color=C_BASE, linestyle="--",
            linewidth=1.2, label="Random baseline (AUC = 0.500)")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color=C_BASE)

    ax.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate  (Sensitivity / Recall)", fontsize=11)
    ax.set_title("ROC-AUC Curve — Global Models on Time-Split Test Set",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    n_pos = int(y_test.sum())
    ax.text(0.98, 0.08,
            f"n={len(y_test):,}  |  {n_pos} closures ({y_test.mean():.1%})",
            transform=ax.transAxes, ha="right", fontsize=9, color="#888")
    plt.tight_layout()
    save("26_roc_auc_curve.png")


# ── Figure 27: FP / FN error profile ─────────────────────────────────────────

def plot_fp_fn_profile(test_df, xgb_prob, feat_cols):
    print("[27] FP/FN error profile...")

    thr    = _opt_threshold(test_df[TARGET_COL].values, xgb_prob)
    y_true = test_df[TARGET_COL].values
    y_pred = (xgb_prob >= thr).astype(int)

    mask_tp = (y_pred == 1) & (y_true == 1)
    mask_fp = (y_pred == 1) & (y_true == 0)
    mask_fn = (y_pred == 0) & (y_true == 1)

    tp_df = test_df[mask_tp]
    fp_df = test_df[mask_fp]
    fn_df = test_df[mask_fn]
    print(f"  TP={len(tp_df)}  FP={len(fp_df)}  FN={len(fn_df)}")

    # Top 10 features by XGB importance (from feat_cols, not booster names)
    import joblib
    xgb_saved = joblib.load(MODEL_DIR / "xgboost_global.pkl")
    imp_map   = dict(zip(xgb_saved.get_booster().feature_names,
                         xgb_saved.feature_importances_))
    top10 = sorted(
        [c for c in feat_cols if c in imp_map],
        key=lambda c: imp_map[c], reverse=True,
    )[:10]

    groups = {
        "True Pos — caught closures":    tp_df,
        "False Pos — false alarms":      fp_df,
        "False Neg — missed closures":   fn_df,
    }
    colors = {
        "True Pos — caught closures":   "#1DB954",
        "False Pos — false alarms":     "#EF9F27",
        "False Neg — missed closures":  "#E84855",
    }

    means = pd.DataFrame({
        name: grp[top10].mean() for name, grp in groups.items()
    })
    std_all = test_df[top10].std().replace(0, 1)
    means_z = means.div(std_all, axis=0)

    x     = np.arange(len(top10))
    width = 0.26
    fig, ax = plt.subplots(figsize=(14, 5))

    for k, (name, color) in enumerate(colors.items()):
        ax.bar(x + (k - 1) * width, means_z[name], width,
               label=f"{name}  (n={len(groups[name])})",
               color=color, alpha=0.82)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in top10], fontsize=8)
    ax.set_ylabel("Mean feature value (z-score vs test set)", fontsize=10)
    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=9)
    ax.set_title(
        f"Error Profile — What Distinguishes Caught vs Missed Closures?\n"
        f"XGBoost at threshold={thr:.2f}  |  top 10 features by importance",
        fontweight="bold",
    )
    plt.tight_layout()
    save("27_fp_fn_error_profile.png")


# ── Figure 29: Precision@K and lift curve ─────────────────────────────────────

def plot_precision_at_k(y_test: np.ndarray, xgb_prob: np.ndarray, lr_prob: np.ndarray):
    print("[29] Precision@K and lift curve...")
    base_rate = y_test.mean()
    K_values  = [10, 25, 50, 100, 200, 500]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel 1: Precision@K bar chart ───────────────────────────────────────
    ax = axes[0]
    x  = np.arange(len(K_values))
    w  = 0.35

    for offset, label, prob, color in [
        (-w/2, "XGBoost",             xgb_prob, C_XGB),
        ( w/2, "Logistic Regression", lr_prob,  C_LR),
    ]:
        sorted_idx = np.argsort(prob)[::-1]
        y_sorted   = y_test[sorted_idx]
        prec_at_k  = [y_sorted[:k].mean() for k in K_values]
        bars = ax.bar(x + offset, prec_at_k, w, label=label, color=color, alpha=0.85)
        for bar, p in zip(bars, prec_at_k):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{p:.0%}", ha="center", va="bottom", fontsize=8)

    ax.axhline(base_rate, color=C_BASE, linestyle="--", linewidth=1.2,
               label=f"Random baseline ({base_rate:.1%})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"@{k}" for k in K_values])
    ax.set_ylabel("Precision (fraction closed)", fontsize=10)
    ax.set_title("Precision@K — Top-K Riskiest Restaurants", fontweight="bold")
    ax.legend(fontsize=9)
    all_prec = [y_test[np.argsort(p)[::-1]][:k].mean()
                for p in [xgb_prob, lr_prob] for k in K_values]
    ax.set_ylim(0, min(1.0, max(all_prec) * 1.25))

    # ── Panel 2: Cumulative lift / gain curve ────────────────────────────────
    ax2   = axes[1]
    n     = len(y_test)
    steps = np.arange(1, n + 1)
    idx_20 = max(1, int(n * 0.20)) - 1

    gain_curves = {}
    for label, prob, color in [
        ("XGBoost",             xgb_prob, C_XGB),
        ("Logistic Regression", lr_prob,  C_LR),
    ]:
        sorted_idx   = np.argsort(prob)[::-1]
        y_sorted     = y_test[sorted_idx]
        cum_closures = np.cumsum(y_sorted)
        total_closed = y_test.sum()
        gain         = cum_closures / total_closed
        gain_curves[label] = (gain, color)
        ax2.plot(steps / n * 100, gain * 100, color=color, linewidth=2, label=label)

    ax2.plot([0, 100], [0, 100], color=C_BASE, linestyle="--", linewidth=1.2,
             label="Random baseline")
    ax2.fill_between([0, 100], [0, 100], alpha=0.04, color=C_BASE)

    for label, (gain, color) in gain_curves.items():
        gain_20 = gain[idx_20]
        ax2.scatter(20, gain_20 * 100, color=color, s=70, zorder=5)
        ax2.annotate(f"{gain_20:.0%}", xy=(20, gain_20 * 100),
                     xytext=(23, gain_20 * 100 - 4),
                     fontsize=8, color=color)

    ax2.set_xlabel("% of restaurants reviewed (sorted by risk)", fontsize=10)
    ax2.set_ylabel("% of closures captured", fontsize=10)
    ax2.set_title("Cumulative Gain (Lift) Curve", fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.set_xlim(0, 100); ax2.set_ylim(0, 100)

    n_pos = int(y_test.sum())
    plt.suptitle(
        f"Ranking Quality — Time-Split Test Set  "
        f"(n={n:,} · {n_pos} closures · base rate={base_rate:.1%})",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    save("29_precision_at_k.png")


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

    # Compute test-set probabilities once, reused by figs 21-27
    xgb_feat = xgb_model.get_booster().feature_names
    X_test_xgb = test_df.reindex(columns=xgb_feat).fillna(medians)
    X_test_lr  = test_df[feat_cols].fillna(medians)
    y_test     = test_df[TARGET_COL].values
    xgb_prob   = xgb_model.predict_proba(X_test_xgb)[:, 1]
    lr_prob    = lr_pipe.predict_proba(X_test_lr)[:, 1]

    # Figs 21-27
    plot_pr_roc(train_df, test_df, feat_cols, xgb_model, lr_pipe)
    plot_shap_beeswarm(test_df, xgb_model)
    plot_feature_importance(xgb_model, lr_pipe, feat_cols)
    plot_top5_feature_profiles(all_df)
    plot_confusion_matrices(y_test, xgb_prob, lr_prob)
    plot_roc_standalone(y_test, xgb_prob, lr_prob)
    plot_fp_fn_profile(test_df, xgb_prob, feat_cols)
    plot_precision_at_k(y_test, xgb_prob, lr_prob)

    print("\nDone.")


if __name__ == "__main__":
    main()
