"""
07_model_analysis.py — Deep EDA and model analysis.
Reads features.parquet, ensemble_predictions.parquet, models/xgboost.pkl.
Saves 9 figures to figures/.

Run:
    python 07_model_analysis.py
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import shap
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

from config_00 import PROC_DIR, MODEL_DIR, FIG_DIR, TARGET_COL, COVID_START, COVID_END

plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
PALETTE = {0: "#2E86AB", 1: "#E84855"}

COVID_START = pd.Timestamp(COVID_START)
COVID_END   = pd.Timestamp(COVID_END)
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

    # Drop all columns that preds already has (except business_id join key)
    overlap = set(preds.columns) - {"business_id"}
    test = preds.merge(
        feat.drop(columns=list(overlap), errors="ignore"),
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


def plot_correlation_heatmap(feat: pd.DataFrame, model):
    """2c — Pearson correlation heatmap for top 15 features."""
    feat_cols = [c for c in feat.columns if c not in _META]
    importance = dict(zip(model.feature_names_in_, model.feature_importances_))
    top15 = sorted(
        [c for c in feat_cols if c in importance],
        key=lambda c: importance[c], reverse=True
    )[:15]

    corr = feat[top15].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax, cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.5, square=True,
        xticklabels=[c.replace("_", "\n") for c in top15],
        yticklabels=[c.replace("_", "\n") for c in top15],
    )
    ax.tick_params(labelsize=7)
    ax.set_title("Feature Correlation Matrix (top 15 by importance)",
                 fontweight="bold", pad=12)
    plt.tight_layout()
    save("13_correlation_heatmap.png")


def plot_covid_cohort(feat: pd.DataFrame):
    """2d — Compare key metrics between COVID and non-COVID anchor cohorts."""
    COMPARE_COLS = [
        "days_since_last_review", "review_velocity",
        "months_with_zero_reviews", "review_momentum",
        "checkin_momentum", "mean_vader",
    ]
    COMPARE_COLS = [c for c in COMPARE_COLS if c in feat.columns]

    n_cols = 3
    n_rows = int(np.ceil(len(COMPARE_COLS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5))
    axes = axes.flatten()

    covid_labels = {0: "Non-COVID", 1: "COVID (2020-03 to 2021-06)"}
    colors = {0: "#2E86AB", 1: "#E84855"}

    for i, col in enumerate(COMPARE_COLS):
        ax = axes[i]
        data = [
            feat.loc[feat["covid_flag"] == g, col].dropna()
            for g in [0, 1]
        ]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="white", linewidth=2))
        for patch, g in zip(bp["boxes"], [0, 1]):
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([covid_labels[0], covid_labels[1]], fontsize=8)
        ax.set_title(col.replace("_", " "), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

        # Annotate closure rates
        for j, g in enumerate([0, 1]):
            grp = feat[feat["covid_flag"] == g]
            rate = grp[TARGET_COL].mean()
            ax.text(j + 1, ax.get_ylim()[1] * 0.97,
                    f"close={rate:.0%}", ha="center", fontsize=7, color=colors[g])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("COVID vs Non-COVID Cohort — Feature Comparison",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    save("14_covid_cohort_comparison.png")


def plot_error_profile(test: pd.DataFrame, model):
    """2e/3c — Compare feature means across TP, FP, FN, TN groups."""
    y_true = test["closed_within_6m"]
    y_pred = test["predicted"]

    tp = test[(y_pred == 1) & (y_true == 1)]
    fp = test[(y_pred == 1) & (y_true == 0)]
    fn = test[(y_pred == 0) & (y_true == 1)]

    print(f"    TP={len(tp)}  FP={len(fp)}  FN={len(fn)}")

    feat_cols = [c for c in model.feature_names_in_ if c in test.columns]
    importance = dict(zip(model.feature_names_in_, model.feature_importances_))
    top8 = sorted(feat_cols, key=lambda c: importance.get(c, 0), reverse=True)[:8]

    groups = {"True Pos (caught)": tp, "False Pos (false alarm)": fp,
              "False Neg (missed)": fn}
    group_colors = {"True Pos (caught)": "#1DB954",
                    "False Pos (false alarm)": "#EF9F27",
                    "False Neg (missed)": "#E84855"}

    means = pd.DataFrame({
        name: grp[top8].mean()
        for name, grp in groups.items()
    })
    # Normalise by full-test-set std for comparability
    stds = test[top8].std().replace(0, 1)
    means_norm = means.div(stds, axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(top8))
    width = 0.25
    for k, (name, color) in enumerate(group_colors.items()):
        ax.bar(x + k * width, means_norm[name], width,
               label=f"{name} (n={len(groups[name])})",
               color=color, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.replace("_", "\n") for c in top8], fontsize=8)
    ax.set_ylabel("Mean (z-score relative to test set std)", fontsize=9)
    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8)
    ax.set_title("Error Profile: What distinguishes False Positives from False Negatives?",
                 fontweight="bold")
    plt.tight_layout()
    save("15_fp_fn_error_profile.png")


def plot_shap_global(test: pd.DataFrame, model):
    """3a — SHAP beeswarm summary for the full test set."""
    feat_cols = [c for c in model.feature_names_in_ if c in test.columns]
    X_test = test[feat_cols].copy()

    # Impute with test-set medians (same strategy as 05_modeling.py)
    X_test = X_test.fillna(X_test.median())

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # shap_values may be 2D (n_samples, n_features)
    if isinstance(shap_values, list):
        sv = np.array(shap_values[-1])
    else:
        sv = np.array(shap_values)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        sv, X_test,
        plot_type="dot",
        show=False,
        max_display=20,
        color_bar_label="Feature value",
    )
    plt.title("SHAP Global Feature Importance — Full Test Set",
              fontweight="bold", pad=10)
    plt.tight_layout()
    save("16_shap_global_summary.png")


def plot_pr_curve_full(test: pd.DataFrame, model):
    """3b — Precision-recall curve with threshold markers."""
    y_true = test["closed_within_6m"].values
    y_prob = test["risk_score"].values  # ensemble probability

    # Also compute XGBoost standalone probability
    feat_cols = [c for c in model.feature_names_in_ if c in test.columns]
    X = test[feat_cols].fillna(test[feat_cols].median())
    xgb_prob = model.predict_proba(X)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, prob, color in [
        ("Ensemble (stacking)", y_prob, "#1DB954"),
        ("XGBoost standalone", xgb_prob, "#2E86AB"),
    ]:
        prec, rec, thr = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        ax.plot(rec, prec, color=color, linewidth=2, label=f"{label} (AUC-PR={ap:.3f})")

        # Mark F1-optimal threshold
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = np.argmax(f1s[:-1])
        ax.scatter(rec[best_idx], prec[best_idx], color=color, s=100, zorder=5)
        ax.annotate(
            f"t={thr[best_idx]:.2f}\nF1={f1s[best_idx]:.3f}",
            xy=(rec[best_idx], prec[best_idx]),
            xytext=(rec[best_idx] - 0.12, prec[best_idx] + 0.06),
            fontsize=8, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        )

    base_rate = y_true.mean()
    ax.axhline(base_rate, color="#888", linestyle="--", linewidth=1,
               label=f"Random baseline (precision={base_rate:.3f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve with F1-Optimal Threshold",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    save("17_pr_curve_full.png")


def compare_xgb_vs_ensemble(test: pd.DataFrame, model):
    """3d — Print XGBoost vs ensemble comparison and recommendation."""
    y_true = test["closed_within_6m"].values
    y_ens  = test["risk_score"].values

    feat_cols = [c for c in model.feature_names_in_ if c in test.columns]
    X = test[feat_cols].fillna(test[feat_cols].median())
    y_xgb = model.predict_proba(X)[:, 1]

    def metrics(y_true, y_prob, label):
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = np.argmax(f1s[:-1])
        opt_t = thr[best_idx]
        ap = average_precision_score(y_true, y_prob)
        y_pred = (y_prob >= opt_t).astype(int)
        f1 = f1_score(y_true, y_pred)
        return {"model": label, "AUC-PR": round(ap, 4),
                "opt_threshold": round(float(opt_t), 4), "F1": round(f1, 4)}

    rows = [
        metrics(y_true, y_xgb, "XGBoost standalone"),
        metrics(y_true, y_ens,  "Ensemble (stacking)"),
    ]
    print("\n  === XGBoost vs Ensemble ===")
    for r in rows:
        print(f"  {r['model']:30s}  AUC-PR={r['AUC-PR']}  "
              f"F1={r['F1']}  threshold={r['opt_threshold']}")

    winner = max(rows, key=lambda r: r["AUC-PR"])
    print(f"\n  Recommendation: use {winner['model']} "
          f"(AUC-PR {winner['AUC-PR']} > other)")


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
    print("[2c] Correlation heatmap...")
    plot_correlation_heatmap(feat, model)

    print("[2d] COVID cohort analysis...")
    plot_covid_cohort(feat)

    print("[2e/3c] FP/FN error profile...")
    plot_error_profile(test, model)

    print("[3a] SHAP global summary (may take ~30s)...")
    plot_shap_global(test, model)

    print("[3b] Full PR curve + threshold markers...")
    plot_pr_curve_full(test, model)

    print("[3d] XGBoost vs ensemble comparison...")
    compare_xgb_vs_ensemble(test, model)


if __name__ == "__main__":
    main()
