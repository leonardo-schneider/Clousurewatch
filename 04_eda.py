"""
04_eda.py
──────────
Exploratory Data Analysis — plots saved to figures/

Run:
    python 04_eda.py
"""

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config_00 import PROC_DIR, FIG_DIR, TARGET_COL

plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
PALETTE = {0: "#2E86AB", 1: "#E84855"}   # blue=open, red=closed


def save(name: str):
    p = FIG_DIR / name
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")


def main():
    print("=" * 60)
    print("STEP 4 — EDA")
    print("=" * 60)

    feat = pd.read_parquet(PROC_DIR / "features.parquet")
    labeled = pd.read_parquet(PROC_DIR / "labeled_businesses.parquet")

    # ── 1. Class balance ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = feat[TARGET_COL].value_counts()
    ax.bar(["Open (0)", "Closed (1)"], counts.values,
           color=[PALETTE[0], PALETTE[1]], edgecolor="white", width=0.5)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, f"{v:,}\n({v/len(feat):.1%})", ha="center", fontsize=10)
    ax.set_title("Class Distribution", fontweight="bold")
    ax.set_ylabel("Count")
    save("01_class_balance.png")

    # ── 2. Anchor date distribution ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    feat["anchor_date"].hist(bins=40, ax=ax, color="#555", edgecolor="white")
    ax.set_title("Distribution of Anchor Dates", fontweight="bold")
    ax.set_xlabel("Anchor Date")
    ax.set_ylabel("Count")
    save("02_anchor_date_dist.png")

    # ── 3. Key numeric features by label ──────────────────────────────────
    num_features = [
        "mean_stars_obs", "stars_delta_3m", "review_velocity",
        "review_velocity_slope", "days_since_last_review",
        "mean_vader", "vader_trend_slope", "pct_1star",
        "n_checkins_obs", "price_range",
    ]
    num_features = [f for f in num_features if f in feat.columns]

    n = len(num_features)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(num_features):
        ax = axes[i]
        for label, grp in feat.groupby(TARGET_COL):
            vals = grp[col].dropna()
            ax.hist(vals, bins=30, alpha=0.6, density=True,
                    color=PALETTE[label],
                    label="Closed" if label == 1 else "Open",
                    edgecolor="none")
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions: Open vs Closed", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save("03_feature_distributions.png")

    # ── 4. Correlation heatmap ─────────────────────────────────────────────
    corr_cols = num_features + [TARGET_COL]
    corr_data = feat[corr_cols].dropna()
    corr = corr_data.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.3, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    save("04_correlation_heatmap.png")

    # ── 5. Rating trend by label (violin) ────────────────────────────────
    plot_df = feat[["stars_delta_3m", TARGET_COL]].dropna()
    plot_df["Status"] = plot_df[TARGET_COL].map({0: "Open", 1: "Closed"})

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.violinplot(data=plot_df, x="Status", y="stars_delta_3m",
                   palette={"Open": PALETTE[0], "Closed": PALETTE[1]},
                   inner="quartile", ax=ax)
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_title("Rating Trend (Last 3m − First 3m)\nby Business Status", fontweight="bold")
    ax.set_ylabel("Stars Δ")
    save("05_rating_trend_violin.png")

    # ── 6. Review drought flag rate ────────────────────────────────────────
    drought = feat.groupby(TARGET_COL)["review_drought_flag"].mean().reset_index()
    drought["Status"] = drought[TARGET_COL].map({0: "Open", 1: "Closed"})

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(drought["Status"], drought["review_drought_flag"] * 100,
           color=[PALETTE[0], PALETTE[1]], width=0.4)
    ax.set_title("Review Drought Flag Rate\n(No reviews in last 90 days)", fontweight="bold")
    ax.set_ylabel("% of Businesses")
    for i, v in enumerate(drought["review_drought_flag"] * 100):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=11)
    save("06_drought_flag.png")

    # ── 7. City-level closure rate ─────────────────────────────────────────
    city_stats = (
        feat.groupby("city")[TARGET_COL]
        .agg(["sum", "count"])
        .assign(closure_rate=lambda x: x["sum"] / x["count"])
        .sort_values("closure_rate", ascending=False)
        .head(20)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E84855" if r > city_stats["closure_rate"].median() else "#2E86AB"
              for r in city_stats["closure_rate"]]
    ax.barh(city_stats["city"], city_stats["closure_rate"] * 100, color=colors)
    ax.axvline(city_stats["closure_rate"].median() * 100, color="black",
               ls="--", lw=1, label="Median")
    ax.set_xlabel("Closure Rate (%)")
    ax.set_title("Top 20 Cities by Closure Rate", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save("07_city_closure_rate.png")

    print(f"\n  All figures saved to {FIG_DIR}/")
    print("\n  KEY OBSERVATIONS TO HIGHLIGHT IN PRESENTATION:")
    for col in ["review_drought_flag", "stars_delta_3m", "vader_trend_slope"]:
        if col in feat.columns:
            g = feat.groupby(TARGET_COL)[col].mean()
            print(f"    {col}: Open={g.get(0, np.nan):.3f}  Closed={g.get(1, np.nan):.3f}")


if __name__ == "__main__":
    main()
