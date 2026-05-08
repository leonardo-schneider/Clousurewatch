"""
12_kfold_experiment.py -- Random 10-fold CV vs LOMO CV comparison.

Uses the same XGB + LR hyperparameters as the LOMO global model so the
only variable is fold assignment: random split vs held-out-metro split.

Expected finding: random k-fold scores higher because each fold shares
city distributions between train and test — the model never has to
generalise to an unseen geography.

Output:
    models/kfold_results.json
    figures/28_kfold_vs_lomo.png

Run:
    python 12_kfold_experiment.py
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

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
from xgboost import XGBClassifier

from config_00 import TARGET_COL, RANDOM_SEED, MODEL_DIR, FIG_DIR

plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_KFOLD = "#1DB954"   # green
C_LOMO  = "#2E86AB"   # blue
C_LR_K  = "#F4A261"   # orange (LR k-fold)
C_LR_L  = "#E84855"   # red    (LR LOMO)

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

XGB_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=10, eval_metric="aucpr",
    random_state=RANDOM_SEED, verbosity=0,
)
LR_C = 10.0


def load_all_metros() -> pd.DataFrame:
    frames = []
    for metro, ddir in METROS.items():
        df = pd.read_parquet(Path(ddir) / "features.parquet")
        df["metro"] = metro
        df["anchor_date"] = pd.to_datetime(df["anchor_date"])
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    return all_df[all_df["anchor_date"] <= LATEST_ANCHOR].copy()


def get_feat_cols(df):
    return [c for c in df.columns if c not in META_COLS]


def eval_fold(model, X_test, y_test):
    prob    = model.predict_proba(X_test)[:, 1]
    auc_pr  = float(average_precision_score(y_test, prob))
    auc_roc = float(roc_auc_score(y_test, prob))
    prec, rec, thr = precision_recall_curve(y_test, prob)
    f1s    = 2 * prec * rec / (prec + rec + 1e-9)
    opt    = float(thr[np.argmax(f1s[:-1])])
    f1     = float(f1_score(y_test, (prob >= opt).astype(int)))
    return {"AUC_PR": round(auc_pr, 4), "AUC_ROC": round(auc_roc, 4), "F1": round(f1, 4)}


def run_kfold(all_df):
    print("[2] Running 10-fold stratified CV...")
    feat_cols = get_feat_cols(all_df)
    y = all_df[TARGET_COL].values

    skf    = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    folds  = []
    metros_in_fold = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(all_df, y)):
        train_df = all_df.iloc[train_idx]
        test_df  = all_df.iloc[test_idx]

        # Metro composition of this test fold
        metro_counts = test_df["metro"].value_counts().to_dict()

        # Fit imputation on train only
        medians  = train_df[feat_cols].median()
        X_train  = train_df[feat_cols].fillna(medians)
        y_train  = train_df[TARGET_COL].values
        X_test   = test_df[feat_cols].fillna(medians)
        y_test   = test_df[TARGET_COL].values

        # XGBoost
        xgb = XGBClassifier(**XGB_PARAMS)
        xgb.fit(X_train, y_train)
        xgb_metrics = eval_fold(xgb, X_test, y_test)

        # Logistic Regression
        lr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=LR_C, class_weight="balanced",
                max_iter=1000, random_state=RANDOM_SEED,
            )),
        ])
        lr_pipe.fit(X_train, y_train)
        lr_metrics = eval_fold(lr_pipe, X_test, y_test)

        folds.append({
            "fold": fold_i + 1,
            "n_train": len(train_df),
            "n_test":  len(test_df),
            "closure_rate": round(float(y_test.mean()), 4),
            "metro_mix": metro_counts,
            "xgb": xgb_metrics,
            "lr":  lr_metrics,
        })

        print(f"  Fold {fold_i+1:2d}  "
              f"[XGB] AUC-PR={xgb_metrics['AUC_PR']:.4f}  AUC-ROC={xgb_metrics['AUC_ROC']:.4f}  "
              f"[LR]  AUC-PR={lr_metrics['AUC_PR']:.4f}  AUC-ROC={lr_metrics['AUC_ROC']:.4f}")

    xgb_prs  = [f["xgb"]["AUC_PR"]  for f in folds]
    xgb_rocs = [f["xgb"]["AUC_ROC"] for f in folds]
    lr_prs   = [f["lr"]["AUC_PR"]   for f in folds]
    lr_rocs  = [f["lr"]["AUC_ROC"]  for f in folds]

    aggregate = {
        "xgb": {
            "mean_AUC_PR":  round(float(np.mean(xgb_prs)),  4),
            "std_AUC_PR":   round(float(np.std(xgb_prs)),   4),
            "mean_AUC_ROC": round(float(np.mean(xgb_rocs)), 4),
            "std_AUC_ROC":  round(float(np.std(xgb_rocs)),  4),
        },
        "lr": {
            "mean_AUC_PR":  round(float(np.mean(lr_prs)),   4),
            "std_AUC_PR":   round(float(np.std(lr_prs)),    4),
            "mean_AUC_ROC": round(float(np.mean(lr_rocs)),  4),
            "std_AUC_ROC":  round(float(np.std(lr_rocs)),   4),
        },
    }

    print(f"\n  [XGB] AUC-PR={aggregate['xgb']['mean_AUC_PR']:.4f}"
          f" +/-{aggregate['xgb']['std_AUC_PR']:.4f}"
          f"  AUC-ROC={aggregate['xgb']['mean_AUC_ROC']:.4f}"
          f" +/-{aggregate['xgb']['std_AUC_ROC']:.4f}")
    print(f"  [LR ] AUC-PR={aggregate['lr']['mean_AUC_PR']:.4f}"
          f" +/-{aggregate['lr']['std_AUC_PR']:.4f}"
          f"  AUC-ROC={aggregate['lr']['mean_AUC_ROC']:.4f}"
          f" +/-{aggregate['lr']['std_AUC_ROC']:.4f}")

    return {"folds": folds, "aggregate": aggregate}


def plot_comparison(kfold_results):
    print("\n[3] Plotting comparison figure...")
    lomo = json.load(open(MODEL_DIR / "lomo_results.json"))

    kf_agg  = kfold_results["aggregate"]
    lo_agg  = lomo["aggregate"]

    # ── Left panel: mean AUC-PR comparison bar ────────────────────────────────
    # ── Right panel: per-fold distributions (box/strip) ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel 1: grouped bar (AUC-PR + AUC-ROC for each method) ---
    ax = axes[0]
    methods   = ["XGB\n10-fold", "XGB\nLOMO", "LR\n10-fold", "LR\nLOMO"]
    means_pr  = [kf_agg["xgb"]["mean_AUC_PR"],  lo_agg["xgb"]["mean_AUC_PR"],
                 kf_agg["lr"]["mean_AUC_PR"],   lo_agg["lr"]["mean_AUC_PR"]]
    stds_pr   = [kf_agg["xgb"]["std_AUC_PR"],   lo_agg["xgb"]["std_AUC_PR"],
                 kf_agg["lr"]["std_AUC_PR"],    lo_agg["lr"]["std_AUC_PR"]]
    colors_pr = [C_KFOLD, C_LOMO, C_LR_K, C_LR_L]

    bars = ax.bar(methods, means_pr, color=colors_pr, alpha=0.82, width=0.5,
                  yerr=stds_pr, capsize=5, error_kw=dict(elinewidth=1.5, ecolor="#555"))
    for bar, m, s in zip(bars, means_pr, stds_pr):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Mean AUC-PR  (mean ± std across folds)", fontsize=10)
    ax.set_title("AUC-PR: Random 10-fold vs LOMO CV", fontweight="bold")
    ax.set_ylim(0, max(means_pr) * 1.35)

    # Annotate lift
    lift = (kf_agg["xgb"]["mean_AUC_PR"] - lo_agg["xgb"]["mean_AUC_PR"])
    ax.annotate(
        f"+{lift:.3f} optimism\nbias (k-fold vs LOMO)",
        xy=(0, kf_agg["xgb"]["mean_AUC_PR"]),
        xytext=(1.2, kf_agg["xgb"]["mean_AUC_PR"] + 0.04),
        fontsize=8.5, color="#ccc",
        arrowprops=dict(arrowstyle="->", color="#ccc", lw=0.9),
    )

    # --- Panel 2: per-fold AUC-PR distribution ---
    ax2 = axes[1]

    kf_xgb_pr = [f["xgb"]["AUC_PR"] for f in kfold_results["folds"]]
    lo_xgb_pr = [f["xgb"]["AUC_PR"] for f in lomo["folds"]]
    kf_lr_pr  = [f["lr"]["AUC_PR"]  for f in kfold_results["folds"]]
    lo_lr_pr  = [f["lr"]["AUC_PR"]  for f in lomo["folds"]]

    positions = [1, 2, 3.5, 4.5]
    data      = [kf_xgb_pr, lo_xgb_pr, kf_lr_pr, lo_lr_pr]
    colors_bx = [C_KFOLD, C_LOMO, C_LR_K, C_LR_L]
    labels_bx = ["XGB 10-fold", "XGB LOMO", "LR 10-fold", "LR LOMO"]

    bp = ax2.boxplot(data, positions=positions, patch_artist=True,
                     widths=0.6, medianprops=dict(color="white", linewidth=2),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2),
                     flierprops=dict(marker="o", markersize=4, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors_bx):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Overlay individual points
    for pos, vals, color in zip(positions, data, colors_bx):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax2.scatter(pos + jitter, vals, color=color, s=28, alpha=0.7, zorder=5)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels_bx, fontsize=9)
    ax2.set_ylabel("AUC-PR per fold", fontsize=10)
    ax2.set_title("Per-Fold AUC-PR Distribution", fontweight="bold")
    ax2.set_xlim(0.2, 5.3)

    plt.suptitle(
        "Random 10-Fold vs Leave-One-Metro-Out: How Much Does Geography Matter?\n"
        "Same hyperparameters, same data — only the fold assignment differs",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    p = FIG_DIR / "28_kfold_vs_lomo.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p}")


def main():
    print("=" * 60)
    print("STEP 12 -- Random 10-Fold vs LOMO Experiment")
    print("=" * 60)

    print("[1] Loading all 9 metros...")
    all_df = load_all_metros()
    print(f"  Total: {len(all_df):,} restaurants  "
          f"Closed: {all_df[TARGET_COL].sum():,} ({all_df[TARGET_COL].mean():.1%})")

    kfold_results = run_kfold(all_df)

    out_path = MODEL_DIR / "kfold_results.json"
    with open(out_path, "w") as f:
        json.dump(kfold_results, f, indent=2, default=lambda x: None)
    print(f"\n  Results saved -> {out_path}")

    plot_comparison(kfold_results)

    # Summary
    kf = kfold_results["aggregate"]
    lo = json.load(open(MODEL_DIR / "lomo_results.json"))["aggregate"]
    print("\n  === SUMMARY ===")
    print(f"  {'Method':20s}  AUC-PR           AUC-ROC")
    print(f"  {'XGB 10-fold':20s}  {kf['xgb']['mean_AUC_PR']:.4f}+/-{kf['xgb']['std_AUC_PR']:.4f}"
          f"  {kf['xgb']['mean_AUC_ROC']:.4f}+/-{kf['xgb']['std_AUC_ROC']:.4f}")
    print(f"  {'XGB LOMO':20s}  {lo['xgb']['mean_AUC_PR']:.4f}+/-{lo['xgb']['std_AUC_PR']:.4f}"
          f"  {lo['xgb']['mean_AUC_ROC']:.4f}+/-{lo['xgb']['std_AUC_ROC']:.4f}")
    print(f"  {'LR 10-fold':20s}  {kf['lr']['mean_AUC_PR']:.4f}+/-{kf['lr']['std_AUC_PR']:.4f}"
          f"  {kf['lr']['mean_AUC_ROC']:.4f}+/-{kf['lr']['std_AUC_ROC']:.4f}")
    print(f"  {'LR LOMO':20s}  {lo['lr']['mean_AUC_PR']:.4f}+/-{lo['lr']['std_AUC_PR']:.4f}"
          f"  {lo['lr']['mean_AUC_ROC']:.4f}+/-{lo['lr']['std_AUC_ROC']:.4f}")
    xgb_bias = kf["xgb"]["mean_AUC_PR"] - lo["xgb"]["mean_AUC_PR"]
    lr_bias  = kf["lr"]["mean_AUC_PR"]  - lo["lr"]["mean_AUC_PR"]
    print(f"\n  Optimism bias (k-fold minus LOMO):  XGB={xgb_bias:+.4f}  LR={lr_bias:+.4f}")
    print("  -> k-fold inflates AUC-PR because test folds share city distributions with train.")
    print("     LOMO is the correct evaluation for real-world cross-city deployment.")


if __name__ == "__main__":
    main()
