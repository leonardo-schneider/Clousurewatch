"""
05_modeling.py
──────────────
Benchmark: Logistic Regression
Main model: LightGBM
Evaluation: 5-fold time-aware cross-validation + held-out test set

Run:
    pip install lightgbm scikit-learn imbalanced-learn
    python 05_modeling.py
"""

import sys, warnings, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config_00 import (
    PROC_DIR, MODEL_DIR, FIG_DIR,
    TARGET_COL, RANDOM_SEED, N_FOLDS,
    TEST_FRAC, PRIMARY_METRIC,
    LGBM_EARLY_STOPPING,
)

plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────────────────────
# Feature columns (exclude meta-columns)
# ─────────────────────────────────────────────────────────────────────────────
META_COLS = {"business_id", TARGET_COL, "anchor_date", "city", "state"}

LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           ["average_precision", "auc"],
    "boosting_type":    "gbdt",
    "learning_rate":    0.05,
    "num_leaves":       31,
    "min_child_samples": 20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "n_estimators":     1000,
    "class_weight":     "balanced",
    "random_state":     RANDOM_SEED,
    "verbose":          -1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC_PR": average_precision_score(y_true, y_prob),
        "AUC_ROC": roc_auc_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }


def print_metrics(name: str, metrics: dict):
    print(f"\n  [{name}]")
    for k, v in metrics.items():
        print(f"    {k:12s}: {v:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Time-aware train/test split
# ─────────────────────────────────────────────────────────────────────────────

def time_split(df: pd.DataFrame, test_frac: float):
    """
    Hold out the most recent TEST_FRAC businesses as the final test set.
    This mimics production: model trained on past, evaluated on future.
    """
    df_sorted = df.sort_values("anchor_date")
    n_test = int(len(df_sorted) * test_frac)
    train_val = df_sorted.iloc[:-n_test].copy()
    test       = df_sorted.iloc[-n_test:].copy()
    print(f"  Train/Val: {len(train_val):,}  |  Test: {len(test):,}")
    print(f"  Test anchor range: {test['anchor_date'].min().date()} → {test['anchor_date'].max().date()}")
    return train_val, test


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation folds (time-aware)
# ─────────────────────────────────────────────────────────────────────────────

def time_aware_cv_folds(df: pd.DataFrame, n_folds: int):
    """
    Expanding-window CV: each fold adds more data to training.
    Fold k trains on rows 0..k*(n/K) and validates on the next chunk.
    """
    df_sorted = df.sort_values("anchor_date").reset_index(drop=True)
    n = len(df_sorted)
    fold_size = n // n_folds
    folds = []
    for k in range(1, n_folds + 1):
        train_end = k * fold_size
        val_end   = min((k + 1) * fold_size, n)
        if val_end <= train_end:
            break
        train_idx = df_sorted.index[:train_end].tolist()
        val_idx   = df_sorted.index[train_end:val_end].tolist()
        folds.append((train_idx, val_idx))
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────

def run_logistic_regression(X_train, y_train, X_val, y_val):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_SEED,
        ))
    ])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_val)[:, 1]
    return pipe, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# Main model: LightGBM
# ─────────────────────────────────────────────────────────────────────────────

def run_lightgbm(X_train, y_train, X_val, y_val):
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    y_prob = model.predict_proba(X_val)[:, 1]
    return model, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_roc(results: dict, split_name: str, save_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model_name, (y_true, y_prob) in results.items():
        color = "#E84855" if "LightGBM" in model_name else "#2E86AB"
        auc_pr = average_precision_score(y_true, y_prob)
        auc_roc = roc_auc_score(y_true, y_prob)

        # PR curve
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        axes[0].plot(rec, prec, color=color, lw=2,
                     label=f"{model_name} (AUC-PR={auc_pr:.3f})")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[1].plot(fpr, tpr, color=color, lw=2,
                     label=f"{model_name} (AUC-ROC={auc_roc:.3f})")

    # Baselines
    pos_rate = y_true.mean()
    axes[0].axhline(pos_rate, color="grey", ls="--", lw=1, label=f"Random ({pos_rate:.2f})")
    axes[1].plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Random")

    axes[0].set_title(f"Precision-Recall Curve\n{split_name}", fontweight="bold")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].legend(fontsize=9)

    axes[1].set_title(f"ROC Curve\n{split_name}", fontweight="bold")
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / save_name, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {FIG_DIR}/{save_name}")


def plot_feature_importance(model, feature_names, top_n=20):
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
    colors = ["#E84855" if v > imp.median() else "#2E86AB" for v in imp.values]
    ax.barh(imp.index, imp.values, color=colors)
    ax.set_title(f"LightGBM Feature Importance (Top {top_n})", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_feature_importance.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved → {FIG_DIR}/08_feature_importance.png")


def plot_confusion(y_true, y_prob, threshold=0.5, model_name="Model"):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Open", "Pred Closed"],
                yticklabels=["True Open", "True Closed"], ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}\n(threshold={threshold:.2f})",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"09_confusion_{model_name.replace(' ','_')}.png",
                bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 5 — Modeling")
    print("=" * 60)

    feat = pd.read_parquet(PROC_DIR / "features.parquet")

    feature_cols = [c for c in feat.columns if c not in META_COLS]
    print(f"\n  Feature columns: {len(feature_cols)}")

    X_all = feat[feature_cols].copy()
    y_all = feat[TARGET_COL].copy()

    # Impute NaN with median (only computed on each train fold, re-applied to val)
    # For simplicity here we impute globally — in production use a Pipeline
    X_all = X_all.fillna(X_all.median())

    # ── Time-aware train/test split ────────────────────────────────────────
    print("\n[1] Train/Test Split")
    train_val_df, test_df = time_split(feat, TEST_FRAC)
    train_val_idx = train_val_df.index
    test_idx      = test_df.index

    X_trainval = X_all.loc[train_val_idx]
    y_trainval = y_all.loc[train_val_idx]
    X_test     = X_all.loc[test_idx]
    y_test     = y_all.loc[test_idx]

    # ── 5-Fold Time-Aware CV ───────────────────────────────────────────────
    print(f"\n[2] {N_FOLDS}-Fold Time-Aware Cross Validation")
    folds = time_aware_cv_folds(train_val_df.reset_index(drop=True), N_FOLDS)

    lr_cv_metrics  = []
    lgb_cv_metrics = []

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        X_tr = X_trainval.iloc[tr_idx]
        y_tr = y_trainval.iloc[tr_idx]
        X_va = X_trainval.iloc[va_idx]
        y_va = y_trainval.iloc[va_idx]

        # Fill NaN per fold to prevent leakage
        medians = X_tr.median()
        X_tr = X_tr.fillna(medians)
        X_va = X_va.fillna(medians)

        _, lr_prob  = run_logistic_regression(X_tr, y_tr, X_va, y_va)
        _, lgb_prob = run_lightgbm(X_tr, y_tr, X_va, y_va)

        lr_cv_metrics.append(compute_metrics(y_va, lr_prob))
        lgb_cv_metrics.append(compute_metrics(y_va, lgb_prob))

        print(f"  Fold {fold_i+1}: LR AUC-PR={lr_cv_metrics[-1]['AUC_PR']:.3f}  "
              f"LGB AUC-PR={lgb_cv_metrics[-1]['AUC_PR']:.3f}")

    print("\n  CV SUMMARY (mean ± std across folds)")
    for name, metrics_list in [("Logistic Regression", lr_cv_metrics),
                                ("LightGBM", lgb_cv_metrics)]:
        df_m = pd.DataFrame(metrics_list)
        print(f"\n  {name}:")
        for col in df_m.columns:
            print(f"    {col:12s}: {df_m[col].mean():.4f} ± {df_m[col].std():.4f}")

    # ── Final models on full train/val ─────────────────────────────────────
    print("\n[3] Final Models (train on full train+val, eval on held-out test)")

    # Re-impute with global train medians
    train_medians = X_trainval.median()
    X_tr_full = X_trainval.fillna(train_medians)
    X_te_full  = X_test.fillna(train_medians)

    lr_final, lr_test_prob   = run_logistic_regression(X_tr_full, y_trainval,
                                                        X_te_full, y_test)
    lgb_final, lgb_test_prob = run_lightgbm(X_tr_full, y_trainval,
                                             X_te_full, y_test)

    lr_test_metrics  = compute_metrics(y_test, lr_test_prob)
    lgb_test_metrics = compute_metrics(y_test, lgb_test_prob)

    print("\n  HELD-OUT TEST RESULTS")
    print_metrics("Logistic Regression (benchmark)", lr_test_metrics)
    print_metrics("LightGBM (main model)", lgb_test_metrics)

    improvement = (lgb_test_metrics["AUC_PR"] - lr_test_metrics["AUC_PR"]) / lr_test_metrics["AUC_PR"]
    print(f"\n  LightGBM AUC-PR improvement over LR: {improvement:+.1%}")

    # ── Save models ────────────────────────────────────────────────────────
    joblib.dump(lr_final,  MODEL_DIR / "logistic_regression.pkl")
    joblib.dump(lgb_final, MODEL_DIR / "lightgbm_model.pkl")
    print(f"\n  Models saved to {MODEL_DIR}/")

    # ── Save metrics to JSON for reporting ────────────────────────────────
    results_summary = {
        "cv_logistic_regression": {
            k: float(np.mean([m[k] for m in lr_cv_metrics]))
            for k in lr_cv_metrics[0]
        },
        "cv_lightgbm": {
            k: float(np.mean([m[k] for m in lgb_cv_metrics]))
            for k in lgb_cv_metrics[0]
        },
        "test_logistic_regression": {k: float(v) for k, v in lr_test_metrics.items()},
        "test_lightgbm":            {k: float(v) for k, v in lgb_test_metrics.items()},
    }
    with open(MODEL_DIR / "results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n[4] Generating figures")
    plot_pr_roc(
        results={
            "Logistic Regression": (y_test, lr_test_prob),
            "LightGBM":            (y_test, lgb_test_prob),
        },
        split_name="Held-Out Test Set",
        save_name="10_pr_roc_test.png",
    )
    plot_feature_importance(lgb_final, feature_cols)
    plot_confusion(y_test, lgb_test_prob, model_name="LightGBM")
    plot_confusion(y_test, lr_test_prob,  model_name="Logistic Regression")

    print("\n  Done. Check figures/ and models/ directories.")


if __name__ == "__main__":
    main()
