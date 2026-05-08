"""
05_modeling.py -- Final model: Logistic Regression benchmark + XGBoost.

Pipeline:
  1. Load features.parquet for all 9 metros
  2. Time-based 80/20 train/test split (no random splits -- anti-leakage)
  3. 5-fold StratifiedKFold CV on training set to tune hyperparameters
  4. Retrain both models on full training set with best params
  5. Report metrics: CV folds, train set, test set
  6. Save figures: PR curves, ROC curves (train vs test, per model)
  7. Save models: xgboost_final.pkl, logistic_regression_final.pkl

Primary metric: AUC-PR (correct for imbalanced data; 10% closure rate).
Secondary metric: AUC-ROC, F1.
"""
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_curve, roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config_00 import TARGET_COL, MODEL_DIR, FIG_DIR, RANDOM_SEED
from app_helpers import add_null_flags

plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Constants ─────────────────────────────────────────────────────────────────
LATEST_ANCHOR = pd.Timestamp("2020-06-01")
TEST_FRAC     = 0.20
N_FOLDS       = 5
META_COLS     = {"business_id", TARGET_COL, "anchor_date", "city", "state", "metro"}

METRO_DIRS = {
    "tampa":         Path("data/processed"),
    "philadelphia":  Path("data/processed_philly"),
    "indianapolis":  Path("data/processed_indianapolis"),
    "tucson":        Path("data/processed_tucson"),
    "nashville":     Path("data/processed_nashville"),
    "new_orleans":   Path("data/processed_new_orleans"),
    "saint_louis":   Path("data/processed_saint_louis"),
    "reno":          Path("data/processed_reno"),
    "boise":         Path("data/processed_boise"),
}

XGB_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr,
     "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": w,
     "scale_pos_weight": 10, "eval_metric": "aucpr",
     "random_state": RANDOM_SEED, "verbosity": 0}
    for n in [300, 500]
    for d in [3, 4, 6]
    for lr in [0.05, 0.1]
    for w in [3, 5, 10]
]
LR_C_GRID = [0.01, 0.1, 1.0, 10.0]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_all() -> pd.DataFrame:
    frames = []
    for metro, d in METRO_DIRS.items():
        p = d / "features.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p} -- run 03_feature_engineering.py first")
        df = pd.read_parquet(p)
        df = add_null_flags(df)
        emb = d / "review_embeddings.parquet"
        if emb.exists():
            df = df.merge(pd.read_parquet(emb), on="business_id", how="left")
        df["metro"]       = metro
        df["anchor_date"] = pd.to_datetime(df["anchor_date"])
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    return all_df[all_df["anchor_date"] <= LATEST_ANCHOR].copy()


def time_split(df: pd.DataFrame):
    df_s   = df.sort_values("anchor_date").reset_index(drop=True)
    n_test = max(1, int(len(df_s) * TEST_FRAC))
    return df_s.iloc[:-n_test].copy(), df_s.iloc[-n_test:].copy()


# ── 5-fold CV ─────────────────────────────────────────────────────────────────
def kfold_tune(X_train: np.ndarray, y_train: np.ndarray):
    """
    StratifiedKFold on training data.
    Returns (best_xgb_params, best_lr_c, xgb_fold_scores, lr_fold_scores).
    Medians are refit per fold to prevent leakage.
    """
    skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=False)
    xgb_scores = np.zeros(len(XGB_GRID))
    lr_scores  = np.zeros(len(LR_C_GRID))
    fold_xgb   = []
    fold_lr    = []

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr,  X_val  = X_train[tr_idx], X_train[val_idx]
        y_tr,  y_val  = y_train[tr_idx], y_train[val_idx]

        med     = np.nanmedian(X_tr, axis=0)
        X_tr_f  = np.where(np.isnan(X_tr),  med, X_tr)
        X_val_f = np.where(np.isnan(X_val), med, X_val)

        best_fold_xgb = -1.0
        for pi, params in enumerate(XGB_GRID):
            m  = XGBClassifier(**params)
            m.fit(X_tr_f, y_tr, verbose=False)
            pr = average_precision_score(y_val, m.predict_proba(X_val_f)[:, 1])
            xgb_scores[pi] += pr
            if pr > best_fold_xgb:
                best_fold_xgb = pr
        fold_xgb.append(round(best_fold_xgb, 4))

        sc       = StandardScaler()
        X_tr_sc  = sc.fit_transform(X_tr_f)
        X_val_sc = sc.transform(X_val_f)
        best_fold_lr = -1.0
        for ci, c in enumerate(LR_C_GRID):
            m  = LogisticRegression(C=c, class_weight="balanced", solver="lbfgs",
                                    max_iter=1000, random_state=RANDOM_SEED)
            m.fit(X_tr_sc, y_tr)
            pr = average_precision_score(y_val, m.predict_proba(X_val_sc)[:, 1])
            lr_scores[ci] += pr
            if pr > best_fold_lr:
                best_fold_lr = pr
        fold_lr.append(round(best_fold_lr, 4))

        print(f"  Fold {fold_i+1}/{N_FOLDS}  XGB={best_fold_xgb:.4f}  LR={best_fold_lr:.4f}")

    best_xgb = XGB_GRID[int(np.argmax(xgb_scores))]
    best_c   = LR_C_GRID[int(np.argmax(lr_scores))]
    return best_xgb, best_c, fold_xgb, fold_lr


# ── Metric helpers ────────────────────────────────────────────────────────────
def metrics(y_true, prob):
    auc_pr  = float(average_precision_score(y_true, prob))
    auc_roc = float(roc_auc_score(y_true, prob))
    p, r, thr = precision_recall_curve(y_true, prob)
    f1s     = 2 * p * r / (p + r + 1e-9)
    opt_thr = float(thr[np.argmax(f1s[:-1])])
    f1      = float(f1_score(y_true, (prob >= opt_thr).astype(int)))
    return {"AUC_PR": round(auc_pr,4), "AUC_ROC": round(auc_roc,4),
            "F1": round(f1,4), "threshold": round(opt_thr,4)}


# ── Figures ───────────────────────────────────────────────────────────────────
def save_pr_fig(y_tr, y_te, p_tr, p_te, title, path):
    baseline_tr = y_tr.mean()
    baseline_te = y_te.mean()
    subtitle    = (f"Train n={len(y_tr):,} ({baseline_tr:.1%} closure) | "
                   f"Test n={len(y_te):,} ({baseline_te:.1%} closure)")
    prec_tr, rec_tr, _ = precision_recall_curve(y_tr, p_tr)
    prec_te, rec_te, _ = precision_recall_curve(y_te, p_te)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec_tr, prec_tr, color="#2E86AB", lw=2,
            label=f"Train  AUC-PR = {average_precision_score(y_tr, p_tr):.3f}")
    ax.plot(rec_te, prec_te, color="#E84855", lw=2, ls="--",
            label=f"Test   AUC-PR = {average_precision_score(y_te, p_te):.3f}")
    ax.axhline(baseline_tr, color="#2E86AB", ls=":", lw=1.2, alpha=0.5,
               label=f"Train baseline ({baseline_tr:.3f})")
    ax.axhline(baseline_te, color="#E84855", ls=":", lw=1.2, alpha=0.5,
               label=f"Test baseline ({baseline_te:.3f})")
    ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(f"Precision-Recall Curve\n{title} - Training vs Test", fontweight="bold")
    ax.legend(fontsize=9)
    fig.suptitle(subtitle, fontsize=9, y=0.01)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {path}")


def save_roc_fig(y_tr, y_te, p_tr, p_te, title, path):
    baseline_tr = y_tr.mean()
    baseline_te = y_te.mean()
    subtitle    = (f"Train n={len(y_tr):,} ({baseline_tr:.1%} closure) | "
                   f"Test n={len(y_te):,} ({baseline_te:.1%} closure)")
    fpr_tr, tpr_tr, _ = roc_curve(y_tr, p_tr)
    fpr_te, tpr_te, _ = roc_curve(y_te, p_te)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_tr, tpr_tr, color="#2E86AB", lw=2,
            label=f"Train  AUC-ROC = {roc_auc_score(y_tr, p_tr):.3f}")
    ax.plot(fpr_te, tpr_te, color="#E84855", lw=2, ls="--",
            label=f"Test   AUC-ROC = {roc_auc_score(y_te, p_te):.3f}")
    ax.plot([0,1],[0,1], color="gray", ls="--", lw=1.2, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(f"ROC Curve\n{title} - Training vs Test", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    fig.suptitle(subtitle, fontsize=9, y=0.01)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STEP 05 -- Final Model: LR Benchmark + XGBoost")
    print("=" * 60)

    print("\n[1] Loading all metro features...")
    all_df    = load_all()
    feat_cols = [c for c in all_df.columns if c not in META_COLS]
    print(f"  {len(all_df):,} restaurants  {int(all_df[TARGET_COL].sum())} closed "
          f"({all_df[TARGET_COL].mean():.1%})  {len(feat_cols)} features")

    print("\n[2] Time-based 80/20 train/test split...")
    train_df, test_df = time_split(all_df)
    medians   = train_df[feat_cols].median()
    X_train   = train_df[feat_cols].fillna(medians).values
    y_train   = train_df[TARGET_COL].values
    X_test    = test_df[feat_cols].fillna(medians).values
    y_test    = test_df[TARGET_COL].values
    print(f"  Train: {len(train_df):,}  ({train_df['anchor_date'].min().date()} "
          f"to {train_df['anchor_date'].max().date()})  closure={y_train.mean():.1%}")
    print(f"  Test:  {len(test_df):,}  ({test_df['anchor_date'].min().date()} "
          f"to {test_df['anchor_date'].max().date()})  closure={y_test.mean():.1%}")

    print(f"\n[3] 5-fold CV tuning ({len(XGB_GRID)} XGB combos x {N_FOLDS} folds)...")
    best_xgb_params, best_lr_c, fold_xgb, fold_lr = kfold_tune(X_train, y_train)
    print(f"\n  Best XGB: n={best_xgb_params['n_estimators']} "
          f"d={best_xgb_params['max_depth']} lr={best_xgb_params['learning_rate']} "
          f"mcw={best_xgb_params['min_child_weight']}")
    print(f"  Best LR C={best_lr_c}")
    print(f"  XGB fold AUC-PR: {fold_xgb}  mean={np.mean(fold_xgb):.4f}")
    print(f"  LR  fold AUC-PR: {fold_lr}   mean={np.mean(fold_lr):.4f}")

    print("\n[4] Retraining on full training set...")
    xgb_model = XGBClassifier(**best_xgb_params)
    xgb_model.fit(X_train, y_train, verbose=False)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    lr_model   = LogisticRegression(C=best_lr_c, class_weight="balanced",
                                    solver="lbfgs", max_iter=1000,
                                    random_state=RANDOM_SEED)
    lr_model.fit(X_train_sc, y_train)

    print("\n[5] Metrics across CV / Train / Test")
    prob_xgb_train = xgb_model.predict_proba(X_train)[:, 1]
    prob_xgb_test  = xgb_model.predict_proba(X_test)[:, 1]
    prob_lr_train  = lr_model.predict_proba(X_train_sc)[:, 1]
    prob_lr_test   = lr_model.predict_proba(X_test_sc)[:, 1]

    xgb_cv    = {"AUC_PR": round(float(np.mean(fold_xgb)),4),
                 "std":    round(float(np.std(fold_xgb)),4)}
    lr_cv     = {"AUC_PR": round(float(np.mean(fold_lr)),4),
                 "std":    round(float(np.std(fold_lr)),4)}
    xgb_train = metrics(y_train, prob_xgb_train)
    xgb_test  = metrics(y_test,  prob_xgb_test)
    lr_train  = metrics(y_train, prob_lr_train)
    lr_test   = metrics(y_test,  prob_lr_test)

    print(f"\n  {'Model':22s} {'Sample':8s} {'AUC-PR':>8} {'AUC-ROC':>9} {'F1':>7}")
    print("  " + "-" * 58)
    print(f"  {'Logistic Regression':22s} {'CV':8s} {lr_cv['AUC_PR']:8.4f} +/-{lr_cv['std']:.4f}")
    print(f"  {'':22s} {'Train':8s} {lr_train['AUC_PR']:8.4f} {lr_train['AUC_ROC']:9.4f} {lr_train['F1']:7.4f}")
    print(f"  {'':22s} {'Test':8s} {lr_test['AUC_PR']:8.4f} {lr_test['AUC_ROC']:9.4f} {lr_test['F1']:7.4f}")
    print(f"  {'XGBoost':22s} {'CV':8s} {xgb_cv['AUC_PR']:8.4f} +/-{xgb_cv['std']:.4f}")
    print(f"  {'':22s} {'Train':8s} {xgb_train['AUC_PR']:8.4f} {xgb_train['AUC_ROC']:9.4f} {xgb_train['F1']:7.4f}")
    print(f"  {'':22s} {'Test':8s} {xgb_test['AUC_PR']:8.4f} {xgb_test['AUC_ROC']:9.4f} {xgb_test['F1']:7.4f}")

    print("\n[6] Saving models...")
    joblib.dump(xgb_model, Path(MODEL_DIR) / "xgboost_final.pkl")
    joblib.dump(lr_model,  Path(MODEL_DIR) / "logistic_regression_final.pkl")
    joblib.dump(scaler,    Path(MODEL_DIR) / "lr_scaler_final.pkl")
    print(f"  Saved -> models/xgboost_final.pkl")
    print(f"  Saved -> models/logistic_regression_final.pkl")

    results = {
        "split": {"train_n": len(train_df), "test_n": len(test_df),
                  "train_closure_rate": round(float(y_train.mean()),4),
                  "test_closure_rate":  round(float(y_test.mean()),4),
                  "train_date_range": [str(train_df["anchor_date"].min().date()),
                                       str(train_df["anchor_date"].max().date())],
                  "test_date_range":  [str(test_df["anchor_date"].min().date()),
                                       str(test_df["anchor_date"].max().date())]},
        "cv_folds": N_FOLDS,
        "logistic_regression": {"cv": lr_cv, "train": lr_train, "test": lr_test,
                                 "best_C": best_lr_c},
        "xgboost": {"cv": xgb_cv, "train": xgb_train, "test": xgb_test,
                    "best_params": {k: v for k, v in best_xgb_params.items()
                                    if k not in ("eval_metric","random_state","verbosity")}},
    }
    out_json = Path(MODEL_DIR) / "final_results.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Saved -> {out_json}")

    print("\n[7] Generating figures...")
    save_pr_fig(y_train, y_test, prob_xgb_train, prob_xgb_test,
                "XGBoost", Path(FIG_DIR) / "35_train_pr_curve.png")
    save_roc_fig(y_train, y_test, prob_xgb_train, prob_xgb_test,
                 "XGBoost", Path(FIG_DIR) / "35b_train_roc_curve.png")
    save_pr_fig(y_train, y_test, prob_lr_train, prob_lr_test,
                "Logistic Regression", Path(FIG_DIR) / "36_train_pr_curve_lr.png")
    save_roc_fig(y_train, y_test, prob_lr_train, prob_lr_test,
                 "Logistic Regression", Path(FIG_DIR) / "36b_train_roc_curve_lr.png")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
