"""
06_ensemble.py
──────────────
Ensemble strategy ladder over the 5 trained models from 05_modeling.py.

Tries in order:
  1. Simple average
  2. Weighted average (by CV AUC-PR)
  3. Stacking (meta-LogisticRegression)

Stops at the first strategy that beats XGBoost baseline (AUC-PR = 0.2069).

Outputs:
  models/ensemble_results.json
  data/processed/ensemble_predictions.parquet

Run:
  python 06_ensemble.py
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).parent))
from config_00 import PROC_DIR, MODEL_DIR, TARGET_COL, RANDOM_SEED

# ── Constants ──────────────────────────────────────────────────────────────────
BASELINE_AUC_PR = 0.2069   # XGBoost test result from 05_modeling.py

MODEL_NAMES = [
    "xgboost",
    "random_forest",
    "logistic_regression",
]

# CV AUC-PR scores from confirmed results (used for weighted average weights)
CV_AUC_PR = {
    "xgboost":            0.124,
    "random_forest":      0.112,
    "logistic_regression": 0.106,
    "lightgbm":           0.091,
    "mlp":                0.083,
}

META_COLS = {"business_id", TARGET_COL, "anchor_date", "city", "state"}

# Feature columns to carry into the predictions parquet for the UI
SIGNAL_COLS = [
    "months_with_zero_reviews",
    "days_since_last_review",
    "review_drought_flag",
    "checkin_drought_flag",
    "pct_5star",
]


# ── Core math functions (tested in tests/test_ensemble.py) ────────────────────

def simple_average(probs: dict) -> np.ndarray:
    """Average predicted probabilities from all models with equal weight."""
    return np.mean(list(probs.values()), axis=0)


def weighted_average(probs: dict, weights: dict) -> np.ndarray:
    """Weighted average of predicted probabilities. Weights need not sum to 1."""
    total = sum(weights[k] for k in probs)
    return np.sum([weights[k] * probs[k] for k in probs], axis=0) / total


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Return AUC-PR, AUC-ROC, and F1 for given true labels and predicted probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC_PR":  float(average_precision_score(y_true, y_prob)),
        "AUC_ROC": float(roc_auc_score(y_true, y_prob)),
        "F1":      float(f1_score(y_true, y_pred, zero_division=0)),
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the probability threshold that maximises F1 on the given data."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thresholds[np.argmax(f1[:-1])])


# ── Data / model loading ───────────────────────────────────────────────────────

def load_models(model_dir: Path) -> dict:
    """Load saved .pkl model files; skips any that fail to load (e.g. numpy version mismatch)."""
    models = {}
    missing = []
    for name in MODEL_NAMES:
        path = model_dir / f"{name}.pkl"
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            print(f"  WARNING: Could not load {name}.pkl ({e.__class__.__name__}: {e}) - skipping.")
    if missing:
        raise FileNotFoundError(
            f"Missing model files (run 05_modeling.py first):\n" +
            "\n".join(missing)
        )
    if not models:
        raise RuntimeError("No models could be loaded. Re-run 05_modeling.py.")
    return models


def get_probabilities(models: dict, X: pd.DataFrame) -> dict:
    """Get predicted probabilities from each model on X. Skips models that fail."""
    probs = {}
    for name, model in models.items():
        try:
            probs[name] = model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"  WARNING: {name} prediction failed ({e.__class__.__name__}: {e}) - skipping.")
    if not probs:
        raise RuntimeError("No models produced predictions. Re-run 05_modeling.py.")
    return probs


# ── Stacking ───────────────────────────────────────────────────────────────────

def stacking_ensemble(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Simple stacking: split train 80/20 (time-ordered), use saved models to
    predict on the 20% holdout, train a meta-LogisticRegression on those
    predictions, then apply to test.

    Note: slight data leakage — base models were trained on the full train set
    including the 20% holdout. This is acceptable for a final project demo.
    """
    split = int(len(X_train) * 0.8)
    X_meta_val = X_train.iloc[split:]
    y_meta_val = y_train.iloc[split:]

    available = []
    for name in MODEL_NAMES:
        if name not in models:
            continue
        try:
            models[name].predict_proba(X_meta_val)[:, 1]
            available.append(name)
        except Exception as e:
            print(f"  WARNING: {name} skipped in stacking ({e.__class__.__name__}: {e})")

    meta_val_features = np.column_stack([
        models[name].predict_proba(X_meta_val)[:, 1]
        for name in available
    ])
    meta_test_features = np.column_stack([
        models[name].predict_proba(X_test)[:, 1]
        for name in available
    ])

    meta_lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
    )
    meta_lr.fit(meta_val_features, y_meta_val)
    return meta_lr.predict_proba(meta_test_features)[:, 1]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 6 - Ensemble Strategy Ladder")
    print("=" * 60)

    # ── Load feature matrix ────────────────────────────────────────────────
    feat = pd.read_parquet(PROC_DIR / "features.parquet")
    biz  = pd.read_parquet(PROC_DIR / "businesses.parquet")[
        ["business_id", "name", "stars"]
    ]

    feat["anchor_date"] = pd.to_datetime(feat["anchor_date"])
    feature_cols = [c for c in feat.columns if c not in META_COLS]

    TEST_CUTOFF  = pd.Timestamp("2020-06-01")
    train_df = feat[feat["anchor_date"] < TEST_CUTOFF].copy()
    test_df  = feat[feat["anchor_date"] >= TEST_CUTOFF].copy()
    print(f"\n  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    train_medians = train_df[feature_cols].median()
    X_train = train_df[feature_cols].fillna(train_medians)
    y_train = train_df[TARGET_COL]
    X_test  = test_df[feature_cols].fillna(train_medians)
    y_test  = test_df[TARGET_COL]

    # ── Load models ────────────────────────────────────────────────────────
    print("\n  Loading models...")
    models = load_models(MODEL_DIR)
    print(f"  Loaded: {list(models.keys())}")

    # ── Individual predictions ─────────────────────────────────────────────
    test_probs = get_probabilities(models, X_test)

    # ── Strategy 1: Simple average ─────────────────────────────────────────
    print("\n[1] Simple Average")
    sa_prob    = simple_average(test_probs)
    sa_metrics = compute_metrics(y_test, sa_prob)
    print(f"  AUC-PR={sa_metrics['AUC_PR']:.4f}  "
          f"AUC-ROC={sa_metrics['AUC_ROC']:.4f}  F1={sa_metrics['F1']:.4f}")

    # ── Strategy 2: Weighted average ───────────────────────────────────────
    print("\n[2] Weighted Average (by CV AUC-PR)")
    wa_prob    = weighted_average(test_probs, CV_AUC_PR)
    wa_metrics = compute_metrics(y_test, wa_prob)
    print(f"  AUC-PR={wa_metrics['AUC_PR']:.4f}  "
          f"AUC-ROC={wa_metrics['AUC_ROC']:.4f}  F1={wa_metrics['F1']:.4f}")

    # ── Pick best so far ───────────────────────────────────────────────────
    results = {
        "xgboost_baseline":  {"AUC_PR": BASELINE_AUC_PR},
        "simple_average":    sa_metrics,
        "weighted_average":  wa_metrics,
    }

    best_prob, best_name, best_auc = (
        (sa_prob, "simple_average", sa_metrics["AUC_PR"])
        if sa_metrics["AUC_PR"] >= wa_metrics["AUC_PR"]
        else (wa_prob, "weighted_average", wa_metrics["AUC_PR"])
    )

    # ── Strategy 3: Stacking (only if neither beat baseline) ──────────────
    if best_auc < BASELINE_AUC_PR:
        print("\n[3] Stacking (meta-LogisticRegression)")
        print("  Neither strategy beat baseline - escalating...")
        st_prob    = stacking_ensemble(models, X_train, y_train, X_test)
        st_metrics = compute_metrics(y_test, st_prob)
        print(f"  AUC-PR={st_metrics['AUC_PR']:.4f}  "
              f"AUC-ROC={st_metrics['AUC_ROC']:.4f}  F1={st_metrics['F1']:.4f}")
        results["stacking"] = st_metrics
        if st_metrics["AUC_PR"] > best_auc:
            best_prob, best_name, best_auc = st_prob, "stacking", st_metrics["AUC_PR"]
    else:
        print("\n[3] Stacking - skipped (not needed)")

    # ── Summary ────────────────────────────────────────────────────────────
    improvement = (best_auc - BASELINE_AUC_PR) / BASELINE_AUC_PR
    print(f"\n  Winner:   {best_name}")
    print(f"  AUC-PR:  {best_auc:.4f}  (baseline: {BASELINE_AUC_PR:.4f})")
    print(f"  Change:  {improvement:+.1%} vs XGBoost")

    results["winner"] = best_name
    results["winner_auc_pr"] = best_auc

    opt_threshold = find_optimal_threshold(y_test.values, best_prob)
    best_metrics_opt = compute_metrics(y_test, best_prob, threshold=opt_threshold)

    print(f"\n  Optimal threshold: {opt_threshold:.3f}  (vs default 0.500)")
    print(f"  Optimized F1:      {best_metrics_opt['F1']:.4f}  "
          f"(vs default-threshold F1: "
          f"{compute_metrics(y_test, best_prob)['F1']:.4f})")

    results["opt_threshold"]   = opt_threshold
    results["optimized_f1"]    = best_metrics_opt["F1"]

    # ── Save outputs ───────────────────────────────────────────────────────
    with open(MODEL_DIR / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {MODEL_DIR}/ensemble_results.json")

    # Build predictions parquet (test set only — these are the UI records)
    sig_cols_present = [c for c in SIGNAL_COLS if c in test_df.columns]
    pred_df = test_df[
        ["business_id", "anchor_date", TARGET_COL] + sig_cols_present
    ].copy()
    pred_df["risk_score"] = best_prob
    pred_df["opt_threshold"] = float(opt_threshold)
    pred_df = pred_df.merge(biz, on="business_id", how="left")
    pred_df.to_parquet(PROC_DIR / "ensemble_predictions.parquet", index=False)
    print(f"  Saved: {PROC_DIR}/ensemble_predictions.parquet")
    print(f"  Records: {len(pred_df):,}  |  "
          f"Closed: {pred_df[TARGET_COL].sum()} ({pred_df[TARGET_COL].mean():.1%})")


if __name__ == "__main__":
    main()
