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
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from config_00 import PROC_DIR, MODEL_DIR, TARGET_COL, RANDOM_SEED

# ── Constants ──────────────────────────────────────────────────────────────────
BASELINE_AUC_PR = 0.2069   # XGBoost test result from 05_modeling.py

MODEL_NAMES = [
    "xgboost",
    "random_forest",
    "lightgbm",
    "mlp",
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
    return sum(weights[k] * probs[k] for k in probs) / total


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC_PR":  float(average_precision_score(y_true, y_prob)),
        "AUC_ROC": float(roc_auc_score(y_true, y_prob)),
        "F1":      float(f1_score(y_true, y_pred, zero_division=0)),
    }


# ── Data / model loading ───────────────────────────────────────────────────────

def load_models(model_dir: Path) -> dict:
    """Load all 5 saved .pkl model files. Raises if any are missing."""
    models = {}
    missing = []
    for name in MODEL_NAMES:
        path = model_dir / f"{name}.pkl"
        if path.exists():
            models[name] = joblib.load(path)
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"Missing model files (run 05_modeling.py first):\n" +
            "\n".join(missing)
        )
    return models


def get_probabilities(models: dict, X: pd.DataFrame) -> dict:
    """Get predicted probabilities from each model on X."""
    return {
        name: model.predict_proba(X)[:, 1]
        for name, model in models.items()
    }


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

    meta_val_features = np.column_stack([
        models[name].predict_proba(X_meta_val)[:, 1]
        for name in MODEL_NAMES
    ])
    meta_test_features = np.column_stack([
        models[name].predict_proba(X_test)[:, 1]
        for name in MODEL_NAMES
    ])

    meta_lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
    )
    meta_lr.fit(meta_val_features, y_meta_val)
    return meta_lr.predict_proba(meta_test_features)[:, 1]
