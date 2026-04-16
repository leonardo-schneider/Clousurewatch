"""
app.py
------
Streamlit dashboard -- Restaurant Closure Risk Predictor.

Run:
    streamlit run app.py

Requires:
    data/processed/ensemble_predictions.parquet  (run 06_ensemble.py first)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from config_00 import PROC_DIR, TARGET_COL

# -- Feature metadata ----------------------------------------------------------

SIGNAL_COLS = [
    "months_with_zero_reviews",
    "days_since_last_review",
    "review_drought_flag",
    "checkin_drought_flag",
    "pct_5star",
]

SIGNAL_LABELS = {
    "months_with_zero_reviews": "Months with zero reviews",
    "days_since_last_review":   "Days since last review",
    "review_drought_flag":      "Review drought triggered",
    "checkin_drought_flag":     "Check-in drought triggered",
    "pct_5star":                "% of 5-star reviews",
}

# True = high value means more risk, False = high value means less risk
SIGNAL_RISK_DIR = {
    "months_with_zero_reviews": True,
    "days_since_last_review":   True,
    "review_drought_flag":      True,
    "checkin_drought_flag":     True,
    "pct_5star":                False,
}


# -- Helper functions (unit tested in tests/test_app_helpers.py) ---------------

def risk_color(score: float) -> str:
    """Return hex color for a given risk score."""
    if score >= 0.60:
        return "#e94560"
    elif score >= 0.30:
        return "#f7a440"
    return "#4caf50"


def risk_label(score: float) -> str:
    """Return HIGH / MEDIUM / LOW label."""
    if score >= 0.60:
        return "HIGH"
    elif score >= 0.30:
        return "MEDIUM"
    return "LOW"


def risk_badge(score: float) -> str:
    """Return emoji color circle for sidebar list."""
    if score >= 0.60:
        return "🔴"
    elif score >= 0.30:
        return "🟠"
    return "🟢"


def percentile_rank(series: pd.Series, value: float) -> float:
    """Return fraction of series values <= value (0.0-1.0)."""
    return float((series.dropna() <= value).mean())


def load_predictions() -> pd.DataFrame:
    """
    Load ensemble predictions parquet.
    Raises FileNotFoundError with a clear message if the file doesn't exist.
    """
    pred_path = PROC_DIR / "ensemble_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {pred_path}\n"
            "Run the ensemble script first:  python 06_ensemble.py"
        )
    return pd.read_parquet(pred_path)
