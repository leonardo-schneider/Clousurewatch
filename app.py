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
import numpy as np              # used in render_detail (Task 7)
import plotly.graph_objects as go  # used in render_detail (Task 7)
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


# -- Sidebar -------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame) -> pd.Series:
    """
    Render the risk-ranked restaurant list in the sidebar.
    Returns the currently selected restaurant row as a pd.Series.
    """
    st.sidebar.title("Closure Watch")
    st.sidebar.caption("Tampa Bay - Restaurant Closure Risk")

    search = st.sidebar.text_input("Search by name", "")

    df_sorted = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    if search.strip():
        mask = df_sorted["name"].fillna("").str.contains(search.strip(), case=False, regex=False)
        df_sorted = df_sorted[mask].reset_index(drop=True)

    if df_sorted.empty:
        st.sidebar.warning("No restaurants match your search.")
        st.stop()

    selected_idx = st.sidebar.radio(
        "Restaurants by Risk",
        range(len(df_sorted)),
        format_func=lambda i: (
            f"{risk_badge(df_sorted.iloc[i]['risk_score'])} "
            f"{df_sorted.iloc[i]['name']} -- "
            f"{df_sorted.iloc[i]['risk_score']:.0%}"
        ),
        label_visibility="collapsed",
    )
    return df_sorted.iloc[selected_idx]


# -- App entry point -----------------------------------------------------------

def main():
    st.set_page_config(
        page_title="ClosureWatch",
        layout="wide",
        page_icon="🍽",
    )

    try:
        df = load_predictions()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    restaurant = render_sidebar(df)
    st.session_state["selected_id"] = restaurant["business_id"]

    # Detail panel placeholder -- replaced in Task 7
    st.markdown(f"## {restaurant.get('name', 'Unknown')}")
    st.caption("Detail panel coming in Task 7")


if __name__ == "__main__":
    main()
