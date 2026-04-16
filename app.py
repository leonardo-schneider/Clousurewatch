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


# -- Detail panel --------------------------------------------------------------

def render_detail(restaurant: pd.Series, df: pd.DataFrame) -> None:
    """Render the right-side detail panel for the selected restaurant."""
    color = risk_color(restaurant["risk_score"])
    label = risk_label(restaurant["risk_score"])

    # -- Header row ------------------------------------------------------------
    col_name, col_gauge = st.columns([3, 1])

    with col_name:
        st.markdown(f"## {restaurant.get('name', 'Unknown')}")
        stars = restaurant.get("stars", None)
        city  = restaurant.get("city", "")
        star_str = f"Yelp {stars} stars" if pd.notna(stars) else "Yelp rating N/A"
        st.caption(f"{city} - {star_str}")

    with col_gauge:
        st.markdown(
            f"""
            <div style='text-align:center;background:#16213e;border-radius:8px;
                        padding:14px;margin-top:4px'>
              <div style='font-size:38px;font-weight:bold;color:{color}'>
                {restaurant['risk_score']:.0%}
              </div>
              <div style='color:#aaa;font-size:11px'>closure risk</div>
              <div style='color:{color};font-size:12px;font-weight:bold'>{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # -- Key signals -----------------------------------------------------------
    st.subheader("Key Signals")
    present_cols = [
        c for c in SIGNAL_COLS
        if c in restaurant.index and pd.notna(restaurant[c])
    ]

    for col in present_cols:
        val     = restaurant[col]
        is_risk = SIGNAL_RISK_DIR[col]
        name    = SIGNAL_LABELS[col]

        if col in ("review_drought_flag", "checkin_drought_flag"):
            triggered = bool(val)
            icon = "warning" if (triggered and is_risk) else "check"
            state = "Yes" if triggered else "No"
            if triggered and is_risk:
                st.warning(f"**{name}**: {state}")
            else:
                st.success(f"**{name}**: {state}")
        elif col == "days_since_last_review":
            if is_risk:
                st.warning(f"**{name}**: {val:.0f} days")
            else:
                st.success(f"**{name}**: {val:.0f} days")
        elif col == "pct_5star":
            if val > 0.5:
                st.success(f"**{name}**: {val:.0%}")
            else:
                st.warning(f"**{name}**: {val:.0%}")
        else:
            if is_risk:
                st.warning(f"**{name}**: {val:.1f}")
            else:
                st.success(f"**{name}**: {val:.1f}")

    st.divider()

    # -- Feature bar chart (percentile-normalized) -----------------------------
    st.subheader("Feature Contributions")
    st.caption(
        "Bar length = restaurant percentile in this dataset "
        "(0% = lowest, 100% = highest for risk-increasing features)"
    )

    chart_data = {}
    chart_colors = []

    for col in present_cols:
        pct = percentile_rank(df[col], restaurant[col])
        # For risk-decreasing features (pct_5star), invert so bar = risk contribution
        display_pct = pct if SIGNAL_RISK_DIR[col] else (1.0 - pct)
        chart_data[SIGNAL_LABELS[col]] = display_pct
        chart_colors.append(risk_color(1.0) if SIGNAL_RISK_DIR[col] else "#4caf50")

    fig = go.Figure(go.Bar(
        x=list(chart_data.values()),
        y=list(chart_data.keys()),
        orientation="h",
        marker_color=chart_colors,
        text=[f"{v:.0%}" for v in chart_data.values()],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#eeeeee", size=12),
        margin=dict(l=0, r=60, t=10, b=0),
        height=220,
        xaxis=dict(
            range=[0, 1.15],
            tickformat=".0%",
            gridcolor="#0f3460",
            showgrid=True,
        ),
        yaxis=dict(gridcolor="#0f3460"),
    )
    st.plotly_chart(fig, use_container_width=True)


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

    render_detail(restaurant, df)


if __name__ == "__main__":
    main()
