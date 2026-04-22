"""
ClosureWatch - Tampa Bay Restaurant Failure Prediction
Spotify-inspired dark dashboard: deep black, #1DB954 green accent,
bold condensed type, album-art energy applied to risk data.
"""

from __future__ import annotations

import os as _os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app_helpers import (
    risk_color,
    risk_label,
    risk_badge,
    percentile_rank,
    outcome_banner_html,
    compute_shap_row,
)

# Page config must be first Streamlit command.
st.set_page_config(
    page_title="ClosureWatch · Tampa Bay",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

SPOTIFY_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&family=DM+Mono:wght@400;500&display=swap');

:root {
    --sp-green:      #1DB954;
    --sp-green-dim:  #158843;
    --sp-black:      #121212;
    --sp-surface:    #181818;
    --sp-elevated:   #282828;
    --sp-elevated2:  #3e3e3e;
    --sp-text:       #FFFFFF;
    --sp-text-sub:   #B3B3B3;
    --sp-text-hint:  #535353;
    --sp-danger:     #E24B4A;
    --sp-danger-dim: #5c1a1a;
    --sp-warn:       #EF9F27;
    --sp-warn-dim:   #4a3000;
    --sp-safe:       #1DB954;
    --sp-safe-dim:   #0d3320;
}

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif !important;
    background-color: var(--sp-black) !important;
    color: var(--sp-text) !important;
}

.stApp { background-color: var(--sp-black) !important; }
section[data-testid="stSidebar"] {
    background-color: var(--sp-black) !important;
    border-right: 1px solid #282828 !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 100% !important; }

.stTextInput input {
    background-color: var(--sp-elevated) !important;
    border: 1px solid var(--sp-elevated2) !important;
    border-radius: 500px !important;
    color: var(--sp-text) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
}
.stTextInput input:focus {
    border-color: var(--sp-text) !important;
    box-shadow: none !important;
}
.stTextInput input::placeholder { color: var(--sp-text-hint) !important; }

.stRadio > div { gap: 0 !important; }
.stRadio label {
    background: transparent !important;
    border-radius: 4px !important;
    cursor: pointer;
    padding: 6px 8px 6px 4px !important;
    display: flex !important;
    align-items: center !important;
}
.stRadio label:hover { background: var(--sp-elevated) !important; }
.stRadio div[role="radiogroup"] { gap: 0 !important; }

[data-testid="metric-container"] {
    background: var(--sp-surface) !important;
    border-radius: 8px !important;
    padding: 1rem 1.25rem !important;
    border: none !important;
}
[data-testid="metric-container"] label {
    color: var(--sp-text-sub) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 800 !important;
    color: var(--sp-text) !important;
}

hr { border-color: #282828 !important; margin: 1rem 0 !important; }

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--sp-black); }
::-webkit-scrollbar-thumb { background: var(--sp-elevated2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #666; }

.js-plotly-plot .plotly { background: transparent !important; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.cw-header {
    background: linear-gradient(180deg, #1a1a1a 0%, var(--sp-black) 100%);
    border-bottom: 1px solid #282828;
    padding: 1.25rem 0 1rem 0;
    margin-bottom: 1.5rem;
}
.cw-brand {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--sp-green);
    margin-bottom: 2px;
}
.cw-title {
    font-size: 28px;
    font-weight: 900;
    color: var(--sp-text);
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.cw-sub {
    font-size: 13px;
    color: var(--sp-text-sub);
    margin-top: 4px;
}

.risk-display {
    text-align: center;
    padding: 1.5rem;
    background: var(--sp-surface);
    border-radius: 8px;
    position: relative;
    overflow: hidden;
}
.risk-display::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.risk-high::before { background: var(--sp-danger); }
.risk-med::before { background: var(--sp-warn); }
.risk-low::before { background: var(--sp-green); }

.risk-pct {
    font-size: 72px;
    font-weight: 900;
    line-height: 1;
    letter-spacing: 0;
}
.risk-high .risk-pct { color: var(--sp-danger); }
.risk-med .risk-pct { color: var(--sp-warn); }
.risk-low .risk-pct { color: var(--sp-green); }

.risk-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 6px;
}
.risk-high .risk-label { color: var(--sp-danger); }
.risk-med .risk-label { color: var(--sp-warn); }
.risk-low .risk-label { color: var(--sp-green); }

.risk-sublabel {
    font-size: 12px;
    color: var(--sp-text-sub);
    margin-top: 4px;
}

.signal-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 6px;
    font-size: 13px;
    background: var(--sp-surface);
    transition: background 0.15s;
}
.signal-row:hover { background: var(--sp-elevated); }
.signal-name { color: var(--sp-text-sub); font-weight: 500; }
.signal-val { color: var(--sp-text); font-weight: 700; font-family: 'DM Mono', monospace; }
.signal-triggered { color: var(--sp-danger); }
.signal-safe { color: var(--sp-green); }

.rest-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    border-radius: 6px;
    cursor: pointer;
    margin-bottom: 2px;
    transition: background 0.1s;
}
.rest-item:hover { background: var(--sp-elevated); }
.rest-item.active { background: var(--sp-elevated2); }
.rest-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.rest-name { font-size: 13px; font-weight: 600; color: var(--sp-text); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rest-pct { font-size: 12px; font-weight: 700; font-family: 'DM Mono', monospace; }

.section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--sp-text-hint);
    margin-bottom: 10px;
    margin-top: 4px;
}

.yelp-line {
    font-size: 12px;
    color: var(--sp-text-hint);
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 4px;
}
.yelp-stars { color: #FF3B2F; }

.sidebar-brand {
    padding: 20px 16px 12px 16px;
    border-bottom: 1px solid #282828;
    margin-bottom: 12px;
}
.sidebar-brand-name {
    font-size: 18px;
    font-weight: 900;
    letter-spacing: -0.5px;
    color: var(--sp-text);
}
.sidebar-brand-sub {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--sp-green);
    margin-top: 2px;
}

.now-watching {
    padding: 10px 14px;
    background: var(--sp-surface);
    border-radius: 8px;
    margin-bottom: 12px;
    border-left: 3px solid var(--sp-green);
}
.now-watching-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--sp-green);
    margin-bottom: 3px;
}
.now-watching-name {
    font-size: 13px;
    font-weight: 700;
    color: var(--sp-text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.cw-card {
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    overflow: hidden;
    cursor: pointer;
    transition: border-color 0.2s;
    background: #181818;
    margin-bottom: 4px;
}
.cw-card:hover { border-color: #1db954; }
.cw-card.selected { border: 2px solid #1db954; }
.cw-card-img {
    width: 100%;
    height: 140px;
    object-fit: cover;
    display: block;
}
.cw-card-img-placeholder {
    width: 100%;
    height: 140px;
    background: #2a2a2a;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #535353;
    font-size: 12px;
}
.cw-card-body { padding: 10px 12px 12px; }
.cw-card-name {
    font-size: 13px;
    font-weight: 600;
    color: #fff;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 2px;
}
.cw-card-meta { font-size: 11px; color: #b3b3b3; margin-bottom: 8px; }
.cw-card-stats {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #b3b3b3;
}
.cw-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 20px;
    letter-spacing: 0.04em;
}
.cw-badge-HIGH { background: #3d0000; color: #ff4444; }
.cw-badge-MEDIUM { background: #3d2800; color: #ffaa00; }
.cw-badge-LOW { background: #003d1a; color: #1db954; }
</style>
"""

st.markdown(SPOTIFY_CSS, unsafe_allow_html=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load ensemble predictions parquet. Falls back to synthetic demo data."""
    path = Path("data/processed/ensemble_predictions.parquet")
    if path.exists():
        df = pd.read_parquet(path)
        rename = {}
        for col in df.columns:
            low = col.lower()
            if "name" in low:
                rename[col] = "name"
            if "risk" in low or "prob" in low or "score" in low:
                rename[col] = "risk_score"
            if "stars" in low or "rating" in low:
                rename[col] = "stars"
        df = df.rename(columns=rename)
        if "risk_score" not in df.columns:
            num = df.select_dtypes("number").columns
            if len(num):
                df["risk_score"] = df[num[0]]
        if "name" not in df.columns:
            obj = df.select_dtypes("object").columns
            if len(obj):
                df["name"] = df[obj[0]]
        df["risk_pct"] = (df["risk_score"] * 100).round(1)
    else:
        rng = np.random.default_rng(42)
        names = [
            "Woody's Liquor & Fine Wine", "Winn-Dixie Deli", "Dunkin'",
            "Top China", "Stacie's Cottage Cafe", "Chicago Deli & Coney Dogs",
            "Walgreens Bistro", "Baskin-Robbins", "Taco Bell (Dale Mabry)",
            "Taco Bell (Fowler)", "China Wok", "Spartan Manor",
            "Unks Bar B Que", "Slizzy Mcgee's", "ABC Fine Wine Spirits",
            "Beef 'O' Brady's", "Denny's Ybor", "IHOP Brandon",
            "Checkers", "Golden Wok", "Sushi Time", "La Teresita",
            "El Cap", "Mel's Hot Dogs", "Bern's Steak House",
            "The Columbia Restaurant", "Oxford Exchange", "Ulele",
            "Daily Eats", "Ciccio's",
        ]
        risk = sorted(rng.beta(2, 5, size=len(names)) * 0.85 + 0.10, reverse=True)
        stars = rng.uniform(1.0, 4.5, size=len(names)).round(1)
        months_zero = rng.integers(0, 6, size=len(names))
        days_last = rng.integers(10, 210, size=len(names))
        review_drought = (days_last > 90).astype(int)
        checkin_drought = rng.integers(0, 2, size=len(names))
        pct_5star = rng.uniform(0, 0.6, size=len(names)).round(2)
        feat_names = [
            "months_with_zero_reviews", "days_since_last_review",
            "review_drought_flag", "checkin_drought_flag",
            "pct_5star", "review_velocity_trend", "checkin_velocity",
            "avg_sentiment_score", "review_count_12m",
        ]
        feat_matrix = rng.uniform(-1, 1, size=(len(names), len(feat_names)))
        for i, r in enumerate(risk):
            feat_matrix[i] += (r - 0.5) * 0.5

        df = pd.DataFrame({
            "name": names,
            "risk_score": risk,
            "risk_pct": [round(r * 100, 1) for r in risk],
            "stars": stars,
            "months_with_zero_reviews": months_zero,
            "days_since_last_review": days_last,
            "review_drought_flag": review_drought,
            "checkin_drought_flag": checkin_drought,
            "pct_5star": pct_5star,
        })
        for j, fn in enumerate(feat_names):
            df[f"feat_{fn}"] = feat_matrix[:, j].clip(-1, 1)

    return df.sort_values("risk_pct", ascending=False).reset_index(drop=True)


@st.cache_resource
def load_xgb_model():
    import joblib
    path = Path("models/xgboost.pkl")
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data
def load_feature_matrix() -> pd.DataFrame | None:
    path = Path("data/processed/features.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data
def load_photo_index():
    path = Path("data/processed/photo_index.parquet")
    if path.exists():
        return pd.read_parquet(path).set_index("business_id")
    return pd.DataFrame()


df = load_data()
_xgb_model = load_xgb_model()
_feat_matrix = load_feature_matrix()

_parquet_path = Path("data/processed/ensemble_predictions.parquet")
_batch_date = (
    pd.Timestamp(_os.path.getmtime(_parquet_path), unit="s").strftime("%b %Y")
    if _parquet_path.exists()
    else "Unknown"
)


def risk_tier(pct: float) -> tuple[str, str, str]:
    """Return tier name, CSS class, and hex color. pct is 0–100."""
    frac = pct / 100.0
    name = risk_label(frac).replace("MEDIUM", "ELEVATED")
    css  = {"HIGH": "risk-high", "ELEVATED": "risk-med", "LOW": "risk-low"}[name]
    col  = risk_color(frac)
    return name, css, col


def stars_html(rating: float) -> str:
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty


FEAT_LABELS = {
    "months_with_zero_reviews": "Months with zero reviews",
    "days_since_last_review": "Days since last review",
    "review_drought_flag": "Review drought flag",
    "checkin_drought_flag": "Check-in drought flag",
    "pct_5star": "% of 5-star reviews",
    "review_velocity_trend": "Review velocity trend",
    "checkin_velocity": "Check-in velocity",
    "avg_sentiment_score": "Avg sentiment score",
    "review_count_12m": "Review count (12m)",
}


def restaurant_card_html(row, photo_index, selected=False):
    bid = row["business_id"]
    risk_pct = row["risk_pct"]
    tier, css_class, hex_color = risk_tier(risk_pct)

    # Photo
    photo_html = ""
    if bid in photo_index.index:
        photo_path = photo_index.loc[bid, "path"]
        if Path(photo_path).exists():
            import base64
            with open(photo_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            photo_html = f'<img class="cw-card-img" src="data:image/jpeg;base64,{b64}" />'
    if not photo_html:
        cat = str(row.get("categories", "")).split(",")[0].strip()[:20]
        photo_html = f'<div class="cw-card-img-placeholder">{cat if cat else "No photo"}</div>'

    selected_class = "selected" if selected else ""
    stars = row.get("stars_yelp_global", row.get("stars_yelp", row.get("stars", 0)))

    return f"""
    <div class="cw-card {selected_class}">
        {photo_html}
        <div class="cw-card-body">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                <div class="cw-card-name">{row['name']}</div>
                <span class="cw-badge cw-badge-{tier}">{tier}</span>
            </div>
            <div class="cw-card-meta">{row.get('city', 'Tampa')} &nbsp;·&nbsp; {stars:.1f}★</div>
            <div class="cw-card-stats">
                <span style="color:{hex_color};font-weight:700">{risk_pct:.1f}% risk</span>
                <span>{int(row.get('days_since_last_review', 0))}d silent</span>
            </div>
        </div>
    </div>
    """

# Session state bootstrap must happen before the sidebar renders.
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = int(df.index[0])
if "tier_filter" not in st.session_state:
    st.session_state.tier_filter = "All"


st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] .stButton button {
    background: var(--sp-elevated) !important;
    border: 1px solid transparent !important;
    border-radius: 500px !important;
    color: var(--sp-text-sub) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 4px 12px !important;
    height: auto !important;
    min-height: 28px !important;
    transition: all 0.15s !important;
}
div[data-testid="stHorizontalBlock"] .stButton button:hover {
    background: var(--sp-elevated2) !important;
    color: var(--sp-text) !important;
    border-color: var(--sp-elevated2) !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: transparent !important;
    border: none !important;
    border-radius: 4px !important;
    color: var(--sp-text-sub) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-align: left !important;
    padding: 6px 10px !important;
    height: auto !important;
    min-height: 32px !important;
    width: 100% !important;
    transition: background 0.1s, color 0.1s !important;
    justify-content: flex-start !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: var(--sp-elevated) !important;
    color: var(--sp-text) !important;
}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {
    color: var(--sp-text-hint) !important;
    font-size: 10px !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stThumb"] {
    background-color: var(--sp-green) !important;
    border-color: var(--sp-green) !important;
}
.stSlider [role="slider"] {
    background-color: var(--sp-green) !important;
}
section[data-testid="stSidebar"] .stSelectbox select,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
    background-color: var(--sp-elevated) !important;
    border: 1px solid var(--sp-elevated2) !important;
    border-radius: 6px !important;
    color: var(--sp-text) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-name">ClosureWatch</div>
            <div class="sidebar-brand-sub">Tampa Bay · Live Risk Feed</div>
        </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "Search restaurants",
        placeholder="�?  Search restaurants…",
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="section-label" style="padding:0 2px;margin-top:10px">Filter by tier</div>',
        unsafe_allow_html=True,
    )

    pill_cols = st.columns(4)
    pills = [
        ("All", "All", "#B3B3B3"),
        ("🔴 High", "HIGH", "#E24B4A"),
        ("🟡 Mid", "ELEVATED", "#EF9F27"),
        ("🟢 Low", "LOW", "#1DB954"),
    ]
    for col, (label, value, _color) in zip(pill_cols, pills):
        with col:
            if st.button(label, key=f"pill_{value}"):
                st.session_state.tier_filter = value

    active = st.session_state.tier_filter
    tier_colors_map = {
        "All": "#B3B3B3",
        "HIGH": "#E24B4A",
        "ELEVATED": "#EF9F27",
        "LOW": "#1DB954",
    }
    active_color = tier_colors_map.get(active, "#B3B3B3")
    st.markdown(
        f'<div style="font-size:10px;color:{active_color};font-weight:700;'
        f'letter-spacing:1.5px;padding:4px 2px 2px 2px">'
        f'▸ SHOWING: {active.upper()}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label" style="padding:0 2px;margin-top:10px">Risk range</div>',
        unsafe_allow_html=True,
    )
    risk_range = st.slider(
        "Risk range",
        min_value=0,
        max_value=100,
        value=(0, 100),
        step=5,
        format="%d%%",
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="section-label" style="padding:0 2px;margin-top:10px">Sort by</div>',
        unsafe_allow_html=True,
    )
    sort_by = st.selectbox(
        "Sort by",
        ["Risk ↓ (highest first)", "Risk ↑ (lowest first)", "Name A–Z", "Name Z–A"],
        label_visibility="collapsed",
    )

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    filtered = df.copy()
    if query:
        filtered = filtered[filtered["name"].str.lower().str.contains(query.lower(), na=False)]

    if active == "HIGH":
        filtered = filtered[filtered["risk_pct"] >= 60]
    elif active == "ELEVATED":
        filtered = filtered[(filtered["risk_pct"] >= 30) & (filtered["risk_pct"] < 60)]
    elif active == "LOW":
        filtered = filtered[filtered["risk_pct"] < 30]

    filtered = filtered[
        (filtered["risk_pct"] >= risk_range[0]) &
        (filtered["risk_pct"] <= risk_range[1])
    ]

    if sort_by.startswith("Risk ↓"):
        filtered = filtered.sort_values("risk_pct", ascending=False)
    elif sort_by.startswith("Risk ↑"):
        filtered = filtered.sort_values("risk_pct", ascending=True)
    elif sort_by.startswith("Name A"):
        filtered = filtered.sort_values("name", ascending=True)
    elif sort_by.startswith("Name Z"):
        filtered = filtered.sort_values("name", ascending=False)

    n_high = int((filtered["risk_pct"] >= 60).sum())
    n_mid = int(((filtered["risk_pct"] >= 30) & (filtered["risk_pct"] < 60)).sum())
    n_low = int((filtered["risk_pct"] < 30).sum())

    st.markdown(f"""
    <div style="display:flex;gap:8px;padding:6px 2px;margin-bottom:6px">
        <div style="flex:1;text-align:center;background:#5c1a1a;border-radius:6px;padding:5px 0">
            <div style="font-size:16px;font-weight:800;color:#E24B4A">{n_high}</div>
            <div style="font-size:9px;font-weight:700;letter-spacing:1px;color:rgba(226,75,74,0.6)">HIGH</div>
        </div>
        <div style="flex:1;text-align:center;background:#4a3000;border-radius:6px;padding:5px 0">
            <div style="font-size:16px;font-weight:800;color:#EF9F27">{n_mid}</div>
            <div style="font-size:9px;font-weight:700;letter-spacing:1px;color:rgba(239,159,39,0.6)">MID</div>
        </div>
        <div style="flex:1;text-align:center;background:#0d3320;border-radius:6px;padding:5px 0">
            <div style="font-size:16px;font-weight:800;color:#1DB954">{n_low}</div>
            <div style="font-size:9px;font-weight:700;letter-spacing:1px;color:rgba(29,185,84,0.6)">LOW</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="font-size:10px;color:#535353;padding:0 2px">
            {len(filtered)} shown · {len(df)} total<br>
            XGBoost ensemble · AUC-ROC 0.700
        </div>
    """, unsafe_allow_html=True)


# ── Card grid ──────────────────────────────────────────────────────
photo_index = load_photo_index()

st.markdown("### Restaurants")

# Show top N cards based on current filter + sort
CARDS_PER_PAGE = 12
display_df = filtered.head(CARDS_PER_PAGE).reset_index(drop=True)

cols = st.columns(3)
for i, (_, row) in enumerate(display_df.iterrows()):
    col = cols[i % 3]
    with col:
        bid = row["business_id"]
        is_selected = (st.session_state.selected_idx is not None and
                       df.loc[st.session_state.selected_idx, "business_id"] == bid)

        card_html = restaurant_card_html(row, photo_index, selected=is_selected)
        st.markdown(card_html, unsafe_allow_html=True)

        if st.button(
            f"View details \u2192 {row['name'][:25]}",
            key=f"card_{bid}",
            use_container_width=True,
        ):
            idx = df[df["business_id"] == bid].index[0]
            st.session_state.selected_idx = idx
            st.rerun()

st.markdown("""
<style>
div[data-testid="stButton"] > button {
    background: transparent;
    border: 1px solid #2a2a2a;
    color: #b3b3b3;
    font-size: 11px;
    padding: 4px 8px;
    border-radius: 0 0 12px 12px;
    margin-top: -4px;
    transition: color 0.15s, border-color 0.15s;
}
div[data-testid="stButton"] > button:hover {
    color: #1db954;
    border-color: #1db954;
}
</style>
""", unsafe_allow_html=True)

st.divider()
# ── existing detail panel follows here unchanged ──

sel_idx = st.session_state.selected_idx
if sel_idx not in df.index:
    sel_idx = int(df.index[0])
    st.session_state.selected_idx = sel_idx

selected = df.loc[sel_idx]
tier_name, tier_css, dot_color = risk_tier(selected["risk_pct"])

_banner_html = outcome_banner_html(selected)
st.markdown(f"""
<div class="cw-header">
    {_banner_html}
    <div class="cw-brand">📡 ClosureWatch</div>
    <div class="cw-title">{selected['name']}</div>
    <div class="cw-sub">
        Tampa Bay Metro · Yelp Academic Dataset · 6-month closure prediction
    </div>
</div>
""", unsafe_allow_html=True)

col_risk, col_metrics = st.columns([1, 2.5], gap="large")

with col_risk:
    yelp_stars = selected.get("stars", 2.5)
    star_full = "★" * int(yelp_stars)
    star_empty = "☆" * (5 - int(yelp_stars))
    st.markdown(f"""
    <div class="risk-display {tier_css}">
        <div style="font-size:10px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;
                    color:var(--sp-text-hint);margin-bottom:8px">Closure Risk Score</div>
        <div class="risk-pct">{selected['risk_pct']:.0f}<span style="font-size:36px">%</span></div>
        <div class="risk-label">{tier_name}</div>
        <div class="risk-sublabel" style="margin-top:12px">
            Yelp <span style="color:#FF3B2F">{star_full}{star_empty}</span>
            &nbsp;{yelp_stars} stars
        </div>
        <div style="margin-top:14px;padding-top:12px;border-top:1px solid #282828;">
            <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;
                        text-transform:uppercase;color:var(--sp-text-hint);margin-bottom:4px">
                Risk Percentile
            </div>
            <div style="font-size:20px;font-weight:800;color:{dot_color}">
                Top {max(1, 100 - int(percentile_rank(df["risk_pct"], selected["risk_pct"]) * 100))}%
            </div>
            <div style="font-size:10px;color:var(--sp-text-hint);margin-top:2px">
                riskiest in Tampa Bay
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_metrics:
    m1, m2, m3 = st.columns(3)
    days_val = int(selected.get("days_since_last_review", 0))
    months_val = float(selected.get("months_with_zero_reviews", 0))
    pct_5 = float(selected.get("pct_5star", 0))
    with m1:
        st.metric("Days Since Last Review", f"{days_val}d",
                  delta=None, help="Days elapsed from last review to anchor date")
    with m2:
        st.metric("Zero-Review Months", f"{months_val:.1f}",
                  help="Months in observation window with no new reviews")
    with m3:
        st.metric("5-Star Review Rate", f"{pct_5 * 100:.0f}%",
                  help="Fraction of all reviews that are 5-star")

    d1, d2 = st.columns(2)
    review_drought = bool(selected.get("review_drought_flag", 0))
    checkin_drought = bool(selected.get("checkin_drought_flag", 0))
    with d1:
        flag_color = "#E24B4A" if review_drought else "#1DB954"
        flag_text = "TRIGGERED" if review_drought else "CLEAR"
        st.markdown(f"""
        <div style="background:var(--sp-surface);border-radius:8px;padding:1rem 1.25rem;
                    border-left:3px solid {flag_color}">
            <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;
                        text-transform:uppercase;color:var(--sp-text-hint);margin-bottom:4px">
                Review Drought
            </div>
            <div style="font-size:20px;font-weight:800;color:{flag_color}">{flag_text}</div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        flag_color2 = "#E24B4A" if checkin_drought else "#1DB954"
        flag_text2 = "TRIGGERED" if checkin_drought else "CLEAR"
        st.markdown(f"""
        <div style="background:var(--sp-surface);border-radius:8px;padding:1rem 1.25rem;
                    border-left:3px solid {flag_color2}">
            <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;
                        text-transform:uppercase;color:var(--sp-text-hint);margin-bottom:4px">
                Check-in Drought
            </div>
            <div style="font-size:20px;font-weight:800;color:{flag_color2}">{flag_text2}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

chart_col, dist_col = st.columns([1.6, 1], gap="large")

with chart_col:
    st.markdown('<div class="section-label">Feature Contributions (SHAP)</div>', unsafe_allow_html=True)

    _business_id = selected.get("business_id", None)
    shap_vals, feat_cols = None, None
    if _business_id and _xgb_model is not None and _feat_matrix is not None:
        shap_vals, feat_cols = compute_shap_row(_xgb_model, _feat_matrix, str(_business_id))

    if shap_vals is not None:
        pairs = sorted(zip(shap_vals, feat_cols), key=lambda x: abs(x[0]), reverse=True)
        vals_s   = [p[0] for p in pairs]
        labels_s = [
            FEAT_LABELS.get(p[1], p[1].replace("_", " ").title())
            for p in pairs
        ]
        colors_s = ["#E24B4A" if v > 0 else "#1DB954" for v in vals_s]

        fig_feat = go.Figure(go.Bar(
            x=vals_s,
            y=labels_s,
            orientation="h",
            marker_color=colors_s,
            marker_line_width=0,
        ))
        fig_feat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Montserrat", color="#B3B3B3", size=12),
            margin=dict(l=0, r=16, t=4, b=4),
            height=280,
            xaxis=dict(
                zeroline=True, zerolinecolor="#3e3e3e", zerolinewidth=1,
                gridcolor="#282828", tickfont=dict(size=11, color="#535353"),
                title=dict(
                    text="← lowers risk  ·  raises risk →",
                    font=dict(size=10, color="#535353"),
                ),
            ),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11, color="#B3B3B3")),
            bargap=0.3,
        )
        st.plotly_chart(fig_feat, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown(
            '<div style="color:#535353;font-size:13px;padding:2rem">'
            "SHAP model not available — run 05_modeling.py first, or select a restaurant with matching data.</div>",
            unsafe_allow_html=True,
        )

with dist_col:
    st.markdown('<div class="section-label">Risk Distribution</div>', unsafe_allow_html=True)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df["risk_pct"],
        nbinsx=30,
        marker_color="#282828",
        name="All restaurants",
    ))
    fig_dist.add_vline(
        x=selected["risk_pct"],
        line_color=dot_color,
        line_width=2,
        annotation_text=f"{selected['risk_pct']:.0f}%",
        annotation_font_color=dot_color,
        annotation_font_size=11,
        annotation_font_family="Montserrat",
    )
    fig_dist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Montserrat", color="#B3B3B3", size=11),
        margin=dict(l=0, r=0, t=4, b=4),
        height=280,
        showlegend=False,
        bargap=0.05,
        xaxis=dict(
            gridcolor="#282828", tickfont=dict(size=10, color="#535353"),
            title=dict(text="Closure risk %", font=dict(size=10, color="#535353")),
        ),
        yaxis=dict(
            gridcolor="#282828", tickfont=dict(size=10, color="#535353"),
            title=dict(text="Count", font=dict(size=10, color="#535353")),
        ),
    )
    st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

st.markdown(f"""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #282828;
            display:flex;justify-content:space-between;align-items:center;
            font-size:11px;color:#535353">
    <span>ClosureWatch · ML Final Project · Tampa Bay, FL</span>
    <span>Yelp Academic Dataset · XGBoost Ensemble · AUC-ROC 0.700 · 5,143 restaurants</span>
    <span style="color:#B3B3B3;font-weight:700">📦 BATCH · {_batch_date}</span>
</div>
""", unsafe_allow_html=True)
