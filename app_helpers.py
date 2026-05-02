"""Pure helper functions for the ClosureWatch dashboard. No Streamlit imports."""
from __future__ import annotations
import pandas as pd


def risk_color(pct: float) -> str:
    """Hex color for a risk score in [0.0, 1.0]."""
    if pct >= 0.60:
        return "#e94560"
    if pct >= 0.30:
        return "#f7a440"
    return "#4caf50"


def risk_label(pct: float) -> str:
    """Tier label for a risk score in [0.0, 1.0]."""
    if pct >= 0.60:
        return "HIGH"
    if pct >= 0.30:
        return "MEDIUM"
    return "LOW"


def risk_badge(pct: float) -> str:
    """Emoji badge for a risk score in [0.0, 1.0]."""
    if pct >= 0.60:
        return "🔴"
    if pct >= 0.30:
        return "🟠"
    return "🟢"


def percentile_rank(series: pd.Series, value: float) -> float:
    """Fraction of values in series that are <= value."""
    return float((series <= value).mean())


def outcome_banner_html(row: pd.Series) -> str:
    """
    Return an HTML string for the ground truth outcome banner.
    Returns empty string if closed_within_6m column is not present.
    """
    if "closed_within_6m" not in row.index:
        return ""

    anchor_str = ""
    if "anchor_date" in row.index and pd.notna(row["anchor_date"]):
        anchor_str = pd.Timestamp(row["anchor_date"]).strftime("Anchor: %b %Y")

    if int(row["closed_within_6m"]) == 1:
        return (
            '<div style="background:#5c1a1a;border-left:4px solid #E24B4A;'
            'padding:6px 12px;border-radius:4px;margin-bottom:12px;'
            'display:flex;justify-content:space-between;align-items:center">'
            '<span style="color:#E24B4A;font-size:11px;font-weight:700">'
            "✓ OUTCOME KNOWN · PERMANENTLY CLOSED</span>"
            f'<span style="color:#666;font-size:10px">{anchor_str}</span>'
            "</div>"
        )
    return (
        '<div style="background:#0d3320;border-left:4px solid #1DB954;'
        'padding:6px 12px;border-radius:4px;margin-bottom:12px;'
        'display:flex;justify-content:space-between;align-items:center">'
        '<span style="color:#1DB954;font-size:11px;font-weight:700">'
        "✓ OUTCOME KNOWN · STILL OPEN</span>"
        f'<span style="color:#666;font-size:10px">{anchor_str}</span>'
        "</div>"
    )


def compute_shap_row(model, feature_matrix: "pd.DataFrame", business_id: str):
    """
    Compute SHAP values for one restaurant. Returns (shap_values_list, feature_names_list)
    or (None, None) if the model or row is unavailable.
    """
    import shap
    import numpy as np

    if model is None or feature_matrix is None:
        return None, None

    # Use model's own feature names so the column list always matches training exactly,
    # regardless of extra columns (lat/lon, risk_score, etc.) added by the app.
    booster = model.get_booster()
    feat_cols = booster.feature_names  # list in training order

    feature_matrix = feature_matrix.copy()
    feature_matrix["anchor_date"] = pd.to_datetime(feature_matrix["anchor_date"])

    row = feature_matrix[feature_matrix["business_id"] == business_id]
    if row.empty:
        return None, None

    TEST_CUTOFF = "2020-06-01"
    train_rows = feature_matrix[feature_matrix["anchor_date"] < TEST_CUTOFF]
    available = [c for c in feat_cols if c in feature_matrix.columns]
    train_medians = train_rows[available].median()
    X = row.reindex(columns=feat_cols).fillna(train_medians)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    # sv may be 1D array, 2D array, or list-of-arrays depending on XGBoost version
    if isinstance(sv, list):
        # list of arrays (one per class) — take positive class
        arr = np.array(sv[-1])
    else:
        arr = np.array(sv)
    if arr.ndim == 2:
        arr = arr[0]
    return arr.tolist(), feat_cols
