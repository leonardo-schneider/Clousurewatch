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
