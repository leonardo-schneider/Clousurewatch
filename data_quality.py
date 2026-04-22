"""
data_quality.py — Pure functions for data quality checks.
Imported by 04b_data_quality.py and tests/test_data_quality.py.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

# COVID period boundaries (same as config_00.COVID_START / COVID_END)
_COVID_START = pd.Timestamp("2020-03-01")
_COVID_END   = pd.Timestamp("2021-06-01")

_VALID_PRICE_RANGE = {1.0, 2.0, 3.0, 4.0}

_EXPECTED_SCHEMA = {
    "business_id":           "object",
    "closed_within_6m":      "numeric",
    "anchor_date":           "datetime",
    "days_since_last_review":"numeric",
    "n_reviews_obs":         "numeric",
    "price_range":           "numeric",
    "review_velocity":       "numeric",
}


def check_duplicates(df: pd.DataFrame) -> dict:
    """Return dict with n_duplicates and ok flag."""
    n_dup = int(df["business_id"].duplicated().sum())
    return {"n_duplicates": n_dup, "ok": n_dup == 0}


def add_covid_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Return copy of df with integer covid_flag column."""
    out = df.copy()
    dates = pd.to_datetime(out["anchor_date"])
    out["covid_flag"] = ((dates >= _COVID_START) & (dates <= _COVID_END)).astype(int)
    return out


def check_price_range(df: pd.DataFrame) -> dict:
    """Return dict with n_invalid, invalid_values list, and ok flag."""
    valid_mask = df["price_range"].isna() | df["price_range"].isin(_VALID_PRICE_RANGE)
    bad = df.loc[~valid_mask, "price_range"]
    return {
        "n_invalid": int(len(bad)),
        "invalid_values": sorted(bad.unique().tolist()),
        "ok": len(bad) == 0,
    }


def detect_outliers(
    df: pd.DataFrame, cols: list[str], k: float = 3.0
) -> pd.DataFrame:
    """
    Return DataFrame of outlier rows using IQR method (value > Q3 + k*IQR).
    Columns: business_id, feature, value, threshold.
    """
    results = []
    for col in cols:
        s = df[col].dropna()
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        upper = q3 + k * iqr
        mask = df[col] > upper
        if mask.any():
            chunk = df.loc[mask, ["business_id", col]].copy()
            chunk = chunk.rename(columns={col: "value"})
            chunk["feature"] = col
            chunk["threshold"] = round(upper, 3)
            results.append(chunk[["business_id", "feature", "value", "threshold"]])
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["business_id", "feature", "value", "threshold"]
    )


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return list of schema violation strings. Empty list = clean."""
    issues = []
    for col, expected in _EXPECTED_SCHEMA.items():
        if col not in df.columns:
            issues.append(f"MISSING column: {col}")
            continue
        if expected == "datetime":
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                issues.append(f"TYPE {col}: expected datetime, got {df[col].dtype}")
        elif expected == "numeric":
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"TYPE {col}: expected numeric, got {df[col].dtype}")
        elif expected == "object":
            if not pd.api.types.is_object_dtype(df[col]):
                issues.append(f"TYPE {col}: expected object/str, got {df[col].dtype}")
    return issues
