"""Tests for data_quality.py quality check functions."""
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_quality import (
    check_duplicates,
    add_covid_flag,
    check_price_range,
    detect_outliers,
    validate_schema,
)


def _sample_df():
    return pd.DataFrame({
        "business_id":           ["A", "B", "C"],
        "anchor_date":           pd.to_datetime(["2019-01-01", "2020-06-01", "2021-01-01"]),
        "closed_within_6m":      [0, 1, 0],
        "days_since_last_review":[10.0, 200.0, 50.0],
        "n_reviews_obs":         [5.0, 3.0, 100.0],
        "price_range":           [1.0, 2.0, 4.0],
        "review_velocity":       [0.5, 0.2, 8.0],
    })


# ── 1a duplicates ──────────────────────────────────────────────────────────
def test_check_duplicates_clean():
    result = check_duplicates(_sample_df())
    assert result["n_duplicates"] == 0
    assert result["ok"] is True


def test_check_duplicates_finds_dup():
    df = _sample_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    result = check_duplicates(df)
    assert result["n_duplicates"] == 1
    assert result["ok"] is False


# ── 1b COVID flag ──────────────────────────────────────────────────────────
def test_add_covid_flag_labels_correctly():
    df = _sample_df()
    out = add_covid_flag(df)
    # 2019-01-01 → not covid; 2020-06-01 → covid; 2021-01-01 → covid
    assert list(out["covid_flag"]) == [0, 1, 1]


def test_add_covid_flag_does_not_mutate_input():
    df = _sample_df()
    add_covid_flag(df)
    assert "covid_flag" not in df.columns


# ── 1c price_range ─────────────────────────────────────────────────────────
def test_check_price_range_valid():
    result = check_price_range(_sample_df())
    assert result["ok"] is True
    assert result["n_invalid"] == 0


def test_check_price_range_catches_invalid():
    df = _sample_df()
    df.loc[0, "price_range"] = 7.0
    result = check_price_range(df)
    assert result["ok"] is False
    assert result["n_invalid"] == 1


# ── 1d outliers ────────────────────────────────────────────────────────────
def test_detect_outliers_finds_extreme():
    df = _sample_df()
    # review_velocity: values [0.5, 0.2, 8.0]. 8.0 is extreme.
    out = detect_outliers(df, ["review_velocity"], k=0.5)
    assert len(out) > 0
    assert "review_velocity" in out["feature"].values


def test_detect_outliers_clean_data():
    df = _sample_df()
    out = detect_outliers(df, ["price_range"], k=3.0)
    assert len(out) == 0


# ── 1e schema ──────────────────────────────────────────────────────────────
def test_validate_schema_clean():
    issues = validate_schema(_sample_df())
    assert issues == []


def test_validate_schema_missing_column():
    df = _sample_df().drop(columns=["closed_within_6m"])
    issues = validate_schema(df)
    assert any("closed_within_6m" in i for i in issues)
