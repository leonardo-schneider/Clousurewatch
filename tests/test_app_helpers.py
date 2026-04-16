"""Unit tests for app.py helper functions."""
import importlib.util
from pathlib import Path
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location("app", Path(__file__).parent.parent / "app.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

risk_color    = _mod.risk_color
risk_label    = _mod.risk_label
risk_badge    = _mod.risk_badge
percentile_rank = _mod.percentile_rank


class TestRiskColor:
    def test_high_risk_red(self):
        assert risk_color(0.75) == "#e94560"

    def test_medium_risk_orange(self):
        assert risk_color(0.45) == "#f7a440"

    def test_low_risk_green(self):
        assert risk_color(0.15) == "#4caf50"

    def test_boundary_60_is_red(self):
        assert risk_color(0.60) == "#e94560"

    def test_boundary_30_is_orange(self):
        assert risk_color(0.30) == "#f7a440"


class TestRiskLabel:
    def test_high(self):
        assert risk_label(0.80) == "HIGH"

    def test_medium(self):
        assert risk_label(0.50) == "MEDIUM"

    def test_low(self):
        assert risk_label(0.10) == "LOW"


class TestRiskBadge:
    def test_high_is_red_circle(self):
        assert risk_badge(0.70) == "🔴"

    def test_medium_is_orange_circle(self):
        assert risk_badge(0.45) == "🟠"

    def test_low_is_green_circle(self):
        assert risk_badge(0.20) == "🟢"


class TestPercentileRank:
    def test_middle_value(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert percentile_rank(series, 3.0) == pytest.approx(0.6)

    def test_max_value(self):
        series = pd.Series([1.0, 2.0, 3.0])
        assert percentile_rank(series, 3.0) == pytest.approx(1.0)

    def test_min_value(self):
        series = pd.Series([1.0, 2.0, 3.0])
        assert percentile_rank(series, 1.0) == pytest.approx(1 / 3)
