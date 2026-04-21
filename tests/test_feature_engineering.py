"""Unit tests for momentum helpers in 03_feature_engineering.py."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "feat_eng",
    Path(__file__).parent.parent / "03_feature_engineering.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_momentum = _mod.compute_momentum


class TestComputeMomentum:
    def test_declining(self):
        assert compute_momentum(10, 2) < 1.0

    def test_growing(self):
        assert compute_momentum(2, 10) > 1.0

    def test_zero_first_half(self):
        assert compute_momentum(0, 5) == pytest.approx(5.0)

    def test_both_zero(self):
        assert compute_momentum(0, 0) == pytest.approx(0.0)

    def test_equal_halves(self):
        assert compute_momentum(5, 5) == pytest.approx(5 / 6)

    def test_returns_float(self):
        assert isinstance(compute_momentum(3, 3), float)
