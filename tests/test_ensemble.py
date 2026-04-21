"""Unit tests for 06_ensemble.py core math functions."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

# 06_ensemble.py starts with a digit — use importlib to load it
_spec = importlib.util.spec_from_file_location(
    "ensemble_06",
    Path(__file__).parent.parent / "06_ensemble.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

simple_average = _mod.simple_average
weighted_average = _mod.weighted_average
compute_metrics = _mod.compute_metrics


class TestSimpleAverage:
    def test_equal_weights(self):
        probs = {
            "a": np.array([0.2, 0.8, 0.5]),
            "b": np.array([0.4, 0.6, 0.3]),
        }
        result = simple_average(probs)
        np.testing.assert_allclose(result, np.array([0.3, 0.7, 0.4]))

    def test_single_model(self):
        probs = {"only": np.array([0.1, 0.9])}
        result = simple_average(probs)
        np.testing.assert_allclose(result, np.array([0.1, 0.9]))

    def test_returns_ndarray(self):
        probs = {"a": np.array([0.5]), "b": np.array([0.5])}
        assert isinstance(simple_average(probs), np.ndarray)


class TestWeightedAverage:
    def test_known_result(self):
        probs = {
            "a": np.array([0.2, 0.8]),
            "b": np.array([0.4, 0.6]),
        }
        weights = {"a": 0.7, "b": 0.3}
        result = weighted_average(probs, weights)
        expected = 0.7 * np.array([0.2, 0.8]) + 0.3 * np.array([0.4, 0.6])
        np.testing.assert_allclose(result, expected)

    def test_equal_weights_matches_simple_average(self):
        probs = {
            "a": np.array([0.3, 0.7]),
            "b": np.array([0.5, 0.5]),
        }
        weights = {"a": 0.5, "b": 0.5}
        result = weighted_average(probs, weights)
        np.testing.assert_allclose(result, simple_average(probs))


class TestComputeMetrics:
    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.01, 0.02, 0.98, 0.99])
        metrics = compute_metrics(y_true, y_prob)
        assert metrics["AUC_PR"] > 0.99
        assert metrics["AUC_ROC"] > 0.99

    def test_returns_required_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.3, 0.7, 0.4, 0.6])
        metrics = compute_metrics(y_true, y_prob)
        assert set(metrics.keys()) == {"AUC_PR", "AUC_ROC", "F1"}

    def test_values_are_floats(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.2, 0.8])
        metrics = compute_metrics(y_true, y_prob)
        for v in metrics.values():
            assert isinstance(v, float)


find_optimal_threshold = _mod.find_optimal_threshold


class TestFindOptimalThreshold:
    def test_returns_float_in_unit_interval(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 100)
        y_prob  = rng.uniform(0, 1, 100)
        t = find_optimal_threshold(y_true, y_prob)
        assert isinstance(t, float)
        assert 0.0 <= t <= 1.0

    def test_perfect_separation_threshold_near_midpoint(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob  = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
        t = find_optimal_threshold(y_true, y_prob)
        assert 0.3 <= t <= 0.6

    def test_imbalanced_threshold_below_half(self):
        rng = np.random.default_rng(1)
        y_true = np.array([1] * 10 + [0] * 90)
        y_prob  = np.where(y_true == 1,
                           rng.uniform(0.3, 0.9, 100),
                           rng.uniform(0.0, 0.5, 100))
        t = find_optimal_threshold(y_true, y_prob)
        assert t < 0.5
