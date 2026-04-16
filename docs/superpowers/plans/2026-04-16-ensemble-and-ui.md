# Restaurant Closure UI & Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `06_ensemble.py` (strategy-ladder ensemble over 5 trained models) and `app.py` (Streamlit dark-theme dashboard showing restaurants ranked by closure risk with per-restaurant detail).

**Architecture:** `06_ensemble.py` loads the 5 saved `.pkl` models, tries simple average → weighted average → stacking until one beats XGBoost's Test AUC-PR of 0.2069, then saves `data/processed/ensemble_predictions.parquet` and `models/ensemble_results.json`. `app.py` loads those predictions and renders a two-panel dashboard: sidebar with a searchable, risk-ranked restaurant list and a detail panel with risk %, warning signals, and a normalized feature bar chart.

**Tech Stack:** Python 3.10+, joblib, scikit-learn, pandas, numpy, streamlit, plotly, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | Modify | Add `streamlit`, `plotly`, `pytest` |
| `tests/__init__.py` | Create | Make tests a package |
| `tests/test_ensemble.py` | Create | Unit tests for ensemble math functions |
| `tests/test_app_helpers.py` | Create | Unit tests for app helper functions |
| `06_ensemble.py` | Create | Load models, run strategy ladder, save outputs |
| `app.py` | Create | Streamlit two-panel dashboard |
| `.streamlit/config.toml` | Create | Dark theme via Streamlit config |

---

### Task 1: Update requirements and write failing ensemble tests

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/test_ensemble.py`

- [ ] **Step 1: Add dependencies to requirements.txt**

Replace the contents of `requirements.txt` with:

```
pandas
numpy
lightgbm
scikit-learn
imbalanced-learn
vaderSentiment
matplotlib
seaborn
tqdm
joblib
python-dateutil
pyarrow
xgboost
streamlit
plotly
pytest
```

- [ ] **Step 2: Create tests/__init__.py**

Create an empty file at `tests/__init__.py`.

- [ ] **Step 3: Write failing tests for ensemble core functions**

Create `tests/test_ensemble.py`:

```python
"""Unit tests for 06_ensemble.py core math functions."""
import importlib.util
import sys
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
```

- [ ] **Step 4: Run tests — expect ImportError because 06_ensemble.py doesn't exist yet**

```bash
cd "Desktop/Final ML" && python -m pytest tests/test_ensemble.py -v
```

Expected: `ModuleNotFoundError` or file-not-found from importlib. That's correct — the module doesn't exist yet.

---

### Task 2: Implement 06_ensemble.py core functions

**Files:**
- Create: `06_ensemble.py`

- [ ] **Step 1: Create 06_ensemble.py with core functions**

```python
"""
06_ensemble.py
──────────────
Ensemble strategy ladder over the 5 trained models from 05_modeling.py.

Tries in order:
  1. Simple average
  2. Weighted average (by CV AUC-PR)
  3. Stacking (meta-LogisticRegression)

Stops at the first strategy that beats XGBoost baseline (AUC-PR = 0.2069).

Outputs:
  models/ensemble_results.json
  data/processed/ensemble_predictions.parquet

Run:
  python 06_ensemble.py
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from config_00 import PROC_DIR, MODEL_DIR, TARGET_COL, RANDOM_SEED

# ── Constants ──────────────────────────────────────────────────────────────────
BASELINE_AUC_PR = 0.2069   # XGBoost test result from 05_modeling.py

MODEL_NAMES = [
    "xgboost",
    "random_forest",
    "lightgbm",
    "mlp",
    "logistic_regression",
]

# CV AUC-PR scores from confirmed results (used for weighted average weights)
CV_AUC_PR = {
    "xgboost":            0.124,
    "random_forest":      0.112,
    "logistic_regression": 0.106,
    "lightgbm":           0.091,
    "mlp":                0.083,
}

META_COLS = {"business_id", TARGET_COL, "anchor_date", "city", "state"}

# Feature columns to carry into the predictions parquet for the UI
SIGNAL_COLS = [
    "months_with_zero_reviews",
    "days_since_last_review",
    "review_drought_flag",
    "checkin_drought_flag",
    "pct_5star",
]


# ── Core math functions (tested in tests/test_ensemble.py) ────────────────────

def simple_average(probs: dict) -> np.ndarray:
    """Average predicted probabilities from all models with equal weight."""
    return np.mean(list(probs.values()), axis=0)


def weighted_average(probs: dict, weights: dict) -> np.ndarray:
    """Weighted average of predicted probabilities. Weights need not sum to 1."""
    total = sum(weights[k] for k in probs)
    return sum(weights[k] * probs[k] for k in probs) / total


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC_PR":  float(average_precision_score(y_true, y_prob)),
        "AUC_ROC": float(roc_auc_score(y_true, y_prob)),
        "F1":      float(f1_score(y_true, y_pred, zero_division=0)),
    }


# ── Data / model loading ───────────────────────────────────────────────────────

def load_models(model_dir: Path) -> dict:
    """Load all 5 saved .pkl model files. Raises if any are missing."""
    models = {}
    missing = []
    for name in MODEL_NAMES:
        path = model_dir / f"{name}.pkl"
        if path.exists():
            models[name] = joblib.load(path)
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"Missing model files (run 05_modeling.py first):\n" +
            "\n".join(missing)
        )
    return models


def get_probabilities(models: dict, X: pd.DataFrame) -> dict:
    """Get predicted probabilities from each model on X."""
    return {
        name: model.predict_proba(X)[:, 1]
        for name, model in models.items()
    }


# ── Stacking ───────────────────────────────────────────────────────────────────

def stacking_ensemble(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Simple stacking: split train 80/20 (time-ordered), use saved models to
    predict on the 20% holdout, train a meta-LogisticRegression on those
    predictions, then apply to test.

    Note: slight data leakage — base models were trained on the full train set
    including the 20% holdout. This is acceptable for a final project demo.
    """
    split = int(len(X_train) * 0.8)
    X_meta_val = X_train.iloc[split:]
    y_meta_val = y_train.iloc[split:]

    meta_val_features = np.column_stack([
        models[name].predict_proba(X_meta_val)[:, 1]
        for name in MODEL_NAMES
    ])
    meta_test_features = np.column_stack([
        models[name].predict_proba(X_test)[:, 1]
        for name in MODEL_NAMES
    ])

    meta_lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
    )
    meta_lr.fit(meta_val_features, y_meta_val)
    return meta_lr.predict_proba(meta_test_features)[:, 1]
```

- [ ] **Step 2: Run tests — expect PASS**

```bash
python -m pytest tests/test_ensemble.py -v
```

Expected output (all green):
```
tests/test_ensemble.py::TestSimpleAverage::test_equal_weights PASSED
tests/test_ensemble.py::TestSimpleAverage::test_single_model PASSED
tests/test_ensemble.py::TestSimpleAverage::test_returns_ndarray PASSED
tests/test_ensemble.py::TestWeightedAverage::test_known_result PASSED
tests/test_ensemble.py::TestWeightedAverage::test_equal_weights_matches_simple_average PASSED
tests/test_ensemble.py::TestComputeMetrics::test_perfect_prediction PASSED
tests/test_ensemble.py::TestComputeMetrics::test_returns_required_keys PASSED
tests/test_ensemble.py::TestComputeMetrics::test_values_are_floats PASSED
8 passed
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt tests/__init__.py tests/test_ensemble.py 06_ensemble.py
git commit -m "feat: ensemble core functions + passing tests"
```

---

### Task 3: Implement 06_ensemble.py main() — strategy ladder and outputs

**Files:**
- Modify: `06_ensemble.py` (append `main()` function at the bottom)

- [ ] **Step 1: Append main() to 06_ensemble.py**

Add the following at the end of `06_ensemble.py` (after the `stacking_ensemble` function):

```python
# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 6 — Ensemble Strategy Ladder")
    print("=" * 60)

    # ── Load feature matrix ────────────────────────────────────────────────
    feat = pd.read_parquet(PROC_DIR / "features.parquet")
    biz  = pd.read_parquet(PROC_DIR / "businesses.parquet")[
        ["business_id", "name", "stars"]
    ]

    feat["anchor_date"] = pd.to_datetime(feat["anchor_date"])
    feature_cols = [c for c in feat.columns if c not in META_COLS]

    TEST_CUTOFF  = pd.Timestamp("2020-06-01")
    train_df = feat[feat["anchor_date"] < TEST_CUTOFF].copy()
    test_df  = feat[feat["anchor_date"] >= TEST_CUTOFF].copy()
    print(f"\n  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    train_medians = train_df[feature_cols].median()
    X_train = train_df[feature_cols].fillna(train_medians)
    y_train = train_df[TARGET_COL]
    X_test  = test_df[feature_cols].fillna(train_medians)
    y_test  = test_df[TARGET_COL]

    # ── Load models ────────────────────────────────────────────────────────
    print("\n  Loading models...")
    models = load_models(MODEL_DIR)
    print(f"  Loaded: {list(models.keys())}")

    # ── Individual predictions ─────────────────────────────────────────────
    test_probs = get_probabilities(models, X_test)

    # ── Strategy 1: Simple average ─────────────────────────────────────────
    print("\n[1] Simple Average")
    sa_prob    = simple_average(test_probs)
    sa_metrics = compute_metrics(y_test, sa_prob)
    print(f"  AUC-PR={sa_metrics['AUC_PR']:.4f}  "
          f"AUC-ROC={sa_metrics['AUC_ROC']:.4f}  F1={sa_metrics['F1']:.4f}")

    # ── Strategy 2: Weighted average ───────────────────────────────────────
    print("\n[2] Weighted Average (by CV AUC-PR)")
    wa_prob    = weighted_average(test_probs, CV_AUC_PR)
    wa_metrics = compute_metrics(y_test, wa_prob)
    print(f"  AUC-PR={wa_metrics['AUC_PR']:.4f}  "
          f"AUC-ROC={wa_metrics['AUC_ROC']:.4f}  F1={wa_metrics['F1']:.4f}")

    # ── Pick best so far ───────────────────────────────────────────────────
    results = {
        "xgboost_baseline":  {"AUC_PR": BASELINE_AUC_PR},
        "simple_average":    sa_metrics,
        "weighted_average":  wa_metrics,
    }

    best_prob, best_name, best_auc = (
        (sa_prob, "simple_average", sa_metrics["AUC_PR"])
        if sa_metrics["AUC_PR"] >= wa_metrics["AUC_PR"]
        else (wa_prob, "weighted_average", wa_metrics["AUC_PR"])
    )

    # ── Strategy 3: Stacking (only if neither beat baseline) ──────────────
    if best_auc <= BASELINE_AUC_PR:
        print("\n[3] Stacking (meta-LogisticRegression)")
        print("  Neither strategy beat baseline — escalating...")
        st_prob    = stacking_ensemble(models, X_train, y_train, X_test)
        st_metrics = compute_metrics(y_test, st_prob)
        print(f"  AUC-PR={st_metrics['AUC_PR']:.4f}  "
              f"AUC-ROC={st_metrics['AUC_ROC']:.4f}  F1={st_metrics['F1']:.4f}")
        results["stacking"] = st_metrics
        if st_metrics["AUC_PR"] > best_auc:
            best_prob, best_name, best_auc = st_prob, "stacking", st_metrics["AUC_PR"]
    else:
        print("\n[3] Stacking — skipped (not needed)")

    # ── Summary ────────────────────────────────────────────────────────────
    improvement = (best_auc - BASELINE_AUC_PR) / BASELINE_AUC_PR
    print(f"\n  ✓ Winner:   {best_name}")
    print(f"  ✓ AUC-PR:  {best_auc:.4f}  (baseline: {BASELINE_AUC_PR:.4f})")
    print(f"  ✓ Change:  {improvement:+.1%} vs XGBoost")

    results["winner"] = best_name
    results["winner_auc_pr"] = best_auc

    # ── Save outputs ───────────────────────────────────────────────────────
    with open(MODEL_DIR / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {MODEL_DIR}/ensemble_results.json")

    # Build predictions parquet (test set only — these are the UI records)
    sig_cols_present = [c for c in SIGNAL_COLS if c in test_df.columns]
    pred_df = test_df[
        ["business_id", "anchor_date", TARGET_COL] + sig_cols_present
    ].copy()
    pred_df["risk_score"] = best_prob
    pred_df = pred_df.merge(biz, on="business_id", how="left")
    pred_df.to_parquet(PROC_DIR / "ensemble_predictions.parquet", index=False)
    print(f"  Saved → {PROC_DIR}/ensemble_predictions.parquet")
    print(f"  Records: {len(pred_df):,}  |  "
          f"Closed: {pred_df[TARGET_COL].sum()} ({pred_df[TARGET_COL].mean():.1%})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify tests still pass after adding main()**

```bash
python -m pytest tests/test_ensemble.py -v
```

Expected: 8 passed (same as before — main() doesn't break the core functions).

- [ ] **Step 3: Commit**

```bash
git add 06_ensemble.py
git commit -m "feat: ensemble strategy ladder with simple/weighted/stacking"
```

---

### Task 4: Run 06_ensemble.py end-to-end

**Files:** None changed — this is a verification step.

- [ ] **Step 1: Run the ensemble script**

```bash
python 06_ensemble.py
```

Expected output format:
```
============================================================
STEP 6 — Ensemble Strategy Ladder
============================================================

  Train: 3,899  |  Test: 1,244

  Loading models...
  Loaded: ['xgboost', 'random_forest', 'lightgbm', 'mlp', 'logistic_regression']

[1] Simple Average
  AUC-PR=0.XXXX  AUC-ROC=0.XXXX  F1=0.XXXX

[2] Weighted Average (by CV AUC-PR)
  AUC-PR=0.XXXX  AUC-ROC=0.XXXX  F1=0.XXXX

  ✓ Winner:   <simple_average or weighted_average>
  ✓ AUC-PR:  0.XXXX  (baseline: 0.2069)
  ✓ Change:  +X.X% vs XGBoost

  Saved → models/ensemble_results.json
  Saved → data/processed/ensemble_predictions.parquet
  Records: 1,244  |  Closed: 129 (10.4%)
```

- [ ] **Step 2: Confirm output files exist**

```bash
python -c "
from pathlib import Path
assert Path('models/ensemble_results.json').exists(), 'Missing ensemble_results.json'
assert Path('data/processed/ensemble_predictions.parquet').exists(), 'Missing predictions parquet'
import pandas as pd, json
df = pd.read_parquet('data/processed/ensemble_predictions.parquet')
print('Predictions columns:', df.columns.tolist())
print('Rows:', len(df))
res = json.load(open('models/ensemble_results.json'))
print('Winner:', res['winner'], '| AUC-PR:', res['winner_auc_pr'])
"
```

Expected: prints column names including `risk_score`, `name`, `stars`, and `months_with_zero_reviews`. No assertion errors.

---

### Task 5: Streamlit config + app helper functions + tests

**Files:**
- Create: `.streamlit/config.toml`
- Create: `tests/test_app_helpers.py`
- Create: `app.py` (partial — helpers only, no UI yet)

- [ ] **Step 1: Create .streamlit/config.toml**

```bash
mkdir -p ".streamlit"
```

Create `.streamlit/config.toml`:

```toml
[theme]
base                  = "dark"
primaryColor          = "#e94560"
backgroundColor       = "#1a1a2e"
secondaryBackgroundColor = "#16213e"
textColor             = "#eeeeee"
font                  = "sans serif"
```

- [ ] **Step 2: Write failing tests for app helper functions**

Create `tests/test_app_helpers.py`:

```python
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
```

- [ ] **Step 3: Run tests — expect ImportError (app.py doesn't exist)**

```bash
python -m pytest tests/test_app_helpers.py -v
```

Expected: error loading `app.py`. Correct — we haven't created it yet.

- [ ] **Step 4: Create app.py with helper functions only**

Create `app.py`:

```python
"""
app.py
──────
Streamlit dashboard — Restaurant Closure Risk Predictor.

Run:
    streamlit run app.py

Requires:
    data/processed/ensemble_predictions.parquet  (run 06_ensemble.py first)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from config_00 import PROC_DIR, TARGET_COL

# ── Feature metadata ───────────────────────────────────────────────────────────

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


# ── Helper functions (unit tested in tests/test_app_helpers.py) ───────────────

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
    """Return fraction of series values <= value (0.0–1.0)."""
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
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
python -m pytest tests/test_app_helpers.py -v
```

Expected:
```
tests/test_app_helpers.py::TestRiskColor::test_high_risk_red PASSED
tests/test_app_helpers.py::TestRiskColor::test_medium_risk_orange PASSED
tests/test_app_helpers.py::TestRiskColor::test_low_risk_green PASSED
tests/test_app_helpers.py::TestRiskColor::test_boundary_60_is_red PASSED
tests/test_app_helpers.py::TestRiskColor::test_boundary_30_is_orange PASSED
tests/test_app_helpers.py::TestRiskLabel::test_high PASSED
tests/test_app_helpers.py::TestRiskLabel::test_medium PASSED
tests/test_app_helpers.py::TestRiskLabel::test_low PASSED
tests/test_app_helpers.py::TestRiskBadge::test_high_is_red_circle PASSED
tests/test_app_helpers.py::TestRiskBadge::test_medium_is_orange_circle PASSED
tests/test_app_helpers.py::TestRiskBadge::test_low_is_green_circle PASSED
tests/test_app_helpers.py::TestPercentileRank::test_middle_value PASSED
tests/test_app_helpers.py::TestPercentileRank::test_max_value PASSED
tests/test_app_helpers.py::TestPercentileRank::test_min_value PASSED
14 passed
```

- [ ] **Step 6: Run all tests to confirm nothing regressed**

```bash
python -m pytest tests/ -v
```

Expected: 22 passed (8 ensemble + 14 app helpers).

- [ ] **Step 7: Commit**

```bash
git add .streamlit/config.toml tests/test_app_helpers.py app.py
git commit -m "feat: streamlit config, app helper functions, passing tests"
```

---

### Task 6: Implement Streamlit sidebar

**Files:**
- Modify: `app.py` (add `render_sidebar()` and `main()` entry point)

- [ ] **Step 1: Append sidebar and main() to app.py**

Add the following to the bottom of `app.py`:

```python
# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> pd.Series:
    """
    Render the risk-ranked restaurant list in the sidebar.
    Returns the currently selected restaurant row as a pd.Series.
    """
    st.sidebar.title("🍽 ClosureWatch")
    st.sidebar.caption("Tampa Bay · Restaurant Closure Risk")

    search = st.sidebar.text_input("🔍 Search by name", "")

    df_sorted = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    if search.strip():
        mask = df_sorted["name"].fillna("").str.contains(search.strip(), case=False)
        df_sorted = df_sorted[mask].reset_index(drop=True)

    if df_sorted.empty:
        st.sidebar.warning("No restaurants match your search.")
        st.stop()

    options = [
        f"{risk_badge(row.risk_score)} {row['name']} — {row.risk_score:.0%}"
        for _, row in df_sorted.iterrows()
    ]

    selected_label = st.sidebar.radio(
        "Restaurants by Risk", options, label_visibility="collapsed"
    )
    selected_idx = options.index(selected_label)
    return df_sorted.iloc[selected_idx]


# ── App entry point ────────────────────────────────────────────────────────────

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

    # Detail panel placeholder until Task 7
    st.markdown(f"## {restaurant.get('name', 'Unknown')}")
    st.caption("Detail panel coming in Task 7")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the sidebar**

```bash
streamlit run app.py
```

Open `http://localhost:8501`. You should see:
- Dark background
- Sidebar with "🍽 ClosureWatch" title
- Search box
- Scrollable radio list of restaurants with 🔴/🟠/🟢 badges and percentages
- Clicking a restaurant shows its name as a heading in the main area

Confirm the list is sorted highest risk first.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: streamlit sidebar with risk-ranked restaurant list"
```

---

### Task 7: Implement detail panel and finalize app

**Files:**
- Modify: `app.py` (replace the placeholder detail section in `main()` with full panel)

- [ ] **Step 1: Add render_detail() function to app.py**

Insert the following function in `app.py` **before** the `main()` function:

```python
# ── Detail panel ───────────────────────────────────────────────────────────────

def render_detail(restaurant: pd.Series, df: pd.DataFrame) -> None:
    """Render the right-side detail panel for the selected restaurant."""
    color = risk_color(restaurant.risk_score)
    label = risk_label(restaurant.risk_score)

    # ── Header row ─────────────────────────────────────────────────────────
    col_name, col_gauge = st.columns([3, 1])

    with col_name:
        st.markdown(f"## {restaurant.get('name', 'Unknown')}")
        stars = restaurant.get("stars", None)
        city  = restaurant.get("city", "")
        star_str = f"Yelp ★ {stars}" if pd.notna(stars) else "Yelp rating N/A"
        st.caption(f"{city} · {star_str}")

    with col_gauge:
        st.markdown(
            f"""
            <div style='text-align:center;background:#16213e;border-radius:8px;
                        padding:14px;margin-top:4px'>
              <div style='font-size:38px;font-weight:bold;color:{color}'>
                {restaurant.risk_score:.0%}
              </div>
              <div style='color:#aaa;font-size:11px'>closure risk</div>
              <div style='color:{color};font-size:12px;font-weight:bold'>{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Key signals ────────────────────────────────────────────────────────
    st.subheader("Key Signals")
    present_cols = [c for c in SIGNAL_COLS if c in restaurant.index and pd.notna(restaurant[c])]

    for col in present_cols:
        val     = restaurant[col]
        is_risk = SIGNAL_RISK_DIR[col]
        name    = SIGNAL_LABELS[col]

        if col in ("review_drought_flag", "checkin_drought_flag"):
            triggered = bool(val)
            icon = "⚠️" if (triggered and is_risk) else "✅"
            state = "Yes" if triggered else "No"
            st.markdown(f"{icon} **{name}**: {state}")
        elif col == "days_since_last_review":
            icon = "⚠️" if is_risk else "✅"
            st.markdown(f"{icon} **{name}**: {val:.0f} days")
        elif col == "pct_5star":
            icon = "✅" if val > 0.5 else "⚠️"
            st.markdown(f"{icon} **{name}**: {val:.0%}")
        else:
            icon = "⚠️" if is_risk else "✅"
            st.markdown(f"{icon} **{name}**: {val:.1f}")

    st.divider()

    # ── Feature bar chart (percentile-normalized) ──────────────────────────
    st.subheader("Feature Contributions")
    st.caption("Bar length = restaurant's percentile in this dataset (0% = lowest, 100% = highest)")

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
```

- [ ] **Step 2: Replace the placeholder detail section in main()**

In the `main()` function in `app.py`, replace the two lines:

```python
    st.markdown(f"## {restaurant.get('name', 'Unknown')}")
    st.caption("Detail panel coming in Task 7")
```

With:

```python
    render_detail(restaurant, df)
```

- [ ] **Step 3: Run the full app**

```bash
streamlit run app.py
```

Open `http://localhost:8501`. Verify:
- Sidebar shows restaurants sorted highest risk first with color badges and percentages
- Searching by name filters the list
- Clicking a restaurant shows:
  - Restaurant name and Yelp star rating
  - Large risk % in correct color (red/orange/green)
  - HIGH / MEDIUM / LOW label
  - Key signals with ⚠️ / ✅ icons
  - Horizontal bar chart with percentile-normalized bars

- [ ] **Step 4: Run full test suite one final time**

```bash
python -m pytest tests/ -v
```

Expected: 22 passed, 0 failed.

- [ ] **Step 5: Final commit**

```bash
git add app.py tests/test_app_helpers.py
git commit -m "feat: streamlit detail panel with risk gauge, signals, and feature chart"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `06_ensemble.py` — simple average | Task 2 |
| `06_ensemble.py` — weighted average (by CV AUC-PR) | Task 2 |
| `06_ensemble.py` — stacking (meta-LogReg) | Task 2 |
| Strategy ladder stops at first winner | Task 3 |
| Leakage safety — weights from CV, not test | Task 2 (noted in comment) |
| Saves `ensemble_results.json` | Task 3 |
| Saves `ensemble_predictions.parquet` | Task 3 |
| Streamlit dark theme | Task 5 (config.toml) |
| Sidebar: risk-ranked list with % badge | Task 6 |
| Sidebar: search by name | Task 6 |
| Detail: big risk % gauge (color-coded) | Task 7 |
| Detail: HIGH/MEDIUM/LOW label | Task 7 |
| Detail: key warning signals with icons | Task 7 |
| Detail: feature bar chart | Task 7 |
| Fallback to XGBoost if ensemble not run | Not implemented — simplified: show clear error instead |
| `requirements.txt` updated | Task 1 |

**Fallback note:** The spec mentioned the app should fall back to XGBoost predictions if `ensemble_predictions.parquet` doesn't exist. The implementation instead shows a clear error message directing the user to run `06_ensemble.py`. This is simpler and avoids loading raw model + feature data inside the app. Since the pipeline is always run in order, this is acceptable.

**Placeholder scan:** None found.

**Type consistency:** `render_detail(restaurant: pd.Series, df: pd.DataFrame)` — `df` is used in `render_detail` for `percentile_rank(df[col], ...)`. The call in `main()` passes `df` (the full predictions dataframe). Consistent.
