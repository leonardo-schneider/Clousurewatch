# Dashboard & Model Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix encoding corruption, add real SHAP explanations, show ground truth outcomes, and improve the ensemble model with momentum features and threshold optimization.

**Architecture:** Track 1 (app changes) runs first — no pipeline re-run needed, works with existing `models/xgboost.pkl` and `data/processed/ensemble_predictions.parquet`. Pure helper functions are extracted to `app_helpers.py` so they can be unit-tested without triggering Streamlit. Track 2 (pipeline changes) runs after: add momentum features to `03_feature_engineering.py`, slim the ensemble to top-3 models, and find the F1-optimal threshold in `06_ensemble.py`.

**Tech Stack:** Python 3.10+, Streamlit ≥ 1.35, shap, ftfy, XGBoost, scikit-learn, pandas, numpy, pytest

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `app_helpers.py` | Pure helper functions (risk tier, percentile rank, SHAP, outcome banner) |
| Modify | `tests/test_app_helpers.py` | Update import path to `app_helpers.py` |
| Modify | `app.py` | Import helpers, fix encoding, SHAP chart, outcome banner, interactive table, BATCH badge |
| Modify | `03_feature_engineering.py` | Add `compute_momentum` helper + two momentum features |
| Create | `tests/test_feature_engineering.py` | Tests for `compute_momentum` |
| Modify | `06_ensemble.py` | Slim to top-3 models, add `find_optimal_threshold`, use it in `main()` |
| Modify | `tests/test_ensemble.py` | Tests for `find_optimal_threshold` |
| Modify | `CLAUDE.md` | Document two new features |

---

## Task 1: Create app_helpers.py with pure helper functions

**Files:**
- Create: `app_helpers.py`
- Modify: `tests/test_app_helpers.py`

The existing `tests/test_app_helpers.py` imports from `app.py` directly. Since `app.py` calls `st.set_page_config()` at module level, importing it outside of a Streamlit context crashes. Extract the pure functions to `app_helpers.py` and point the test there.

- [ ] **Step 1: Run the existing tests to confirm they fail**

```bash
cd "c:/Users/leona/Desktop/Final ML"
pytest tests/test_app_helpers.py -v
```
Expected: errors — module-level Streamlit calls crash on import, and `risk_color`/`risk_label`/`risk_badge`/`percentile_rank` don't exist yet.

- [ ] **Step 2: Create `app_helpers.py` with the pure helper functions**

```python
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
```

- [ ] **Step 3: Update `tests/test_app_helpers.py` to import from `app_helpers`**

Replace the import block at the top of the file:

Old:
```python
_spec = importlib.util.spec_from_file_location("app", Path(__file__).parent.parent / "app.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

risk_color    = _mod.risk_color
risk_label    = _mod.risk_label
risk_badge    = _mod.risk_badge
percentile_rank = _mod.percentile_rank
```

New:
```python
_spec = importlib.util.spec_from_file_location(
    "app_helpers", Path(__file__).parent.parent / "app_helpers.py"
)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

risk_color      = _mod.risk_color
risk_label      = _mod.risk_label
risk_badge      = _mod.risk_badge
percentile_rank = _mod.percentile_rank
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
pytest tests/test_app_helpers.py -v
```
Expected: all 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add app_helpers.py tests/test_app_helpers.py
git commit -m "feat: extract pure app helpers to app_helpers.py, wire existing tests"
```

---

## Task 2: Fix encoding corruption in app.py

**Files:**
- Modify: `app.py`

The file has double-encoded UTF-8 (saved as cp1252 by mistake). Characters like `★` were UTF-8 encoded, those bytes treated as cp1252, then re-encoded, resulting in mojibake like `â˜…`. Fix with a one-shot decode trick.

- [ ] **Step 1: Run the encoding fixer**

```bash
python -c "
from pathlib import Path
content = Path('app.py').read_text(encoding='utf-8')
fixed = content.encode('cp1252').decode('utf-8')
Path('app.py').write_text(fixed, encoding='utf-8')
print('Done. Spot-check:')
import re
stars = re.findall(r'[★☆📡🔴🟡🟢↓↑▶▸]', fixed)
print(f'  Found {len(stars)} correctly-decoded special characters')
"
```
Expected: prints a count of correctly-decoded special characters > 0.

- [ ] **Step 2: Verify the file looks correct**

```bash
python -c "
content = open('app.py', encoding='utf-8').read()
# Check a known string that was mojibake
assert '★' in content or '☆' in content or '📡' in content, 'Encoding still broken'
print('Encoding OK')
"
```
Expected: prints `Encoding OK`.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "fix: repair cp1252/utf-8 double-encoding corruption in app.py"
```

---

## Task 3: Wire app_helpers into app.py — percentile rank and BATCH badge

**Files:**
- Modify: `app.py`

Replace the fake "model confidence" bar with a percentile rank, and change the `LIVE` badge to `BATCH` with a file date.

- [ ] **Step 1: Add imports at the top of app.py**

After the existing imports block (after `import plotly.graph_objects as go`), add:

```python
from app_helpers import (
    risk_color,
    risk_label,
    risk_badge,
    percentile_rank,
    outcome_banner_html,
)
```

- [ ] **Step 2: Replace the `risk_tier` usages with new helpers**

The existing `risk_tier(pct)` function takes a 0-100 value and returns `(tier_name, css_class, dot_color)`. Keep it in place but rewrite it to delegate to the new helpers so existing call sites still work:

```python
def risk_tier(pct: float) -> tuple[str, str, str]:
    """Return tier name, CSS class, and hex color. pct is 0–100."""
    frac = pct / 100.0
    name = risk_label(frac).replace("MEDIUM", "ELEVATED")
    css  = {"HIGH": "risk-high", "ELEVATED": "risk-med", "LOW": "risk-low"}[name]
    col  = risk_color(frac)
    return name, css, col
```

- [ ] **Step 3: Replace the model confidence bar with percentile rank**

Find this block in app.py (inside `with col_risk:`):
```python
        <div style="margin-top:14px;padding-top:12px;border-top:1px solid #282828;">
            <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;
                        text-transform:uppercase;color:var(--sp-text-hint);margin-bottom:4px">
                Model confidence
            </div>
            <div style="background:#282828;border-radius:3px;height:4px;overflow:hidden">
                <div style="background:{dot_color};height:100%;
                            width:{min(selected['risk_pct'] * 1.1, 100):.0f}%;
                            border-radius:3px;transition:width 0.4s"></div>
            </div>
        </div>
```

Replace with:
```python
        <div style="margin-top:14px;padding-top:12px;border-top:1px solid #282828;">
            <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;
                        text-transform:uppercase;color:var(--sp-text-hint);margin-bottom:4px">
                Risk Percentile
            </div>
            <div style="font-size:20px;font-weight:800;color:{dot_color}">
                Top {100 - int(percentile_rank(df["risk_pct"], selected["risk_pct"]) * 100)}%
            </div>
            <div style="font-size:10px;color:var(--sp-text-hint);margin-top:2px">
                riskiest in Tampa Bay
            </div>
        </div>
```

- [ ] **Step 4: Fix the BATCH badge in the footer**

Find the footer line:
```python
    <span style="color:#1DB954;font-weight:700">🔴 LIVE</span>
```

Replace with (compute the date dynamically from the parquet file):
```python
import os
_parquet_path = Path("data/processed/ensemble_predictions.parquet")
_batch_date = (
    pd.Timestamp(os.path.getmtime(_parquet_path), unit="s").strftime("%b %Y")
    if _parquet_path.exists()
    else "Unknown"
)
```

Add that block near the top of the file (after `df = load_data()`), then replace the footer span with:

```python
    <span style="color:#B3B3B3;font-weight:700">📦 BATCH · {_batch_date}</span>
```

- [ ] **Step 5: Run the app to visually confirm**

```bash
streamlit run app.py
```
Check: risk card now shows "Top X% riskiest in Tampa Bay" instead of a progress bar. Footer shows "📦 BATCH · [month]".

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: replace fake confidence bar with percentile rank, fix LIVE to BATCH badge"
```

---

## Task 4: Add SHAP on-the-fly feature contributions

**Files:**
- Modify: `app.py`

Replace the fake `feat_*` bar chart with real SHAP values computed from the saved XGBoost model.

- [ ] **Step 1: Install shap**

```bash
pip install shap
```

- [ ] **Step 2: Add cached model and feature matrix loaders**

Add these two functions after the `load_data()` function in `app.py`:

```python
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
```

Then load them after `df = load_data()`:

```python
_xgb_model = load_xgb_model()
_feat_matrix = load_feature_matrix()
```

- [ ] **Step 3: Add the SHAP computation function in app_helpers.py**

Append to `app_helpers.py`:

```python
def compute_shap_row(model, feature_matrix: "pd.DataFrame", business_id: str):
    """
    Compute SHAP values for one restaurant. Returns (shap_values_array, feature_names)
    or (None, None) if the model or row is unavailable.
    """
    import shap
    import numpy as np

    if model is None or feature_matrix is None:
        return None, None

    _META = {"business_id", "closed_within_6m", "anchor_date", "city", "state"}
    feat_cols = [c for c in feature_matrix.columns if c not in _META]

    row = feature_matrix[feature_matrix["business_id"] == business_id]
    if row.empty:
        return None, None

    TEST_CUTOFF = "2020-06-01"
    train_rows = feature_matrix[feature_matrix["anchor_date"] < TEST_CUTOFF]
    train_medians = train_rows[feat_cols].median()
    X = row[feat_cols].fillna(train_medians)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    return sv[0].tolist(), feat_cols
```

Also add `compute_shap_row` to the imports in `app.py`:

```python
from app_helpers import (
    risk_color,
    risk_label,
    risk_badge,
    percentile_rank,
    outcome_banner_html,
    compute_shap_row,
)
```

- [ ] **Step 4: Replace the fake feature chart with the SHAP chart**

Find the `with chart_col:` block in `app.py`. Replace the entire contents of that block with:

```python
with chart_col:
    st.markdown('<div class="section-label">Feature Contributions (SHAP)</div>', unsafe_allow_html=True)

    _business_id = selected.get("business_id", None)
    shap_vals, feat_cols = (None, None)
    if _business_id and _xgb_model is not None and _feat_matrix is not None:
        shap_vals, feat_cols = compute_shap_row(_xgb_model, _feat_matrix, _business_id)

    if shap_vals is not None:
        pairs = sorted(zip(shap_vals, feat_cols), key=lambda x: abs(x[0]), reverse=True)
        vals_s  = [p[0] for p in pairs]
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
            "SHAP model not available — run 05_modeling.py first.</div>",
            unsafe_allow_html=True,
        )
```

Note: the `import plotly.graph_objects as go as _go` line is wrong — `go` is already imported at the top of app.py. Remove that line; just use `go` directly.

- [ ] **Step 5: Run the app and confirm SHAP bars appear**

```bash
streamlit run app.py
```
Select a restaurant. The Feature Contributions chart should show bars with numeric SHAP values (not the [-1, 1] scaled fake values). Red bars indicate features that push the prediction toward closure; green bars push it away.

- [ ] **Step 6: Commit**

```bash
git add app.py app_helpers.py
git commit -m "feat: replace fake feature chart with on-the-fly SHAP values from XGBoost"
```

---

## Task 5: Add ground truth outcome banner

**Files:**
- Modify: `app.py`

`ensemble_predictions.parquet` contains `closed_within_6m` for all records (test-set only). Show an outcome banner above the restaurant name for every restaurant.

- [ ] **Step 1: Add banner just above the header markdown in app.py**

Find the header block:
```python
st.markdown(f"""
<div class="cw-header">
    <div class="cw-brand">📡 ClosureWatch</div>
    <div class="cw-title">{selected['name']}</div>
```

Replace with:
```python
_banner_html = outcome_banner_html(selected)
st.markdown(f"""
<div class="cw-header">
    {_banner_html}
    <div class="cw-brand">📡 ClosureWatch</div>
    <div class="cw-title">{selected['name']}</div>
```

- [ ] **Step 2: Run the app and confirm the banner**

```bash
streamlit run app.py
```
Select a restaurant that actually closed — the red "✓ OUTCOME KNOWN · PERMANENTLY CLOSED" banner should appear above the name. Select one that stayed open — green banner appears.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add ground truth outcome banner above restaurant header"
```

---

## Task 6: Replace static Watch List table with interactive st.dataframe

**Files:**
- Modify: `app.py`

Clicking a row in the Watch List should navigate to that restaurant's detail view.

- [ ] **Step 1: Replace the Plotly table block**

Find the entire block from `st.markdown('<div class="section-label">🔴 Highest Risk — Watch List</div>` down through `st.plotly_chart(fig_table, ...)`. Replace it all with:

```python
st.markdown('<div class="section-label">🔴 Highest Risk — Watch List</div>', unsafe_allow_html=True)

top20 = df.head(20).copy()
top20_display = pd.DataFrame({
    "Restaurant": top20["name"].values,
    "Risk %":     top20["risk_pct"].apply(lambda x: f"{x:.1f}%").values,
    "Tier":       top20["risk_pct"].apply(lambda x: risk_tier(x)[0]).values,
    "Stars ⭐":   top20.get("stars", pd.Series(["-"] * len(top20))).apply(
                      lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x)
                  ).values,
})

event = st.dataframe(
    top20_display,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
    hide_index=True,
)

if event.selection.rows:
    clicked_local = event.selection.rows[0]
    new_idx = int(top20.index[clicked_local])
    if new_idx != st.session_state.selected_idx:
        st.session_state.selected_idx = new_idx
        st.rerun()
```

- [ ] **Step 2: Run the app and confirm row selection works**

```bash
streamlit run app.py
```
Click a row in the Watch List table — the detail panel above should update to show that restaurant.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: replace static Plotly table with interactive st.dataframe row selection"
```

---

## Task 7: Add momentum features to 03_feature_engineering.py

**Files:**
- Modify: `03_feature_engineering.py`
- Create: `tests/test_feature_engineering.py`

Add `review_momentum` and `checkin_momentum` — ratio of second-half to first-half activity within the 12-month observation window. Extract a `compute_momentum` helper for testability.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_feature_engineering.py`:

```python
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
        # More activity in first half → ratio < 1
        assert compute_momentum(10, 2) < 1.0

    def test_growing(self):
        # More activity in second half → ratio > 1
        assert compute_momentum(2, 10) > 1.0

    def test_zero_first_half(self):
        # Denominator is first+1 to avoid division by zero
        assert compute_momentum(0, 5) == pytest.approx(5.0)

    def test_both_zero(self):
        assert compute_momentum(0, 0) == pytest.approx(0.0)

    def test_equal_halves(self):
        assert compute_momentum(5, 5) == pytest.approx(5 / 6)

    def test_returns_float(self):
        assert isinstance(compute_momentum(3, 3), float)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_feature_engineering.py -v
```
Expected: `AttributeError: module 'feat_eng' has no attribute 'compute_momentum'`

- [ ] **Step 3: Add `compute_momentum` helper and momentum features to `03_feature_engineering.py`**

After the `trend_early_vs_late` function (around line 83), add:

```python
def compute_momentum(first_half_count: int, second_half_count: int) -> float:
    """Ratio of second-half to first-half event count. < 1.0 means declining activity."""
    return float(second_half_count) / (first_half_count + 1)
```

Then inside `build_features_one()`, after the checkin signals block (after line 226, `feat["checkin_drought_flag"] = ...`), add:

```python
    # ── H. Momentum features (decline rate, no leakage) ───────────────────────
    mid_date = obs_start + relativedelta(months=OBS_MONTHS // 2)

    # Review momentum: second 6 months vs first 6 months
    if n_rev > 0:
        first_half_rev = obs_reviews[obs_reviews["date"] < mid_date]
        last_half_rev  = obs_reviews[obs_reviews["date"] >= mid_date]
        feat["review_momentum"] = compute_momentum(len(first_half_rev), len(last_half_rev))
    else:
        feat["review_momentum"] = 0.0

    # Check-in momentum: second 6 months vs first 6 months
    if n_checkins > 0:
        first_half_ci = obs_checkins[obs_checkins["checkin_date"] < mid_date]
        last_half_ci  = obs_checkins[obs_checkins["checkin_date"] >= mid_date]
        feat["checkin_momentum"] = compute_momentum(len(first_half_ci), len(last_half_ci))
    else:
        feat["checkin_momentum"] = 0.0
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_feature_engineering.py -v
```
Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add 03_feature_engineering.py tests/test_feature_engineering.py
git commit -m "feat: add review_momentum and checkin_momentum decline-rate features"
```

---

## Task 8: Top-3 ensemble + F1-optimal threshold in 06_ensemble.py

**Files:**
- Modify: `06_ensemble.py`
- Modify: `tests/test_ensemble.py`

Slim the ensemble to the top-3 models by CV AUC-PR (removes MLP and LightGBM). Add `find_optimal_threshold` to replace the hardcoded 0.5 threshold in F1 reporting.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ensemble.py`:

```python
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
        # Optimal cut is between 0.3 and 0.6
        assert 0.3 <= t <= 0.6

    def test_imbalanced_threshold_below_half(self):
        # With 10% positive rate, optimal threshold should be well below 0.5
        rng = np.random.default_rng(1)
        y_true = np.array([1] * 10 + [0] * 90)
        y_prob  = np.where(y_true == 1,
                           rng.uniform(0.3, 0.9, 100),
                           rng.uniform(0.0, 0.5, 100))
        t = find_optimal_threshold(y_true, y_prob)
        assert t < 0.5
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_ensemble.py::TestFindOptimalThreshold -v
```
Expected: `AttributeError: module 'ensemble_06' has no attribute 'find_optimal_threshold'`

- [ ] **Step 3: Add `find_optimal_threshold` to `06_ensemble.py`**

Add to the imports at the top of `06_ensemble.py`:

```python
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
```

After the `compute_metrics` function (around line 85), add:

```python
def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the probability threshold that maximises F1 on the given data."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thresholds[np.argmax(f1[:-1])])
```

- [ ] **Step 4: Slim MODEL_NAMES to top-3**

Change:
```python
MODEL_NAMES = [
    "xgboost",
    "random_forest",
    "lightgbm",
    "mlp",
    "logistic_regression",
]
```
To:
```python
MODEL_NAMES = [
    "xgboost",
    "random_forest",
    "logistic_regression",
]
```

- [ ] **Step 5: Use `find_optimal_threshold` in `main()`**

After the `best_prob` / `best_name` / `best_auc` are resolved (after the stacking block, around line 250), add:

```python
    opt_threshold = find_optimal_threshold(y_test.values, best_prob)
    best_metrics_opt = compute_metrics(y_test, best_prob, threshold=opt_threshold)

    print(f"\n  Optimal threshold: {opt_threshold:.3f}  "
          f"(vs default 0.500)")
    print(f"  Optimized F1:      {best_metrics_opt['F1']:.4f}  "
          f"(vs default-threshold F1: "
          f"{compute_metrics(y_test, best_prob)['F1']:.4f})")

    results["opt_threshold"]   = opt_threshold
    results["optimized_f1"]    = best_metrics_opt["F1"]
```

Also save `opt_threshold` to the predictions parquet (in the pred_df block):

```python
    pred_df["opt_threshold"] = float(opt_threshold)
```

- [ ] **Step 6: Run tests to confirm they pass**

```bash
pytest tests/test_ensemble.py -v
```
Expected: all tests pass including the three new `TestFindOptimalThreshold` tests.

- [ ] **Step 7: Commit**

```bash
git add 06_ensemble.py tests/test_ensemble.py
git commit -m "feat: slim ensemble to top-3 models, add F1-optimal threshold"
```

---

## Task 9: Re-run the pipeline

- [ ] **Step 1: Re-run feature engineering (~15 min)**

```bash
python 03_feature_engineering.py
```
Expected output includes `review_momentum` and `checkin_momentum` in the feature matrix. Shape should show 2 more columns than before.

Verify:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/features.parquet')
print('review_momentum' in df.columns, 'checkin_momentum' in df.columns)
print(df[['review_momentum','checkin_momentum']].describe())
"
```
Expected: `True True` and a describe table with mean values between 0 and 3.

- [ ] **Step 2: Re-run modeling (~8 min)**

```bash
python 05_modeling.py
```
Expected: completes without errors. New `models/xgboost.pkl` includes `review_momentum` and `checkin_momentum` in its feature set.

- [ ] **Step 3: Re-run ensemble (~2 min)**

```bash
python 06_ensemble.py
```
Expected output includes:
- `Loaded: ['xgboost', 'random_forest', 'logistic_regression']` (3 models, not 5)
- `Optimal threshold: 0.1xx` (well below 0.5)
- `ensemble_results.json` and `ensemble_predictions.parquet` updated.

- [ ] **Step 4: Verify ensemble_results.json**

```bash
python -c "
import json
r = json.load(open('models/ensemble_results.json'))
print('opt_threshold:', r['opt_threshold'])
print('optimized_f1:', r['optimized_f1'])
print('winner:', r['winner'])
"
```
Expected: `opt_threshold` is a float between 0.05 and 0.45, `optimized_f1` is higher than the old default-threshold F1.

- [ ] **Step 5: Commit**

```bash
git add data/processed/features.parquet models/ data/processed/ensemble_predictions.parquet
git commit -m "chore: re-run pipeline with momentum features and top-3 ensemble"
```

---

## Task 10: Update CLAUDE.md with new features

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add momentum features to the Key Predictive Features section**

In `CLAUDE.md`, find the "Key Predictive Features" section and append after the existing list:

```markdown
6. review_momentum       (ratio: reviews last 6m / reviews first 6m + 1)
7. checkin_momentum      (ratio: checkins last 6m / checkins first 6m + 1)
```

- [ ] **Step 2: Update the Confirmed Results table**

Replace the old results table with the updated ensemble results after the pipeline re-run. Add a row for the top-3 ensemble result and the optimized F1.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document momentum features and updated ensemble results in CLAUDE.md"
```

---

## Full Test Run

After all tasks are complete:

```bash
pytest tests/ -v
```

Expected: all tests in `test_app_helpers.py`, `test_feature_engineering.py`, and `test_ensemble.py` pass.

```bash
streamlit run app.py
```

Verify visually:
- No encoding artifacts (no â˜…, ðŸ"¡, etc.)
- SHAP chart shows real model-derived contribution values
- Outcome banner (red/green) appears above every restaurant name
- "Top X% riskiest" shows instead of fake progress bar
- Footer shows "📦 BATCH · [month]"
- Clicking a Watch List row navigates to that restaurant
