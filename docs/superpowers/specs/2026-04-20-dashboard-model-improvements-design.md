# ClosureWatch — Dashboard & Model Improvements

**Date:** 2026-04-20  
**Scope:** Two independent tracks — app fixes (no pipeline re-run) and pipeline improvements (re-run 03→05→06).

---

## Track 1 — App Changes (`app.py`)

No pipeline re-run required. Works with existing `xgboost.pkl` and `ensemble_predictions.parquet`.

**Dependency:** `pip install shap`

### 1. Encoding Fix
Re-save `app.py` as UTF-8 and replace all mojibake string literals throughout the file (e.g., `â˜…` → `★`, `ðŸ"¡` → `📡`). Pure find-and-replace, no logic changes.

### 2. SHAP Feature Contributions (on-the-fly)
- Load `models/xgboost.pkl` and the full feature matrix parquet once at app startup via `@st.cache_resource`.
- On restaurant selection, run `shap.TreeExplainer` on that single row (~50ms).
- Replace the current fake scaled-value bar chart with real SHAP values.
- Red bars = raise risk (positive SHAP), green bars = lower risk (negative SHAP).
- X-axis shows actual SHAP contribution values, not arbitrary [-1, 1] range.
- Feature order: sorted by absolute SHAP value descending (most impactful first).

### 3. Ground Truth Outcome Banner
- `ensemble_predictions.parquet` already contains `closed_within_6m`.
- When a restaurant is selected and this column is present, render a banner above the restaurant name.
- Three states:
  - **Closed:** dark red banner, red left border — "✓ OUTCOME KNOWN · PERMANENTLY CLOSED"
  - **Still open:** dark green banner, green left border — "✓ OUTCOME KNOWN · STILL OPEN"
  - **Unknown:** gray banner — "OUTCOME UNKNOWN · PREDICTION ONLY"
- Banner includes anchor date formatted as "Anchor: Mon YYYY".

### 4. Replace "Model Confidence" Bar with Percentile Rank
- Remove the current bar (which is just `risk_pct × 1.1`, meaningless).
- Replace with a **percentile rank**: compute where this restaurant sits among all restaurants (e.g., "Top 8% riskiest in Tampa Bay").
- Computed at load time from the full dataset.

### 5. Fix Footer "LIVE" Badge
- Change `🔴 LIVE` to `📦 BATCH` with the parquet file's last-modified date (e.g., "BATCH · Apr 2026").

### 6. Interactive Watch List Table
- Replace the static Plotly table with `st.dataframe` using `selection_mode="single-row"` (Streamlit ≥ 1.35).
- Clicking a row sets `selected_idx` in session state and triggers `st.rerun()`.
- Preserves existing columns: Rank, Restaurant, Closure Risk, Tier, Yelp Stars.

---

## Track 2 — Pipeline Changes

Re-run order: `03_feature_engineering.py` (~15 min) → `05_modeling.py` (~8 min) → `06_ensemble.py` (~2 min).  
Scripts `01_load_filter.py` and `02_build_labels.py` do not need re-running.

### 1. Decline-Rate Features (`03_feature_engineering.py`)
Add two momentum features inside `build_features_one()`. Both use only `date < anchor_date` (no leakage).

Split the 12-month observation window at the midpoint (6 months):
- `review_momentum = reviews_last_6m / (reviews_first_6m + 1)` — values < 1.0 indicate slowing, < 0.5 indicate severe decline.
- `checkin_momentum = checkins_last_6m / (checkins_first_6m + 1)` — same for foot traffic.

Document new features in the Features section of `CLAUDE.md`.

### 2. Top-3 Weighted Ensemble (`06_ensemble.py`)
- Change `MODEL_NAMES` to only the top 3 models by CV AUC-PR: `["xgboost", "random_forest", "logistic_regression"]`.
- Remove `"lightgbm"` (CV AUC-PR 0.091) and `"mlp"` (CV AUC-PR 0.083) from the ensemble.
- Weighted average by CV AUC-PR scores remains the same logic.
- Stacking fallback (`[3]`) remains in place if top-3 weighted average doesn't beat baseline.

### 3. Threshold Optimization (`06_ensemble.py`)
- After computing ensemble probabilities, find the F1-maximizing threshold using `precision_recall_curve`.
- Expected optimal threshold: ~0.15–0.25 (well below the default 0.5 for imbalanced data).
- Save `opt_threshold` and `optimized_f1` to `ensemble_results.json`.
- Use `opt_threshold` (not 0.5) when computing the final reported F1 metric.
- Save `opt_threshold` value into the predictions parquet as a scalar metadata column so the app can reference it for coloring.

---

## Success Criteria

- App renders without encoding artifacts in any browser.
- SHAP chart shows real model-derived values; bars have correct sign relative to risk.
- Ground truth banner appears for all test-set restaurants (1,244 records).
- Clicking a Watch List table row navigates to that restaurant's detail view.
- Footer shows BATCH + date, not LIVE.
- Percentile rank is accurate (Top X% computed from full dataset).
- `review_momentum` and `checkin_momentum` appear in `features.parquet` and model feature importance.
- `ensemble_results.json` contains `opt_threshold` and `optimized_f1`.
- Pipeline re-run completes without errors.
