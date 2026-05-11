'# Multi-Metro Pipeline + LOMO CV — Design Spec

**Date:** 2026-04-30
**Project:** Restaurant Failure Prediction (Yelp Academic Dataset)

---

## Goal

Run the existing Tampa pipeline against all remaining Yelp metros, then evaluate generalization via Leave-One-Metro-Out (LOMO) cross-validation with per-fold XGBoost hyperparameter tuning. Output a global model trained on all US metros and evaluate it on Edmonton as an out-of-distribution test.

---

## Scope

**In scope (Spec A):**
- `09_multi_metro.py` — generic city pipeline
- `10_lomo_cv.py` — LOMO CV + global model
- `has_photo` binary feature backfilled into all metro feature matrices

**Out of scope (Spec B, separate):**
- `03b_photo_features.py` (photo count, category distribution, CLIP embeddings)

---

## Metros

**LOMO pool (9 US metros):**

| City | State | ~Restaurants |
|---|---|---|
| Tampa | FL | 3,923 |
| Philadelphia | PA | 7,528 |
| Indianapolis | IN | 3,668 |
| Tucson | AZ | 3,338 |
| Nashville | TN | 3,302 |
| New Orleans | LA | 3,149 |
| Saint Louis | MO | ~2,968 |
| Reno | NV | 1,915 |
| Boise | ID | 1,171 |

**OOD test (post-LOMO):**
- Edmonton, AB — only non-US metro, ~2,785 restaurants

---

## File Structure

```
09_multi_metro.py                        # new — generic pipeline
10_lomo_cv.py                            # new — LOMO CV + global model

data/processed_{city_slug}/              # one dir per new metro
  businesses.parquet
  reviews.parquet
  checkins.parquet
  tips.parquet
  labeled_businesses.parquet
  features.parquet                       # includes has_photo column

data/processed/features.parquet          # Tampa — backfilled with has_photo
data/processed_philly/features.parquet   # Philly — backfilled with has_photo

models/lomo_results.json                 # per-metro metrics + aggregated summary
models/xgboost_global.pkl               # global model trained on all 9 metros
figures/19_lomo_cv_results.png          # AUC-PR + AUC-ROC bars per metro
```

City slug rule: lowercase, spaces → underscores (e.g. `saint_louis`, `new_orleans`).

---

## `09_multi_metro.py` Design

### Interface

```bash
python 09_multi_metro.py --city Nashville --state TN
python 09_multi_metro.py --city "Saint Louis" --state MO
python 09_multi_metro.py --city Edmonton --state AB   # Edmonton OOD
```

### Key behaviour

- **City filtering:** match on both `city` AND `state` fields from business JSON to handle duplicate city names (Saint Louis / St. Louis problem)
- **Category filter:** same as existing pipeline — "Restaurants" or "Food" in categories
- **LATEST_ANCHOR:** `2020-06-01` for all metros (conservative cap, avoids dataset-edge label artifact seen in Philadelphia 2021 cohort)
- **EARLIEST_ANCHOR:** `2016-01-01` (same as Tampa)
- **Checkpoint loading:** if `businesses.parquet` exists in output dir → skip JSON reads; if `labeled_businesses.parquet` exists → skip labeling; always re-run feature engineering to pick up `has_photo`
- **Labeling:** 3-way review-recency rule from `02_build_labels.py` (`build_label` function) — not `is_open` flag
- **Feature engineering:** imports and calls `build_features_one` from `03_feature_engineering.py` with observation-window-filtered data per business
- **`has_photo`:** join labeled businesses against `data/processed/photo_index.parquet` before feature engineering; add `has_photo` as a column on each row so `build_features_one` returns it like any metadata field
- **Minimum threshold:** if fewer than 50 labeled restaurants, print warning and exit cleanly
- **`--backfill-only` flag:** when passed, skips all JSON loading and feature engineering — only joins `photo_index.parquet` onto existing `features.parquet` for Tampa and Philadelphia and re-saves

### Output

Prints summary: N labeled, N closed, closure rate, features shape. Saves all parquets to `data/processed_{city_slug}/`.

---

## `has_photo` Feature

**Source:** `data/processed/photo_index.parquet` (36,680 businesses, built by `photo_index.py`)

**Definition:** `has_photo = 1` if `business_id` appears in photo index, else `0`

**Temporal note:** Photo timestamps are not available in the raw data, so this is a static binary — not temporally filtered to `date < anchor_date`. It is a proxy for owner engagement (businesses that ever uploaded photos to Yelp). Document this limitation in CLAUDE.md.

**Backfill:** After computing `has_photo` for each new metro, also add it to:
- `data/processed/features.parquet` (Tampa)
- `data/processed_philly/features.parquet` (Philadelphia)

This ensures the full LOMO pool has a consistent feature set.

---

## `10_lomo_cv.py` Design

### Data loading

```python
METROS = {
    "tampa":         "data/processed",
    "philadelphia":  "data/processed_philly",
    "indianapolis":  "data/processed_indianapolis",
    "tucson":        "data/processed_tucson",
    "nashville":     "data/processed_nashville",
    "new_orleans":   "data/processed_new_orleans",
    "saint_louis":   "data/processed_saint_louis",
    "reno":          "data/processed_reno",
    "boise":         "data/processed_boise",
}
EDMONTON_DIR = "data/processed_edmonton"
```

Load each metro's `features.parquet`, add a `metro` column, concatenate into master DataFrame.

### LOMO fold structure (9 folds)

For each held-out metro:

```
train_pool = concat of all other 8 metro feature matrices

Time-based train/val split within train_pool:
  latest 20% by anchor_date → val
  remaining 80%             → train

Fit imputation medians on train only
Apply medians to train, val, and held-out metro (no data leak)

Tune XGBoost hyperparameters on val AUC-PR
  (same search space as 05_modeling.py — n_estimators, max_depth,
   learning_rate, subsample, colsample_bytree, min_child_weight)

Retrain best params on full train_pool (train + val combined)
Apply train_pool medians to held-out metro

Evaluate on held-out metro:
  AUC-PR, AUC-ROC, F1 at val-threshold, N restaurants, closure rate
```

### Global model

After all 9 folds:
- Identify fold with highest val AUC-PR → use its hyperparameters
- Train XGBoost on all 9 metros combined (full dataset, no held-out)
- Save as `models/xgboost_global.pkl`
- Evaluate on Edmonton: AUC-PR, AUC-ROC, F1

### Results output

`models/lomo_results.json`:
```json
{
  "folds": [
    {"metro": "tampa", "AUC_PR": ..., "AUC_ROC": ..., "F1": ..., "n": ..., "closure_rate": ...},
    ...
  ],
  "aggregate": {"mean_AUC_PR": ..., "std_AUC_PR": ..., "mean_AUC_ROC": ..., "std_AUC_ROC": ...},
  "global_model": {"train_metros": 9, "test_metro": "edmonton", "AUC_PR": ..., "AUC_ROC": ...}
}
```

### Figure (`figures/19_lomo_cv_results.png`)

Grouped horizontal bar chart:
- One row per metro, sorted by AUC-PR descending
- Two bars per metro: AUC-PR (blue) and AUC-ROC (green)
- Vertical reference line: Tampa single-city AUC-PR (0.203) and AUC-ROC (0.694)
- Title: "LOMO CV — Generalization Across 9 US Metros"

---

## Anti-Leakage Checklist

- Imputation medians fit on training portion of train_pool only ✓
- Hyperparameters tuned on val, not on held-out metro ✓
- `has_photo` is static (no anchor filtering possible) — documented ✓
- No review/checkin/tip data from after anchor_date in features ✓
- Edmonton evaluated only against global model, never seen during tuning ✓

---

## Running Order (Thursday)

```bash
# Step 1 — Run all 8 new metros in parallel (separate terminals)
python 09_multi_metro.py --city Indianapolis --state IN
python 09_multi_metro.py --city Tucson       --state AZ
python 09_multi_metro.py --city Nashville    --state TN
python 09_multi_metro.py --city "New Orleans" --state LA
python 09_multi_metro.py --city "Saint Louis" --state MO
python 09_multi_metro.py --city Reno         --state NV
python 09_multi_metro.py --city Boise        --state ID
python 09_multi_metro.py --city Edmonton     --state AB

# Step 2 — Backfill has_photo into Tampa + Philly
python 09_multi_metro.py --backfill-only

# Step 3 — LOMO CV (after all metros complete)
python 10_lomo_cv.py
```
