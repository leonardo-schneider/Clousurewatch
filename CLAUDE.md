# ClosureWatch — CLAUDE.md

> Auto-read by Claude Code. Describes the full project state, pipeline,
> conventions, and results. Update this file after every major change.

---

## Project Overview

**Goal:** Binary classification — given 12 months of Yelp behavioral signals
for a restaurant, predict whether it will permanently close in the next 6 months.

**Dataset:** Yelp Academic Dataset (https://www.yelp.com/dataset)
**Geography:** Tampa Bay primary → expanded to 9 US metros + Edmonton (Canada)
**Stakeholder framing:** SME credit underwriting (Kabbage/BlueVine style) —
predicting business failure from public behavioral data, same way alternative
lenders use transaction signals.

**App:** ClosureWatch — Streamlit dashboard with photo card grid, SHAP
explanations, risk percentile, and outcome banners.

---

## Repository Layout

```
restaurant_failure/
├── CLAUDE.md                  ← you are here
├── config_00.py               ← all global constants — edit this first
│
├── 01_load_filter.py          ← load Yelp JSON → filtered Parquet (Tampa)
├── 02_build_labels.py         ← anchor dates + leakage-safe binary labels
├── 03_feature_engineering.py  ← time-windowed features (VADER, velocity, trend)
├── 04_eda.py                  ← basic EDA plots → figures/
├── 04b_data_quality.py        ← data quality diagnostics (1A-1E)
├── 05_modeling.py             ← algorithm shootout (5 models) + CV + test eval
├── 06_ensemble.py             ← stacking ensemble (XGB + RF + LR)
├── 07_model_analysis.py       ← SHAP global, error analysis, threshold curves
├── 08_philadelphia.py         ← Philadelphia zero-shot pipeline
├── 09_multi_metro.py          ← multi-metro pipeline (all 9 metros + Edmonton)
├── 10_lomo_cv.py              ← Leave-One-Metro-Out CV + global model
│
├── photo_index.py             ← builds business_id → best photo path lookup
├── app.py                     ← Streamlit dashboard (ClosureWatch)
├── app_helpers.py             ← risk color/label/SHAP helpers
│
├── data/
│   ├── raw/                   ← Yelp JSON source files
│   ├── raw_photos/            ← photos.json + photos/*.jpg (200k images)
│   ├── processed/             ← Tampa processed Parquets
│   ├── processed_philly/      ← Philadelphia
│   ├── processed_indianapolis/
│   ├── processed_nashville/
│   ├── processed_new_orleans/
│   ├── processed_saint_louis/
│   ├── processed_tucson/
│   ├── processed_reno/
│   ├── processed_boise/
│   └── processed_edmonton/    ← Canada OOD test
│
├── models/
│   ├── xgboost.pkl            ← Tampa-only XGBoost (v1)
│   ├── xgboost_global.pkl     ← Global 9-metro XGBoost (v3, FINAL)
│   ├── logistic_regression.pkl← Benchmark model
│   ├── random_forest.pkl      ← Part of ensemble
│   ├── lightgbm_model.pkl     ← Algorithm shootout (referenced by app)
│   ├── ensemble_results.json  ← Stacking ensemble metrics
│   ├── lomo_results.json      ← LOMO CV results (all 9 metros)
│   ├── results_summary.json   ← Tampa single-city results
│   ├── philadelphia_results.json
│   └── leaderboard.csv        ← Algorithm shootout leaderboard
│
└── figures/                   ← all plots (safe to delete and regenerate)
    └── eda_deep/              ← deep EDA figures (06_eda_deep.py output)
```

---

## Raw Data Files

| File | Location | Status |
|---|---|---|
| `yelp_academic_dataset_business.json` | `data/raw/Yelp JSON/` | ✅ Required |
| `yelp_academic_dataset_review.json` | `data/raw/Yelp JSON/` | ✅ Required |
| `yelp_academic_dataset_checkin.json` | `data/raw/Yelp JSON/` | ✅ Required |
| `yelp_academic_dataset_tip.json` | `data/raw/Yelp JSON/` | ✅ Required |
| `yelp_academic_dataset_photos.json` | `data/raw_photos/` | ✅ Indexed |
| `photos/*.jpg` | `data/raw_photos/photos/` | ✅ 200k images |

**Photo coverage:** 3,294 / 5,143 Tampa restaurants (64%). `photo_index.parquet`
maps `business_id → best photo path` (food label preferred). `has_photo` is
included as a feature in `09_multi_metro.py`.

---

## Temporal Design — Anti-Leakage Rules

```
|--- obs_start ---|--- OBSERVATION WINDOW (12m) ---|--- anchor_date ---|--- OUTCOME (6m) ---|
                         Features built here                ↑                Label here
                                                     80th pct review date
```

**Hard rules — never violate:**

1. Features use only `date < anchor_date` — no exceptions
2. Anchor = 80th percentile review date, NOT last review
3. Imputation medians fit on training fold only
4. Train/test split is time-based (anchor_date cutoff)
5. LOMO CV: hold out entire metro, train on rest
6. LATEST_ANCHOR = 2020-06-01 for all metros (prevents dataset-edge label noise)
7. Label rule: last_review > outcome_end + 3m → open (0);
   last_review ≤ outcome_end → closed (1); else ambiguous → 0

---

## Key Config Values (config_00.py)

| Constant | Value | Notes |
|---|---|---|
| `OBS_MONTHS` | 12 | Observation window |
| `OUTCOME_MONTHS` | 6 | Prediction horizon |
| `LATEST_ANCHOR` | `2020-06-01` | Prevents 2021 edge artifact |
| `EARLIEST_ANCHOR` | `2016-01-01` | Drop, not clip, out-of-range |
| `TARGET_COL` | `closed_within_6m` | Label column |
| `PRIMARY_METRIC` | `average_precision` | AUC-PR |
| `ENCODING` | `utf-8` | All file reads |
| `RAW_DIR` | `data/raw/Yelp JSON` | Source JSON location |

---

## Features (48 total + has_photo = 49 in multi-metro)

### Volume & Velocity
- `n_reviews_obs`, `review_velocity`, `review_velocity_slope`
- `months_with_zero_reviews`, `days_since_last_review`
- `review_drought_flag` (binary: silent 90+ days)
- `review_momentum` — reviews_last_6m / (reviews_first_6m + 1)

### Rating Signals
- `mean_stars_obs`, `std_stars_obs`, `rating_trend_slope`
- `stars_delta_3m` — last 3m minus first 3m avg
- `pct_1star`, `pct_5star`
- `stars_recent_3m`, `stars_early_3m`

### VADER Sentiment
- `mean_vader`, `vader_trend_slope`, `vader_trend_3m`
- `pct_negative_vader`, `vader_stars_divergence`

### Checkin Signals
- `n_checkins_obs`, `checkin_velocity`, `checkin_velocity_slope`
- `checkin_drought_flag`, `checkin_momentum`

### Reviewer Quality
- `mean_review_useful`, `mean_review_funny`, `mean_review_cool`
- `pct_high_quality_reviews`

### Tips
- `n_tips_obs`, `tip_velocity`, `mean_tip_compliments`

### Business Metadata
- `price_range`, `open_days_per_week`, `stars_yelp_global`
- `is_fast_food`, `is_bar`, `is_cafe`, `is_pizza`, `is_mexican`, `is_seafood`

### Photo (multi-metro only)
- `has_photo` — binary: business has at least 1 Yelp photo

---

## Model Evolution

| Version | Training | Test AUC-PR | Test AUC-ROC |
|---|---|---|---|
| v1 Logistic Regression (benchmark) | Tampa only | 0.157 | 0.646 |
| v1 XGBoost (Tampa-only) | Tampa only | 0.207 | 0.700 |
| v2 Zero-shot Philadelphia | Tampa model → Philly | 0.142 | 0.617 |
| **v3 Global LOMO (FINAL)** | **8 metros → 1 held out** | **0.360 mean** | **0.781 mean** |

---

## Algorithm Shootout (Tampa-only, held-out test)

| Model | CV AUC-PR | Test AUC-PR | Test AUC-ROC | Test F1 |
|---|---|---|---|---|
| XGBoost ✓ | 0.124 ± 0.039 | 0.207 | 0.700 | 0.245 |
| Random Forest | 0.112 ± 0.054 | 0.199 | 0.695 | 0.135 |
| LightGBM | 0.091 ± 0.021 | 0.198 | 0.670 | 0.209 |
| MLP Neural Net | 0.083 ± 0.010 | 0.159 | 0.619 | 0.162 |
| Logistic Regression | 0.106 ± 0.035 | 0.157 | 0.646 | 0.255 |

**Ensemble (stacking XGB + RF + LR):**
- AUC-PR: 0.200 | AUC-ROC: 0.694
- Optimal threshold: 0.27 (F1-optimized) | F1: 0.299

---

## LOMO CV Results — 9 US Metros

*Leave-One-Metro-Out: train on 8 metros, test on held-out metro*

| Metro | AUC-PR | AUC-ROC | F1 | N |
|---|---|---|---|---|
| New Orleans | 0.4926 | 0.8388 | 0.4273 | 1,786 |
| Reno | 0.4718 | 0.8273 | 0.4810 | 921 |
| Indianapolis | 0.4400 | 0.7942 | 0.4276 | 2,098 |
| Tucson | 0.4159 | 0.8066 | 0.4049 | 1,825 |
| Philadelphia | 0.3768 | 0.7793 | 0.3767 | 4,258 |
| Nashville | 0.3523 | 0.7898 | 0.3579 | 1,776 |
| Boise | 0.3016 | 0.7461 | 0.3117 | 627 |
| Saint Louis | 0.2600 | 0.7528 | 0.2513 | 1,274 |
| Tampa | 0.1310 | 0.6953 | 0.1583 | 5,143 |
| **Mean** | **0.360 ±0.108** | **0.781 ±0.042** | | ~19,708 |

**Edmonton OOD (Canada, global model):** AUC-PR=0.245, AUC-ROC=0.655

### Key Findings
- Global model is **77% better** than Tampa-only (AUC-PR 0.360 vs 0.203)
- Tampa is the hardest metro — lowest closure rate (6.3%) makes
  calibration from other markets difficult
- Silence signal generalizes to Canada without retraining
- Cross-metro AUC-ROC consistently above 0.74 (except Tampa + Boise)

---

## Cross-City Zero-Shot (Tampa model → Philadelphia)

| Metric | Tampa | Philadelphia |
|---|---|---|
| AUC-ROC | 0.694 | 0.617 (-11%) |
| AUC-PR | 0.203 | 0.142 (-30%) |

**Conclusion:** Ranking signal partially universal; calibration is local.
Global LOMO model solves the calibration problem.

---

## Data Quality Notes

- Zero duplicates by business_id in all metros
- Price range valid (1–4 only); Philly coverage 84.9% vs Tampa 90.2%
- COVID uplift: Tampa +3.1pp, Philadelphia +5.7pp closure rate
- 2021 cohort excluded from all metros (dataset-edge artifact)
- Philadelphia review density ~40% lower than Tampa per restaurant
- High-null features (VADER trend, stars_delta, tip_compliments) are
  intentional — require minimum review count, handled by LightGBM/XGBoost natively

---

## Dataset Geography

9 distinct US metros + Edmonton (Canada). NOT a nationwide sample:

| Metro | States/Cities | ~Restaurants |
|---|---|---|
| Philadelphia | PA + NJ suburbs | ~7,500 |
| Tampa Bay | FL (Tampa + suburbs) | ~5,000 |
| Indianapolis | IN + suburbs | ~4,000 |
| Tucson | AZ | ~3,300 |
| Nashville | TN + suburbs | ~3,500 |
| New Orleans | LA + Metairie/Kenner | ~3,200 |
| Saint Louis | MO | ~3,000 |
| Reno | NV + Sparks | ~2,400 |
| Boise | ID + Meridian | ~1,600 |
| Edmonton | AB (OOD test) | ~2,800 |

**Limitation:** No NYC, LA, Chicago, Houston — biased toward Sun Belt
and Midwest mid-size cities.

---

## Business Operating Point (Tampa, threshold=0.27)

- Restaurants flagged high risk: 205 / 1,244 (16.5%)
- Closures caught: 50 / 129 (38.8% recall)
- Lift over random: 6×
- Optimized F1: 0.299

---

## Assignment Requirements

| Requirement | Status | Where |
|---|---|---|
| Novel dataset (not Kaggle) | ✅ | Yelp Academic Dataset |
| Data cleaning + EDA | ✅ | `01_load_filter.py`, `04_eda.py`, `04b_data_quality.py` |
| Literature review | ⏳ | Presentation slides |
| Benchmark model | ✅ | Logistic Regression, `05_modeling.py` |
| ML model + CV | ✅ | XGBoost, 5-fold + LOMO CV |
| Train/val/test split | ✅ | Time-based + metro-based |
| Correct metrics | ✅ | AUC-PR primary (imbalanced) |
| Presentation (10-15 slides) | ⏳ | Due Tuesday |
| Extra credit GenAI | ⏳ | Optional |

---

## Notes for Claude Code

- **Always check `config_00.py` first** before modifying paths or constants
- **LATEST_ANCHOR = 2020-06-01** for all metros — do not change
- **Do not use random train/test splits** anywhere
- **Final model is `xgboost_global.pkl`** — trained on all 9 metros combined
- **Benchmark is `logistic_regression.pkl`** — Tampa only
- Photo features: `has_photo` is live in `09_multi_metro.py`.
  Additional photo features (`photo_recency_days`, `n_food_photos`) not yet implemented
- To add new features: edit `build_features_one()` in `03_feature_engineering.py`
  AND `build_features()` in `09_multi_metro.py` to keep them in sync
- Edmonton is OOD test only — do not include in LOMO training folds
- `data/processed_*/` directories: do not edit manually, re-run upstream script
