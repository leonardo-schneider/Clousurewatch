# Restaurant Failure Prediction — CLAUDE.md

> This file is read by Claude Code automatically. It describes the project,
> pipeline, conventions, and guidance for any agentic work in this repository.

---

## Project Overview

**Goal:** Binary classification — given 12 months of Yelp behavioral signals
for a restaurant, predict whether it will permanently close in the next 6 months.

**Dataset:** Yelp Academic Dataset (https://www.yelp.com/dataset)
**Geography:** Tampa, FL primary; expand to all Florida if Tampa yields < 200 closures.
**Stakeholder framing:** SME credit underwriting (think Kabbage/BlueVine) —
predicting business failure from public behavioral data.

---

## Repository Layout

```
restaurant_failure/
├── CLAUDE.md                     ← you are here
├── config_00.py                  ← all global constants; edit this first
├── 01_load_filter.py             ← load Yelp JSON → filtered Parquet
├── 02_build_labels.py            ← anchor dates + leakage-safe binary labels
├── 03_feature_engineering.py     ← time-windowed features (VADER, velocity, trend)
├── 04_eda.py                     ← EDA plots → figures/
├── 05_modeling.py                ← LogReg benchmark + LightGBM + CV + test eval
│
├── data/
│   ├── raw/                      ← drop ALL Yelp source files here (see below)
│   └── processed/                ← auto-generated Parquet files (do not edit)
├── models/                       ← saved model artifacts + results_summary.json
└── figures/                      ← all plots (auto-generated, safe to delete)
```

---

## Raw Data Files (data/raw/)

Place all Yelp JSON files here before running any script.

| File | Status | Used in |
|---|---|---|
| `yelp_academic_dataset_business.json` | Required | `01_load_filter.py` |
| `yelp_academic_dataset_review.json` | Required | `01_load_filter.py` |
| `yelp_academic_dataset_checkin.json` | Required | `01_load_filter.py` |
| `yelp_academic_dataset_tip.json` | Required | `01_load_filter.py` |
| `yelp_academic_dataset_photos.json` | **Downloaded, not yet used** | Future work (see below) |

### Photo Data — Future Work

The photos file (`yelp_academic_dataset_photos.json`) has been downloaded and
is available in `data/raw/`. It is **not used in the current pipeline** but is
reserved for a potential future feature engineering module.

**Planned use cases (do not implement until instructed):**
- Photo count and recency as a proxy for owner engagement
- Photo category distribution (food vs interior vs exterior) as a quality signal
- Image-based quality scoring via a vision model (e.g. CLIP embeddings)
- Decline in photo upload frequency as a leading closure indicator

When implementing photo features, create a new script `03b_photo_features.py`
and merge its output into the main feature matrix before `04_eda.py`. All photo
features must be filtered to `date < anchor_date` to prevent leakage.

---

## Pipeline — Run Order

Always run scripts in numerical order. Each script depends on the output of
the previous one.

```bash
python 01_load_filter.py           # ~5-10 min
python 02_build_labels.py          # ~2 min
python 03_feature_engineering.py   # ~10-20 min (VADER is slow)
python 04_eda.py                   # ~1 min
python 05_modeling.py              # ~5-10 min
```

To re-run only from a specific step, the upstream Parquet files are preserved
and do not need to be regenerated unless the upstream script changes.

---

## Temporal Design — Anti-Leakage Rules

This is the most critical correctness constraint in the project.

```
Timeline per restaurant:

|--- obs_start ---|--- OBSERVATION WINDOW (12m) ---|--- anchor_date ---|--- OUTCOME WINDOW (6m) ---|--- outcome_end
                          Features built here               ↑                   Label determined here
                                                      Anchor date
                                                 (80th pct review date)
```

**Hard rules — never violate these:**

1. **Features use only data where `date < anchor_date`.**
   Any feature accidentally using data on or after `anchor_date` is a leak.

2. **The anchor is the 80th percentile review date, not the last review.**
   Using the last review as anchor would expose the review drought signal trivially.

3. **Imputation medians are computed on the training fold only**, then applied
   to validation. Never fit imputers on the full dataset.

4. **Train/test split is time-based** (most recent 15% by anchor date = test).
   Never use random splits — they leak future business states into training.

5. **Cross-validation folds are expanding-window**, not random k-fold.

---

## Key Config Values (config_00.py)

| Constant | Value | Notes |
|---|---|---|
| `OBS_MONTHS` | 12 | Observation window length |
| `OUTCOME_MONTHS` | 6 | How far ahead we predict |
| `LATEST_ANCHOR` | `2021-06-01` | Adjust to dataset coverage |
| `EARLIEST_ANCHOR` | `2016-01-01` | Needs 12m history before |
| `MIN_CLOSED_THRESHOLD` | 200 | Tampa floor before FL expansion |
| `N_FOLDS` | 5 | CV folds |
| `TEST_FRAC` | 0.15 | Held-out test fraction |
| `TARGET_COL` | `closed_within_6m` | Label column name |
| `PRIMARY_METRIC` | `average_precision` | AUC-PR is primary |

---

## Confirmed Results — Full Algorithm Shootout

| Model | CV AUC-PR | Test AUC-PR | Test AUC-ROC | Test F1 |
|---|---|---|---|---|
| **XGBoost** (winner) | 0.124 ± 0.039 | **0.2069** | **0.7002** | 0.2448 |
| Random Forest | 0.112 ± 0.054 | 0.1994 | 0.6948 | 0.1347 |
| LightGBM | 0.091 ± 0.021 | 0.1977 | 0.6699 | 0.2093 |
| MLP Neural Network | 0.083 ± 0.010 | 0.1592 | 0.6187 | 0.1622 |
| Logistic Regression (benchmark) | 0.106 ± 0.035 | 0.1574 | 0.6458 | 0.2553 |

- Base rate (random AUC-PR): 0.063
- XGBoost is 3.3x better than random on AUC-PR
- XGBoost +31.4% AUC-PR improvement over Logistic Regression benchmark
- MLP note: lowest CV variance (±0.010) — most stable but peaks lower
- Dataset: 5,143 restaurants, 326 closures (6.3%), Tampa Bay metro
- Test set: 1,244 businesses, 129 closures (10.4%), anchored 2020-06 onward

## Key Predictive Features (from LightGBM importance)

Top signals in order:
1. months_with_zero_reviews  (corr=0.56 with label)
2. days_since_last_review    (corr=0.56)
3. review_drought_flag       (corr=0.43)
4. checkin_drought_flag      (corr=0.34)
5. pct_5star                 (corr=0.29)

---

## Dependencies

```bash
pip install pandas numpy lightgbm scikit-learn imbalanced-learn \
            vaderSentiment matplotlib seaborn tqdm joblib \
            python-dateutil pyarrow
```

Python 3.10+ recommended.

---

## Assignment Requirements Mapping

| Requirement | Where addressed |
|---|---|
| Novel dataset (not Kaggle) | Yelp Academic Dataset |
| Data cleaning + EDA | `01_load_filter.py`, `04_eda.py` |
| Literature review | Presentation slides (manual) |
| Benchmark model | Logistic Regression in `05_modeling.py` |
| ML model with CV | LightGBM with 5-fold expanding-window CV |
| Train/val/test split | `05_modeling.py` (time-based) |
| Correct metrics | AUC-PR primary (imbalanced classification) |
| 10-15 slide presentation | To be built after modeling is complete |
| Extra credit (GenAI comparison) | `06_genai_comparison.py` (not yet created) |

---

## Notes for Claude Code

- **Always check `config_00.py` first** before modifying any path or constant.
- **Do not modify files in `data/processed/`** — delete and re-run the upstream
  script instead.
- **Do not use random train/test splits** anywhere in this project.
- If adding new features, add them to `03_feature_engineering.py` inside
  `build_features_one()` and document them in the Features section of this file.
- If the Tampa sample is too small after running `02_build_labels.py`, set
  `PRIMARY_CITIES = []` and `FALLBACK_STATES = ["FL"]` in `config_00.py`.
- Photo features: **do not implement until explicitly requested**. See Photo
  Data section above.
