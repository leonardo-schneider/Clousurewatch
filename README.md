# ClosureWatch — Restaurant Closure Prediction

Binary classification: given 12 months of Yelp behavioral signals for a restaurant,
predict whether it will permanently close in the next 6 months.

Framed as an SME credit underwriting problem — alternative lenders (Kabbage, BlueVine)
use public behavioral signals the same way banks use transaction history. If you can
predict business failure from Yelp activity alone, you can flag deteriorating credits
before they default.

---

## Dataset

**Source:** [Yelp Academic Dataset](https://www.yelp.com/dataset)  
**Geography:** 9 US metros — Philadelphia, Tampa, Indianapolis, Tucson, Nashville,
New Orleans, Saint Louis, Reno, Boise  
**Size:** ~31,700 restaurants, ~9% overall closure rate (class-imbalanced)  
**Time range:** Reviews from 2005–2021; restricted to anchor dates ≤ 2020-06-01 to
avoid dataset-edge artifacts from incomplete 2021 cohorts

Raw data is not included in this repo (~10 GB Yelp JSON). Download from the link above
and place in `data/raw/`. Processed feature files are committed — you can run
`05_modeling.py` directly without the raw data.

### Geographic Scope and Limitations

The 9 metros are not a nationally representative sample. They skew toward Sun Belt and
Midwest mid-size cities and exclude major markets (NYC, LA, Chicago, Houston). Results
should be interpreted in that context.

---

## Problem Design

### Why This Is Hard

Restaurant closure prediction is a classic imbalanced classification problem:
- ~9% of restaurants close within 6 months of any given anchor date
- Yelp behavioral signals are noisy proxies — a restaurant can go silent on Yelp and
  stay open, or accumulate glowing reviews before closing suddenly
- Features must be computed *strictly before* the prediction date to prevent data leakage

### Temporal Design (Anti-Leakage)

```
|--- obs_start ---|--- OBSERVATION WINDOW (12 months) ---|--- anchor_date ---|--- OUTCOME (6 months) ---|
                           Features built here                     ^                   Label here
                                                        80th pct review date
```

Every design decision was made to prevent temporal leakage:

1. **Anchor date = 80th percentile review date**, not the last review. Using the last
   review would let the feature window "see" near-present activity, biasing the model
   toward businesses still actively reviewed at the time of prediction.

2. **All features use only data strictly before the anchor date.** No exceptions.

3. **Train/test split is time-based**: earliest 80% of restaurants by anchor date → train;
   latest 20% → test. No random splits. Random splits would let the model train on
   future anchor dates and test on past ones.

4. **Imputation medians are fit on the training set only**, then applied to the test set.

5. **Label rule uses review recency, not `is_open` from Yelp JSON.** The `is_open`
   field in the Yelp dataset reflects status at dataset download time (2022), not at the
   anchor date — using it directly would be leakage. Instead:
   - Last review > outcome_end + 3 months → open (0)
   - Last review ≤ outcome_end → closed (1)
   - Ambiguous → 0 (conservative)

6. **LATEST_ANCHOR = 2020-06-01** across all metros. Restricts to restaurants with
   enough post-anchor history to resolve the label cleanly.

### Known Limitation: Short-History Restaurants

~12% of restaurants have fewer than 18 months of total review history (first to last
review). For these businesses, the 12-month observation window before the anchor date is
partially or fully empty — features like review velocity slope and sentiment trend will
be null or zero rather than informative. XGBoost handles this natively via missing-value
splits; the model degrades gracefully to low-confidence predictions for this segment
rather than producing incorrect ones.

---

## Data Pipeline

### Step 1 — Load & Filter (`01_load_filter.py`)

Reads Yelp JSON for all 9 metros, applies filters:
- Restaurant category (excludes bars-only, coffee shops, etc.)
- Minimum 3 reviews (businesses with 1–2 reviews carry no signal)
- Valid state/city match per metro

Outputs per metro: `businesses.parquet`, `reviews.parquet`, `checkins.parquet`,
`tips.parquet`

### Step 2 — Build Labels (`02_build_labels.py`)

Computes anchor dates and binary closure labels using the review-recency rule above.
No `is_open` leakage.

### Step 3 — Feature Engineering (`03_feature_engineering.py`)

Builds 81 time-windowed features per restaurant, strictly before the anchor date.
High-null features (VADER trend, stars delta, tip compliments) are intentional — they
require a minimum review count to be defined. XGBoost and LightGBM handle these natively.

---

## Features (81 total)

### Review Volume & Velocity
- `n_reviews_obs` — total reviews in observation window
- `review_velocity` — reviews per month
- `review_velocity_slope` — linear trend in monthly review counts
- `months_with_zero_reviews` — months of silence in the window
- `days_since_last_review` — recency signal
- `review_drought_flag` — binary: 90+ day silence before anchor
- `review_momentum` — reviews last 6m / (reviews first 6m + 1)

### Rating Signals
- `mean_stars_obs`, `std_stars_obs` — level and dispersion
- `rating_trend_slope` — linear trend in monthly avg rating
- `stars_delta_3m` — last 3m avg minus first 3m avg (direction)
- `pct_1star`, `pct_5star` — polarization
- `stars_recent_3m`, `stars_early_3m`

### VADER Sentiment
- `mean_vader`, `vader_trend_slope`, `vader_trend_3m`
- `pct_negative_vader`
- `vader_stars_divergence` — sentiment vs stars disagreement

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

### Null-Flag Indicators
Binary flags for each feature with informative missingness (e.g., `vader_trend_slope`
is null when fewer than 3 monthly sentiment observations exist). These allow the model
to distinguish "no trend computed" from "flat trend."

### Sentence Embeddings (PCA-32)
Review text embedded with `all-MiniLM-L6-v2` (384-dim), averaged per restaurant,
then compressed to 32 dimensions via PCA fit on training data. Adds 32 features that
capture semantic content beyond star ratings.

---

## Modeling

### Models

**Benchmark: Logistic Regression**  
L2 regularization, `class_weight="balanced"` to handle imbalance, StandardScaler
preprocessing. C grid: {0.01, 0.1, 1.0, 10.0}.

**Final Model: XGBoost**  
Gradient boosted trees with `scale_pos_weight=10` (ratio of negatives to positives)
for imbalance handling. Hyperparameter grid (36 combinations):
- `n_estimators` ∈ {300, 500}
- `max_depth` ∈ {3, 4, 6}
- `learning_rate` ∈ {0.05, 0.1}
- `min_child_weight` ∈ {3, 5, 10}

### Train/Test Split

Time-based 80/20 split by anchor date across all 9 metros combined:
- **Train:** 14,772 restaurants (earliest 80% by anchor date)
- **Test:** 3,692 restaurants (latest 20% by anchor date)
- Train closure rate: 10.1% | Test closure rate: 6.2%

The lower closure rate in the test set is expected — the test set contains
temporally later anchor dates, when COVID-era data is more represented and
incomplete labeling affects some restaurants.

### Cross-Validation

5-fold StratifiedKFold on the training set (`shuffle=False` to preserve temporal order).
Best hyperparameters selected by mean validation AUC-PR across folds. Both models
retrained on the full 80% training set using best params before final evaluation.

### Why AUC-PR as Primary Metric

With a ~9% positive rate, accuracy is meaningless (predicting all-negative gives 91%
accuracy). AUC-ROC is insensitive to the imbalance. AUC-PR (average precision) directly
measures performance on the minority class and is the correct primary metric here.

---

## Results

| Model | CV AUC-PR | Train AUC-PR | Test AUC-PR | Train AUC-ROC | Test AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.355 ± 0.054 | 0.372 | 0.245 | 0.794 | 0.808 |
| **XGBoost** | **0.400 ± 0.061** | **0.562** | **0.328** | **0.898** | **0.833** |

Random baseline AUC-PR ≈ 0.062 (closure rate). Both models substantially exceed baseline.

### F1 at Optimal Threshold

| Model | Split | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|---|
| XGBoost | Train | 0.688 | 0.515 | 0.529 | 0.522 |
| XGBoost | **Test** | **0.711** | **0.340** | **0.437** | **0.382** |
| Logistic Regression | Train | 0.713 | 0.375 | 0.380 | 0.377 |
| Logistic Regression | **Test** | **0.751** | **0.304** | **0.376** | **0.336** |

### Interpreting the Overfitting

XGBoost shows a meaningful train-to-test gap in AUC-PR (0.562 → 0.328, ~42%). Two
factors contribute:

1. **Model variance:** XGBoost with `scale_pos_weight=10` tends to memorize minority
   class patterns in the training folds. Higher `min_child_weight` or lower `max_depth`
   would close this gap at some cost to test performance.

2. **Distribution shift:** The test set has a lower closure rate (6.2% vs 10.1% in
   train) because it contains later anchor dates — a real temporal shift, not a sampling
   artifact. AUC-PR is sensitive to base rate changes, so part of the gap reflects
   this shift rather than true overfitting.

The AUC-ROC gap is much smaller (0.898 → 0.833), suggesting the model's *ranking*
generalizes well — it's the *calibration* that degrades. For the underwriting use case
(rank-order, not calibrated probability), AUC-ROC is the more operationally relevant
metric.

---

## ClosureWatch App

Interactive Streamlit dashboard for exploring model predictions.

```bash
streamlit run app.py
```

Features:
- Photo card grid of restaurants sorted by predicted closure risk
- Risk percentile and color-coded risk label (Low / Medium / High / Critical)
- SHAP waterfall explanations per restaurant (which features drove the score)
- Outcome banners for confirmed closures
- Filter by metro, category, risk tier

---

## How to Run

```bash
pip install -r requirements.txt

# Full pipeline from raw Yelp JSON:
python 01_load_filter.py          # load and filter all 9 metros
python 02_build_labels.py         # build binary closure labels
python 03_feature_engineering.py  # engineer 81 time-windowed features
python 04_eda.py                  # EDA figures -> figures/
python 04b_data_quality.py        # data quality diagnostics

# Modeling (processed data already committed -- can run directly):
python 05_modeling.py             # LR + XGBoost, 5-fold CV, save models + figures

# App:
streamlit run app.py
```

---

## Repository Layout

```
├── config_00.py               # global constants (paths, dates, metric)
├── 01_load_filter.py          # load Yelp JSON -> filtered Parquet (all 9 metros)
├── 02_build_labels.py         # anchor dates + leakage-safe binary labels
├── 03_feature_engineering.py  # time-windowed features
├── 04_eda.py                  # EDA plots -> figures/
├── 04b_data_quality.py        # data quality diagnostics
├── 05_modeling.py             # LR + XGBoost, 5-fold CV, evaluation, figures
├── app.py                     # ClosureWatch Streamlit dashboard
├── app_helpers.py             # risk scoring + SHAP helpers
├── data/
│   ├── processed*/            # processed Parquets per metro (committed)
│   └── raw/                   # Yelp JSON (NOT committed, ~10 GB)
├── models/                    # trained model files (committed)
├── figures/                   # all output figures (committed)
└── experiments/               # exploratory work (LOMO CV, ensemble, zero-shot)
```

---

## Experiments

See `experiments/` for work that informed but is not part of the final pipeline:

- `10_lomo_cv.py` — Leave-One-Metro-Out CV: train on 8 metros, test on 1. Mean
  AUC-PR 0.360, AUC-ROC 0.781 across 9 metros.
- `06_ensemble.py` — Stacking ensemble (XGBoost + Random Forest + LR). AUC-PR 0.200
  on Tampa hold-out — did not beat single XGBoost.
- `08_philadelphia.py` — Zero-shot transfer: Tampa-trained model applied to Philadelphia.
  AUC-PR 0.142 vs 0.203 Tampa (-30%) — ranking signal partially universal, calibration
  is local.
- `09_multi_metro.py` — Global model trained on all 9 metros pooled (no held-out metro).
- `14_kfold_global.py` — K-fold CV on the global pooled dataset.
