# ClosureWatch -- Restaurant Closure Prediction

Binary classification: given 12 months of Yelp behavioral signals for a restaurant,
predict whether it will permanently close in the next 6 months.
Framed as an SME credit underwriting problem -- alternative lenders use public behavioral
signals the same way banks use transaction history.

## Dataset

**Source:** [Yelp Academic Dataset](https://www.yelp.com/dataset)
**Geography:** 9 US metros -- Philadelphia, Tampa, Indianapolis, Tucson, Nashville,
New Orleans, Saint Louis, Reno, Boise
**Size:** ~18,500 restaurants, ~10% closure rate (class-imbalanced)

Raw data is not included in this repo (download from the link above and place in `data/raw/`).
Processed feature files are committed -- you can run `05_modeling.py` directly.

## How to Run

```bash
pip install -r requirements.txt

# Data pipeline (only needed if starting from raw JSON):
python 01_load_filter.py          # load and filter all metros from Yelp JSON
python 02_build_labels.py         # build binary closure labels
python 03_feature_engineering.py  # engineer 81 time-windowed features

# Analysis:
python 04_eda.py                  # EDA figures -> figures/
python 04b_data_quality.py        # data quality diagnostics

# Modeling (processed data already committed -- can run directly):
python 05_modeling.py             # LR benchmark + XGBoost, 5-fold CV, figures, saved models

# App:
streamlit run app.py              # launch ClosureWatch dashboard
```

## Results

| Model | CV AUC-PR | Test AUC-PR | Test AUC-ROC |
|---|---|---|---|
| Logistic Regression (benchmark) | 0.355 +/- 0.054 | 0.245 | 0.808 |
| **XGBoost (5-fold CV tuned)** | **0.400 +/- 0.061** | **0.328** | **0.833** |

Primary metric is AUC-PR (area under the precision-recall curve) -- the correct choice
for imbalanced binary classification. Train/test split is time-based (no random splits)
to prevent temporal leakage.

## Features (81 total)

Time-windowed signals computed strictly before the anchor date:
- Review velocity, drought flags, momentum
- Rating trend, VADER sentiment trend
- Checkin signals, tip signals
- Reviewer quality metrics
- Business metadata (price range, category, hours)
- Null-flag indicators for informatively-missing features
- Sentence-embedding PCA features (MiniLM-L6-v2, PCA-32)

## Temporal Design

```
|--- obs_start ---|--- OBSERVATION WINDOW (12m) ---|--- anchor_date ---|--- OUTCOME (6m) ---|
                         Features built here                                    Label here
```

Anchor date = 80th percentile review date (not the last review -- prevents leakage).
All features use only data strictly before the anchor date.

## Experiments

See `experiments/` for exploratory work: LOMO cross-validation, stacking ensemble,
Philadelphia zero-shot transfer, and the original Tampa-only pipeline.
