# ClosureWatch — Repo Cleanup & Academic Submission Design

**Date:** 2026-05-07
**Due:** 2026-05-14

---

## Goal

Reorganize the ClosureWatch repo into a clean, linear academic pipeline that maps directly to the assignment rubric. Experimental work is preserved in `experiments/` but not part of the graded submission. Raw data stays off GitHub; processed results and code are committed.

---

## Repo Structure (Final State)

### Main pipeline (root)

| Script | Purpose |
|---|---|
| `README.md` | Dataset source, setup instructions, how to run, results summary |
| `config_00.py` | Global constants (unchanged) |
| `requirements.txt` | Python dependencies |
| `01_load_filter.py` | Load & clean all 9 metros from Yelp JSON (replaces Tampa-only version) |
| `02_build_labels.py` | Anchor dates + binary labels for all metros |
| `03_feature_engineering.py` | Time-windowed features for all metros |
| `04_eda.py` | EDA plots → figures/ |
| `04b_data_quality.py` | Data quality diagnostics |
| `05_modeling.py` | **Final model**: LR benchmark + XGBoost, 80/20 split, 5-fold CV, figures, saved models |
| `app.py` | ClosureWatch Streamlit dashboard |
| `app_helpers.py` | Risk color/label/SHAP helpers |
| `photo_index.py` | Business → best photo path lookup |

### experiments/ (moved, not deleted)

All exploratory scripts that are not part of the graded pipeline:

- `01_load_filter_tampa.py` (original Tampa-only loader)
- `06_ensemble.py`
- `07_model_analysis.py`
- `08_philadelphia.py`
- `09_multi_metro.py`
- `10_lomo_cv.py`
- `14_kfold_global.py`
- Any other numbered scripts not in the main pipeline
- `experiments/README.md` — one-line note: "Exploratory work, not part of the main pipeline."

### What gets committed to GitHub

| Path | Committed? | Reason |
|---|---|---|
| `data/processed*/` | Yes | Processed parquets (~small), enables reproducibility without raw data |
| `models/xgboost_final.pkl` | Yes | Final model for app |
| `models/logistic_regression_final.pkl` | Yes | Benchmark model |
| `figures/` | Yes | All final figures |
| `data/raw/` | No (.gitignored) | ~10GB Yelp JSON, download from yelp.com/dataset |
| `data/raw_photos/` | No (.gitignored) | 200k images |

---

## 05_modeling.py — Final Modeling Script

Single script that covers the full ML pipeline, satisfying all rubric requirements:

### Steps

1. **Load** processed parquets for all 9 metros (output of `03_feature_engineering.py`)
2. **Split** — time-based 80/20: earliest 80% by anchor_date → train, latest 20% → test. No random splits.
3. **5-Fold CV on training set** (StratifiedKFold, shuffle=False to preserve time order):
   - Logistic Regression: sweeps C ∈ {0.01, 0.1, 1.0, 10.0}
   - XGBoost: grid over n_estimators, max_depth, learning_rate, min_child_weight
   - Selects best params by mean val AUC-PR across folds
4. **Retrain** both models on full 80% training set with best params
5. **Evaluate** on all three samples and report AUC-PR, AUC-ROC, F1:
   - CV validation folds (mean ± std)
   - Training set
   - Held-out test set
6. **Save figures**: PR curves (train vs test), ROC curves (train vs test), confusion matrices, feature importance
7. **Save models**: `models/xgboost_final.pkl`, `models/logistic_regression_final.pkl`

### Metrics summary (from existing runs)

| Model | CV AUC-PR | Train AUC-PR | Test AUC-PR | Train AUC-ROC | Test AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.353 | 0.372 | 0.245 | 0.794 | 0.808 |
| XGBoost | 0.399 | 0.562 | 0.328 | 0.898 | 0.833 |

---

## README Structure

```
# ClosureWatch — Restaurant Closure Prediction

[One-paragraph problem description and business framing]

## Dataset
Source: Yelp Academic Dataset (https://www.yelp.com/dataset)
9 US metros, ~18,000 restaurants.
Raw data not included — download and place in data/raw/.

## How to Run
pip install -r requirements.txt

python 01_load_filter.py
python 02_build_labels.py
python 03_feature_engineering.py
python 04_eda.py
python 05_modeling.py
streamlit run app.py

## Results
Benchmark (Logistic Regression): AUC-PR 0.245 | AUC-ROC 0.808
Final model (XGBoost, 5-fold CV): AUC-PR 0.328 | AUC-ROC 0.833

## Experiments
See experiments/ for exploratory work (LOMO CV, ensemble, Philadelphia zero-shot).
```

---

## Assignment Rubric Mapping

| Requirement | Where it lives |
|---|---|
| Novel dataset (not Kaggle) | README + `01_load_filter.py` |
| Data cleaning | `01_load_filter.py`, `02_build_labels.py` |
| EDA + plots | `04_eda.py`, `04b_data_quality.py`, `figures/` |
| Redundant features discussion | `04_eda.py` + correlation heatmap figure |
| Benchmark model (Logistic Regression) | `05_modeling.py` |
| Correct metrics for imbalanced data | `05_modeling.py` (AUC-PR primary) |
| ML model (XGBoost) | `05_modeling.py` |
| Train/validation/test split | `05_modeling.py` (80/20 time-based) |
| K-fold CV for hyperparameters | `05_modeling.py` (5-fold StratifiedKFold) |
| Compare metrics across all three samples | `05_modeling.py` output + figures |

---

## app.py Update

`app.py` currently loads `xgboost_global.pkl` (trained on 100% of data). It must be updated to load `xgboost_final.pkl` (trained on the 80% training split) so the app and the modeling script are consistent.

---

## Out of Scope (for this cleanup)

- Streamlit Cloud deployment (after submission)
- Adding new features or models
- Changing any modeling decisions
