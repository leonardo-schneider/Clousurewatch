# Restaurant Closure Predictor — UI & Ensemble Design

**Date:** 2026-04-16  
**Project:** ML Final — Restaurant Failure Prediction  
**Status:** Approved

---

## Overview

Two additions to the existing pipeline:

1. **`06_ensemble.py`** — combines the 5 trained models into an ensemble, tries strategies in order of complexity until one beats the XGBoost baseline (Test AUC-PR = 0.2069)
2. **`app.py`** — Streamlit dark-theme dashboard where users can browse Tampa Bay restaurants ranked by closure risk and inspect individual predictions

No existing scripts are modified. The pipeline run order becomes:

```
python 01_load_filter.py
python 02_build_labels.py
python 03_feature_engineering.py
python 04_eda.py
python 05_modeling.py
python 06_ensemble.py        ← new
streamlit run app.py         ← new
```

---

## Section 1: Ensemble (`06_ensemble.py`)

### Goal

Improve on XGBoost's Test AUC-PR of 0.2069 by combining all 5 trained models. Try strategies in order; stop when one beats the baseline.

### Models Used

All 5 `.pkl` files saved by `05_modeling.py` in `models/`:

| File | Model |
|---|---|
| `xgboost.pkl` | XGBoost (baseline winner) |
| `random_forest.pkl` | Random Forest |
| `lightgbm.pkl` | LightGBM |
| `mlp.pkl` | MLP Neural Network |
| `logistic_regression.pkl` | Logistic Regression |

### Strategy Ladder

Tried in order. Stop at the first strategy that beats XGBoost's Test AUC-PR = 0.2069.

**Strategy 1 — Simple Average**  
Average the predicted probabilities from all 5 models with equal weight.

```
ensemble_prob = mean([prob_xgb, prob_rf, prob_lgb, prob_mlp, prob_lr])
```

**Strategy 2 — Weighted Average**  
Weight each model by its CV AUC-PR score (from `results_summary.json`), normalized to sum to 1.

```
weights = cv_auc_pr_per_model / sum(cv_auc_pr_per_model)
ensemble_prob = sum(weights * probs)
```

CV AUC-PR values (from confirmed results):

| Model | CV AUC-PR |
|---|---|
| XGBoost | 0.124 |
| Logistic Regression | 0.106 |
| Random Forest | 0.112 |
| LightGBM | 0.091 |
| MLP | 0.083 |

**Strategy 3 — Stacking**  
Train a Logistic Regression meta-model on out-of-fold (OOF) predictions from the 5-fold time-aware CV. Meta-model is trained only on train data; test predictions come from models trained on full train set.

Leakage safety: weights and meta-model derived from CV folds on train data only — test set is never seen until final evaluation.

### Outputs

- `models/ensemble_results.json` — metrics (AUC-PR, AUC-ROC, F1) for each strategy and the winning strategy name
- `data/processed/ensemble_predictions.parquet` — one row per restaurant with columns: `business_id`, `name`, `city`, `stars`, `risk_score` (winning ensemble probability), `label` (true closure), `anchor_date`, and the top 5 feature values used for signal display in the UI

---

## Section 2: Streamlit App (`app.py`)

### Goal

Interactive dark-theme dashboard that lets any user look up Tampa Bay restaurants and see their predicted closure risk, key warning signals, and feature contributions.

### Layout

**Two-panel design:**

```
┌──────────────────┬─────────────────────────────────────────┐
│  SIDEBAR (left)  │  DETAIL PANEL (right)                   │
│                  │                                         │
│  Search bar      │  Restaurant name + Yelp info            │
│  ─────────────   │  Big risk % gauge  (color-coded)        │
│  Mario's  81% 🔴 │                                         │
│  Sunrise  73% 🔴 │  KEY SIGNALS                            │
│  Gulf T.  58% 🟠 │  ⚠ 112 days no reviews                  │
│  Anchor   44% 🟠 │  ⚠ Review velocity dropped 68%         │
│  Bay Sus  12% 🟢 │  ✓ Sentiment score stable               │
│                  │                                         │
│                  │  TOP FEATURES (horizontal bar chart)    │
└──────────────────┴─────────────────────────────────────────┘
```

**Sidebar:**
- Restaurants sorted descending by `risk_score`
- Each row shows: name, neighborhood, risk % badge (red ≥60%, orange 30–59%, green <30%)
- Search box filters the list by name (case-insensitive)
- Clicking a row sets `st.session_state["selected_id"]`

**Detail panel (right):**
- Restaurant name, city, Yelp star rating
- Large risk percentage (colored: red/orange/green matching badge)
- Risk label: HIGH / MEDIUM / LOW
- Key signals section: top 3–5 features that pushed the score up or down, shown as human-readable sentences (e.g. "No reviews in last 112 days")
- Horizontal bar chart (Plotly) of top feature contributions, colored red (risk-increasing) or green (risk-decreasing)

### Data Source

Loads `data/processed/ensemble_predictions.parquet`. If the file doesn't exist (ensemble not run yet), falls back to loading `models/xgboost.pkl` and running predictions inline on `data/processed/features.parquet`.

### Theme

Dark: navy/dark blue background (`#1a1a2e`, `#16213e`, `#0f3460`), red accent (`#e94560`), orange warning (`#f7a440`), green safe (`#4caf50`).

Implemented via `st.set_page_config` + custom CSS injected with `st.markdown`.

### Dependencies

Add to `requirements.txt`:
```
streamlit
plotly
```

---

## Section 3: File Structure Changes

```
Desktop/Final ML/
├── 06_ensemble.py                          ← NEW
├── app.py                                  ← NEW
├── docs/superpowers/specs/
│   └── 2026-04-16-...-design.md           ← NEW (this file)
├── requirements.txt                        ← add streamlit, plotly
└── ... (all existing files unchanged)
```

---

## Success Criteria

- `06_ensemble.py` runs after `05_modeling.py` without errors, prints a leaderboard of ensemble strategies, and saves predictions
- At least one ensemble strategy is evaluated; results saved to `models/ensemble_results.json`
- `streamlit run app.py` launches without errors
- Sidebar shows restaurants sorted by risk with color-coded badges
- Clicking a restaurant shows the detail panel with risk %, signals, and feature bar chart
- App works with or without the ensemble having been run (fallback to XGBoost)
