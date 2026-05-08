# ClosureWatch Repo Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the ClosureWatch repo into a clean, linear academic pipeline (01→05 + app) with experimental work archived in `experiments/`, a proper final model trained on 80/20 split with 5-fold CV, and a README the professor can follow.

**Architecture:** Seven numbered scripts (01–05 + EDA) feed sequentially from raw Yelp JSON → processed parquets → final models → figures. `app.py` loads the final model. Experimental scripts (LOMO, ensemble, Philadelphia, etc.) move to `experiments/` unchanged.

**Tech Stack:** Python 3.10+, pandas, XGBoost, scikit-learn, matplotlib, joblib, streamlit

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Rename | `01_load_filter.py` → `experiments/01_load_filter_tampa.py` | Original Tampa-only loader (archived) |
| Rewrite | `01_load_filter.py` | Load & filter all 9 metros from raw Yelp JSON |
| Rewrite | `02_build_labels.py` | Build anchor dates + binary labels for all metros |
| Modify | `03_feature_engineering.py` | Add `__main__` block to loop over all metros |
| Keep | `04_eda.py`, `04b_data_quality.py` | EDA stays Tampa-based (explicit in comments) |
| Create | `05_modeling.py` | LR + XGBoost, 80/20 split, 5-fold CV, figures, saved models |
| Modify | `app.py` | Change model path to `xgboost_final.pkl` |
| Modify | `.gitignore` | Exclude raw data, raw photos, pycache |
| Create | `README.md` | Dataset source, setup, how to run, results |
| Create | `experiments/README.md` | One-line note on experimental work |
| Move | 6 scripts | `06_ensemble.py`, `07_model_analysis.py`, `08_philadelphia.py`, `09_multi_metro.py`, `10_lomo_cv.py`, `14_kfold_global.py` → `experiments/` |

---

## Task 1: Create `experiments/` and move experimental scripts

**Files:**
- Create: `experiments/README.md`
- Move: `06_ensemble.py`, `07_model_analysis.py`, `08_philadelphia.py`, `09_multi_metro.py`, `10_lomo_cv.py`, `14_kfold_global.py`, `01_load_filter.py`

- [ ] **Step 1: Create experiments/ folder and its README**

```bash
mkdir experiments
```

Write `experiments/README.md`:
```markdown
# Experiments

Exploratory work — not part of the main graded pipeline.

| Script | Purpose |
|---|---|
| `01_load_filter_tampa.py` | Original Tampa-only data loader |
| `06_ensemble.py` | Stacking ensemble (XGB + RF + LR) |
| `07_model_analysis.py` | SHAP global, error analysis, threshold curves |
| `08_philadelphia.py` | Philadelphia zero-shot transfer |
| `09_multi_metro.py` | Multi-metro feature pipeline (per-city CLI) |
| `10_lomo_cv.py` | Leave-One-Metro-Out cross-validation |
| `14_kfold_global.py` | Early global kfold prototype |
```

- [ ] **Step 2: Move experimental scripts**

```bash
git mv 06_ensemble.py experiments/06_ensemble.py
git mv 07_model_analysis.py experiments/07_model_analysis.py
git mv 08_philadelphia.py experiments/08_philadelphia.py
git mv 09_multi_metro.py experiments/09_multi_metro.py
git mv 10_lomo_cv.py experiments/10_lomo_cv.py
git mv 14_kfold_global.py experiments/14_kfold_global.py
git mv 01_load_filter.py experiments/01_load_filter_tampa.py
```

- [ ] **Step 3: Commit**

```bash
git add experiments/
git commit -m "refactor: move experimental scripts to experiments/"
```

---

## Task 2: Rewrite `01_load_filter.py` — global loader for all 9 metros

**Files:**
- Create: `01_load_filter.py`

Reference: `experiments/09_multi_metro.py` → `load_businesses()` and `load_interactions()` have the exact JSON parsing logic to borrow.

- [ ] **Step 1: Write `01_load_filter.py`**

```python
"""
01_load_filter.py -- Load and filter raw Yelp JSON for all 9 metros.

Reads the single Yelp Academic Dataset (all cities in one JSON) and
filters each metro by city+state. Saves per-metro raw parquets to
data/processed_{metro}/.

Run once after downloading the dataset:
    python 01_load_filter.py

Output per metro:
    data/processed_{metro}/businesses.parquet
    data/processed_{metro}/reviews.parquet
    data/processed_{metro}/checkins.parquet
    data/processed_{metro}/tips.parquet
"""
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path

from config_00 import RAW_DIR

ENCODING = "utf-8"

# (metro_key, city_name, state_code, output_dir)
METRO_CONFIGS = [
    ("tampa",          "Tampa",        "FL", Path("data/processed")),
    ("philadelphia",   "Philadelphia", "PA", Path("data/processed_philly")),
    ("indianapolis",   "Indianapolis", "IN", Path("data/processed_indianapolis")),
    ("tucson",         "Tucson",       "AZ", Path("data/processed_tucson")),
    ("nashville",      "Nashville",    "TN", Path("data/processed_nashville")),
    ("new_orleans",    "New Orleans",  "LA", Path("data/processed_new_orleans")),
    ("saint_louis",    "Saint Louis",  "MO", Path("data/processed_saint_louis")),
    ("reno",           "Reno",         "NV", Path("data/processed_reno")),
    ("boise",          "Boise",        "ID", Path("data/processed_boise")),
]


def load_businesses(city: str, state: str) -> pd.DataFrame:
    path = RAW_DIR / "yelp_academic_dataset_business.json"
    rows = []
    with open(path, encoding=ENCODING) as f:
        for line in f:
            r = json.loads(line)
            if r.get("city") != city or r.get("state") != state:
                continue
            cats = r.get("categories") or ""
            if "Restaurants" not in cats and "Food" not in cats:
                continue
            price_raw = (r.get("attributes") or {}).get("RestaurantsPriceRange2")
            try:
                price_range = float(price_raw) if price_raw else None
            except (ValueError, TypeError):
                price_range = None
            hours = r.get("hours") or {}
            open_days = sum(1 for v in hours.values() if v) if hours else None
            rows.append({
                "business_id":        r["business_id"],
                "name":               r.get("name", ""),
                "city":               r.get("city", ""),
                "state":              r.get("state", ""),
                "is_open":            int(r.get("is_open", 1)),
                "stars_yelp":         float(r.get("stars", 0)),
                "review_count_yelp":  int(r.get("review_count", 0)),
                "price_range":        price_range,
                "open_days_per_week": open_days,
                "categories":         cats,
                "latitude":           r.get("latitude"),
                "longitude":          r.get("longitude"),
            })
    return pd.DataFrame(rows)


def load_interactions(bids: set):
    """Returns (reviews, checkins, tips) DataFrames for the given business_id set."""
    rev_rows, ci_rows, tip_rows = [], [], []

    with open(RAW_DIR / "yelp_academic_dataset_review.json", encoding=ENCODING) as f:
        for line in f:
            r = json.loads(line)
            if r["business_id"] not in bids:
                continue
            rev_rows.append({
                "business_id": r["business_id"],
                "date":        pd.Timestamp(r["date"]),
                "stars":       float(r["stars"]),
                "text":        r.get("text", ""),
                "useful":      int(r.get("useful", 0)),
                "funny":       int(r.get("funny", 0)),
                "cool":        int(r.get("cool", 0)),
            })

    with open(RAW_DIR / "yelp_academic_dataset_checkin.json", encoding=ENCODING) as f:
        for line in f:
            r = json.loads(line)
            if r["business_id"] not in bids:
                continue
            for ts in r.get("date", "").split(", "):
                ts = ts.strip()
                if ts:
                    ci_rows.append({"business_id": r["business_id"],
                                    "checkin_date": pd.Timestamp(ts)})

    with open(RAW_DIR / "yelp_academic_dataset_tip.json", encoding=ENCODING) as f:
        for line in f:
            r = json.loads(line)
            if r["business_id"] not in bids:
                continue
            tip_rows.append({"business_id": r["business_id"],
                             "date":              pd.Timestamp(r["date"]),
                             "compliment_count":  int(r.get("compliment_count", 0))})

    return pd.DataFrame(rev_rows), pd.DataFrame(ci_rows), pd.DataFrame(tip_rows)


def process_metro(metro_key, city, state, out):
    print(f"\n[{metro_key}] {city}, {state}")
    out.mkdir(parents=True, exist_ok=True)

    biz = load_businesses(city, state)
    print(f"  {len(biz):,} restaurants")
    if biz.empty:
        print("  WARNING: no businesses found — check city/state spelling")
        return

    reviews, checkins, tips = load_interactions(set(biz["business_id"]))
    print(f"  reviews={len(reviews):,}  checkins={len(checkins):,}  tips={len(tips):,}")

    biz.to_parquet(out / "businesses.parquet", index=False)
    reviews.to_parquet(out / "reviews.parquet",   index=False)
    checkins.to_parquet(out / "checkins.parquet", index=False)
    tips.to_parquet(out / "tips.parquet",         index=False)
    print(f"  Saved -> {out}/")


if __name__ == "__main__":
    print("Loading raw Yelp JSON for all 9 metros...")
    print("NOTE: Each pass reads the full JSON (~3-5 GB). Runtime ~30-60 min total.")
    for metro_key, city, state, out in METRO_CONFIGS:
        process_metro(metro_key, city, state, out)
    print("\nDone. Run 02_build_labels.py next.")
```

- [ ] **Step 2: Verify script is importable (no syntax errors)**

```bash
python -c "import importlib; importlib.import_module('01_load_filter')" 2>&1 || python -c "
import ast, sys
with open('01_load_filter.py') as f:
    ast.parse(f.read())
print('Syntax OK')
"
```
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add 01_load_filter.py
git commit -m "feat: rewrite 01_load_filter.py as global loader for all 9 metros"
```

---

## Task 3: Rewrite `02_build_labels.py` — global label builder

**Files:**
- Modify: `02_build_labels.py`

Reference: `experiments/09_multi_metro.py` → `build_labels()` function (lines ~139-200) has the exact 3-way review-recency label logic.

- [ ] **Step 1: Read the current `02_build_labels.py` and `experiments/09_multi_metro.py` `build_labels()` function**

Read both files to understand the existing logic before overwriting.

- [ ] **Step 2: Rewrite `02_build_labels.py`**

```python
"""
02_build_labels.py -- Build anchor dates and binary closure labels for all 9 metros.

Label rule (review-recency, no is_open leakage):
  last_review > outcome_end + 3m  → 0 (still active)
  last_review <= outcome_end      → 1 (closed)
  ambiguous window                → 0

Reads:  data/processed_{metro}/businesses.parquet
        data/processed_{metro}/reviews.parquet
Writes: data/processed_{metro}/businesses_labeled.parquet
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta

from config_00 import OBS_MONTHS, OUTCOME_MONTHS, EARLIEST_ANCHOR, TARGET_COL

LATEST_ANCHOR = pd.Timestamp("2020-06-01")

METRO_DIRS = {
    "tampa":         Path("data/processed"),
    "philadelphia":  Path("data/processed_philly"),
    "indianapolis":  Path("data/processed_indianapolis"),
    "tucson":        Path("data/processed_tucson"),
    "nashville":     Path("data/processed_nashville"),
    "new_orleans":   Path("data/processed_new_orleans"),
    "saint_louis":   Path("data/processed_saint_louis"),
    "reno":          Path("data/processed_reno"),
    "boise":         Path("data/processed_boise"),
}


def build_labels(biz: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    earliest = pd.Timestamp(EARLIEST_ANCHOR)
    rev_groups = reviews.groupby("business_id")
    rows = []
    for _, b in biz.iterrows():
        bid = b["business_id"]
        if bid not in rev_groups.groups:
            continue
        rev = rev_groups.get_group(bid).sort_values("date")
        if rev.empty:
            continue
        last_review = rev["date"].iloc[-1]
        n_reviews   = len(rev)

        # Anchor = 80th percentile review date (anti-leakage)
        anchor_date = rev["date"].quantile(0.80, interpolation="nearest")
        if anchor_date < earliest or anchor_date > LATEST_ANCHOR:
            continue

        obs_start   = anchor_date - relativedelta(months=OBS_MONTHS)
        outcome_end = anchor_date + relativedelta(months=OUTCOME_MONTHS)

        if last_review > outcome_end + relativedelta(months=3):
            label = 0
        elif last_review <= outcome_end:
            label = 1
        else:
            label = 0

        rows.append({
            **b.to_dict(),
            "anchor_date":   anchor_date,
            "obs_start":     obs_start,
            "outcome_end":   outcome_end,
            "last_review":   last_review,
            "n_reviews_raw": n_reviews,
            TARGET_COL:      label,
        })
    return pd.DataFrame(rows)


def process_metro(metro_key: str, d: Path) -> None:
    biz_path = d / "businesses.parquet"
    rev_path = d / "reviews.parquet"
    if not biz_path.exists() or not rev_path.exists():
        print(f"  [{metro_key}] SKIP — run 01_load_filter.py first")
        return

    biz     = pd.read_parquet(biz_path)
    reviews = pd.read_parquet(rev_path)
    labeled = build_labels(biz, reviews)
    n_closed = int(labeled[TARGET_COL].sum())
    print(f"  [{metro_key}] {len(labeled):,} labeled  closed={n_closed} ({n_closed/len(labeled):.1%})")

    labeled.to_parquet(d / "businesses_labeled.parquet", index=False)


if __name__ == "__main__":
    print("Building labels for all 9 metros...")
    for metro_key, d in METRO_DIRS.items():
        process_metro(metro_key, d)
    print("\nDone. Run 03_feature_engineering.py next.")
```

- [ ] **Step 3: Verify syntax**

```bash
python -c "
import ast
with open('02_build_labels.py') as f:
    ast.parse(f.read())
print('Syntax OK')
"
```
Expected: `Syntax OK`

- [ ] **Step 4: Commit**

```bash
git add 02_build_labels.py
git commit -m "feat: rewrite 02_build_labels.py as global label builder for all 9 metros"
```

---

## Task 4: Update `03_feature_engineering.py` — add global `__main__` block

**Files:**
- Modify: `03_feature_engineering.py`

The feature engineering functions in `03_feature_engineering.py` already work on individual metros (they are called by `09_multi_metro.py`). We just need to add a `__main__` block that loops over all metros so the script can be run standalone.

- [ ] **Step 1: Read `03_feature_engineering.py` to find the main entry point and key function names**

Look for: the existing `if __name__ == "__main__":` block (Tampa-only), and the main feature-building function (e.g., `build_features()` or `build_features_one()`).

- [ ] **Step 2: Replace the `__main__` block**

Find the existing `if __name__ == "__main__":` block at the bottom of `03_feature_engineering.py` and replace it with:

```python
if __name__ == "__main__":
    """
    Build features for all 9 metros. Reads businesses_labeled.parquet +
    reviews/checkins/tips parquets per metro. Writes features.parquet.
    Run after 02_build_labels.py.
    """
    from pathlib import Path

    METRO_DIRS = {
        "tampa":         Path("data/processed"),
        "philadelphia":  Path("data/processed_philly"),
        "indianapolis":  Path("data/processed_indianapolis"),
        "tucson":        Path("data/processed_tucson"),
        "nashville":     Path("data/processed_nashville"),
        "new_orleans":   Path("data/processed_new_orleans"),
        "saint_louis":   Path("data/processed_saint_louis"),
        "reno":          Path("data/processed_reno"),
        "boise":         Path("data/processed_boise"),
    }

    print("Building features for all 9 metros...")
    for metro_key, d in METRO_DIRS.items():
        labeled_path = d / "businesses_labeled.parquet"
        if not labeled_path.exists():
            print(f"  [{metro_key}] SKIP — run 02_build_labels.py first")
            continue
        print(f"  [{metro_key}] ...")
        # build_metro_features() is the existing function in this file.
        # Check the function name at the top of 03_feature_engineering.py and use it here.
        features = build_metro_features(d)   # adjust name if different
        features.to_parquet(d / "features.parquet", index=False)
        n = len(features)
        n_c = int(features["closed_within_6m"].sum())
        print(f"    {n:,} restaurants  closed={n_c} ({n_c/n:.1%})  saved -> {d}/features.parquet")
    print("\nDone. Run 04_eda.py next.")
```

> **Note for implementer:** Read `03_feature_engineering.py` first to confirm the actual function name that builds features for a metro directory. It may be `build_features()`, `build_metro_features()`, or similar. Adjust the call accordingly.

- [ ] **Step 3: Verify syntax**

```bash
python -c "
import ast
with open('03_feature_engineering.py') as f:
    ast.parse(f.read())
print('Syntax OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add 03_feature_engineering.py
git commit -m "feat: add global __main__ loop to 03_feature_engineering.py for all 9 metros"
```

---

## Task 5: Write `05_modeling.py` — final model (LR + XGBoost, 5-fold CV, 80/20 split)

**Files:**
- Create: `05_modeling.py`
- Output: `models/xgboost_final.pkl`, `models/logistic_regression_final.pkl`
- Output figures: `figures/35_train_pr_curve.png`, `figures/35b_train_roc_curve.png`, `figures/36_train_pr_curve_lr.png`, `figures/36b_train_roc_curve_lr.png`

This is the heart of the assignment. It demonstrates: proper split, k-fold CV, benchmark vs ML model, correct metrics for imbalanced data.

- [ ] **Step 1: Write `05_modeling.py`**

```python
"""
05_modeling.py -- Final model: Logistic Regression benchmark + XGBoost.

Pipeline:
  1. Load features.parquet for all 9 metros
  2. Time-based 80/20 train/test split (no random splits -- anti-leakage)
  3. 5-fold StratifiedKFold CV on training set to tune hyperparameters
  4. Retrain both models on full training set with best params
  5. Report metrics: CV folds, train set, test set
  6. Save figures: PR curves, ROC curves (train vs test, per model)
  7. Save models: xgboost_final.pkl, logistic_regression_final.pkl

Primary metric: AUC-PR (correct for imbalanced data; 10% closure rate).
Secondary metric: AUC-ROC, F1.
"""
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_curve, roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config_00 import TARGET_COL, MODEL_DIR, FIG_DIR, RANDOM_SEED
from app_helpers import add_null_flags

plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Constants ─────────────────────────────────────────────────────────────────
LATEST_ANCHOR = pd.Timestamp("2020-06-01")
TEST_FRAC     = 0.20
N_FOLDS       = 5
META_COLS     = {"business_id", TARGET_COL, "anchor_date", "city", "state", "metro"}

METRO_DIRS = {
    "tampa":         Path("data/processed"),
    "philadelphia":  Path("data/processed_philly"),
    "indianapolis":  Path("data/processed_indianapolis"),
    "tucson":        Path("data/processed_tucson"),
    "nashville":     Path("data/processed_nashville"),
    "new_orleans":   Path("data/processed_new_orleans"),
    "saint_louis":   Path("data/processed_saint_louis"),
    "reno":          Path("data/processed_reno"),
    "boise":         Path("data/processed_boise"),
}

XGB_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr,
     "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": w,
     "scale_pos_weight": 10, "eval_metric": "aucpr",
     "random_state": RANDOM_SEED, "verbosity": 0}
    for n in [300, 500]
    for d in [3, 4, 6]
    for lr in [0.05, 0.1]
    for w in [3, 5, 10]
]
LR_C_GRID = [0.01, 0.1, 1.0, 10.0]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_all() -> pd.DataFrame:
    frames = []
    for metro, d in METRO_DIRS.items():
        p = d / "features.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p} -- run 03_feature_engineering.py first")
        df = pd.read_parquet(p)
        df = add_null_flags(df)
        emb = d / "review_embeddings.parquet"
        if emb.exists():
            df = df.merge(pd.read_parquet(emb), on="business_id", how="left")
        df["metro"]       = metro
        df["anchor_date"] = pd.to_datetime(df["anchor_date"])
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    return all_df[all_df["anchor_date"] <= LATEST_ANCHOR].copy()


def time_split(df: pd.DataFrame):
    df_s   = df.sort_values("anchor_date").reset_index(drop=True)
    n_test = max(1, int(len(df_s) * TEST_FRAC))
    return df_s.iloc[:-n_test].copy(), df_s.iloc[-n_test:].copy()


# ── 5-fold CV ─────────────────────────────────────────────────────────────────
def kfold_tune(X_train: np.ndarray, y_train: np.ndarray):
    """
    StratifiedKFold on training data.
    Returns (best_xgb_params, best_lr_c, xgb_fold_scores, lr_fold_scores).
    Medians are refit per fold to prevent leakage.
    """
    skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=False)
    xgb_scores = np.zeros(len(XGB_GRID))
    lr_scores  = np.zeros(len(LR_C_GRID))
    fold_xgb   = []   # best val AUC-PR per fold (for reporting)
    fold_lr    = []

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr,  X_val  = X_train[tr_idx], X_train[val_idx]
        y_tr,  y_val  = y_train[tr_idx], y_train[val_idx]

        # Impute with training-fold medians only (anti-leakage)
        med     = np.nanmedian(X_tr, axis=0)
        X_tr_f  = np.where(np.isnan(X_tr),  med, X_tr)
        X_val_f = np.where(np.isnan(X_val), med, X_val)

        # XGBoost grid
        best_fold_xgb = -1.0
        for pi, params in enumerate(XGB_GRID):
            m  = XGBClassifier(**params)
            m.fit(X_tr_f, y_tr, verbose=False)
            pr = average_precision_score(y_val, m.predict_proba(X_val_f)[:, 1])
            xgb_scores[pi] += pr
            if pr > best_fold_xgb:
                best_fold_xgb = pr
        fold_xgb.append(round(best_fold_xgb, 4))

        # LR grid (scale within fold)
        sc       = StandardScaler()
        X_tr_sc  = sc.fit_transform(X_tr_f)
        X_val_sc = sc.transform(X_val_f)
        best_fold_lr = -1.0
        for ci, c in enumerate(LR_C_GRID):
            m  = LogisticRegression(C=c, class_weight="balanced", solver="lbfgs",
                                    max_iter=1000, random_state=RANDOM_SEED)
            m.fit(X_tr_sc, y_tr)
            pr = average_precision_score(y_val, m.predict_proba(X_val_sc)[:, 1])
            lr_scores[ci] += pr
            if pr > best_fold_lr:
                best_fold_lr = pr
        fold_lr.append(round(best_fold_lr, 4))

        print(f"  Fold {fold_i+1}/{N_FOLDS}  XGB={best_fold_xgb:.4f}  LR={best_fold_lr:.4f}")

    best_xgb = XGB_GRID[int(np.argmax(xgb_scores))]
    best_c   = LR_C_GRID[int(np.argmax(lr_scores))]
    return best_xgb, best_c, fold_xgb, fold_lr


# ── Metric helpers ────────────────────────────────────────────────────────────
def metrics(y_true, prob):
    auc_pr  = float(average_precision_score(y_true, prob))
    auc_roc = float(roc_auc_score(y_true, prob))
    p, r, thr = precision_recall_curve(y_true, prob)
    f1s     = 2 * p * r / (p + r + 1e-9)
    opt_thr = float(thr[np.argmax(f1s[:-1])])
    f1      = float(f1_score(y_true, (prob >= opt_thr).astype(int)))
    return {"AUC_PR": round(auc_pr,4), "AUC_ROC": round(auc_roc,4),
            "F1": round(f1,4), "threshold": round(opt_thr,4)}


# ── Figures ───────────────────────────────────────────────────────────────────
def save_pr_fig(y_tr, y_te, p_tr, p_te, title, path):
    baseline_tr = y_tr.mean()
    baseline_te = y_te.mean()
    subtitle    = (f"Train n={len(y_tr):,} ({baseline_tr:.1%} closure) | "
                   f"Test n={len(y_te):,} ({baseline_te:.1%} closure)")
    prec_tr, rec_tr, _ = precision_recall_curve(y_tr, p_tr)
    prec_te, rec_te, _ = precision_recall_curve(y_te, p_te)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec_tr, prec_tr, color="#2E86AB", lw=2,
            label=f"Train  AUC-PR = {average_precision_score(y_tr, p_tr):.3f}")
    ax.plot(rec_te, prec_te, color="#E84855", lw=2, ls="--",
            label=f"Test   AUC-PR = {average_precision_score(y_te, p_te):.3f}")
    ax.axhline(baseline_tr, color="#2E86AB", ls=":", lw=1.2, alpha=0.5,
               label=f"Train baseline ({baseline_tr:.3f})")
    ax.axhline(baseline_te, color="#E84855", ls=":", lw=1.2, alpha=0.5,
               label=f"Test baseline ({baseline_te:.3f})")
    ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(f"Precision-Recall Curve\n{title} - Training vs Test", fontweight="bold")
    ax.legend(fontsize=9)
    fig.suptitle(subtitle, fontsize=9, y=0.01)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {path}")


def save_roc_fig(y_tr, y_te, p_tr, p_te, title, path):
    baseline_tr = y_tr.mean()
    baseline_te = y_te.mean()
    subtitle    = (f"Train n={len(y_tr):,} ({baseline_tr:.1%} closure) | "
                   f"Test n={len(y_te):,} ({baseline_te:.1%} closure)")
    fpr_tr, tpr_tr, _ = roc_curve(y_tr, p_tr)
    fpr_te, tpr_te, _ = roc_curve(y_te, p_te)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_tr, tpr_tr, color="#2E86AB", lw=2,
            label=f"Train  AUC-ROC = {roc_auc_score(y_tr, p_tr):.3f}")
    ax.plot(fpr_te, tpr_te, color="#E84855", lw=2, ls="--",
            label=f"Test   AUC-ROC = {roc_auc_score(y_te, p_te):.3f}")
    ax.plot([0,1],[0,1], color="gray", ls="--", lw=1.2, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(f"ROC Curve\n{title} - Training vs Test", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    fig.suptitle(subtitle, fontsize=9, y=0.01)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STEP 05 -- Final Model: LR Benchmark + XGBoost")
    print("=" * 60)

    # 1. Load
    print("\n[1] Loading all metro features...")
    all_df    = load_all()
    feat_cols = [c for c in all_df.columns if c not in META_COLS]
    print(f"  {len(all_df):,} restaurants  {int(all_df[TARGET_COL].sum())} closed "
          f"({all_df[TARGET_COL].mean():.1%})  {len(feat_cols)} features")

    # 2. Split
    print("\n[2] Time-based 80/20 train/test split...")
    train_df, test_df = time_split(all_df)
    medians   = train_df[feat_cols].median()
    X_train   = train_df[feat_cols].fillna(medians).values
    y_train   = train_df[TARGET_COL].values
    X_test    = test_df[feat_cols].fillna(medians).values
    y_test    = test_df[TARGET_COL].values
    print(f"  Train: {len(train_df):,}  ({train_df['anchor_date'].min().date()} "
          f"to {train_df['anchor_date'].max().date()})  closure={y_train.mean():.1%}")
    print(f"  Test:  {len(test_df):,}  ({test_df['anchor_date'].min().date()} "
          f"to {test_df['anchor_date'].max().date()})  closure={y_test.mean():.1%}")

    # 3. 5-fold CV
    print(f"\n[3] 5-fold CV tuning ({len(XGB_GRID)} XGB combos x {N_FOLDS} folds)...")
    best_xgb_params, best_lr_c, fold_xgb, fold_lr = kfold_tune(X_train, y_train)
    print(f"\n  Best XGB: n={best_xgb_params['n_estimators']} "
          f"d={best_xgb_params['max_depth']} lr={best_xgb_params['learning_rate']} "
          f"mcw={best_xgb_params['min_child_weight']}")
    print(f"  Best LR C={best_lr_c}")
    print(f"  XGB fold AUC-PR: {fold_xgb}  mean={np.mean(fold_xgb):.4f}")
    print(f"  LR  fold AUC-PR: {fold_lr}   mean={np.mean(fold_lr):.4f}")

    # 4. Retrain on full training set
    print("\n[4] Retraining on full training set...")
    xgb_model = XGBClassifier(**best_xgb_params)
    xgb_model.fit(X_train, y_train, verbose=False)

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    lr_model   = LogisticRegression(C=best_lr_c, class_weight="balanced",
                                    solver="lbfgs", max_iter=1000,
                                    random_state=RANDOM_SEED)
    lr_model.fit(X_train_sc, y_train)

    # 5. Evaluate
    print("\n[5] Metrics across CV / Train / Test")
    prob_xgb_train = xgb_model.predict_proba(X_train)[:, 1]
    prob_xgb_test  = xgb_model.predict_proba(X_test)[:, 1]
    prob_lr_train  = lr_model.predict_proba(X_train_sc)[:, 1]
    prob_lr_test   = lr_model.predict_proba(X_test_sc)[:, 1]

    xgb_cv    = {"AUC_PR": round(float(np.mean(fold_xgb)),4),
                 "std":    round(float(np.std(fold_xgb)),4)}
    lr_cv     = {"AUC_PR": round(float(np.mean(fold_lr)),4),
                 "std":    round(float(np.std(fold_lr)),4)}
    xgb_train = metrics(y_train, prob_xgb_train)
    xgb_test  = metrics(y_test,  prob_xgb_test)
    lr_train  = metrics(y_train, prob_lr_train)
    lr_test   = metrics(y_test,  prob_lr_test)

    print(f"\n  {'Model':22s} {'Sample':8s} {'AUC-PR':>8} {'AUC-ROC':>9} {'F1':>7}")
    print("  " + "-" * 58)
    print(f"  {'Logistic Regression':22s} {'CV':8s} {lr_cv['AUC_PR']:8.4f} +/-{lr_cv['std']:.4f}")
    print(f"  {'':22s} {'Train':8s} {lr_train['AUC_PR']:8.4f} {lr_train['AUC_ROC']:9.4f} {lr_train['F1']:7.4f}")
    print(f"  {'':22s} {'Test':8s} {lr_test['AUC_PR']:8.4f} {lr_test['AUC_ROC']:9.4f} {lr_test['F1']:7.4f}")
    print(f"  {'XGBoost':22s} {'CV':8s} {xgb_cv['AUC_PR']:8.4f} +/-{xgb_cv['std']:.4f}")
    print(f"  {'':22s} {'Train':8s} {xgb_train['AUC_PR']:8.4f} {xgb_train['AUC_ROC']:9.4f} {xgb_train['F1']:7.4f}")
    print(f"  {'':22s} {'Test':8s} {xgb_test['AUC_PR']:8.4f} {xgb_test['AUC_ROC']:9.4f} {xgb_test['F1']:7.4f}")

    # 6. Save models
    print("\n[6] Saving models...")
    joblib.dump(xgb_model, Path(MODEL_DIR) / "xgboost_final.pkl")
    joblib.dump(lr_model,  Path(MODEL_DIR) / "logistic_regression_final.pkl")
    joblib.dump(scaler,    Path(MODEL_DIR) / "lr_scaler_final.pkl")
    print(f"  Saved -> models/xgboost_final.pkl")
    print(f"  Saved -> models/logistic_regression_final.pkl")

    # 7. Save results JSON
    results = {
        "split": {"train_n": len(train_df), "test_n": len(test_df),
                  "train_closure_rate": round(float(y_train.mean()),4),
                  "test_closure_rate":  round(float(y_test.mean()),4),
                  "train_date_range": [str(train_df['anchor_date'].min().date()),
                                       str(train_df['anchor_date'].max().date())],
                  "test_date_range":  [str(test_df['anchor_date'].min().date()),
                                       str(test_df['anchor_date'].max().date())]},
        "cv_folds": N_FOLDS,
        "logistic_regression": {"cv": lr_cv, "train": lr_train, "test": lr_test,
                                 "best_C": best_lr_c},
        "xgboost": {"cv": xgb_cv, "train": xgb_train, "test": xgb_test,
                    "best_params": {k: v for k, v in best_xgb_params.items()
                                    if k not in ("eval_metric","random_state","verbosity")}},
    }
    out_json = Path(MODEL_DIR) / "final_results.json"
    import json
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Saved -> {out_json}")

    # 8. Figures
    print("\n[7] Generating figures...")
    save_pr_fig(y_train, y_test, prob_xgb_train, prob_xgb_test,
                "XGBoost", Path(FIG_DIR) / "35_train_pr_curve.png")
    save_roc_fig(y_train, y_test, prob_xgb_train, prob_xgb_test,
                 "XGBoost", Path(FIG_DIR) / "35b_train_roc_curve.png")
    save_pr_fig(y_train, y_test, prob_lr_train, prob_lr_test,
                "Logistic Regression", Path(FIG_DIR) / "36_train_pr_curve_lr.png")
    save_roc_fig(y_train, y_test, prob_lr_train, prob_lr_test,
                 "Logistic Regression", Path(FIG_DIR) / "36b_train_roc_curve_lr.png")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```bash
python 05_modeling.py
```

Expected output ends with a metrics table like:
```
  Model                  Sample   AUC-PR   AUC-ROC      F1
  ----------------------------------------------------------
  Logistic Regression    CV       0.353x +/-...
  ...
  XGBoost                Test     0.328x    0.833x  0.382x
```
And `models/xgboost_final.pkl`, `models/logistic_regression_final.pkl`, four figure files should exist.

- [ ] **Step 3: Verify output files exist**

```bash
python -c "
from pathlib import Path
files = [
    'models/xgboost_final.pkl',
    'models/logistic_regression_final.pkl',
    'models/final_results.json',
    'figures/35_train_pr_curve.png',
    'figures/35b_train_roc_curve.png',
    'figures/36_train_pr_curve_lr.png',
    'figures/36b_train_roc_curve_lr.png',
]
for f in files:
    status = 'OK' if Path(f).exists() else 'MISSING'
    print(f'{status}  {f}')
"
```
Expected: all `OK`

- [ ] **Step 4: Commit**

```bash
git add 05_modeling.py models/xgboost_final.pkl models/logistic_regression_final.pkl models/lr_scaler_final.pkl models/final_results.json figures/35_train_pr_curve.png figures/35b_train_roc_curve.png figures/36_train_pr_curve_lr.png figures/36b_train_roc_curve_lr.png
git commit -m "feat: add 05_modeling.py -- LR + XGBoost with 5-fold CV and 80/20 split"
```

---

## Task 6: Update `app.py` to load `xgboost_final.pkl`

**Files:**
- Modify: `app.py` (lines ~398-400)

- [ ] **Step 1: Find and update the model loading block in `app.py`**

Current code (around line 398):
```python
        "models/xgboost_global_calibrated.pkl",
        "models/xgboost_global.pkl",
        "models/xgboost.pkl",
```

Replace with:
```python
        "models/xgboost_final.pkl",
        "models/xgboost_global_calibrated.pkl",
        "models/xgboost_global.pkl",
```

This puts `xgboost_final.pkl` first in the fallback list so the app uses the clean final model, while staying backwards compatible if the file is missing.

- [ ] **Step 2: Verify the app starts without error**

```bash
python -c "import app" 2>&1 | head -5
```
Expected: no import errors (Streamlit will warn about missing browser, that's fine).

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "fix: update app.py to load xgboost_final.pkl as primary model"
```

---

## Task 7: Update `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Read current `.gitignore`**

Check if `data/raw/` and `data/raw_photos/` are already excluded.

- [ ] **Step 2: Ensure these lines are present in `.gitignore`**

```
# Raw Yelp data (large, download from https://www.yelp.com/dataset)
data/raw/
data/raw_photos/

# Python
__pycache__/
*.pyc
*.pyo
.env

# IDE
.vscode/
.idea/

# Temp
*.tmp
plot_train_curves.py
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: update .gitignore to exclude raw data and pycache"
```

---

## Task 8: Write `README.md`

**Files:**
- Create/overwrite: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# ClosureWatch — Restaurant Closure Prediction

Binary classification: given 12 months of Yelp behavioral signals for a restaurant,
predict whether it will permanently close in the next 6 months.
Framed as an SME credit underwriting problem (alternative lenders use public behavioral
signals the same way banks use transaction history).

## Dataset

**Source:** [Yelp Academic Dataset](https://www.yelp.com/dataset)
**Geography:** 9 US metros — Philadelphia, Tampa, Indianapolis, Tucson, Nashville,
New Orleans, Saint Louis, Reno, Boise
**Size:** ~18,500 restaurants, ~10% closure rate (imbalanced)

Raw data is not included in this repo (download from the link above and place in `data/raw/`).
Processed feature files are committed — you can run `05_modeling.py` directly.

## How to Run

```bash
pip install -r requirements.txt

# Data pipeline (only needed if starting from raw JSON):
python 01_load_filter.py        # load & filter all metros from Yelp JSON
python 02_build_labels.py       # build binary closure labels
python 03_feature_engineering.py # engineer 81 time-windowed features

# Analysis:
python 04_eda.py                # EDA figures -> figures/
python 04b_data_quality.py      # data quality diagnostics

# Modeling (processed data already committed, can run directly):
python 05_modeling.py           # LR benchmark + XGBoost, 5-fold CV, figures, saved models

# App:
streamlit run app.py            # launch ClosureWatch dashboard
```

## Results

| Model | CV AUC-PR | Test AUC-PR | Test AUC-ROC |
|---|---|---|---|
| Logistic Regression (benchmark) | 0.353 ± 0.02 | 0.245 | 0.808 |
| **XGBoost (5-fold CV tuned)** | **0.399 ± 0.03** | **0.328** | **0.833** |

Primary metric is AUC-PR (correct for imbalanced binary classification).
Train/test split is time-based (no random splits) to prevent temporal leakage.

## Features (81 total)

Time-windowed signals computed strictly before the anchor date:
- Review velocity, drought flags, momentum
- Rating trend, VADER sentiment trend
- Checkin signals, tip signals
- Reviewer quality metrics
- Business metadata (price range, category, hours)
- Null-flag indicators for informatively-missing features
- Sentence-embedding PCA features (MiniLM-L6-v2)

## Experiments

See `experiments/` for exploratory work: LOMO cross-validation, stacking ensemble,
Philadelphia zero-shot transfer, and the original Tampa-only pipeline.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with dataset source, pipeline instructions, and results"
```

---

## Task 9: Final verification

- [ ] **Step 1: Check root directory is clean**

```bash
ls *.py
```
Expected: only `01_load_filter.py`, `02_build_labels.py`, `03_feature_engineering.py`, `04_eda.py`, `04b_data_quality.py`, `05_modeling.py`, `app.py`, `app_helpers.py`, `photo_index.py`, `config_00.py`

- [ ] **Step 2: Verify models exist**

```bash
python -c "
import joblib
m = joblib.load('models/xgboost_final.pkl')
print('xgboost_final.pkl OK -- type:', type(m).__name__)
m2 = joblib.load('models/logistic_regression_final.pkl')
print('logistic_regression_final.pkl OK -- type:', type(m2).__name__)
"
```

- [ ] **Step 3: Verify all 4 key figures exist**

```bash
python -c "
from pathlib import Path
figs = ['35_train_pr_curve.png','35b_train_roc_curve.png',
        '36_train_pr_curve_lr.png','36b_train_roc_curve_lr.png']
for f in figs:
    p = Path('figures') / f
    print('OK  ' if p.exists() else 'MISSING  ', f)
"
```

- [ ] **Step 4: Final commit**

```bash
git status
git add -A
git commit -m "chore: final cleanup -- clean root pipeline, experiments archived, README complete"
```
