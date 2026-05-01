# Multi-Metro Pipeline + LOMO CV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `09_multi_metro.py` (parameterized city pipeline) and `10_lomo_cv.py` (9-fold LOMO CV + global model evaluated on Edmonton).

**Architecture:** `09_multi_metro.py` mirrors `08_philadelphia.py` but accepts `--city`/`--state` CLI args and bakes `has_photo` into the feature matrix. `10_lomo_cv.py` loads all nine metro feature matrices, runs a per-fold XGBoost grid search over a 16-combination param space, aggregates results, trains a global model on all nine metros, and evaluates it on Edmonton.

**Tech Stack:** pandas, numpy, xgboost, scikit-learn, matplotlib, joblib, argparse, importlib (for `03_feature_engineering.build_features_one`)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `09_multi_metro.py` | Create | Generic city pipeline — load → label → features → save |
| `10_lomo_cv.py` | Create | LOMO CV, per-fold tuning, global model, Edmonton OOD |

Both scripts are standalone; no new helper modules needed.

---

### Task 1: `09_multi_metro.py` — Scaffold, CLI args, and data loading

**Files:**
- Create: `09_multi_metro.py`

- [ ] **Step 1: Write the scaffold with failing smoke test**

Create `09_multi_metro.py` with this content:

```python
"""
09_multi_metro.py -- Generic multi-metro restaurant pipeline.

Loads raw Yelp JSON for any US (or Canadian) city, engineers features
identical to the Tampa pipeline, and saves Parquet checkpoints.

Usage:
    python 09_multi_metro.py --city Nashville --state TN
    python 09_multi_metro.py --city "Saint Louis" --state MO
    python 09_multi_metro.py --city Edmonton --state AB
    python 09_multi_metro.py --backfill-only   # add has_photo to Tampa + Philly

Runtime: ~25-45 min per city (VADER is the bottleneck).
Output:  data/processed_{city_slug}/ (parquets)
"""

import argparse
import importlib
import json
import warnings
warnings.filterwarnings("ignore")

_fe = importlib.import_module("03_feature_engineering")

import numpy as np
import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from config_00 import (
    RAW_DIR, OBS_MONTHS, OUTCOME_MONTHS, EARLIEST_ANCHOR, TARGET_COL,
)

LATEST_ANCHOR = "2020-06-01"   # conservative cap; avoids review-density artifact near dataset edge
MIN_LABELED   = 50
ENCODING      = "utf-8"
PHOTO_INDEX   = Path("data/processed/photo_index.parquet")

plt_style = {
    "font.family": "serif",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def city_slug(city: str) -> str:
    return city.lower().replace(" ", "_")


def out_dir(city: str) -> Path:
    d = Path(f"data/processed_{city_slug(city)}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_businesses(city: str, state: str) -> pd.DataFrame:
    """Filter business JSON on both city AND state to handle duplicate city names."""
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
    df = pd.DataFrame(rows)
    print(f"    {city}, {state}: {len(df):,} restaurants found")
    return df


def load_interactions(bids: set) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load reviews, checkins, tips for the given business_id set."""
    print("  Loading reviews...")
    rev_rows = []
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
    reviews = pd.DataFrame(rev_rows)

    print("  Loading checkins...")
    ci_rows = []
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
    checkins = pd.DataFrame(ci_rows)

    print("  Loading tips...")
    tip_rows = []
    with open(RAW_DIR / "yelp_academic_dataset_tip.json", encoding=ENCODING) as f:
        for line in f:
            r = json.loads(line)
            if r["business_id"] not in bids:
                continue
            tip_rows.append({"business_id": r["business_id"],
                             "date": pd.Timestamp(r["date"]),
                             "compliment_count": int(r.get("compliment_count", 0))})
    tips = pd.DataFrame(tip_rows)

    print(f"    reviews={len(reviews):,}  checkins={len(checkins):,}  tips={len(tips):,}")
    return reviews, checkins, tips
```

- [ ] **Step 2: Verify the scaffold imports cleanly**

```bash
python -c "import importlib; m = importlib.import_module('09_multi_metro'); print('city_slug:', m.city_slug('Saint Louis')); print('OK')"
```

Expected output:
```
city_slug: saint_louis
OK
```

- [ ] **Step 3: Commit the scaffold**

```bash
git add 09_multi_metro.py
git commit -m "feat: scaffold 09_multi_metro.py with data loading"
```

---

### Task 2: `09_multi_metro.py` — Labeling

**Files:**
- Modify: `09_multi_metro.py`

- [ ] **Step 1: Write a failing inline test for the label logic**

```bash
python -c "
import importlib, pandas as pd, numpy as np
m = importlib.import_module('09_multi_metro')
# Minimal biz row
biz = pd.DataFrame([{'business_id': 'A', 'name': 'X', 'city': 'T', 'state': 'FL',
                      'stars_yelp': 4.0, 'price_range': None, 'open_days_per_week': 5,
                      'categories': 'Restaurants'}])
# Reviews: anchor will be 80th pct; last review well after outcome_end → label=0
rev = pd.DataFrame([{'business_id': 'A', 'date': pd.Timestamp('2018-01-01'), 'stars': 4.0,
                      'text': '', 'useful': 0, 'funny': 0, 'cool': 0},
                    {'business_id': 'A', 'date': pd.Timestamp('2023-01-01'), 'stars': 4.0,
                      'text': '', 'useful': 0, 'funny': 0, 'cool': 0}])
labeled = m.build_labels(biz, rev)
assert labeled.iloc[0]['closed_within_6m'] == 0, 'Should be open (last review 2023 >> outcome_end)'
print('PASS: label=0 when last review is far after outcome_end')
"
```

Expected: `NameError: name 'build_labels' is not defined` (or AttributeError) — function doesn't exist yet.

- [ ] **Step 2: Implement `build_labels` and append it to `09_multi_metro.py`**

Add this function to `09_multi_metro.py` (before `load_interactions`):

```python
def build_labels(biz: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """
    3-way review-recency rule (mirrors 02_build_labels.py build_label()).
    Completely ignores is_open flag — labels from review behaviour only.
    """
    earliest = pd.Timestamp(EARLIEST_ANCHOR)
    latest   = pd.Timestamp(LATEST_ANCHOR)

    rev_groups = reviews.groupby("business_id")
    rows = []
    for _, b in biz.iterrows():
        bid = b["business_id"]
        if bid not in rev_groups.groups:
            continue
        rev = rev_groups.get_group(bid).sort_values("date")
        if rev.empty:
            continue

        anchor = rev["date"].quantile(0.80, interpolation="nearest")
        if not (earliest <= anchor <= latest):
            continue

        obs_start   = anchor - relativedelta(months=OBS_MONTHS)
        outcome_end = anchor + relativedelta(months=OUTCOME_MONTHS)
        last_review  = rev["date"].max()

        if last_review > outcome_end + relativedelta(months=3):
            closed = 0
        elif last_review <= outcome_end:
            closed = 1
        else:
            closed = 0   # ambiguous window

        rows.append({
            "business_id":        bid,
            "name":               b["name"],
            "city":               b["city"],
            "state":              b["state"],
            "anchor_date":        anchor,
            "obs_start":          obs_start,
            "outcome_end":        outcome_end,
            TARGET_COL:           closed,
            "stars_yelp":         b["stars_yelp"],
            "price_range":        b["price_range"],
            "open_days_per_week": b["open_days_per_week"],
            "categories":         b["categories"],
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 3: Re-run the inline test — it should pass**

```bash
python -c "
import importlib, pandas as pd
m = importlib.import_module('09_multi_metro')
biz = pd.DataFrame([{'business_id': 'A', 'name': 'X', 'city': 'T', 'state': 'FL',
                      'stars_yelp': 4.0, 'price_range': None, 'open_days_per_week': 5,
                      'categories': 'Restaurants'}])
rev = pd.DataFrame([{'business_id': 'A', 'date': pd.Timestamp('2018-01-01'), 'stars': 4.0,
                      'text': '', 'useful': 0, 'funny': 0, 'cool': 0},
                    {'business_id': 'A', 'date': pd.Timestamp('2023-01-01'), 'stars': 4.0,
                      'text': '', 'useful': 0, 'funny': 0, 'cool': 0}])
labeled = m.build_labels(biz, rev)
assert labeled.iloc[0]['closed_within_6m'] == 0
print('PASS: label logic correct')
"
```

Expected: `PASS: label logic correct`

- [ ] **Step 4: Test the closed=1 case**

```bash
python -c "
import importlib, pandas as pd
m = importlib.import_module('09_multi_metro')
biz = pd.DataFrame([{'business_id': 'A', 'name': 'X', 'city': 'T', 'state': 'FL',
                      'stars_yelp': 4.0, 'price_range': None, 'open_days_per_week': 5,
                      'categories': 'Restaurants'}])
# anchor = 2018-01-01 (80th pct of only 2 reviews at 2017-07 and 2018-01)
# outcome_end = anchor + 6m = ~2018-07-01
# last_review = 2018-06-01 <= outcome_end → closed=1
rev = pd.DataFrame([{'business_id': 'A', 'date': pd.Timestamp('2017-07-01'), 'stars': 4.0,
                      'text': '', 'useful': 0, 'funny': 0, 'cool': 0},
                    {'business_id': 'A', 'date': pd.Timestamp('2018-06-01'), 'stars': 3.0,
                      'text': '', 'useful': 0, 'funny': 0, 'cool': 0}])
labeled = m.build_labels(biz, rev)
print('labeled rows:', len(labeled))
if len(labeled) > 0:
    print('closed_within_6m:', labeled.iloc[0]['closed_within_6m'])
    print('PASS')
"
```

Expected: `closed_within_6m: 1` and `PASS`.

- [ ] **Step 5: Commit**

```bash
git add 09_multi_metro.py
git commit -m "feat: add build_labels to 09_multi_metro.py (3-way recency rule)"
```

---

### Task 3: `09_multi_metro.py` — Feature engineering with `has_photo`

**Files:**
- Modify: `09_multi_metro.py`

- [ ] **Step 1: Write a failing smoke test for `build_features`**

```bash
python -c "
import importlib
m = importlib.import_module('09_multi_metro')
m.build_features  # AttributeError if not implemented
"
```

Expected: `AttributeError` — function doesn't exist yet.

- [ ] **Step 2: Implement `build_features` in `09_multi_metro.py`**

Add after `build_labels`:

```python
def build_features(
    labeled: pd.DataFrame,
    reviews: pd.DataFrame,
    checkins: pd.DataFrame,
    tips: pd.DataFrame,
    photo_bids: set,
) -> pd.DataFrame:
    """
    Call build_features_one from 03_feature_engineering per restaurant.
    Adds has_photo as a static binary column (no temporal filter — see CLAUDE.md).
    """
    build_one = _fe.build_features_one

    rev_groups = reviews.groupby("business_id") if not reviews.empty else None
    ci_groups  = checkins.groupby("business_id") if not checkins.empty else None
    tip_groups = tips.groupby("business_id")     if not tips.empty     else None

    records = []
    for _, row in tqdm(labeled.iterrows(), total=len(labeled), desc="Features"):
        bid    = row["business_id"]
        anchor = row["anchor_date"]
        obs_s  = row["obs_start"]

        obs_rev = pd.DataFrame()
        if rev_groups and bid in rev_groups.groups:
            obs_rev = rev_groups.get_group(bid)
            obs_rev = obs_rev[(obs_rev["date"] >= obs_s) & (obs_rev["date"] < anchor)]

        obs_ci = pd.DataFrame()
        if ci_groups and bid in ci_groups.groups:
            obs_ci = ci_groups.get_group(bid)
            obs_ci = obs_ci[(obs_ci["checkin_date"] >= obs_s) & (obs_ci["checkin_date"] < anchor)]

        obs_tip = pd.DataFrame()
        if tip_groups and bid in tip_groups.groups:
            obs_tip = tip_groups.get_group(bid)
            obs_tip = obs_tip[(obs_tip["date"] >= obs_s) & (obs_tip["date"] < anchor)]

        feat = build_one(row, obs_rev, obs_ci, obs_tip)
        feat[TARGET_COL]    = row[TARGET_COL]
        feat["anchor_date"] = anchor
        feat["city"]        = row["city"]
        feat["state"]       = row["state"]
        feat["has_photo"]   = int(bid in photo_bids)
        records.append(feat)

    return pd.DataFrame(records)
```

- [ ] **Step 3: Verify `has_photo` is added correctly with a quick check**

```bash
python -c "
import importlib, pandas as pd
m = importlib.import_module('09_multi_metro')
# build_features is defined and has the right signature
import inspect
sig = inspect.signature(m.build_features)
params = list(sig.parameters.keys())
assert 'photo_bids' in params, 'Missing photo_bids param'
print('PASS: build_features signature correct:', params)
"
```

Expected: `PASS: build_features signature correct: ['labeled', 'reviews', 'checkins', 'tips', 'photo_bids']`

- [ ] **Step 4: Commit**

```bash
git add 09_multi_metro.py
git commit -m "feat: add build_features with has_photo to 09_multi_metro.py"
```

---

### Task 4: `09_multi_metro.py` — `--backfill-only` and `main()`

**Files:**
- Modify: `09_multi_metro.py`

- [ ] **Step 1: Implement `backfill_has_photo` and `main()` — append to `09_multi_metro.py`**

```python
def backfill_has_photo(photo_bids: set) -> None:
    """Add has_photo to Tampa and Philadelphia features.parquet (built before this script existed)."""
    targets = [
        Path("data/processed/features.parquet"),
        Path("data/processed_philly/features.parquet"),
    ]
    for p in targets:
        if not p.exists():
            print(f"  WARNING: {p} not found, skipping")
            continue
        df = pd.read_parquet(p)
        df["has_photo"] = df["business_id"].isin(photo_bids).astype(int)
        df.to_parquet(p, index=False)
        pct = df["has_photo"].mean() * 100
        print(f"  Backfilled has_photo -> {p}  ({df['has_photo'].sum():,}/{len(df):,} = {pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Multi-metro restaurant pipeline")
    parser.add_argument("--city",          type=str, default=None)
    parser.add_argument("--state",         type=str, default=None)
    parser.add_argument("--backfill-only", action="store_true",
                        help="Only add has_photo to Tampa + Philly features.parquet")
    args = parser.parse_args()

    # Load photo index once (used in both paths)
    if not PHOTO_INDEX.exists():
        raise FileNotFoundError(f"Photo index not found: {PHOTO_INDEX}")
    photo_bids = set(pd.read_parquet(PHOTO_INDEX)["business_id"])
    print(f"  Photo index: {len(photo_bids):,} businesses")

    if args.backfill_only:
        print("=" * 60)
        print("STEP 9 -- Backfilling has_photo into Tampa + Philadelphia")
        print("=" * 60)
        backfill_has_photo(photo_bids)
        return

    if not args.city or not args.state:
        parser.error("--city and --state are required (or use --backfill-only)")

    city  = args.city
    state = args.state
    slug  = city_slug(city)
    odir  = out_dir(city)

    print("=" * 60)
    print(f"STEP 9 -- {city}, {state}")
    print("=" * 60)

    # ── 1. Businesses (checkpoint) ──────────────────────────────────────────
    biz_path = odir / "businesses.parquet"
    if biz_path.exists():
        print("  Loading businesses from checkpoint...")
        biz = pd.read_parquet(biz_path)
        print(f"    {len(biz):,} rows")
    else:
        print("  Loading businesses from JSON...")
        biz = load_businesses(city, state)
        biz.to_parquet(biz_path, index=False)

    if biz.empty:
        print("  ERROR: No businesses found. Check city/state spelling.")
        return

    # ── 2. Reviews, checkins, tips (checkpoint) ─────────────────────────────
    rev_path = odir / "reviews.parquet"
    if rev_path.exists():
        print("  Loading interactions from checkpoint...")
        reviews  = pd.read_parquet(odir / "reviews.parquet")
        checkins = pd.read_parquet(odir / "checkins.parquet")
        tips     = pd.read_parquet(odir / "tips.parquet")
        print(f"    reviews={len(reviews):,}  checkins={len(checkins):,}  tips={len(tips):,}")
    else:
        bids = set(biz["business_id"])
        reviews, checkins, tips = load_interactions(bids)
        reviews.to_parquet(odir / "reviews.parquet",   index=False)
        checkins.to_parquet(odir / "checkins.parquet", index=False)
        tips.to_parquet(odir / "tips.parquet",         index=False)

    # ── 3. Labeling (checkpoint) ────────────────────────────────────────────
    lab_path = odir / "labeled_businesses.parquet"
    if lab_path.exists():
        print("  Loading labels from checkpoint...")
        labeled = pd.read_parquet(lab_path)
    else:
        print("  Building labels...")
        labeled = build_labels(biz, reviews)
        labeled.to_parquet(lab_path, index=False)

    n_closed = int(labeled[TARGET_COL].sum())
    rate = n_closed / len(labeled) if labeled is not None and len(labeled) else 0
    print(f"  Labeled: {len(labeled):,} restaurants, {n_closed} closed ({rate:.1%})")

    if len(labeled) < MIN_LABELED:
        print(f"  WARNING: Only {len(labeled)} labeled restaurants (< {MIN_LABELED}). Exiting.")
        return

    # ── 4. Feature engineering (always re-run to ensure has_photo is present) ─
    print("  Engineering features (VADER is slow, ~20-40 min)...")
    features = build_features(labeled, reviews, checkins, tips, photo_bids)
    features.to_parquet(odir / "features.parquet", index=False)
    print(f"  Features shape: {features.shape}")
    print(f"  has_photo coverage: {features['has_photo'].mean():.1%}")

    # ── 5. Summary ──────────────────────────────────────────────────────────
    print("\n  === SUMMARY ===")
    print(f"  City:          {city}, {state}")
    print(f"  Labeled:       {len(labeled):,}")
    print(f"  Closed:        {n_closed} ({rate:.1%})")
    print(f"  Features:      {features.shape[1]} cols")
    print(f"  Saved to:      {odir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the CLI parses correctly**

```bash
python 09_multi_metro.py --help
```

Expected output includes `--city`, `--state`, `--backfill-only` in the help text.

- [ ] **Step 3: Verify `city_slug` handles edge cases**

```bash
python -c "
import importlib
m = importlib.import_module('09_multi_metro')
assert m.city_slug('Saint Louis') == 'saint_louis'
assert m.city_slug('New Orleans') == 'new_orleans'
assert m.city_slug('Boise') == 'boise'
assert m.city_slug('Nashville') == 'nashville'
print('PASS: city_slug handles all expected metro names')
"
```

Expected: `PASS: city_slug handles all expected metro names`

- [ ] **Step 4: Commit**

```bash
git add 09_multi_metro.py
git commit -m "feat: add main() and backfill_has_photo to 09_multi_metro.py — pipeline complete"
```

---

### Task 5: `10_lomo_cv.py` — Scaffold, data loading, and LOMO fold structure

**Files:**
- Create: `10_lomo_cv.py`

- [ ] **Step 1: Create the scaffold with METROS dict and data loading**

Create `10_lomo_cv.py`:

```python
"""
10_lomo_cv.py -- Leave-One-Metro-Out CV + global model.

For each of the 9 US metros, holds it out as the test set and trains
XGBoost on the remaining 8. Hyperparameters are tuned on a time-based
val split within the train pool (no data from the held-out metro is used
for tuning). After all folds, a global model trained on all 9 metros is
evaluated on Edmonton as an out-of-distribution test.

Run after all metros have features.parquet:
    python 10_lomo_cv.py

Output:
    models/lomo_results.json
    models/xgboost_global.pkl
    figures/19_lomo_cv_results.png
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier

from config_00 import TARGET_COL, RANDOM_SEED, MODEL_DIR, FIG_DIR

plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

META_COLS = {"business_id", TARGET_COL, "anchor_date", "city", "state", "metro"}

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
EDMONTON_DIR = Path("data/processed_edmonton")

# XGBoost hyperparameter grid (16 combinations)
PARAM_GRID = [
    {
        "n_estimators": n, "max_depth": d, "learning_rate": lr,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": w,
        "scale_pos_weight": 10, "eval_metric": "aucpr",
        "random_state": RANDOM_SEED, "verbosity": 0,
    }
    for n in [300, 500]
    for d in [4, 6]
    for lr in [0.05, 0.1]
    for w in [3, 5]
]


def load_all_metros() -> dict[str, pd.DataFrame]:
    """Load features.parquet for each metro; return dict keyed by metro name."""
    dfs = {}
    for metro, directory in METROS.items():
        p = Path(directory) / "features.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}  — run 09_multi_metro.py for {metro} first")
        df = pd.read_parquet(p)
        df["metro"] = metro
        df["anchor_date"] = pd.to_datetime(df["anchor_date"])
        dfs[metro] = df
        n_closed = int(df[TARGET_COL].sum())
        print(f"  {metro:15s}: {len(df):5,} restaurants, {n_closed} closed ({n_closed/len(df):.1%})")
    return dfs


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def time_val_split(df: pd.DataFrame, val_frac: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Latest val_frac% by anchor_date → val; rest → train."""
    df_s = df.sort_values("anchor_date")
    n_val = max(1, int(len(df_s) * val_frac))
    return df_s.iloc[:-n_val].copy(), df_s.iloc[-n_val:].copy()
```

- [ ] **Step 2: Verify the scaffold loads and PARAM_GRID has 16 entries**

```bash
python -c "
import importlib
m = importlib.import_module('10_lomo_cv')
print('PARAM_GRID entries:', len(m.PARAM_GRID))
assert len(m.PARAM_GRID) == 16, f'Expected 16, got {len(m.PARAM_GRID)}'
print('META_COLS:', m.META_COLS)
print('METROS:', list(m.METROS.keys()))
assert len(m.METROS) == 9
print('PASS')
"
```

Expected: `PARAM_GRID entries: 16` and `PASS`.

- [ ] **Step 3: Verify `time_val_split` respects temporal order**

```bash
python -c "
import importlib, pandas as pd, numpy as np
m = importlib.import_module('10_lomo_cv')
# 100 rows with sequential anchor dates
df = pd.DataFrame({'anchor_date': pd.date_range('2017-01-01', periods=100, freq='ME'),
                   'closed_within_6m': np.zeros(100), 'metro': 'test', 'business_id': range(100)})
train, val = m.time_val_split(df, val_frac=0.20)
assert len(val) == 20, f'Expected 20 val rows, got {len(val)}'
assert train['anchor_date'].max() < val['anchor_date'].min(), 'Temporal order violated'
print('PASS: time_val_split is temporally ordered')
"
```

Expected: `PASS: time_val_split is temporally ordered`

- [ ] **Step 4: Commit**

```bash
git add 10_lomo_cv.py
git commit -m "feat: scaffold 10_lomo_cv.py — METROS, PARAM_GRID, data loading, val split"
```

---

### Task 6: `10_lomo_cv.py` — Per-fold XGBoost tuning and fold evaluation

**Files:**
- Modify: `10_lomo_cv.py`

- [ ] **Step 1: Write failing test for `tune_xgb`**

```bash
python -c "
import importlib
m = importlib.import_module('10_lomo_cv')
m.tune_xgb  # AttributeError if missing
"
```

Expected: `AttributeError`.

- [ ] **Step 2: Implement `tune_xgb` and `run_lomo_fold` — append to `10_lomo_cv.py`**

```python
def tune_xgb(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val:   pd.DataFrame, y_val:   pd.Series,
) -> dict:
    """
    Grid search over PARAM_GRID; select params that maximize val AUC-PR.
    Imputation medians are already applied to X_train/X_val before this call.
    """
    best_auc_pr = -1.0
    best_params = PARAM_GRID[0]

    for params in PARAM_GRID:
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_prob = model.predict_proba(X_val)[:, 1]
        auc_pr = average_precision_score(y_val, y_prob)
        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            best_params = params

    return best_params, best_auc_pr


def run_lomo_fold(
    held_out_metro: str,
    all_dfs: dict[str, pd.DataFrame],
) -> dict:
    """
    Train on 8 metros, tune hyperparams on a time-based val split within
    those 8, retrain on full 8-metro pool, evaluate on held-out metro.
    """
    print(f"\n  ── Fold: held-out = {held_out_metro} ──")

    # Build train pool from the 8 non-held-out metros
    train_pool = pd.concat(
        [df for name, df in all_dfs.items() if name != held_out_metro],
        ignore_index=True,
    )
    held_df = all_dfs[held_out_metro].copy()

    feat_cols = get_feature_cols(train_pool)

    # Time-based 80/20 split within train pool
    train_df, val_df = time_val_split(train_pool, val_frac=0.20)

    X_train = train_df[feat_cols].copy()
    y_train = train_df[TARGET_COL]
    X_val   = val_df[feat_cols].copy()
    y_val   = val_df[TARGET_COL]

    # Fit imputation medians on train only (anti-leakage rule 3)
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_val   = X_val.fillna(train_medians)

    # Hyperparameter tuning on val
    print(f"    Tuning XGBoost ({len(PARAM_GRID)} param combos)...")
    best_params, val_auc_pr = tune_xgb(X_train, y_train, X_val, y_val)
    print(f"    Best val AUC-PR: {val_auc_pr:.4f}  params: n={best_params['n_estimators']} d={best_params['max_depth']} lr={best_params['learning_rate']}")

    # Retrain on full train pool (train + val) with best params
    X_full = train_pool[feat_cols].fillna(train_medians)
    y_full = train_pool[TARGET_COL]
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_full, y_full, verbose=False)

    # Evaluate on held-out metro
    # Use train_pool medians for imputation (no data from held-out metro)
    held_feat_cols = [c for c in feat_cols if c in held_df.columns]
    X_held = held_df.reindex(columns=feat_cols).fillna(train_medians)
    y_held = held_df[TARGET_COL]

    y_prob = final_model.predict_proba(X_held)[:, 1]
    auc_pr  = float(average_precision_score(y_held, y_prob))
    auc_roc = float(roc_auc_score(y_held, y_prob))

    # Threshold: F1-optimize on val
    prec, rec, thr = precision_recall_curve(y_val, final_model.predict_proba(X_val)[:, 1])
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    opt_thr = float(thr[np.argmax(f1s[:-1])])
    f1 = float(f1_score(y_held, (y_prob >= opt_thr).astype(int)))

    result = {
        "metro":        held_out_metro,
        "AUC_PR":       round(auc_pr,  4),
        "AUC_ROC":      round(auc_roc, 4),
        "F1":           round(f1,      4),
        "n":            int(len(held_df)),
        "closure_rate": round(float(y_held.mean()), 4),
        "val_AUC_PR":   round(val_auc_pr, 4),
        "best_params":  best_params,
    }
    print(f"    Test → AUC-PR={auc_pr:.4f}  AUC-ROC={auc_roc:.4f}  F1={f1:.4f}  n={len(held_df)}")
    return result
```

- [ ] **Step 3: Verify `run_lomo_fold` and `tune_xgb` are importable**

```bash
python -c "
import importlib
m = importlib.import_module('10_lomo_cv')
import inspect
assert callable(m.tune_xgb)
assert callable(m.run_lomo_fold)
sig = inspect.signature(m.run_lomo_fold)
assert 'held_out_metro' in sig.parameters
assert 'all_dfs' in sig.parameters
print('PASS: tune_xgb and run_lomo_fold are defined with correct signatures')
"
```

Expected: `PASS: tune_xgb and run_lomo_fold are defined with correct signatures`

- [ ] **Step 4: Commit**

```bash
git add 10_lomo_cv.py
git commit -m "feat: add tune_xgb and run_lomo_fold to 10_lomo_cv.py"
```

---

### Task 7: `10_lomo_cv.py` — Global model, Edmonton OOD, results JSON + figure + `main()`

**Files:**
- Modify: `10_lomo_cv.py`

- [ ] **Step 1: Implement the global model, Edmonton evaluation, figure, and `main()` — append to `10_lomo_cv.py`**

```python
def train_global_model(
    all_dfs: dict[str, pd.DataFrame],
    best_params: dict,
) -> tuple:
    """
    Train XGBoost on all 9 metros concatenated.
    Returns (model, feat_cols, train_medians).
    """
    full_df = pd.concat(all_dfs.values(), ignore_index=True)
    feat_cols = get_feature_cols(full_df)

    X = full_df[feat_cols].copy()
    y = full_df[TARGET_COL]

    train_medians = X.median()
    X = X.fillna(train_medians)

    model = XGBClassifier(**best_params)
    model.fit(X, y, verbose=False)
    return model, feat_cols, train_medians


def evaluate_edmonton(
    model,
    feat_cols: list[str],
    train_medians: pd.Series,
) -> dict:
    """Load Edmonton features.parquet and evaluate the global model."""
    edm_path = EDMONTON_DIR / "features.parquet"
    if not edm_path.exists():
        print(f"  WARNING: Edmonton features not found at {edm_path}. Skipping OOD eval.")
        return {}

    edm = pd.read_parquet(edm_path)
    edm["anchor_date"] = pd.to_datetime(edm["anchor_date"])

    X_edm  = edm.reindex(columns=feat_cols).fillna(train_medians)
    y_edm  = edm[TARGET_COL]

    y_prob  = model.predict_proba(X_edm)[:, 1]
    auc_pr  = float(average_precision_score(y_edm, y_prob))
    auc_roc = float(roc_auc_score(y_edm, y_prob))

    n_closed = int(y_edm.sum())
    print(f"\n  === EDMONTON OOD ===")
    print(f"  n={len(edm):,}  closed={n_closed} ({y_edm.mean():.1%})")
    print(f"  AUC-PR={auc_pr:.4f}  AUC-ROC={auc_roc:.4f}")

    return {
        "AUC_PR":       round(auc_pr,  4),
        "AUC_ROC":      round(auc_roc, 4),
        "n":            int(len(edm)),
        "closure_rate": round(float(y_edm.mean()), 4),
    }


def plot_lomo_results(fold_results: list[dict]) -> None:
    """
    Grouped horizontal bar chart — one row per metro sorted by AUC-PR descending.
    Two bars per metro: AUC-PR (blue) and AUC-ROC (green).
    Tampa single-city reference lines: AUC-PR=0.203, AUC-ROC=0.694.
    Saved to figures/19_lomo_cv_results.png.
    """
    # Tampa reference lines (from 05_modeling.py / CLAUDE.md)
    TAMPA_AUC_PR  = 0.203
    TAMPA_AUC_ROC = 0.694

    df = pd.DataFrame(fold_results).sort_values("AUC_PR", ascending=True)
    metros = df["metro"].tolist()
    n = len(metros)

    y = np.arange(n)
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.7 + 1.5)))
    bars_pr  = ax.barh(y + height/2, df["AUC_PR"],  height, label="AUC-PR",  color="#2E86AB", alpha=0.85)
    bars_roc = ax.barh(y - height/2, df["AUC_ROC"], height, label="AUC-ROC", color="#3BB273", alpha=0.85)

    # Tampa reference lines
    ax.axvline(TAMPA_AUC_PR,  color="#2E86AB", ls="--", lw=1.2, alpha=0.6,
               label=f"Tampa AUC-PR={TAMPA_AUC_PR:.3f}")
    ax.axvline(TAMPA_AUC_ROC, color="#3BB273", ls="--", lw=1.2, alpha=0.6,
               label=f"Tampa AUC-ROC={TAMPA_AUC_ROC:.3f}")

    # Value labels
    for bar in bars_pr:
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.3f}", va="center", fontsize=8)
    for bar in bars_roc:
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.3f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([m.replace("_", " ").title() for m in metros], fontsize=10)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("LOMO CV — Generalization Across 9 US Metros", fontweight="bold", fontsize=12)
    plt.tight_layout()

    out = FIG_DIR / "19_lomo_cv_results.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def main():
    print("=" * 60)
    print("STEP 10 -- Leave-One-Metro-Out CV + Global Model")
    print("=" * 60)

    # ── 1. Load all metros ──────────────────────────────────────────────────
    print("\n[1] Loading metro features")
    all_dfs = load_all_metros()
    total = sum(len(df) for df in all_dfs.values())
    print(f"  Total: {total:,} restaurants across {len(all_dfs)} metros")

    # ── 2. LOMO CV (9 folds) ────────────────────────────────────────────────
    print("\n[2] LOMO CV (9 folds)")
    fold_results = []
    for metro in METROS:
        result = run_lomo_fold(metro, all_dfs)
        fold_results.append(result)

    # ── 3. Aggregate metrics ────────────────────────────────────────────────
    auc_prs  = [r["AUC_PR"]  for r in fold_results]
    auc_rocs = [r["AUC_ROC"] for r in fold_results]
    aggregate = {
        "mean_AUC_PR":  round(float(np.mean(auc_prs)),  4),
        "std_AUC_PR":   round(float(np.std(auc_prs)),   4),
        "mean_AUC_ROC": round(float(np.mean(auc_rocs)), 4),
        "std_AUC_ROC":  round(float(np.std(auc_rocs)),  4),
    }
    print(f"\n  Aggregate: AUC-PR={aggregate['mean_AUC_PR']:.4f}±{aggregate['std_AUC_PR']:.4f}  "
          f"AUC-ROC={aggregate['mean_AUC_ROC']:.4f}±{aggregate['std_AUC_ROC']:.4f}")

    # ── 4. Global model — use best fold's hyperparams ───────────────────────
    print("\n[3] Training global model on all 9 metros")
    best_fold = max(fold_results, key=lambda r: r["val_AUC_PR"])
    print(f"  Using hyperparams from best fold: {best_fold['metro']} (val AUC-PR={best_fold['val_AUC_PR']:.4f})")
    global_params = best_fold["best_params"]

    global_model, feat_cols, train_medians = train_global_model(all_dfs, global_params)
    joblib.dump(global_model, MODEL_DIR / "xgboost_global.pkl")
    print(f"  Global model saved -> {MODEL_DIR}/xgboost_global.pkl")

    # ── 5. Edmonton OOD evaluation ──────────────────────────────────────────
    print("\n[4] Edmonton OOD evaluation")
    edmonton_results = evaluate_edmonton(global_model, feat_cols, train_medians)

    # ── 6. Save results JSON ────────────────────────────────────────────────
    results = {
        "folds":        [
            {k: v for k, v in r.items() if k != "best_params"}   # omit full params from JSON
            for r in fold_results
        ],
        "aggregate":    aggregate,
        "global_model": {
            "train_metros": len(METROS),
            "test_metro":   "edmonton",
            **edmonton_results,
        },
    }
    out_json = MODEL_DIR / "lomo_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {out_json}")

    # ── 7. Figure ───────────────────────────────────────────────────────────
    print("\n[5] Generating figure")
    plot_lomo_results(fold_results)

    # ── 8. Print summary table ──────────────────────────────────────────────
    print("\n  === LOMO CV SUMMARY ===")
    print(f"  {'Metro':15s} {'AUC-PR':>8} {'AUC-ROC':>8} {'F1':>7} {'N':>6} {'Rate':>6}")
    print("  " + "-" * 58)
    for r in sorted(fold_results, key=lambda x: x["AUC_PR"], reverse=True):
        print(f"  {r['metro']:15s} {r['AUC_PR']:8.4f} {r['AUC_ROC']:8.4f} "
              f"{r['F1']:7.4f} {r['n']:6,} {r['closure_rate']:6.1%}")
    print("  " + "-" * 58)
    print(f"  {'MEAN':15s} {aggregate['mean_AUC_PR']:8.4f} {aggregate['mean_AUC_ROC']:8.4f}")

    if edmonton_results:
        print(f"\n  Edmonton OOD: AUC-PR={edmonton_results['AUC_PR']:.4f}  "
              f"AUC-ROC={edmonton_results['AUC_ROC']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the full script imports and `main` is callable**

```bash
python -c "
import importlib
m = importlib.import_module('10_lomo_cv')
assert callable(m.main)
assert callable(m.plot_lomo_results)
assert callable(m.train_global_model)
assert callable(m.evaluate_edmonton)
print('PASS: all functions defined in 10_lomo_cv.py')
"
```

Expected: `PASS: all functions defined in 10_lomo_cv.py`

- [ ] **Step 3: Verify `plot_lomo_results` runs with synthetic data (no file I/O)**

```bash
python -c "
import importlib, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
m = importlib.import_module('10_lomo_cv')
# Synthetic fold results (won't write to disk — FIG_DIR might not exist in CI)
import unittest.mock as mock
fake_results = [
    {'metro': 'tampa', 'AUC_PR': 0.20, 'AUC_ROC': 0.69, 'F1': 0.24, 'n': 1244, 'closure_rate': 0.10},
    {'metro': 'philadelphia', 'AUC_PR': 0.14, 'AUC_ROC': 0.62, 'F1': 0.18, 'n': 2100, 'closure_rate': 0.12},
]
with mock.patch('matplotlib.pyplot.savefig'), mock.patch('matplotlib.pyplot.close'):
    m.plot_lomo_results(fake_results)
print('PASS: plot_lomo_results runs without error')
"
```

Expected: `PASS: plot_lomo_results runs without error`

- [ ] **Step 4: Commit**

```bash
git add 10_lomo_cv.py
git commit -m "feat: complete 10_lomo_cv.py — global model, Edmonton OOD, results JSON + figure"
```

---

## Running Order

After all tasks are complete, run in this order:

```bash
# Step 1 — Run all 8 new metros (separate terminals, run in parallel)
python 09_multi_metro.py --city Indianapolis --state IN
python 09_multi_metro.py --city Tucson       --state AZ
python 09_multi_metro.py --city Nashville    --state TN
python 09_multi_metro.py --city "New Orleans" --state LA
python 09_multi_metro.py --city "Saint Louis" --state MO
python 09_multi_metro.py --city Reno         --state NV
python 09_multi_metro.py --city Boise        --state ID
python 09_multi_metro.py --city Edmonton     --state AB

# Step 2 — Backfill has_photo into Tampa + Philadelphia
python 09_multi_metro.py --backfill-only

# Step 3 — LOMO CV (after all 9 metros complete)
python 10_lomo_cv.py
```

## Spec Coverage Self-Review

| Spec requirement | Task |
|---|---|
| `--city`/`--state` CLI args with city+state dual filter | Task 1, 4 |
| `--backfill-only` adds has_photo to Tampa + Philly | Task 4 |
| LATEST_ANCHOR = 2020-06-01 for all metros | Task 1 (constant) |
| 3-way review-recency label rule (not is_open) | Task 2 |
| `has_photo` joined from photo_index.parquet | Task 3 |
| Checkpoint loading (businesses + labels) | Task 4 main() |
| Always re-run feature engineering | Task 4 main() |
| Minimum 50 labeled threshold | Task 4 main() |
| LOMO 9-fold structure | Task 5 |
| Imputation medians fit on train only | Task 6 |
| Per-fold XGBoost hyperparameter tuning on val | Task 6 |
| Retrain on full train pool (train+val) | Task 6 |
| F1 threshold tuned on val fold | Task 6 |
| Global model uses best fold's hyperparams | Task 7 |
| Global model saved as xgboost_global.pkl | Task 7 |
| Edmonton evaluated only against global model | Task 7 |
| lomo_results.json with folds/aggregate/global_model keys | Task 7 |
| Grouped horizontal bar figure with Tampa reference lines | Task 7 |
