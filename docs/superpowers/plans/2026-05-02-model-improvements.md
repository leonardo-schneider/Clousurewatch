# Model Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four improvements to the global model and dashboard: null-flag features, Precision@K figure, sentence-transformer review embeddings, and model calibration.

**Architecture:** Null flags and embeddings extend the feature matrix without changing the pipeline structure — both are joined at load time so existing metro parquets never need to be re-run. After features are updated, LOMO CV is re-run once to produce a new global model, which is then calibrated and swapped into the dashboard.

**Tech Stack:** Python, XGBoost, scikit-learn `CalibratedClassifierCV`, `sentence-transformers` (`all-MiniLM-L6-v2`), `sklearn.decomposition.PCA`, matplotlib, joblib, Streamlit.

---

## File Structure

| File | Change |
|---|---|
| `app_helpers.py` | Add `NULL_FLAG_COLS` list + `add_null_flags(df)` function |
| `10_lomo_cv.py` | Call `add_null_flags()` after loading each metro parquet; join embeddings if present |
| `app.py` | Call `add_null_flags()` in `load_metro_features()`; prefer calibrated model |
| `11_presentation_figures.py` | Add `plot_precision_at_k()` → figure 29 |
| `compute_embeddings.py` | **New** — stream review JSON, compute per-restaurant mean embeddings, PCA-reduce to 32 dims, save per-metro parquets |
| `13_calibrate_model.py` | **New** — time-split calibration of global XGB using isotonic regression |
| `models/xgboost_global_calibrated.pkl` | **New** — produced by `13_calibrate_model.py` |
| `data/processed_{metro}/review_embeddings.parquet` | **New** (×9 metros) — produced by `compute_embeddings.py` |

---

## Task 1: Precision@K and Lift Curve Figure

**Files:**
- Modify: `11_presentation_figures.py` (add function + call in `main()`)

- [ ] **Step 1: Add `plot_precision_at_k` function before the `main()` block**

Insert this function at line 465, immediately before `# ── Main`:

```python
# ── Figure 29: Precision@K and lift curve ─────────────────────────────────────

def plot_precision_at_k(y_test: np.ndarray, xgb_prob: np.ndarray, lr_prob: np.ndarray):
    print("[29] Precision@K and lift curve...")
    base_rate = y_test.mean()
    K_values  = [10, 25, 50, 100, 200, 500]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel 1: Precision@K bar chart ───────────────────────────────────────
    ax = axes[0]
    x  = np.arange(len(K_values))
    w  = 0.35

    for offset, label, prob, color in [
        (-w/2, "XGBoost",             xgb_prob, C_XGB),
        ( w/2, "Logistic Regression", lr_prob,  C_LR),
    ]:
        sorted_idx = np.argsort(prob)[::-1]
        y_sorted   = y_test[sorted_idx]
        prec_at_k  = [y_sorted[:k].mean() for k in K_values]
        bars = ax.bar(x + offset, prec_at_k, w, label=label, color=color, alpha=0.85)
        for bar, p in zip(bars, prec_at_k):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{p:.0%}", ha="center", va="bottom", fontsize=8)

    ax.axhline(base_rate, color=C_BASE, linestyle="--", linewidth=1.2,
               label=f"Random baseline ({base_rate:.1%})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"@{k}" for k in K_values])
    ax.set_ylabel("Precision (fraction closed)", fontsize=10)
    ax.set_title("Precision@K — Top-K Riskiest Restaurants", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, min(1.0, base_rate * 8))

    # ── Panel 2: Cumulative lift / gain curve ────────────────────────────────
    ax2   = axes[1]
    n     = len(y_test)
    steps = np.arange(1, n + 1)

    for label, prob, color in [
        ("XGBoost",             xgb_prob, C_XGB),
        ("Logistic Regression", lr_prob,  C_LR),
    ]:
        sorted_idx   = np.argsort(prob)[::-1]
        y_sorted     = y_test[sorted_idx]
        cum_closures = np.cumsum(y_sorted)
        total_closed = y_test.sum()
        gain         = cum_closures / total_closed        # % of closures captured
        pct_reviewed = steps / n                           # % of restaurants reviewed
        ax2.plot(pct_reviewed * 100, gain * 100, color=color, linewidth=2, label=label)

    # Random baseline (diagonal)
    ax2.plot([0, 100], [0, 100], color=C_BASE, linestyle="--", linewidth=1.2,
             label="Random baseline")
    ax2.fill_between([0, 100], [0, 100], alpha=0.04, color=C_BASE)

    # Mark the "review 20% of restaurants" point
    for label, prob, color in [
        ("XGBoost", xgb_prob, C_XGB),
        ("LR",      lr_prob,  C_LR),
    ]:
        sorted_idx    = np.argsort(prob)[::-1]
        y_sorted      = y_test[sorted_idx]
        cum_closures  = np.cumsum(y_sorted)
        total_closed  = y_test.sum()
        idx_20        = max(1, int(n * 0.20)) - 1
        gain_20       = cum_closures[idx_20] / total_closed
        ax2.scatter(20, gain_20 * 100, color=color, s=70, zorder=5)
        ax2.annotate(f"{gain_20:.0%}", xy=(20, gain_20 * 100),
                     xytext=(23, gain_20 * 100 - 4),
                     fontsize=8, color=color)

    ax2.set_xlabel("% of restaurants reviewed (sorted by risk)", fontsize=10)
    ax2.set_ylabel("% of closures captured", fontsize=10)
    ax2.set_title("Cumulative Gain (Lift) Curve", fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.set_xlim(0, 100); ax2.set_ylim(0, 100)

    n_pos = int(y_test.sum())
    plt.suptitle(
        f"Ranking Quality — Time-Split Test Set  "
        f"(n={n:,} · {n_pos} closures · base rate={base_rate:.1%})",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    save("29_precision_at_k.png")
```

- [ ] **Step 2: Add the call in `main()`, after the existing `plot_fp_fn_profile` call**

Find this line in `main()`:
```python
    plot_fp_fn_profile(test_df, xgb_prob, feat_cols)
```

Replace with:
```python
    plot_fp_fn_profile(test_df, xgb_prob, feat_cols)
    plot_precision_at_k(y_test, xgb_prob, lr_prob)
```

- [ ] **Step 3: Run the script and verify the figure is created**

```bash
python 11_presentation_figures.py
```

Expected: all previous figures regenerate, then:
```
[29] Precision@K and lift curve...
  Saved -> figures\29_precision_at_k.png
```

- [ ] **Step 4: Commit**

```bash
git add 11_presentation_figures.py
git commit -m "feat: add figure 29 - Precision@K and cumulative gain curve"
```

---

## Task 2: Null-Flag Features

**Files:**
- Modify: `app_helpers.py` (add `NULL_FLAG_COLS` + `add_null_flags`)
- Modify: `10_lomo_cv.py` (call `add_null_flags` in `load_all_metros`)
- Modify: `app.py` (call `add_null_flags` in `load_metro_features`)

**Background:** Five features are null by design — not because data is missing but because the underlying count is zero or below a minimum threshold. These nulls are informative signals:

| Feature | Why null | Null rate (Tampa) |
|---|---|---|
| `vader_trend_slope` | < 5 reviews — can't fit a trend | 29.7% |
| `stars_delta_3m` | No reviews in early OR late 3m window | 40.6% |
| `mean_tip_compliments` | Zero tips filed | 49.1% |
| `checkin_velocity_slope` | Zero check-ins | 21.3% |
| `review_velocity_slope` | Zero reviews | 13.1% |

- [ ] **Step 1: Add `NULL_FLAG_COLS` and `add_null_flags` to `app_helpers.py`**

Append to the bottom of `app_helpers.py` (after `compute_shap_row`):

```python

NULL_FLAG_COLS = [
    "vader_trend_slope",
    "stars_delta_3m",
    "mean_tip_compliments",
    "checkin_velocity_slope",
    "review_velocity_slope",
]


def add_null_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary *_is_null indicator columns for features that are null by design.

    These nulls are informative (e.g. zero tips, zero reviews) rather than random
    missingness, so a flag lets the model learn their signal explicitly.
    Called immediately after loading any features.parquet.
    """
    df = df.copy()
    for col in NULL_FLAG_COLS:
        if col in df.columns:
            df[f"{col}_is_null"] = df[col].isna().astype(np.int8)
    return df
```

- [ ] **Step 2: Verify the function works**

```bash
python -c "
import pandas as pd
from app_helpers import add_null_flags
df = pd.read_parquet('data/processed/features.parquet')
df2 = add_null_flags(df)
new_cols = [c for c in df2.columns if c.endswith('_is_null')]
print('New flag columns:', new_cols)
for c in new_cols:
    print(f'  {c}: {df2[c].mean():.1%} flagged')
"
```

Expected output (approximate):
```
New flag columns: ['vader_trend_slope_is_null', 'stars_delta_3m_is_null', 'mean_tip_compliments_is_null', 'checkin_velocity_slope_is_null', 'review_velocity_slope_is_null']
  vader_trend_slope_is_null: 29.7% flagged
  stars_delta_3m_is_null: 40.6% flagged
  mean_tip_compliments_is_null: 49.1% flagged
  checkin_velocity_slope_is_null: 21.3% flagged
  review_velocity_slope_is_null: 13.1% flagged
```

- [ ] **Step 3: Call `add_null_flags` in `10_lomo_cv.py`'s `load_all_metros`**

At the top of `10_lomo_cv.py`, add the import alongside existing imports:
```python
from app_helpers import add_null_flags
```

In `load_all_metros()`, change:
```python
        df = pd.read_parquet(p)
        df["metro"] = metro
        df["anchor_date"] = pd.to_datetime(df["anchor_date"])
```
to:
```python
        df = pd.read_parquet(p)
        df = add_null_flags(df)
        df["metro"] = metro
        df["anchor_date"] = pd.to_datetime(df["anchor_date"])
```

- [ ] **Step 4: Call `add_null_flags` in `app.py`'s `load_metro_features`**

At the top of `app.py`, add to the existing import from `app_helpers`:
```python
from app_helpers import (
    risk_color,
    risk_label,
    percentile_rank,
    outcome_banner_html,
    compute_shap_row,
    add_null_flags,
)
```

Inside `load_metro_features(data_dir)`, add the call immediately after loading the parquet. Find the line:
```python
    feat = pd.read_parquet(Path(data_dir) / "features.parquet")
```
Add the call on the next line:
```python
    feat = pd.read_parquet(Path(data_dir) / "features.parquet")
    feat = add_null_flags(feat)
```

- [ ] **Step 5: Commit (null flags wired up — LOMO re-run happens in Task 4)**

```bash
git add app_helpers.py 10_lomo_cv.py app.py
git commit -m "feat: add null-flag indicators for 5 informatively-null features"
```

---

## Task 3: Sentence-Transformer Review Embeddings

**Files:**
- Create: `compute_embeddings.py`
- Modify: `10_lomo_cv.py` (join embeddings in `load_all_metros` if file exists)

**Approach:** Stream the 5.3 GB review JSON in 100k-row chunks, collecting only `business_id`, `date`, `text`. For each restaurant, sample up to 15 most recent reviews within the observation window, compute a mean sentence embedding (384-dim via `all-MiniLM-L6-v2`), then PCA-reduce to 32 dimensions. PCA is fit on all 9 metros combined to avoid re-fitting per fold.

- [ ] **Step 1: Install `sentence-transformers`**

```bash
pip install sentence-transformers
```

Expected: installs `sentence-transformers`, `torch`, and `transformers`. First run downloads the model (~80 MB).

- [ ] **Step 2: Create `compute_embeddings.py`**

```python
"""
compute_embeddings.py -- Mean sentence embeddings per restaurant, PCA-reduced to 32 dims.

For each restaurant in all 9 training metros:
  1. Collect up to 15 most recent reviews within the observation window.
  2. Compute mean sentence embedding with all-MiniLM-L6-v2 (384-dim).
  3. Fit PCA(32) on all metros combined.
  4. Save per-metro review_embeddings.parquet  (business_id + emb_pc_00..emb_pc_31).

Run once before 10_lomo_cv.py:
    python compute_embeddings.py

Runtime: ~20-40 min (depends on CPU; model download ~80 MB on first run).
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from config_00 import RAW_DIR

REVIEW_JSON   = Path(RAW_DIR) / "yelp_academic_dataset_review.json"
MODEL_NAME    = "all-MiniLM-L6-v2"
MAX_REVIEWS   = 15      # most recent reviews per restaurant
N_COMPONENTS  = 32      # PCA output dimensions
CHUNK_SIZE    = 100_000 # rows per JSON chunk

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


def load_labeled(data_dir: str) -> pd.DataFrame:
    """Return labeled_businesses with business_id, obs_start, anchor_date."""
    df = pd.read_parquet(Path(data_dir) / "labeled_businesses.parquet")
    df["obs_start"]    = pd.to_datetime(df["obs_start"])
    df["anchor_date"]  = pd.to_datetime(df["anchor_date"])
    return df[["business_id", "obs_start", "anchor_date"]]


def stream_reviews(target_bids: set) -> dict[str, list]:
    """
    Stream review JSON and return {business_id: [(date, text), ...]}
    only for businesses in target_bids.
    Keeps all matching rows; caller handles window filtering.
    """
    print(f"  Streaming {REVIEW_JSON} ...")
    reviews: dict[str, list] = {}
    for chunk in tqdm(
        pd.read_json(REVIEW_JSON, lines=True, chunksize=CHUNK_SIZE,
                     encoding="utf-8"),
        desc="  chunks",
    ):
        sub = chunk[chunk["business_id"].isin(target_bids)][
            ["business_id", "date", "text"]
        ].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        for _, row in sub.iterrows():
            reviews.setdefault(row["business_id"], []).append(
                (row["date"], str(row["text"]))
            )
    return reviews


def build_texts(labeled: pd.DataFrame, reviews: dict[str, list]) -> list[tuple]:
    """
    Return [(business_id, concatenated_text), ...] for all restaurants.
    Up to MAX_REVIEWS most recent reviews within observation window.
    Returns empty string for restaurants with no reviews.
    """
    records = []
    for _, row in labeled.iterrows():
        bid  = row["business_id"]
        obs  = row["obs_start"]
        anc  = row["anchor_date"]
        revs = reviews.get(bid, [])
        # Filter to observation window, sort desc, take top MAX_REVIEWS
        in_window = sorted(
            [(d, t) for d, t in revs if obs <= d < anc],
            key=lambda x: x[0], reverse=True,
        )[:MAX_REVIEWS]
        text = " ".join(t for _, t in in_window)
        records.append((bid, text))
    return records


def compute_embeddings(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Batch-encode list of strings; return (n, 384) float32 array."""
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )


def main():
    print("=" * 60)
    print("Computing review embeddings (all-MiniLM-L6-v2)")
    print("=" * 60)

    print("\n[1] Loading sentence-transformer model...")
    model = SentenceTransformer(MODEL_NAME)

    # Collect all business_ids from all 9 metros
    labeled_by_metro = {}
    all_bids: set[str] = set()
    for metro, ddir in METROS.items():
        lb = load_labeled(ddir)
        labeled_by_metro[metro] = lb
        all_bids.update(lb["business_id"].tolist())
    print(f"  Total unique restaurants: {len(all_bids):,}")

    print("\n[2] Streaming reviews JSON (5 GB — takes a few minutes)...")
    all_reviews = stream_reviews(all_bids)
    print(f"  Restaurants with at least 1 review: {len(all_reviews):,}")

    # Build (business_id, text) records for every metro
    all_biz_ids: list[str] = []
    all_texts:   list[str] = []
    metro_slices: dict[str, tuple[int, int]] = {}

    for metro, labeled in labeled_by_metro.items():
        records = build_texts(labeled, all_reviews)
        start = len(all_biz_ids)
        all_biz_ids.extend(r[0] for r in records)
        all_texts.extend(r[1] for r in records)
        metro_slices[metro] = (start, len(all_biz_ids))
        print(f"  {metro:15s}: {len(records):,} restaurants")

    print(f"\n[3] Computing embeddings for {len(all_texts):,} restaurants...")
    raw_embeddings = compute_embeddings(all_texts, model)  # (N, 384)
    print(f"  Embedding matrix shape: {raw_embeddings.shape}")

    print(f"\n[4] Fitting PCA({N_COMPONENTS}) on all metros combined...")
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    reduced = pca.fit_transform(raw_embeddings)  # (N, 32)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  Variance explained by {N_COMPONENTS} components: {explained:.1%}")

    emb_cols = [f"emb_pc_{i:02d}" for i in range(N_COMPONENTS)]

    print("\n[5] Saving per-metro embedding parquets...")
    for metro, ddir in METROS.items():
        start, end = metro_slices[metro]
        bids  = all_biz_ids[start:end]
        vecs  = reduced[start:end]
        df    = pd.DataFrame(vecs, columns=emb_cols)
        df.insert(0, "business_id", bids)
        out   = Path(ddir) / "review_embeddings.parquet"
        df.to_parquet(out, index=False)
        print(f"  Saved -> {out}  ({len(df):,} rows × {N_COMPONENTS} emb cols)")

    print("\nDone. Re-run 10_lomo_cv.py to train with embedding features.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run `compute_embeddings.py` and verify output**

```bash
python compute_embeddings.py
```

Expected final output (approximate):
```
[5] Saving per-metro embedding parquets...
  Saved -> data\processed\review_embeddings.parquet  (5,143 rows × 32 emb cols)
  Saved -> data\processed_philly\review_embeddings.parquet  (4,258 rows × 32 emb cols)
  ...
  Saved -> data\processed_boise\review_embeddings.parquet  (627 rows × 32 emb cols)
```

- [ ] **Step 4: Wire embeddings into `10_lomo_cv.py`'s `load_all_metros`**

In `load_all_metros()`, after the existing `add_null_flags` call (added in Task 2), add the embedding join:

```python
        df = pd.read_parquet(p)
        df = add_null_flags(df)

        # Join sentence-embedding features if available
        emb_path = Path(directory) / "review_embeddings.parquet"
        if emb_path.exists():
            emb = pd.read_parquet(emb_path)
            df  = df.merge(emb, on="business_id", how="left")

        df["metro"] = metro
        df["anchor_date"] = pd.to_datetime(df["anchor_date"])
```

- [ ] **Step 5: Verify the feature count increases**

```bash
python -c "
import pandas as pd
from pathlib import Path
from app_helpers import add_null_flags

df = pd.read_parquet('data/processed/features.parquet')
df = add_null_flags(df)
emb = pd.read_parquet('data/processed/review_embeddings.parquet')
df = df.merge(emb, on='business_id', how='left')
META = {'business_id','closed_within_6m','anchor_date','city','state','metro'}
feat_cols = [c for c in df.columns if c not in META]
print(f'Feature count with null flags + embeddings: {len(feat_cols)}')
# Expected: 49 (original) + 5 (null flags) + 32 (embeddings) = 86
"
```

Expected: `Feature count with null flags + embeddings: 86`

- [ ] **Step 6: Commit**

```bash
git add compute_embeddings.py 10_lomo_cv.py
git commit -m "feat: add sentence-transformer embeddings (MiniLM-L6-v2, PCA-32) and wire into LOMO loader"
```

---

## Task 4: Re-run LOMO CV

**Files:**
- Run: `10_lomo_cv.py` (no code changes — picks up null flags + embeddings automatically)
- Produces: `models/xgboost_global.pkl` (updated), `models/lomo_results.json` (updated)

- [ ] **Step 1: Run LOMO CV**

```bash
python 10_lomo_cv.py
```

Runtime: ~20-30 min (longer than before due to 86 features vs 49).

Expected output (approximate — numbers will change with new features):
```
[3] Aggregate metrics
  [xgb     ] AUC-PR=0.XXXX+/-0.XXXX  AUC-ROC=0.XXXX+/-0.XXXX
  [xgb_cal ] AUC-PR=0.XXXX+/-0.XXXX  AUC-ROC=0.XXXX+/-0.XXXX
  [lr      ] AUC-PR=0.XXXX+/-0.XXXX  AUC-ROC=0.XXXX+/-0.XXXX
  Global XGB saved -> models/xgboost_global.pkl
```

- [ ] **Step 2: Verify new global model uses 86 features**

```bash
python -c "
import joblib
m = joblib.load('models/xgboost_global.pkl')
feat = m.get_booster().feature_names
print(f'Global model feature count: {len(feat)}')
null_flags = [f for f in feat if f.endswith('_is_null')]
emb_feats  = [f for f in feat if f.startswith('emb_pc_')]
print(f'  Null flags: {len(null_flags)} -> {null_flags}')
print(f'  Embedding features: {len(emb_feats)}')
"
```

Expected:
```
Global model feature count: 86
  Null flags: 5 -> ['vader_trend_slope_is_null', ...]
  Embedding features: 32
```

- [ ] **Step 3: Commit results**

```bash
git add models/lomo_results.json
git commit -m "chore: re-run LOMO CV with null flags + sentence embeddings (86 features)"
```

---

## Task 5: Calibrate the Global Model and Update App

**Files:**
- Create: `13_calibrate_model.py`
- Produces: `models/xgboost_global_calibrated.pkl`
- Modify: `app.py` (`load_xgb_model` function)

**Approach:** Time-split all 9 metro features 80/20 by anchor date. The global XGB was trained on all data, so we use it as a pre-fit base estimator and calibrate with isotonic regression on the 20% held-out portion via `CalibratedClassifierCV(cv='prefit')`. This does not require re-training XGB.

- [ ] **Step 1: Create `13_calibrate_model.py`**

```python
"""
13_calibrate_model.py -- Calibrate the global XGBoost model with isotonic regression.

Uses a time-based 20% hold-out of all 9 metros as the calibration set.
Saves models/xgboost_global_calibrated.pkl.

Run after Task 4 (LOMO re-run):
    python 13_calibrate_model.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from app_helpers import add_null_flags

from config_00 import MODEL_DIR, FIG_DIR

LATEST_ANCHOR = pd.Timestamp("2020-06-01")

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

META_COLS = {"business_id", "closed_within_6m", "anchor_date",
             "city", "state", "metro"}


def load_all() -> pd.DataFrame:
    frames = []
    for metro, ddir in METROS.items():
        df = pd.read_parquet(Path(ddir) / "features.parquet")
        df = add_null_flags(df)
        emb_path = Path(ddir) / "review_embeddings.parquet"
        if emb_path.exists():
            emb = pd.read_parquet(emb_path)
            df  = df.merge(emb, on="business_id", how="left")
        df["metro"]       = metro
        df["anchor_date"] = pd.to_datetime(df["anchor_date"])
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    return all_df[all_df["anchor_date"] <= LATEST_ANCHOR].copy()


def time_split(df: pd.DataFrame, val_frac: float = 0.20):
    df_s   = df.sort_values("anchor_date")
    n_cal  = max(1, int(len(df_s) * val_frac))
    return df_s.iloc[:-n_cal].copy(), df_s.iloc[-n_cal:].copy()


def plot_calibration(y_cal, uncal_prob, cal_prob, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.rcParams.update({"font.family": "serif", "figure.dpi": 150})

    for label, prob, color in [
        ("Uncalibrated XGB", uncal_prob, "#2E86AB"),
        ("Calibrated XGB",   cal_prob,   "#1DB954"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_cal, prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2,
                label=label, color=color)

    ax.plot([0, 1], [0, 1], linestyle="--", color="#888",
            linewidth=1.2, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability", fontsize=10)
    ax.set_ylabel("Fraction of positives", fontsize=10)
    ax.set_title("Calibration Curve (Reliability Diagram)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Calibration curve saved -> {out_path}")


def main():
    print("=" * 60)
    print("STEP 13 -- Calibrate Global XGBoost Model")
    print("=" * 60)

    print("\n[1] Loading all 9 metros...")
    all_df    = load_all()
    feat_cols = [c for c in all_df.columns if c not in META_COLS]
    _, cal_df = time_split(all_df, val_frac=0.20)
    print(f"  Calibration set: {len(cal_df):,} restaurants  "
          f"({cal_df['closed_within_6m'].mean():.1%} closure rate)")

    print("\n[2] Loading uncalibrated global model...")
    base_model = joblib.load(MODEL_DIR / "xgboost_global.pkl")
    xgb_feat   = base_model.get_booster().feature_names

    medians    = all_df[xgb_feat].median()
    X_cal      = cal_df.reindex(columns=xgb_feat).fillna(medians)
    y_cal      = cal_df["closed_within_6m"].values

    uncal_prob = base_model.predict_proba(X_cal)[:, 1]
    print(f"  Uncalibrated mean predicted prob: {uncal_prob.mean():.4f}  "
          f"(true rate: {y_cal.mean():.4f})")

    print("\n[3] Fitting isotonic calibration on calibration set...")
    calibrated = CalibratedClassifierCV(
        estimator=base_model, cv="prefit", method="isotonic"
    )
    calibrated.fit(X_cal, y_cal)

    cal_prob = calibrated.predict_proba(X_cal)[:, 1]
    print(f"  Calibrated   mean predicted prob: {cal_prob.mean():.4f}  "
          f"(true rate: {y_cal.mean():.4f})")

    out_path = MODEL_DIR / "xgboost_global_calibrated.pkl"
    joblib.dump(calibrated, out_path)
    print(f"\n[4] Saved -> {out_path}")

    print("\n[5] Plotting reliability diagram...")
    plot_calibration(
        y_cal, uncal_prob, cal_prob,
        FIG_DIR / "30_calibration_curve.png",
    )

    print("\nDone. Update app.py to prefer xgboost_global_calibrated.pkl.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
python 13_calibrate_model.py
```

Expected:
```
[4] Saved -> models\xgboost_global_calibrated.pkl
  Calibration curve saved -> figures\30_calibration_curve.png
```

- [ ] **Step 3: Update `app.py`'s `load_xgb_model` to prefer the calibrated model**

Find the existing `load_xgb_model` function in `app.py` (around line 393):
```python
def load_xgb_model():
    import joblib
    # Prefer global 9-metro model; fall back to Tampa-only
    for path in ["models/xgboost_global.pkl", "models/xgboost.pkl"]:
        p = Path(path)
        if p.exists():
            return joblib.load(p)
    return None
```

Replace with:
```python
def load_xgb_model():
    import joblib
    # Prefer calibrated global model; fall back through hierarchy
    for path in [
        "models/xgboost_global_calibrated.pkl",
        "models/xgboost_global.pkl",
        "models/xgboost.pkl",
    ]:
        p = Path(path)
        if p.exists():
            return joblib.load(p)
    return None
```

- [ ] **Step 4: Verify the app loads the calibrated model**

```bash
python -c "
import sys; sys.path.insert(0,'.')
# Simulate what app.py does
from pathlib import Path
import joblib

model = None
for path in ['models/xgboost_global_calibrated.pkl',
             'models/xgboost_global.pkl', 'models/xgboost.pkl']:
    p = Path(path)
    if p.exists():
        model = joblib.load(p)
        print(f'Loaded: {path}')
        print(f'Type: {type(model).__name__}')
        break
"
```

Expected:
```
Loaded: models/xgboost_global_calibrated.pkl
Type: CalibratedClassifierCV
```

- [ ] **Step 5: Verify SHAP still works with calibrated model**

`compute_shap_row` in `app_helpers.py` calls `model.get_booster()`. `CalibratedClassifierCV` wraps the base estimator — access it via `model.estimator` or `model.calibrated_classifiers_[0].estimator`. Update `compute_shap_row` in `app_helpers.py` to unwrap if needed:

Find in `compute_shap_row`:
```python
    booster = model.get_booster()
```

Replace with:
```python
    # CalibratedClassifierCV wraps the base XGBoost — unwrap if needed
    xgb_base = getattr(model, "estimator", model)
    booster  = xgb_base.get_booster()
```

- [ ] **Step 6: Commit**

```bash
git add 13_calibrate_model.py app.py app_helpers.py
git commit -m "feat: calibrate global XGBoost with isotonic regression, update app to use calibrated model"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Null-flag features: Task 2 adds 5 flags, wired into LOMO + app
- ✅ Precision@K + lift curve: Task 1 adds figure 29 with both panels
- ✅ Sentence-transformer embeddings: Task 3 computes and saves per-metro, Task 4 wires into LOMO
- ✅ Calibration: Task 5 calibrates the model, updates app, updates SHAP helper

**Placeholder scan:** None found — all tasks have complete code blocks.

**Type consistency check:**
- `add_null_flags(df: pd.DataFrame) -> pd.DataFrame` used consistently in Tasks 2 and 5
- `compute_shap_row` unwraps via `getattr(model, "estimator", model)` which works for both `CalibratedClassifierCV` and raw `XGBClassifier`
- `load_all()` in `13_calibrate_model.py` uses same pattern as `12_kfold_experiment.py`

**Dependency order:**
- Task 1 is fully independent
- Task 2 must come before Task 3 (LOMO re-run) and Task 5 (calibration)
- Task 3 must come before Task 4 (LOMO re-run)
- Task 4 must come before Task 5 (calibration uses the new global model)
