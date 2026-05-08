"""
compute_embeddings.py -- Mean sentence embeddings per restaurant, PCA-reduced to 32 dims.

For each restaurant in all 9 training metros:
  1. Collect up to 15 most recent reviews within the observation window.
  2. Compute mean sentence embedding with all-MiniLM-L6-v2 (384-dim).
  3. Fit PCA(32) on non-empty restaurants (see NOTE on LOMO leakage below).
  4. Save per-metro review_embeddings.parquet  (business_id + emb_pc_00..emb_pc_31).
     Restaurants with no in-window reviews get NaN in all emb_pc_* columns.

Run once before 10_lomo_cv.py:
    python compute_embeddings.py

Runtime: ~20-40 min (depends on CPU; model download ~80 MB on first run).
"""
# NOTE: PCA is fit on all 9 metros combined, not per LOMO fold. Since PCA is
# unsupervised (no labels used), this is mild unsupervised leakage — the
# projection axes reflect the test-fold text distribution. The practical impact
# is small but not zero. For strict LOMO purity, refit PCA per fold inside
# 10_lomo_cv.py using only training metros. Accepted as a known limitation.
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
    df["obs_start"]   = pd.to_datetime(df["obs_start"])
    df["anchor_date"] = pd.to_datetime(df["anchor_date"])
    return df[["business_id", "obs_start", "anchor_date"]]


def stream_reviews(target_bids: set) -> dict:
    """
    Stream review JSON and return {business_id: [(date, text), ...]}
    only for businesses in target_bids.
    Keeps all matching rows; caller handles window filtering.
    """
    print(f"  Streaming {REVIEW_JSON} ...")
    reviews = {}
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


def build_texts(labeled: pd.DataFrame, reviews: dict) -> list:
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
        in_window = sorted(
            [(d, t) for d, t in revs if obs <= d < anc],
            key=lambda x: x[0], reverse=True,
        )[:MAX_REVIEWS]
        text = " ".join(t[:500] for _, t in in_window)  # cap per-review text to limit RAM
        records.append((bid, text))
    return records


def compute_embeddings(texts: list, model: SentenceTransformer) -> np.ndarray:
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

    labeled_by_metro = {}
    all_bids = set()
    for metro, ddir in METROS.items():
        lb = load_labeled(ddir)
        labeled_by_metro[metro] = lb
        all_bids.update(lb["business_id"].tolist())
    print(f"  Total unique restaurants: {len(all_bids):,}")

    print("\n[2] Streaming reviews JSON (5 GB -- takes a few minutes)...")
    all_reviews = stream_reviews(all_bids)
    print(f"  Restaurants with at least 1 review: {len(all_reviews):,}")

    all_biz_ids = []
    all_texts   = []
    metro_slices = {}

    for metro, labeled in labeled_by_metro.items():
        records = build_texts(labeled, all_reviews)
        start = len(all_biz_ids)
        all_biz_ids.extend(r[0] for r in records)
        all_texts.extend(r[1] for r in records)
        metro_slices[metro] = (start, len(all_biz_ids))
        print(f"  {metro:15s}: {len(records):,} restaurants")

    n_total   = len(all_texts)
    non_empty = [(i, t) for i, t in enumerate(all_texts) if t.strip()]
    n_empty   = n_total - len(non_empty)
    print(f"\n[3] Computing embeddings for {len(non_empty):,} non-empty restaurants "
          f"({n_empty:,} will get NaN embeddings)...")

    if non_empty:
        ne_indices, ne_texts = zip(*non_empty)
        ne_raw = compute_embeddings(list(ne_texts), model)  # (M, 384)
        print(f"  Embedding matrix shape: {ne_raw.shape}")

        print(f"\n[4] Fitting PCA({N_COMPONENTS}) on non-empty restaurants...")
        pca = PCA(n_components=N_COMPONENTS, random_state=42)
        ne_reduced = pca.fit_transform(ne_raw)  # (M, 32)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  Variance explained by {N_COMPONENTS} components: {explained:.1%}")

        # Reconstruct full (N, 32) array with NaN for empty-text restaurants
        reduced = np.full((n_total, N_COMPONENTS), np.nan, dtype=np.float32)
        for pos, orig_idx in enumerate(ne_indices):
            reduced[orig_idx] = ne_reduced[pos]
    else:
        reduced = np.full((n_total, N_COMPONENTS), np.nan, dtype=np.float32)

    emb_cols = [f"emb_pc_{i:02d}" for i in range(N_COMPONENTS)]

    print("\n[5] Saving per-metro embedding parquets...")
    for metro, ddir in METROS.items():
        start, end = metro_slices[metro]
        bids = all_biz_ids[start:end]
        vecs = reduced[start:end]
        df   = pd.DataFrame(vecs, columns=emb_cols)
        df.insert(0, "business_id", bids)
        out  = Path(ddir) / "review_embeddings.parquet"
        df.to_parquet(out, index=False)
        print(f"  Saved -> {out}  ({len(df):,} rows x {N_COMPONENTS} emb cols)")

    print("\nDone. Re-run 10_lomo_cv.py to train with embedding features.")


if __name__ == "__main__":
    main()
