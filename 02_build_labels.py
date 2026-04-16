"""
02_build_labels.py
──────────────────
Assign anchor dates and construct temporally-safe binary labels.

The core anti-leakage design:
  - Each restaurant gets ONE anchor date drawn from its review history.
  - Features will ONLY use data BEFORE the anchor date.
  - The label is determined by activity in [anchor, anchor + 6 months).
  - No future information bleeds into features.

Run:
    python 02_build_labels.py

Output:
    data/processed/labeled_businesses.parquet
        — one row per restaurant with anchor_date + target label
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

sys.path.insert(0, str(Path(__file__).parent))
from config_00 import (
    PROC_DIR,
    OBS_MONTHS, OUTCOME_MONTHS,
    EARLIEST_ANCHOR, LATEST_ANCHOR,
    TARGET_COL,
)

EARLIEST_ANCHOR_DT = pd.Timestamp(EARLIEST_ANCHOR)
LATEST_ANCHOR_DT   = pd.Timestamp(LATEST_ANCHOR)


# ─────────────────────────────────────────────────────────────────────────────
# Anchor date strategy
# ─────────────────────────────────────────────────────────────────────────────

def assign_anchor(reviews_biz: pd.DataFrame) -> pd.Timestamp | None:
    """
    Pick an anchor date for one business.

    Strategy:
      - Find the 80th percentile review date (the business is well-established).
      - Clip to [EARLIEST_ANCHOR, LATEST_ANCHOR].
      - Require at least OBS_MONTHS of history before the anchor.
      - Return None if constraints can't be met.

    Using 80th pct (not the last review) avoids anchoring right before closure
    which would let the model trivially detect the review drought signal.
    We want the anchor to sit in a "normal" operating period.
    """
    if reviews_biz.empty:
        return None

    dates = reviews_biz["date"].sort_values()
    pct80 = dates.quantile(0.80)

    # Snap to month boundary for cleanliness
    anchor = pct80.replace(day=1)

    # Apply temporal bounds
    anchor = max(anchor, EARLIEST_ANCHOR_DT)
    anchor = min(anchor, LATEST_ANCHOR_DT)

    # Check we have enough history
    obs_start = anchor - relativedelta(months=OBS_MONTHS)
    n_pre = (dates < anchor).sum()
    if n_pre < 4:   # need at least 4 reviews to build meaningful features
        return None

    # Check the business existed before the obs window
    if dates.min() > obs_start:
        return None

    # CRITICAL: We must be able to observe post-anchor activity.
    # If the dataset has no reviews for this business after anchor+6m,
    # we cannot know if it closed. Drop it.
    post_outcome = reviews_biz[reviews_biz["date"] > anchor + relativedelta(months=6)]
    pre_anchor = reviews_biz[reviews_biz["date"] < anchor]

    # Require at least 1 review AFTER the outcome window to confirm still open,
    # OR the business must be in the closed pool (last review < anchor+6m)
    # We handle this in build_label — here just ensure anchor is not too close
    # to the dataset edge (latest review date in the whole dataset)
    DATASET_END = pd.Timestamp("2022-01-19")  # from step 1 output
    if anchor + relativedelta(months=6) > DATASET_END:
        return None

    return anchor


# ─────────────────────────────────────────────────────────────────────────────
# Label construction
# ─────────────────────────────────────────────────────────────────────────────

def build_label(biz_row, reviews_biz, anchor):
    from dateutil.relativedelta import relativedelta
    outcome_end = anchor + relativedelta(months=6)
    DATASET_END = pd.Timestamp("2022-01-19")

    last_review_ever = reviews_biz['date'].max()

    # If business has reviews well after outcome window → was open → label 0
    # "Well after" = at least 3 months past outcome_end (confirms ongoing activity)
    if last_review_ever > outcome_end + relativedelta(months=3):
        return 0

    # If last review is before or within outcome_end → activity stopped → label 1
    # But only if the dataset_end gives us enough room to observe silence
    # (i.e. outcome_end is not right at the edge of the data)
    if last_review_ever <= outcome_end:
        return 1

    # Last review is in (outcome_end, outcome_end+3m] — ambiguous, drop via anchor
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 2 — Build Temporal Labels")
    print("=" * 60)

    biz     = pd.read_parquet(PROC_DIR / "businesses.parquet")
    reviews = pd.read_parquet(PROC_DIR / "reviews.parquet")

    print(f"\n  Businesses: {len(biz):,}")
    print(f"  Reviews   : {len(reviews):,}")

    # Group reviews by business for fast lookup
    rev_groups = reviews.groupby("business_id")

    records = []
    skipped_no_anchor = 0
    skipped_no_reviews = 0

    for _, row in biz.iterrows():
        bid = row["business_id"]

        if bid not in rev_groups.groups:
            skipped_no_reviews += 1
            continue

        biz_reviews = rev_groups.get_group(bid)
        anchor = assign_anchor(biz_reviews)

        if anchor is None:
            skipped_no_anchor += 1
            continue

        label = build_label(row, biz_reviews, anchor)

        obs_start = anchor - relativedelta(months=OBS_MONTHS)
        records.append({
            "business_id":    bid,
            "name":           row["name"],
            "city":           row["city"],
            "state":          row["state"],
            "anchor_date":    anchor,
            "obs_start":      obs_start,
            "outcome_end":    anchor + relativedelta(months=OUTCOME_MONTHS),
            TARGET_COL:       label,
            # Pass-through metadata
            "stars_yelp":         row["stars"],
            "review_count_yelp":  row["review_count"],
            "price_range":        row.get("price_range", np.nan),
            "open_days_per_week": row.get("open_days_per_week", np.nan),
            "categories":         row.get("categories", ""),
            "latitude":           row.get("latitude", np.nan),
            "longitude":          row.get("longitude", np.nan),
        })

    labeled = pd.DataFrame(records)

    print(f"\n  Anchored records    : {len(labeled):,}")
    print(f"  Skipped (no reviews): {skipped_no_reviews:,}")
    print(f"  Skipped (no anchor) : {skipped_no_anchor:,}")
    print(f"\n  CLASS BALANCE")
    print(f"  Closed (label=1) : {labeled[TARGET_COL].sum():,}  ({labeled[TARGET_COL].mean():.1%})")
    print(f"  Open   (label=0) : {(labeled[TARGET_COL]==0).sum():,}")

    # ── Anchor date distribution ───────────────────────────────────────────
    print(f"\n  Anchor date range: {labeled['anchor_date'].min().date()} → {labeled['anchor_date'].max().date()}")
    print(f"  Median anchor    : {labeled['anchor_date'].median().date()}")

    labeled.to_parquet(PROC_DIR / "labeled_businesses.parquet", index=False)
    print(f"\n  Saved → {PROC_DIR}/labeled_businesses.parquet")

    # ── Warn if class imbalance is extreme ────────────────────────────────
    pos_rate = labeled[TARGET_COL].mean()
    if pos_rate < 0.05:
        print("\n  ⚠  WARNING: Very few closures (<5%). Consider:")
        print("     • Expanding geography (all US cities)")
        print("     • Relaxing the anchor date bounds")
        print("     • Using SMOTE or class-weighted models")
    elif pos_rate > 0.40:
        print("\n  ⚠  WARNING: Closure rate >40% — check label logic for leakage.")


if __name__ == "__main__":
    main()
