"""
02_build_labels.py -- Build anchor dates and binary closure labels for all 9 metros.

Label rule (review-recency, no is_open leakage):
  last_review > outcome_end + 3m  -> 0 (still active)
  last_review <= outcome_end      -> 1 (closed)
  ambiguous window                -> 0

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
        print(f"  [{metro_key}] SKIP -- run 01_load_filter.py first")
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
