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
from typing import Tuple
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


def load_interactions(bids: set) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def build_labels(biz: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """
    3-way review-recency rule (mirrors 02_build_labels.py build_label()).
    Completely ignores is_open flag — labels from review behaviour only.

    Rules:
      - last_review > outcome_end + 3 months  → label 0 (restaurant still active)
      - last_review <= outcome_end            → label 1 (closed)
      - everything else (ambiguous window)    → label 0
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
        if rev_groups is not None and bid in rev_groups.groups:
            obs_rev = rev_groups.get_group(bid)
            obs_rev = obs_rev[(obs_rev["date"] >= obs_s) & (obs_rev["date"] < anchor)]

        obs_ci = pd.DataFrame()
        if ci_groups is not None and bid in ci_groups.groups:
            obs_ci = ci_groups.get_group(bid)
            obs_ci = obs_ci[(obs_ci["checkin_date"] >= obs_s) & (obs_ci["checkin_date"] < anchor)]

        obs_tip = pd.DataFrame()
        if tip_groups is not None and bid in tip_groups.groups:
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

    if len(labeled) < MIN_LABELED:
        print(f"  WARNING: Only {len(labeled)} labeled restaurants (< {MIN_LABELED}). Exiting.")
        return

    n_closed = int(labeled[TARGET_COL].sum())
    rate = n_closed / len(labeled)
    print(f"  Labeled: {len(labeled):,} restaurants, {n_closed} closed ({rate:.1%})")

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
