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
