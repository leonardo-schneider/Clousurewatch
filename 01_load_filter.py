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
        print("  WARNING: no businesses found -- check city/state spelling")
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
