"""
01_load_filter.py
─────────────────
Load Yelp Academic Dataset JSON files, filter to Florida restaurants,
and save clean Parquet files for downstream feature engineering.

Run:
    python 01_load_filter.py

Outputs (data/processed/):
    businesses.parquet   — FL restaurants with closure label
    reviews.parquet      — reviews for those businesses
    checkins.parquet     — checkin timeseries
    tips.parquet         — tips / owner responses
"""

import json, sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

# ── allow running from repo root or scripts/ ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config_00 import (
    RAW_DIR, PROC_DIR,
    PRIMARY_CITIES, FALLBACK_STATES, MIN_CLOSED_THRESHOLD,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

RESTAURANT_CATEGORIES = {
    "Restaurants", "Food", "Bars", "Nightlife",
    "Fast Food", "Cafes", "Pizza", "Burgers",
    "Mexican", "Chinese", "Italian", "Sushi Bars",
    "American (Traditional)", "American (New)",
    "Seafood", "Sandwiches", "Breakfast & Brunch",
}

def is_restaurant(cats: str | None) -> bool:
    """Return True if the business is food/restaurant adjacent."""
    if not cats:
        return False
    return bool(set(cats.split(", ")) & RESTAURANT_CATEGORIES)


def stream_json(path: Path, max_rows: int | None = None) -> list[dict]:
    """Read a Yelp JSON file (one JSON object per line)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc=f"Reading {path.name}")):
            if max_rows and i >= max_rows:
                break
            rows.append(json.loads(line))
    return rows


def parse_checkin_dates(date_str: str) -> list[str]:
    """Yelp checkins store comma-separated datetime strings."""
    if not date_str:
        return []
    return [d.strip() for d in date_str.split(",") if d.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load & filter businesses
# ─────────────────────────────────────────────────────────────────────────────

def load_businesses() -> pd.DataFrame:
    path = RAW_DIR / "yelp_academic_dataset_business.json"
    assert path.exists(), f"Missing: {path}\nDownload from https://www.yelp.com/dataset"

    raw = stream_json(path)
    df = pd.DataFrame(raw)

    # ── Keep only restaurants ──────────────────────────────────────────────
    df = df[df["categories"].apply(is_restaurant)].copy()
    print(f"  Restaurants globally: {len(df):,}")

    # ── Geography: try Tampa first, fall back to all FL ───────────────────
    tampa = df[df["city"].isin(PRIMARY_CITIES)]
    n_closed_tampa = (tampa["is_open"] == 0).sum()
    print(f"  Tampa restaurants: {len(tampa):,}  |  closed: {n_closed_tampa:,}")

    if n_closed_tampa >= MIN_CLOSED_THRESHOLD:
        geo = tampa
        geo_label = "Tampa"
    else:
        geo = df[df["state"].isin(FALLBACK_STATES)]
        geo_label = "Florida"
        print(f"  ⚠  Tampa too small — expanding to {geo_label}")

    print(f"  Using {geo_label}: {len(geo):,} restaurants  |  closed: {(geo['is_open']==0).sum():,}")

    # ── Build clean dataframe ─────────────────────────────────────────────
    # Yelp is_open: 1=open, 0=closed (permanently or temporarily)
    keep_cols = [
        "business_id", "name", "city", "state",
        "stars", "review_count", "is_open",
        "categories", "attributes", "hours",
        "latitude", "longitude",
    ]
    geo = geo[[c for c in keep_cols if c in geo.columns]].copy()
    geo["closed_label_raw"] = (geo["is_open"] == 0).astype(int)

    # ── Parse price range from attributes ─────────────────────────────────
    def get_price(attrs):
        if isinstance(attrs, dict):
            return attrs.get("RestaurantsPriceRange2", None)
        return None

    geo["price_range"] = geo["attributes"].apply(get_price)
    geo["price_range"] = pd.to_numeric(geo["price_range"], errors="coerce")

    # ── Parse open days count from hours ──────────────────────────────────
    def open_days(hrs):
        if isinstance(hrs, dict):
            return len([v for v in hrs.values() if v and v != "0:00-0:00"])
        return np.nan

    geo["open_days_per_week"] = geo["hours"].apply(open_days)

    geo = geo.drop(columns=["attributes", "hours"], errors="ignore")

    print(f"\n  Class balance — closed: {geo['closed_label_raw'].mean():.1%}")
    return geo.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Load reviews for filtered businesses
# ─────────────────────────────────────────────────────────────────────────────

def load_reviews(business_ids: set) -> pd.DataFrame:
    path = RAW_DIR / "yelp_academic_dataset_review.json"
    assert path.exists(), f"Missing: {path}"

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Streaming reviews"):
            r = json.loads(line)
            if r["business_id"] in business_ids:
                rows.append({
                    "review_id":   r["review_id"],
                    "business_id": r["business_id"],
                    "user_id":     r["user_id"],
                    "stars":       r["stars"],
                    "date":        r["date"],          # "YYYY-MM-DD HH:MM:SS"
                    "text":        r["text"],
                    "useful":      r.get("useful", 0),
                    "funny":       r.get("funny", 0),
                    "cool":        r.get("cool", 0),
                })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["business_id", "date"]).reset_index(drop=True)
    print(f"  Reviews loaded: {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Load checkins
# ─────────────────────────────────────────────────────────────────────────────

def load_checkins(business_ids: set) -> pd.DataFrame:
    path = RAW_DIR / "yelp_academic_dataset_checkin.json"
    assert path.exists(), f"Missing: {path}"

    rows = []
    for rec in stream_json(path):
        if rec["business_id"] not in business_ids:
            continue
        dates = parse_checkin_dates(rec.get("date", ""))
        for d in dates:
            rows.append({"business_id": rec["business_id"], "checkin_date": d})

    df = pd.DataFrame(rows)
    if df.empty:
        print("  ⚠  No checkins found for these businesses.")
        return df
    df["checkin_date"] = pd.to_datetime(df["checkin_date"])
    df = df.sort_values(["business_id", "checkin_date"]).reset_index(drop=True)
    print(f"  Checkin events loaded: {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Load tips (includes owner responses)
# ─────────────────────────────────────────────────────────────────────────────

def load_tips(business_ids: set) -> pd.DataFrame:
    path = RAW_DIR / "yelp_academic_dataset_tip.json"
    assert path.exists(), f"Missing: {path}"

    rows = []
    for rec in stream_json(path):
        if rec["business_id"] not in business_ids:
            continue
        rows.append({
            "business_id":     rec["business_id"],
            "user_id":         rec["user_id"],
            "text":            rec["text"],
            "date":            rec["date"],
            "compliment_count": rec.get("compliment_count", 0),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  ⚠  No tips found for these businesses.")
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["business_id", "date"]).reset_index(drop=True)
    print(f"  Tips loaded: {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 1 — Load & Filter Yelp Data")
    print("=" * 60)

    # Businesses
    print("\n[1/4] Businesses")
    biz = load_businesses()
    biz.to_parquet(PROC_DIR / "businesses.parquet", index=False)
    print(f"  Saved → {PROC_DIR}/businesses.parquet  ({len(biz):,} rows)")

    bid_set = set(biz["business_id"])

    # Reviews
    print("\n[2/4] Reviews")
    reviews = load_reviews(bid_set)
    reviews.to_parquet(PROC_DIR / "reviews.parquet", index=False)
    print(f"  Saved → {PROC_DIR}/reviews.parquet  ({len(reviews):,} rows)")

    # Checkins
    print("\n[3/4] Checkins")
    checkins = load_checkins(bid_set)
    if not checkins.empty:
        checkins.to_parquet(PROC_DIR / "checkins.parquet", index=False)
        print(f"  Saved → {PROC_DIR}/checkins.parquet  ({len(checkins):,} rows)")

    # Tips
    print("\n[4/4] Tips")
    tips = load_tips(bid_set)
    if not tips.empty:
        tips.to_parquet(PROC_DIR / "tips.parquet", index=False)
        print(f"  Saved → {PROC_DIR}/tips.parquet  ({len(tips):,} rows)")

    # ── Summary stats ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LOAD COMPLETE — Summary")
    print("=" * 60)
    print(f"  Businesses : {len(biz):,}")
    print(f"  Closed (raw label): {biz['closed_label_raw'].sum():,}  ({biz['closed_label_raw'].mean():.1%})")
    if not reviews.empty:
        date_range = f"{reviews['date'].min().date()} → {reviews['date'].max().date()}"
        print(f"  Reviews    : {len(reviews):,}  [{date_range}]")
    if not checkins.empty:
        print(f"  Checkins   : {len(checkins):,}")
    if not tips.empty:
        print(f"  Tips       : {len(tips):,}")
    print()


if __name__ == "__main__":
    main()
