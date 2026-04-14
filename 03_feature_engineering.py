"""
03_feature_engineering.py
──────────────────────────
Build all features using ONLY data in the observation window:
    [anchor_date - 12 months, anchor_date)

Feature groups:
  A. Volume & velocity signals
  B. Rating trend signals
  C. VADER sentiment trend
  D. Review drought & recency
  E. Owner engagement (response rate proxy via tip text)
  F. Reviewer quality signals
  G. Checkin signals
  H. Business metadata (static, no leakage risk)

Run:
    pip install vaderSentiment tqdm
    python 03_feature_engineering.py

Output:
    data/processed/features.parquet
"""

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config_00 import (
    PROC_DIR,
    OBS_MONTHS,
    MIN_REVIEWS_FOR_TREND,
    REVIEW_DROUGHT_DAYS,
    TARGET_COL,
)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    _sia = SentimentIntensityAnalyzer()
except ImportError:
    VADER_AVAILABLE = False
    print("⚠  vaderSentiment not installed. Sentiment features will be NaN.")
    print("   pip install vaderSentiment")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def vader_compound(text: str) -> float:
    if not VADER_AVAILABLE or not isinstance(text, str):
        return np.nan
    return _sia.polarity_scores(text)["compound"]


def safe_slope(series: pd.Series) -> float:
    """OLS slope of a time-indexed series; NaN if too few points."""
    if len(series) < 2:
        return np.nan
    x = np.arange(len(series), dtype=float)
    y = series.values.astype(float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def trend_early_vs_late(series: pd.Series, n: int = 3) -> float:
    """Mean of last N minus mean of first N (positive = improving)."""
    if len(series) < 2 * n:
        return np.nan
    early = series.iloc[:n].mean()
    late  = series.iloc[-n:].mean()
    return float(late - early)


# ─────────────────────────────────────────────────────────────────────────────
# Per-business feature builder
# ─────────────────────────────────────────────────────────────────────────────

def build_features_one(
    row: pd.Series,
    obs_reviews: pd.DataFrame,
    obs_checkins: pd.DataFrame,
    obs_tips: pd.DataFrame,
) -> dict:
    """
    Given pre-filtered observation window data for ONE business,
    return a flat feature dictionary.
    """
    bid = row["business_id"]
    anchor = row["anchor_date"]
    obs_start = row["obs_start"]
    feat = {"business_id": bid}

    # ── A. Volume & velocity ───────────────────────────────────────────────
    n_rev = len(obs_reviews)
    feat["n_reviews_obs"]     = n_rev
    feat["review_velocity"]   = n_rev / OBS_MONTHS   # reviews per month

    if n_rev > 0:
        # Monthly review counts
        obs_reviews = obs_reviews.copy()
        obs_reviews["month"] = obs_reviews["date"].dt.to_period("M")
        monthly = obs_reviews.groupby("month").size()

        feat["review_velocity_slope"]     = safe_slope(monthly)
        feat["review_velocity_cv"]        = monthly.std() / (monthly.mean() + 1e-9)
        feat["months_with_zero_reviews"]  = int((monthly == 0).sum())

        # Recency: days since last review before anchor
        last_review_date = obs_reviews["date"].max()
        feat["days_since_last_review"] = (anchor - last_review_date).days
        feat["review_drought_flag"] = int(
            feat["days_since_last_review"] >= REVIEW_DROUGHT_DAYS
        )
    else:
        feat["review_velocity_slope"]    = np.nan
        feat["review_velocity_cv"]       = np.nan
        feat["months_with_zero_reviews"] = OBS_MONTHS
        feat["days_since_last_review"]   = OBS_MONTHS * 30
        feat["review_drought_flag"]      = 1

    # ── B. Rating signals ──────────────────────────────────────────────────
    if n_rev >= MIN_REVIEWS_FOR_TREND:
        stars = obs_reviews.sort_values("date")["stars"]
        feat["mean_stars_obs"]        = float(stars.mean())
        feat["std_stars_obs"]         = float(stars.std())
        feat["rating_trend_slope"]    = safe_slope(stars)
        feat["rating_trend_3m"]       = trend_early_vs_late(stars, 3)
        feat["pct_1star"]             = float((stars == 1).mean())
        feat["pct_5star"]             = float((stars == 5).mean())
        # Last 3 months vs first 3 months avg
        three_months_ago = anchor - relativedelta(months=3)
        recent = obs_reviews[obs_reviews["date"] >= three_months_ago]["stars"]
        early  = obs_reviews[obs_reviews["date"] <
                             obs_start + relativedelta(months=3)]["stars"]
        feat["stars_recent_3m"]  = float(recent.mean()) if len(recent) else np.nan
        feat["stars_early_3m"]   = float(early.mean())  if len(early)  else np.nan
        feat["stars_delta_3m"]   = (
            feat["stars_recent_3m"] - feat["stars_early_3m"]
            if not np.isnan(feat["stars_recent_3m"]) and not np.isnan(feat["stars_early_3m"])
            else np.nan
        )
    else:
        for k in ["mean_stars_obs", "std_stars_obs", "rating_trend_slope",
                  "rating_trend_3m", "pct_1star", "pct_5star",
                  "stars_recent_3m", "stars_early_3m", "stars_delta_3m"]:
            feat[k] = np.nan

    # ── C. VADER sentiment ─────────────────────────────────────────────────
    if n_rev >= MIN_REVIEWS_FOR_TREND and VADER_AVAILABLE:
        # Sample up to 200 reviews for speed (random but reproducible)
        sample = obs_reviews.sort_values("date")
        if len(sample) > 200:
            sample = sample.sample(200, random_state=42).sort_values("date")

        sample = sample.copy()
        sample["vader"] = sample["text"].apply(vader_compound)

        feat["mean_vader"]          = float(sample["vader"].mean())
        feat["vader_trend_slope"]   = safe_slope(sample["vader"])
        feat["vader_trend_3m"]      = trend_early_vs_late(sample["vader"], 3)
        feat["pct_negative_vader"]  = float((sample["vader"] < -0.05).mean())

        # Vader-stars divergence: if vader is negative but stars are high → suspicious
        stars_z = (sample["stars"] - sample["stars"].mean()) / (sample["stars"].std() + 1e-9)
        vader_z = (sample["vader"] - sample["vader"].mean()) / (sample["vader"].std() + 1e-9)
        feat["vader_stars_divergence"] = float((stars_z - vader_z).abs().mean())
    else:
        for k in ["mean_vader", "vader_trend_slope", "vader_trend_3m",
                  "pct_negative_vader", "vader_stars_divergence"]:
            feat[k] = np.nan

    # ── D. Reviewer quality signals ────────────────────────────────────────
    if n_rev > 0:
        feat["mean_review_useful"] = float(obs_reviews["useful"].mean())
        feat["mean_review_funny"]  = float(obs_reviews["funny"].mean())
        feat["mean_review_cool"]   = float(obs_reviews["cool"].mean())
        # Elite reviewer proxy: reviews with useful > median tend to be from elite
        median_useful = obs_reviews["useful"].median()
        feat["pct_high_quality_reviews"] = float(
            (obs_reviews["useful"] > median_useful).mean()
        )
    else:
        for k in ["mean_review_useful", "mean_review_funny",
                  "mean_review_cool", "pct_high_quality_reviews"]:
            feat[k] = np.nan

    # ── E. Checkin signals ─────────────────────────────────────────────────
    n_checkins = len(obs_checkins)
    feat["n_checkins_obs"] = n_checkins

    if n_checkins > 0:
        obs_checkins = obs_checkins.copy()
        obs_checkins["month"] = obs_checkins["checkin_date"].dt.to_period("M")
        monthly_ci = obs_checkins.groupby("month").size()
        feat["checkin_velocity"]       = n_checkins / OBS_MONTHS
        feat["checkin_velocity_slope"] = safe_slope(monthly_ci)
        feat["checkin_drought_flag"]   = int(
            (anchor - obs_checkins["checkin_date"].max()).days >= REVIEW_DROUGHT_DAYS
        )
    else:
        feat["checkin_velocity"]       = 0.0
        feat["checkin_velocity_slope"] = np.nan
        feat["checkin_drought_flag"]   = 1

    # ── F. Owner engagement via tips ───────────────────────────────────────
    n_tips = len(obs_tips)
    feat["n_tips_obs"] = n_tips

    if n_tips > 0:
        # Yelp tips don't flag owner responses directly, but owner tips
        # tend to be short and formal. Use compliment_count as quality proxy.
        feat["mean_tip_compliments"] = float(obs_tips["compliment_count"].mean())
        feat["tip_velocity"]         = n_tips / OBS_MONTHS
    else:
        feat["mean_tip_compliments"] = np.nan
        feat["tip_velocity"]         = 0.0

    # ── G. Business metadata (no temporal leakage) ─────────────────────────
    feat["stars_yelp_global"]   = row.get("stars_yelp", np.nan)
    feat["price_range"]         = row.get("price_range", np.nan)
    feat["open_days_per_week"]  = row.get("open_days_per_week", np.nan)

    # Category flags (binary)
    cats = str(row.get("categories", ""))
    feat["is_fast_food"]  = int("Fast Food" in cats)
    feat["is_bar"]        = int("Bars" in cats or "Bar" in cats)
    feat["is_cafe"]       = int("Cafe" in cats or "Coffee" in cats)
    feat["is_pizza"]      = int("Pizza" in cats)
    feat["is_mexican"]    = int("Mexican" in cats)
    feat["is_seafood"]    = int("Seafood" in cats)

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 3 — Feature Engineering (leakage-safe)")
    print("=" * 60)

    labeled  = pd.read_parquet(PROC_DIR / "labeled_businesses.parquet")
    reviews  = pd.read_parquet(PROC_DIR / "reviews.parquet")

    checkins_path = PROC_DIR / "checkins.parquet"
    tips_path     = PROC_DIR / "tips.parquet"
    checkins = pd.read_parquet(checkins_path) if checkins_path.exists() else pd.DataFrame()
    tips     = pd.read_parquet(tips_path)     if tips_path.exists()     else pd.DataFrame()

    print(f"\n  Building features for {len(labeled):,} businesses...")

    # Pre-group for O(1) per-business lookup
    rev_groups  = reviews.groupby("business_id")
    ci_groups   = checkins.groupby("business_id") if not checkins.empty else None
    tip_groups  = tips.groupby("business_id")     if not tips.empty     else None

    all_feats = []
    for _, row in tqdm(labeled.iterrows(), total=len(labeled), desc="Feature eng"):
        bid    = row["business_id"]
        anchor = row["anchor_date"]
        obs_s  = row["obs_start"]

        # Filter to observation window (strict: date < anchor)
        obs_rev = (
            rev_groups.get_group(bid)
            if bid in rev_groups.groups else pd.DataFrame()
        )
        if not obs_rev.empty:
            obs_rev = obs_rev[(obs_rev["date"] >= obs_s) & (obs_rev["date"] < anchor)]

        obs_ci = pd.DataFrame()
        if ci_groups and bid in ci_groups.groups:
            obs_ci = ci_groups.get_group(bid)
            obs_ci = obs_ci[(obs_ci["checkin_date"] >= obs_s) & (obs_ci["checkin_date"] < anchor)]

        obs_tip = pd.DataFrame()
        if tip_groups and bid in tip_groups.groups:
            obs_tip = tip_groups.get_group(bid)
            obs_tip = obs_tip[(obs_tip["date"] >= obs_s) & (obs_tip["date"] < anchor)]

        feat = build_features_one(row, obs_rev, obs_ci, obs_tip)
        feat[TARGET_COL]    = row[TARGET_COL]
        feat["anchor_date"] = anchor
        feat["city"]        = row["city"]
        feat["state"]       = row["state"]
        all_feats.append(feat)

    features = pd.DataFrame(all_feats)

    # ── Report coverage ────────────────────────────────────────────────────
    print(f"\n  Feature matrix shape: {features.shape}")
    null_pct = features.isnull().mean().sort_values(ascending=False)
    high_null = null_pct[null_pct > 0.30]
    if not high_null.empty:
        print("\n  High-null features (>30%):")
        for col, pct in high_null.items():
            print(f"    {col:40s}  {pct:.0%} null")

    features.to_parquet(PROC_DIR / "features.parquet", index=False)
    print(f"\n  Saved → {PROC_DIR}/features.parquet")

    # ── Quick sanity check ─────────────────────────────────────────────────
    print(f"\n  Label balance in feature set:")
    print(f"    Closed: {features[TARGET_COL].sum():,}  ({features[TARGET_COL].mean():.1%})")
    print(f"    Open  : {(features[TARGET_COL]==0).sum():,}")


if __name__ == "__main__":
    main()
