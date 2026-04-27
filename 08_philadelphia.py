"""
08_philadelphia.py -- Zero-shot Philadelphia generalization.

Loads raw Yelp data, filters for Philadelphia restaurants,
engineers features, applies the Tampa-trained XGBoost model,
and compares metrics to Tampa test set.

Run:
    python 08_philadelphia.py

Runtime: ~30-40 min (VADER is slow).
Output:  data/processed_philly/ (parquets)
         figures/18_tampa_vs_philly_comparison.png
         models/philadelphia_results.json
"""
import json
import warnings
warnings.filterwarnings("ignore")

import importlib
_fe = importlib.import_module("03_feature_engineering")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, precision_recall_curve,
)
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from config_00 import (
    RAW_DIR, MODEL_DIR, FIG_DIR,
    OBS_MONTHS, OUTCOME_MONTHS, EARLIEST_ANCHOR,
    TARGET_COL,
)

# Cap Philly anchors before COVID distortion in review density
LATEST_ANCHOR = "2020-06-01"

PHILLY_DIR = Path("data/processed_philly")
PHILLY_DIR.mkdir(parents=True, exist_ok=True)

PHILLY_CITIES = {"Philadelphia"}
OPT_THRESHOLD = 0.2704   # Tampa-trained threshold
ENCODING = "utf-8"

plt.rcParams.update({"font.family": "serif", "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})


# ── Load and filter businesses ──────────────────────────────────────────────

def load_philly_businesses() -> pd.DataFrame:
    print("  Loading businesses...")
    path = RAW_DIR / "yelp_academic_dataset_business.json"
    rows = []
    with open(path, encoding=ENCODING) as f:
        for line in f:
            r = json.loads(line)
            if r.get("city") not in PHILLY_CITIES:
                continue
            cats = r.get("categories") or ""
            if "Restaurants" not in cats and "Food" not in cats:
                continue
            hours = r.get("hours") or {}
            open_days = sum(1 for v in hours.values() if v) if hours else None
            price_raw = (r.get("attributes") or {}).get("RestaurantsPriceRange2")
            try:
                price_range = float(price_raw) if price_raw else None
            except (ValueError, TypeError):
                price_range = None
            rows.append({
                "business_id":       r["business_id"],
                "name":              r.get("name", ""),
                "city":              r.get("city", ""),
                "state":             r.get("state", ""),
                "is_open":           int(r.get("is_open", 1)),
                "stars_yelp":        float(r.get("stars", 0)),
                "review_count_yelp": int(r.get("review_count", 0)),
                "price_range":       price_range,
                "open_days_per_week": open_days,
                "categories":        cats,
                "latitude":          r.get("latitude"),
                "longitude":         r.get("longitude"),
            })
    df = pd.DataFrame(rows)
    print(f"    Philadelphia restaurants found: {len(df):,}")
    return df


# ── Labeling ────────────────────────────────────────────────────────────────

def build_labels_philly(biz: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Assign anchor dates and closure labels using same logic as 02_build_labels.py."""
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

        last_review_ever = rev["date"].max()
        # Mirror 02_build_labels.py build_label() — review recency only, not is_open
        if last_review_ever > outcome_end + relativedelta(months=3):
            closed = 0
        elif last_review_ever <= outcome_end:
            closed = 1
        else:
            closed = 0  # ambiguous window

        rows.append({
            "business_id":        bid,
            "name":               b["name"],
            "city":               b["city"],
            "state":              b["state"],
            "anchor_date":        anchor,
            "obs_start":          obs_start,
            "outcome_end":        outcome_end,
            "closed_within_6m":   closed,
            "stars_yelp":         b["stars_yelp"],
            "price_range":        b["price_range"],
            "open_days_per_week": b["open_days_per_week"],
            "categories":         b["categories"],
        })

    return pd.DataFrame(rows)


# ── Feature engineering ─────────────────────────────────────────────────────

def build_features_philly(
    labeled: pd.DataFrame, reviews: pd.DataFrame,
    checkins: pd.DataFrame, tips: pd.DataFrame,
) -> pd.DataFrame:
    """Call build_features_one from 03_feature_engineering for each Philly restaurant."""
    build_one = _fe.build_features_one

    rev_groups = reviews.groupby("business_id")
    ci_groups  = checkins.groupby("business_id") if not checkins.empty else None
    tip_groups = tips.groupby("business_id")     if not tips.empty     else None

    records = []
    for _, row in tqdm(labeled.iterrows(), total=len(labeled), desc="Philly features"):
        bid    = row["business_id"]
        anchor = row["anchor_date"]
        obs_s  = row["obs_start"]

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

        feat = build_one(row, obs_rev, obs_ci, obs_tip)
        feat[TARGET_COL]    = row[TARGET_COL]
        feat["anchor_date"] = anchor
        feat["city"]        = row["city"]
        feat["state"]       = row["state"]
        records.append(feat)

    return pd.DataFrame(records)


# ── Prediction and comparison ───────────────────────────────────────────────

def predict_and_compare(features: pd.DataFrame, model) -> dict:
    """Apply Tampa model to Philly features and compute metrics."""
    feat_cols = list(model.feature_names_in_)
    missing = [c for c in feat_cols if c not in features.columns]
    if missing:
        print(f"    WARNING: {len(missing)} features missing from Philly data: {missing}")

    X = features.reindex(columns=feat_cols).fillna(features.reindex(columns=feat_cols).median())
    y_true = features[TARGET_COL].values
    y_prob = model.predict_proba(X)[:, 1]

    auc_pr  = average_precision_score(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)

    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s      = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    opt_thr  = float(thr[best_idx])
    opt_f1   = float(f1s[best_idx])

    y_pred_fixed = (y_prob >= OPT_THRESHOLD).astype(int)
    f1_fixed     = f1_score(y_true, y_pred_fixed)

    n_closed     = int(y_true.sum())
    flagged      = int((y_prob >= OPT_THRESHOLD).sum())
    caught       = int(((y_prob >= OPT_THRESHOLD) & (y_true == 1)).sum())
    recall_fixed = caught / n_closed if n_closed else 0.0

    return {
        "city":                       "Philadelphia",
        "n_restaurants":              len(features),
        "n_closed":                   n_closed,
        "base_rate":                  round(float(y_true.mean()), 4),
        "AUC_PR":                     round(auc_pr, 4),
        "AUC_ROC":                    round(auc_roc, 4),
        "opt_threshold_local":        round(opt_thr, 4),
        "opt_f1_local":               round(opt_f1, 4),
        "f1_at_tampa_threshold":      round(f1_fixed, 4),
        "flagged_at_tampa_threshold": flagged,
        "caught_at_tampa_threshold":  caught,
        "recall_at_tampa_threshold":  round(recall_fixed, 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 8 -- Philadelphia Zero-Shot Generalization")
    print("=" * 60)

    # ── 1. Load businesses (skip if parquet already exists) ───────────────
    if (PHILLY_DIR / "businesses.parquet").exists():
        print("  Loading businesses from checkpoint...")
        biz = pd.read_parquet(PHILLY_DIR / "businesses.parquet")
        print(f"    Loaded businesses.parquet ({len(biz):,} rows)")
    else:
        biz = load_philly_businesses()
        biz.to_parquet(PHILLY_DIR / "businesses.parquet", index=False)
        print(f"    Saved businesses.parquet ({len(biz):,} rows)")

    # ── 2. Load reviews, checkins, tips (skip if parquets already exist) ──
    philly_bids = set(biz["business_id"])

    if (PHILLY_DIR / "reviews.parquet").exists():
        print("  Loading reviews from checkpoint...")
        reviews = pd.read_parquet(PHILLY_DIR / "reviews.parquet")
        checkins = pd.read_parquet(PHILLY_DIR / "checkins.parquet")
        tips = pd.read_parquet(PHILLY_DIR / "tips.parquet")
        print(f"    reviews={len(reviews):,}  checkins={len(checkins):,}  tips={len(tips):,}")
    else:
        print("  Loading reviews (this takes ~2 min)...")
        rev_rows = []
        with open(RAW_DIR / "yelp_academic_dataset_review.json", encoding=ENCODING) as f:
            for line in f:
                r = json.loads(line)
                if r["business_id"] not in philly_bids:
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
        reviews.to_parquet(PHILLY_DIR / "reviews.parquet", index=False)
        print(f"    Saved reviews.parquet ({len(reviews):,} rows)")

        print("  Loading checkins...")
        ci_rows = []
        with open(RAW_DIR / "yelp_academic_dataset_checkin.json", encoding=ENCODING) as f:
            for line in f:
                r = json.loads(line)
                if r["business_id"] not in philly_bids:
                    continue
                for ts in r.get("date", "").split(", "):
                    ts = ts.strip()
                    if ts:
                        ci_rows.append({"business_id": r["business_id"],
                                        "checkin_date": pd.Timestamp(ts)})
        checkins = pd.DataFrame(ci_rows)
        checkins.to_parquet(PHILLY_DIR / "checkins.parquet", index=False)
        print(f"    Saved checkins.parquet ({len(checkins):,} rows)")

        print("  Loading tips...")
        tip_rows = []
        with open(RAW_DIR / "yelp_academic_dataset_tip.json", encoding=ENCODING) as f:
            for line in f:
                r = json.loads(line)
                if r["business_id"] not in philly_bids:
                    continue
                tip_rows.append({"business_id": r["business_id"],
                                 "date": pd.Timestamp(r["date"]),
                                 "compliment_count": int(r.get("compliment_count", 0))})
        tips = pd.DataFrame(tip_rows)
        tips.to_parquet(PHILLY_DIR / "tips.parquet", index=False)
        print(f"    Saved tips.parquet ({len(tips):,} rows)")

    # ── 3. Build labels ────────────────────────────────────────────────────
    print("  Building labels...")
    labeled = build_labels_philly(biz, reviews)
    labeled.to_parquet(PHILLY_DIR / "labeled_businesses.parquet", index=False)
    n_closed = int(labeled["closed_within_6m"].sum())
    print(f"    Labeled: {len(labeled):,} restaurants, {n_closed} closed ({n_closed/len(labeled):.1%})")

    if len(labeled) < 50:
        print("  ERROR: Too few labeled restaurants. Check anchor date range.")
        return

    # ── 4. Build features (~20 min due to VADER) ──────────────────────────
    print("  Engineering features (VADER is slow, ~20 min)...")
    features = build_features_philly(labeled, reviews, checkins, tips)
    features.to_parquet(PHILLY_DIR / "features.parquet", index=False)
    print(f"    Features shape: {features.shape}")

    # ── 5. Apply Tampa model ───────────────────────────────────────────────
    print("  Loading Tampa-trained XGBoost model...")
    model = joblib.load(MODEL_DIR / "xgboost.pkl")

    print("  Running zero-shot prediction on Philadelphia...")
    philly_results = predict_and_compare(features, model)

    # Tampa test set for comparison: filter features.parquet to test business_ids
    # (ensemble_predictions.parquet contains the 1,244 test-set businesses)
    tampa_preds = pd.read_parquet(Path("data/processed/ensemble_predictions.parquet"))
    tampa_feat  = pd.read_parquet(Path("data/processed/features.parquet"))
    test_bids   = set(tampa_preds["business_id"])
    tampa_test  = tampa_feat[tampa_feat["business_id"].isin(test_bids)].copy()

    feat_cols = [c for c in model.feature_names_in_ if c in tampa_test.columns]
    X_tampa   = tampa_test[feat_cols].fillna(tampa_test[feat_cols].median())
    y_true_t  = tampa_test[TARGET_COL].values
    y_prob_t  = model.predict_proba(X_tampa)[:, 1]

    tampa_results = {
        "city":                  "Tampa Bay",
        "n_restaurants":         len(tampa_test),
        "n_closed":              int(y_true_t.sum()),
        "base_rate":             round(float(y_true_t.mean()), 4),
        "AUC_PR":                round(float(average_precision_score(y_true_t, y_prob_t)), 4),
        "AUC_ROC":               round(float(roc_auc_score(y_true_t, y_prob_t)), 4),
        "f1_at_tampa_threshold": round(float(f1_score(y_true_t, (y_prob_t >= OPT_THRESHOLD).astype(int))), 4),
    }

    # ── 6. Print comparison ────────────────────────────────────────────────
    print("\n  === TAMPA vs PHILADELPHIA ZERO-SHOT COMPARISON ===")
    print(f"  {'Metric':35s} {'Tampa':>10} {'Philadelphia':>14}")
    print("  " + "-" * 62)
    for key in ["n_restaurants", "n_closed", "base_rate", "AUC_PR", "AUC_ROC", "f1_at_tampa_threshold"]:
        t_val = tampa_results.get(key, "--")
        p_val = philly_results.get(key, "--")
        print(f"  {key:35s} {str(t_val):>10} {str(p_val):>14}")

    # ── 7. Save results ────────────────────────────────────────────────────
    combined = {"tampa": tampa_results, "philadelphia": philly_results}
    out_path = MODEL_DIR / "philadelphia_results.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Saved results -> {out_path}")

    # ── 8. Comparison chart ────────────────────────────────────────────────
    metrics_to_plot = ["AUC_PR", "AUC_ROC", "f1_at_tampa_threshold"]
    t_vals = [tampa_results[m] for m in metrics_to_plot]
    p_vals = [philly_results[m] for m in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, t_vals, width, label="Tampa Bay (train)", color="#2E86AB", alpha=0.85)
    ax.bar(x + width/2, p_vals, width, label="Philadelphia (zero-shot)", color="#E84855", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=10)
    ax.set_title("Tampa vs Philadelphia -- Zero-Shot Generalization", fontweight="bold")
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    p = FIG_DIR / "18_tampa_vs_philly_comparison.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p}")


if __name__ == "__main__":
    main()
