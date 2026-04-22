"""
04b_data_quality.py — Data quality audit for features.parquet.
Reads data/processed/features.parquet, runs 5 checks, prints report,
saves data/processed/data_quality_report.json.

Run:
    python 04b_data_quality.py
"""
import json
import pandas as pd
from pathlib import Path

from config_00 import PROC_DIR
from data_quality import (
    check_duplicates,
    add_covid_flag,
    check_price_range,
    detect_outliers,
    validate_schema,
)

OUTLIER_COLS = [
    "n_reviews_obs", "n_checkins_obs", "days_since_last_review",
    "review_velocity", "checkin_velocity",
]


def main():
    print("=" * 60)
    print("STEP 4b -- Data Quality Audit")
    print("=" * 60)

    df = pd.read_parquet(PROC_DIR / "features.parquet")
    df["anchor_date"] = pd.to_datetime(df["anchor_date"])
    print(f"  Loaded features.parquet: {df.shape[0]:,} rows x {df.shape[1]} cols\n")

    report = {}

    # ── 1a Duplicates ─────────────────────────────────────────────────────
    dup = check_duplicates(df)
    report["duplicates"] = dup
    status = "OK" if dup["ok"] else f"FAIL ({dup['n_duplicates']} duplicates)"
    print(f"[1a] Duplicates by business_id: {status}")

    # ── 1b COVID flag ─────────────────────────────────────────────────────
    df = add_covid_flag(df)
    n_covid = int(df["covid_flag"].sum())
    report["covid"] = {
        "n_covid_anchors": n_covid,
        "pct_covid": round(n_covid / len(df), 4),
        "n_non_covid": len(df) - n_covid,
    }
    print(f"[1b] COVID-period anchors (2020-03 to 2021-06): "
          f"{n_covid:,} ({n_covid/len(df):.1%})")
    closed_covid = int(df[df["covid_flag"] == 1]["closed_within_6m"].sum())
    closed_non   = int(df[df["covid_flag"] == 0]["closed_within_6m"].sum())
    rate_covid = closed_covid / n_covid if n_covid else 0
    rate_non   = closed_non / (len(df) - n_covid) if (len(df) - n_covid) else 0
    report["covid"]["closure_rate_covid"] = round(rate_covid, 4)
    report["covid"]["closure_rate_non_covid"] = round(rate_non, 4)
    print(f"       Closure rate COVID cohort:     {rate_covid:.1%}")
    print(f"       Closure rate non-COVID cohort: {rate_non:.1%}")

    # ── 1c price_range ────────────────────────────────────────────────────
    pr = check_price_range(df)
    report["price_range"] = pr
    status = "OK" if pr["ok"] else f"FAIL -- {pr['n_invalid']} invalid values: {pr['invalid_values']}"
    print(f"[1c] price_range validity: {status}")

    # ── 1d Outliers ────────────────────────────────────────────────────────
    outliers = detect_outliers(df, OUTLIER_COLS, k=3.0)
    report["outliers"] = {
        "n_outlier_rows": len(outliers),
        "by_feature": outliers.groupby("feature").size().to_dict() if not outliers.empty else {},
    }
    print(f"[1d] Outliers (IQR k=3): {len(outliers)} flagged rows")
    if not outliers.empty:
        for feat, count in outliers.groupby("feature").size().items():
            thr = outliers[outliers["feature"] == feat]["threshold"].iloc[0]
            print(f"       {feat}: {count} rows above {thr:.1f}")

    # ── 1e Schema validation ───────────────────────────────────────────────
    issues = validate_schema(df)
    report["schema"] = {"issues": issues, "ok": len(issues) == 0}
    status = "OK" if not issues else "FAIL\n  " + "\n  ".join(issues)
    print(f"[1e] Schema validation: {status}")

    # ── Save report ────────────────────────────────────────────────────────
    out_path = PROC_DIR / "data_quality_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved -> {out_path}")


if __name__ == "__main__":
    main()
