"""
13_calibrate_model.py -- Calibrate the global XGBoost model with isotonic regression.

Uses a time-based 20% hold-out of all 9 metros as the calibration set.
Saves models/xgboost_global_calibrated.pkl.

Run after Task 4 (LOMO re-run):
    python 13_calibrate_model.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from app_helpers import add_null_flags

from config_00 import MODEL_DIR, FIG_DIR

LATEST_ANCHOR = pd.Timestamp("2020-06-01")

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

META_COLS = {"business_id", "closed_within_6m", "anchor_date",
             "city", "state", "metro"}


def load_all() -> pd.DataFrame:
    frames = []
    for metro, ddir in METROS.items():
        df = pd.read_parquet(Path(ddir) / "features.parquet")
        df = add_null_flags(df)
        emb_path = Path(ddir) / "review_embeddings.parquet"
        if emb_path.exists():
            emb = pd.read_parquet(emb_path)
            df  = df.merge(emb, on="business_id", how="left")
        df["metro"]       = metro
        df["anchor_date"] = pd.to_datetime(df["anchor_date"])
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    return all_df[all_df["anchor_date"] <= LATEST_ANCHOR].copy()


def time_split(df: pd.DataFrame, val_frac: float = 0.20):
    df_s  = df.sort_values("anchor_date")
    n_cal = max(1, int(len(df_s) * val_frac))
    return df_s.iloc[:-n_cal].copy(), df_s.iloc[-n_cal:].copy()


def plot_calibration(y_cal, uncal_prob, cal_prob, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.rcParams.update({"font.family": "serif", "figure.dpi": 150})

    for label, prob, color in [
        ("Uncalibrated XGB", uncal_prob, "#2E86AB"),
        ("Calibrated XGB",   cal_prob,   "#1DB954"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_cal, prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2,
                label=label, color=color)

    ax.plot([0, 1], [0, 1], linestyle="--", color="#888",
            linewidth=1.2, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability", fontsize=10)
    ax.set_ylabel("Fraction of positives", fontsize=10)
    ax.set_title("Calibration Curve (Reliability Diagram)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Calibration curve saved -> {out_path}")


def main():
    print("=" * 60)
    print("STEP 13 -- Calibrate Global XGBoost Model")
    print("=" * 60)

    print("\n[1] Loading all 9 metros...")
    all_df    = load_all()
    _, cal_df = time_split(all_df, val_frac=0.20)
    print(f"  Calibration set: {len(cal_df):,} restaurants  "
          f"({cal_df['closed_within_6m'].mean():.1%} closure rate)")

    print("\n[2] Loading uncalibrated global model...")
    base_model = joblib.load(Path(MODEL_DIR) / "xgboost_global.pkl")
    xgb_feat   = base_model.get_booster().feature_names

    medians    = all_df.reindex(columns=xgb_feat).median()
    X_cal      = cal_df.reindex(columns=xgb_feat).fillna(medians)
    y_cal      = cal_df["closed_within_6m"].values

    uncal_prob = base_model.predict_proba(X_cal)[:, 1]
    print(f"  Uncalibrated mean predicted prob: {uncal_prob.mean():.4f}  "
          f"(true rate: {y_cal.mean():.4f})")

    print("\n[3] Fitting isotonic calibration on calibration set...")
    calibrated = CalibratedClassifierCV(
        estimator=base_model, cv="prefit", method="isotonic"
    )
    calibrated.fit(X_cal, y_cal)

    cal_prob = calibrated.predict_proba(X_cal)[:, 1]
    print(f"  Calibrated   mean predicted prob: {cal_prob.mean():.4f}  "
          f"(true rate: {y_cal.mean():.4f})")

    out_path = Path(MODEL_DIR) / "xgboost_global_calibrated.pkl"
    joblib.dump(calibrated, out_path)
    print(f"\n[4] Saved -> {out_path}")

    print("\n[5] Plotting reliability diagram...")
    plot_calibration(
        y_cal, uncal_prob, cal_prob,
        Path(FIG_DIR) / "30_calibration_curve.png",
    )

    print("\nDone. Update app.py to prefer xgboost_global_calibrated.pkl.")


if __name__ == "__main__":
    main()
