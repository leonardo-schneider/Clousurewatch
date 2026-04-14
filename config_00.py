"""
Project Configuration
Restaurant Failure Prediction — Yelp Academic Dataset
University ML Final Project
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DIR   = Path("data/raw")          # drop Yelp JSON files here
PROC_DIR  = Path("data/processed")
MODEL_DIR = Path("models")
FIG_DIR   = Path("figures")

for d in [RAW_DIR, PROC_DIR, MODEL_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Geography ──────────────────────────────────────────────────────────────
# Start Tampa; expand to all FL if n_closed < MIN_CLOSED_THRESHOLD
PRIMARY_CITIES  = ["Tampa"]
FALLBACK_STATES = ["FL"]
MIN_CLOSED_THRESHOLD = 200   # minimum closed restaurants needed

# ── Temporal design ────────────────────────────────────────────────────────
# For each business we pick an ANCHOR DATE.
# Observation window  : [anchor - 12 months, anchor)
# Outcome window      : [anchor, anchor + 6 months)
# Label = 1 if business is marked closed AND last review/checkin ≤ anchor+6m
OBS_MONTHS     = 12
OUTCOME_MONTHS = 6

# Latest anchor we allow (leave room for outcome window in the dataset)
# Yelp academic dataset typically runs through ~2022; adjust as needed.
LATEST_ANCHOR  = "2021-06-01"   # anchor + 6 months must be < dataset end
EARLIEST_ANCHOR = "2016-01-01"  # need 12 months of history before anchor

# ── Modeling ───────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
N_FOLDS       = 5
TEST_FRAC     = 0.15   # held-out final test set (time-based split)
VAL_FRAC      = 0.15   # within train, used for early stopping

TARGET_COL    = "closed_within_6m"
PRIMARY_METRIC = "average_precision"  # AUC-PR

LGBM_EARLY_STOPPING = 100

# ── VADER / NLP ────────────────────────────────────────────────────────────
MIN_REVIEWS_FOR_TREND = 4   # need at least N reviews to compute trend features
REVIEW_DROUGHT_DAYS   = 90  # flag if no reviews in last N days of obs window
