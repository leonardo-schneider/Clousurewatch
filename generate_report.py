"""
generate_report.py
------------------
Generates PROJECT_REPORT.pdf in the project root.
Run: python generate_report.py
"""

from fpdf import FPDF
from pathlib import Path

OUT = Path(__file__).parent / "PROJECT_REPORT.pdf"

NAVY   = (26,  26,  46)
BLUE   = (15,  52,  96)
RED    = (233, 69,  96)
ORANGE = (247, 164, 64)
GREEN  = (76,  175, 80)
WHITE  = (238, 238, 238)
LGRAY  = (200, 200, 200)


class PDF(FPDF):

    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 18, "F")
        self.set_y(4)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*RED)
        self.cell(0, 10, "ClosureWatch  |  Restaurant Failure Prediction", align="C")

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*LGRAY)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # ── Helpers ────────────────────────────────────────────────────────────

    def section(self, title):
        self.ln(4)
        self.set_fill_color(*BLUE)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, f"  {title}", fill=True, ln=True)
        self.set_text_color(40, 40, 40)
        self.ln(2)

    def body(self, text, size=10):
        self.set_font("Helvetica", "", size)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, items, indent=8):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        for item in items:
            self.set_x(indent)
            self.multi_cell(0, 5.5, f"-  {item}")
        self.ln(1)

    def kv(self, key, value, key_color=BLUE, val_color=(40,40,40)):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*key_color)
        self.cell(60, 6, key)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*val_color)
        self.multi_cell(0, 6, value)

    def table_row(self, cols, widths, bold=False, bg=None):
        if bg:
            self.set_fill_color(*bg)
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 9)
        self.set_text_color(40, 40, 40)
        for col, w in zip(cols, widths):
            fill = bg is not None
            self.cell(w, 6, str(col), border=1, fill=fill)
        self.ln()

    def colored_badge(self, text, color):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*color)
        self.cell(0, 6, text, ln=True)
        self.set_text_color(40, 40, 40)

    def code(self, lines):
        self.set_fill_color(230, 230, 230)
        self.set_font("Courier", "", 8.5)
        self.set_text_color(30, 30, 30)
        for line in lines:
            self.set_x(10)
            self.cell(0, 5, line, fill=True, ln=True)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.ln(1)


# ── Build PDF ──────────────────────────────────────────────────────────────────

pdf = PDF(orientation="P", unit="mm", format="A4")
pdf.set_auto_page_break(auto=True, margin=16)
pdf.add_page()
pdf.set_margins(14, 20, 14)

# ── Cover / Title ──────────────────────────────────────────────────────────────
pdf.ln(6)
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(*RED)
pdf.cell(0, 12, "ClosureWatch", align="C", ln=True)

pdf.set_font("Helvetica", "B", 14)
pdf.set_text_color(*BLUE)
pdf.cell(0, 8, "Restaurant Failure Prediction - Project Report", align="C", ln=True)

pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 6, "ML Final Project | Tampa Bay, FL | Yelp Academic Dataset", align="C", ln=True)
pdf.ln(4)

# ── 1. Overview ────────────────────────────────────────────────────────────────
pdf.section("1. Project Overview")
pdf.body(
    "ClosureWatch predicts whether a Tampa Bay restaurant will permanently close "
    "within 6 months, using 12 months of behavioral signals from the Yelp Academic "
    "Dataset.  The model is framed as a binary classification problem aimed at "
    "SME credit underwriting (e.g. Kabbage / BlueVine) - predicting business "
    "failure from public behavioral data rather than private financials."
)
pdf.bullet([
    "Dataset: Yelp Academic Dataset (public, not Kaggle)",
    "Geography: Tampa Bay metro area (then all of Florida if < 200 closures)",
    "Label: closed_within_6m  (binary, 6.3% positive rate)",
    "5,143 restaurants  |  326 closures  |  Test set: 1,244 businesses",
])

# ── 2. Pipeline ────────────────────────────────────────────────────────────────
pdf.section("2. Pipeline - Run Order")
pdf.body("All scripts must be run in numerical order. Each depends on the previous output.")
pdf.code([
    "python 01_load_filter.py          # Load Yelp JSON -> filtered Parquet (~5-10 min)",
    "python 02_build_labels.py         # Anchor dates + leakage-safe binary labels",
    "python 03_feature_engineering.py  # Time-windowed features (VADER, velocity, trend)",
    "python 04_eda.py                  # EDA plots -> figures/",
    "python 05_modeling.py             # 5 models with CV + test evaluation",
    "python 06_ensemble.py             # Ensemble strategy ladder (NEW)",
    "streamlit run app.py              # Launch interactive dashboard (NEW)",
])

# ── 3. Leakage-Safe Design ────────────────────────────────────────────────────
pdf.section("3. Temporal Design - Anti-Leakage Rules")
pdf.body(
    "The most critical correctness constraint.  Each restaurant gets an 'anchor date' "
    "(its 80th-percentile review date). Features use only data before that date; "
    "the label is determined by what happens in the 6 months after."
)
pdf.bullet([
    "Features use ONLY data where date < anchor_date",
    "Anchor = 80th-pct review date (not last review - avoids trivial leak)",
    "Imputation medians computed on training fold only",
    "Train/test split is time-based (most recent 15% by anchor date = test)",
    "Cross-validation uses expanding-window folds, not random k-fold",
])

# ── 4. Feature Engineering ────────────────────────────────────────────────────
pdf.section("4. Feature Engineering  (03_feature_engineering.py)")
pdf.body("Top predictive signals extracted from Yelp behavioral data:")

widths = [80, 30, 70]
pdf.table_row(["Feature", "Correlation", "Description"], widths, bold=True, bg=(220,220,240))
rows = [
    ("months_with_zero_reviews", "0.56", "Months in obs. window with no reviews"),
    ("days_since_last_review",   "0.56", "Days from last review to anchor date"),
    ("review_drought_flag",      "0.43", "Binary: drought threshold exceeded"),
    ("checkin_drought_flag",     "0.34", "Binary: check-in drought triggered"),
    ("pct_5star",                "0.29", "Fraction of reviews that are 5-star"),
]
for r in rows:
    pdf.table_row(r, widths)
pdf.ln(2)

# ── 5. Models ─────────────────────────────────────────────────────────────────
pdf.section("5. Models Trained  (05_modeling.py)")
pdf.body(
    "Five classifiers trained on the time-based train split and evaluated on "
    "the held-out test set. Primary metric: AUC-PR (area under precision-recall "
    "curve), chosen because the dataset is highly imbalanced (6.3% positive)."
)

widths2 = [58, 26, 26, 26, 28]
pdf.table_row(
    ["Model", "CV AUC-PR", "Test AUC-PR", "AUC-ROC", "Test F1"],
    widths2, bold=True, bg=(220,220,240)
)
model_rows = [
    ("XGBoost (winner)", "0.124 ± 0.039", "0.2069", "0.7002", "0.2448"),
    ("Random Forest",    "0.112 ± 0.054", "0.1994", "0.6948", "0.1347"),
    ("LightGBM",         "0.091 ± 0.021", "0.1977", "0.6699", "0.2093"),
    ("MLP Neural Net",   "0.083 ± 0.010", "0.1592", "0.6187", "0.1622"),
    ("Logistic Regr.",   "0.106 ± 0.035", "0.1574", "0.6458", "0.2553"),
]
for i, r in enumerate(model_rows):
    bg = (240, 248, 240) if i == 0 else None
    pdf.table_row(r, widths2, bg=bg)

pdf.ln(2)
pdf.body("Base rate (random AUC-PR): 0.063  |  XGBoost is 3.3x better than random")

# ── 6. Ensemble ───────────────────────────────────────────────────────────────
pdf.section("6. Ensemble Strategy  (06_ensemble.py)  - NEW")
pdf.body(
    "A strategy ladder that tries three approaches in order and stops at the "
    "first one that beats the XGBoost baseline (AUC-PR = 0.2069)."
)
pdf.bullet([
    "Strategy 1 - Simple Average: equal-weight average of all model probabilities",
    "Strategy 2 - Weighted Average: weighted by CV AUC-PR score per model",
    "Strategy 3 - Stacking: meta-LogisticRegression trained on OOF predictions",
])
pdf.body(
    "The script is resilient to sklearn version mismatches: if a model's "
    "predict_proba() fails (e.g. due to a version bump), it is skipped with "
    "a warning and the ensemble continues with the remaining models."
)
pdf.body("Outputs:")
pdf.bullet([
    "models/ensemble_results.json         - metrics for each strategy + winner",
    "data/processed/ensemble_predictions.parquet  - risk scores per restaurant",
])

# ── 7. Streamlit App ──────────────────────────────────────────────────────────
pdf.section("7. Streamlit Dashboard  (app.py)  - NEW")
pdf.body(
    "An interactive dark-themed dashboard built with Streamlit + Plotly. "
    "Launch with:  streamlit run app.py"
)
pdf.bullet([
    "Sidebar: all Tampa Bay restaurants sorted by predicted closure risk (%)",
    "Color-coded badges: RED >= 60%  |  ORANGE 30-59%  |  GREEN < 30%",
    "Search bar: case-insensitive filter by restaurant name",
    "Detail panel: risk gauge, key warning signals, Plotly feature bar chart",
    "Feature bar chart: percentile-normalized contributions (red = risk-increasing, "
    "green = risk-decreasing)",
    "Data source: data/processed/ensemble_predictions.parquet",
])

# ── 8. File Structure ─────────────────────────────────────────────────────────
pdf.section("8. File Structure")
pdf.code([
    "Desktop/Final ML/",
    "  config_00.py                  Global constants (paths, dates, hyperparams)",
    "  01_load_filter.py             Yelp JSON -> filtered Parquet",
    "  02_build_labels.py            Anchor dates + binary labels",
    "  03_feature_engineering.py     Time-windowed feature extraction",
    "  04_eda.py                     EDA plots",
    "  05_modeling.py                5 models, CV, test evaluation",
    "  06_ensemble.py                Ensemble strategy ladder  [NEW]",
    "  app.py                        Streamlit dashboard        [NEW]",
    "  generate_report.py            This report generator      [NEW]",
    "  PROJECT_REPORT.pdf            This document              [NEW]",
    "  requirements.txt              Python dependencies",
    "  CLAUDE.md                     Project guide for Claude Code",
    "  .streamlit/config.toml        Dark theme config",
    "  data/raw/                     Yelp source JSON files (not committed)",
    "  data/processed/               Auto-generated Parquet files",
    "  models/                       Saved .pkl files + results JSON",
    "  figures/                      EDA plots (auto-generated)",
    "  tests/                        pytest unit tests",
    "    test_ensemble.py            Tests for 06_ensemble.py math functions",
    "    test_app_helpers.py         Tests for app.py helper functions",
    "  docs/superpowers/",
    "    specs/  2026-04-16-restaurant-closure-ui-ensemble-design.md",
    "    plans/  2026-04-16-ensemble-and-ui.md",
])

# ── 9. Tests ──────────────────────────────────────────────────────────────────
pdf.section("9. Unit Tests")
pdf.body("Run with:  pytest tests/ -v")
pdf.bullet([
    "test_ensemble.py  (8 tests): simple_average, weighted_average, compute_metrics",
    "test_app_helpers.py  (14 tests): risk_color, risk_label, risk_badge, percentile_rank",
])

# ── 10. How to Run ────────────────────────────────────────────────────────────
pdf.section("10. Quick-Start Guide")
pdf.body("Prerequisites: Anaconda Python environment with the packages in requirements.txt")
pdf.code([
    "# 1. Place Yelp JSON files in data/raw/",
    "# 2. Run the full pipeline",
    "python 01_load_filter.py",
    "python 02_build_labels.py",
    "python 03_feature_engineering.py",
    "python 04_eda.py",
    "python 05_modeling.py",
    "python 06_ensemble.py",
    "",
    "# 3. Launch the dashboard",
    "streamlit run app.py",
    "",
    "# 4. Run tests",
    "pytest tests/ -v",
])

pdf.output(str(OUT))
print(f"Saved: {OUT}")
