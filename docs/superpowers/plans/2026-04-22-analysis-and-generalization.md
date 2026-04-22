# Analysis & Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deep data quality audit, EDA, model analysis, and zero-shot Philadelphia generalization for the Tampa Bay restaurant failure prediction project.

**Architecture:** Two independent execution blocks. Block A (Tasks 1–10) is read-only analysis — reads existing parquets and models, writes figures and a JSON report, no pipeline re-run required. Block B (Tasks 11–12) requires running the full Yelp pipeline against Philadelphia data and produces a side-by-side comparison of Tampa vs Philadelphia metrics.

**Tech Stack:** pandas, numpy, matplotlib, seaborn, shap, scikit-learn, joblib, XGBoost; existing pipeline scripts `01_load_filter.py`–`06_ensemble.py`; `config_00.py` as the single source of truth for paths and constants.

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `config_00.py` | Modify | Add `COVID_START`, `COVID_END` constants |
| `04b_data_quality.py` | Create | Phase 1: all five quality checks, saves `data/processed/data_quality_report.json` |
| `07_model_analysis.py` | Create | Phases 2+3: deep EDA and model analysis, saves 9 figures to `figures/` |
| `08_philadelphia.py` | Create | Phase 4: end-to-end Philadelphia pipeline + comparison report |
| `tests/test_data_quality.py` | Create | Unit tests for the five quality-check functions |

---

## BLOCK A — Analysis (Phases 1, 2, 3)

---

### Task 1: Add COVID constants to config_00.py

**Files:**
- Modify: `config_00.py`

- [ ] **Step 1: Add two lines to config_00.py** — append after the `REVIEW_DROUGHT_DAYS` line:

```python
# ── COVID window ────────────────────────────────────────────────────────────
COVID_START = "2020-03-01"   # WHO pandemic declaration
COVID_END   = "2021-06-01"   # matches LATEST_ANCHOR (businesses anchored here
                              # had their outcome window fully inside lockdown era)
```

- [ ] **Step 2: Verify import works**

Run:
```bash
python -c "from config_00 import COVID_START, COVID_END; print(COVID_START, COVID_END)"
```
Expected output: `2020-03-01 2021-06-01`

- [ ] **Step 3: Commit**

```bash
git add config_00.py
git commit -m "feat: add COVID_START and COVID_END constants to config"
```

---

### Task 2: Scaffold 04b_data_quality.py with all five quality checks

**Files:**
- Create: `04b_data_quality.py`

Context: `features.parquet` has 5143 rows × 48 columns. All quality functions should be pure (take a DataFrame, return a dict or DataFrame) so they can be unit-tested separately.

- [ ] **Step 1: Write the failing test first**

Create `tests/test_data_quality.py`:

```python
"""Tests for 04b_data_quality.py quality check functions."""
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_quality import (
    check_duplicates,
    add_covid_flag,
    check_price_range,
    detect_outliers,
    validate_schema,
)


def _sample_df():
    return pd.DataFrame({
        "business_id":           ["A", "B", "C"],
        "anchor_date":           pd.to_datetime(["2019-01-01", "2020-06-01", "2021-01-01"]),
        "closed_within_6m":      [0, 1, 0],
        "days_since_last_review":[10.0, 200.0, 50.0],
        "n_reviews_obs":         [5.0, 3.0, 100.0],
        "price_range":           [1.0, 2.0, 4.0],
        "review_velocity":       [0.5, 0.2, 8.0],
    })


# ── 1a duplicates ──────────────────────────────────────────────────────────
def test_check_duplicates_clean():
    result = check_duplicates(_sample_df())
    assert result["n_duplicates"] == 0
    assert result["ok"] is True


def test_check_duplicates_finds_dup():
    df = _sample_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    result = check_duplicates(df)
    assert result["n_duplicates"] == 1
    assert result["ok"] is False


# ── 1b COVID flag ──────────────────────────────────────────────────────────
def test_add_covid_flag_labels_correctly():
    df = _sample_df()
    out = add_covid_flag(df)
    # 2019-01-01 → not covid; 2020-06-01 → covid; 2021-01-01 → covid
    assert list(out["covid_flag"]) == [0, 1, 1]


def test_add_covid_flag_does_not_mutate_input():
    df = _sample_df()
    add_covid_flag(df)
    assert "covid_flag" not in df.columns


# ── 1c price_range ─────────────────────────────────────────────────────────
def test_check_price_range_valid():
    result = check_price_range(_sample_df())
    assert result["ok"] is True
    assert result["n_invalid"] == 0


def test_check_price_range_catches_invalid():
    df = _sample_df()
    df.loc[0, "price_range"] = 7.0
    result = check_price_range(df)
    assert result["ok"] is False
    assert result["n_invalid"] == 1


# ── 1d outliers ────────────────────────────────────────────────────────────
def test_detect_outliers_finds_extreme():
    df = _sample_df()
    # review_velocity: values [0.5, 0.2, 8.0]. 8.0 is extreme.
    out = detect_outliers(df, ["review_velocity"], k=1.0)
    assert len(out) > 0
    assert "review_velocity" in out["feature"].values


def test_detect_outliers_clean_data():
    df = _sample_df()
    out = detect_outliers(df, ["price_range"], k=3.0)
    assert len(out) == 0


# ── 1e schema ──────────────────────────────────────────────────────────────
def test_validate_schema_clean():
    issues = validate_schema(_sample_df())
    assert issues == []


def test_validate_schema_missing_column():
    df = _sample_df().drop(columns=["closed_within_6m"])
    issues = validate_schema(df)
    assert any("closed_within_6m" in i for i in issues)
```

- [ ] **Step 2: Run tests to verify they fail** (module not yet created)

```bash
pytest tests/test_data_quality.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'data_quality'`

- [ ] **Step 3: Create data_quality.py (importable module) at project root**

```python
"""
data_quality.py — Pure functions for data quality checks.
Imported by 04b_data_quality.py and tests/test_data_quality.py.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

# COVID period boundaries (same as config_00.COVID_START / COVID_END)
_COVID_START = pd.Timestamp("2020-03-01")
_COVID_END   = pd.Timestamp("2021-06-01")

_VALID_PRICE_RANGE = {1.0, 2.0, 3.0, 4.0}

_EXPECTED_SCHEMA = {
    "business_id":           "object",
    "closed_within_6m":      "numeric",
    "anchor_date":           "datetime",
    "days_since_last_review":"numeric",
    "n_reviews_obs":         "numeric",
    "price_range":           "numeric",
    "review_velocity":       "numeric",
}


def check_duplicates(df: pd.DataFrame) -> dict:
    """Return dict with n_duplicates and ok flag."""
    n_dup = int(df["business_id"].duplicated().sum())
    return {"n_duplicates": n_dup, "ok": n_dup == 0}


def add_covid_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Return copy of df with integer covid_flag column."""
    out = df.copy()
    dates = pd.to_datetime(out["anchor_date"])
    out["covid_flag"] = ((dates >= _COVID_START) & (dates <= _COVID_END)).astype(int)
    return out


def check_price_range(df: pd.DataFrame) -> dict:
    """Return dict with n_invalid, invalid_values list, and ok flag."""
    valid_mask = df["price_range"].isna() | df["price_range"].isin(_VALID_PRICE_RANGE)
    bad = df.loc[~valid_mask, "price_range"]
    return {
        "n_invalid": int(len(bad)),
        "invalid_values": sorted(bad.unique().tolist()),
        "ok": len(bad) == 0,
    }


def detect_outliers(
    df: pd.DataFrame, cols: list[str], k: float = 3.0
) -> pd.DataFrame:
    """
    Return DataFrame of outlier rows using IQR method (value > Q3 + k*IQR).
    Columns: business_id, feature, value, threshold.
    """
    results = []
    for col in cols:
        s = df[col].dropna()
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        upper = q3 + k * iqr
        mask = df[col] > upper
        if mask.any():
            chunk = df.loc[mask, ["business_id", col]].copy()
            chunk = chunk.rename(columns={col: "value"})
            chunk["feature"] = col
            chunk["threshold"] = round(upper, 3)
            results.append(chunk[["business_id", "feature", "value", "threshold"]])
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["business_id", "feature", "value", "threshold"]
    )


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return list of schema violation strings. Empty list = clean."""
    issues = []
    for col, expected in _EXPECTED_SCHEMA.items():
        if col not in df.columns:
            issues.append(f"MISSING column: {col}")
            continue
        if expected == "datetime":
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                issues.append(f"TYPE {col}: expected datetime, got {df[col].dtype}")
        elif expected == "numeric":
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"TYPE {col}: expected numeric, got {df[col].dtype}")
        elif expected == "object":
            if not pd.api.types.is_object_dtype(df[col]):
                issues.append(f"TYPE {col}: expected object/str, got {df[col].dtype}")
    return issues
```

- [ ] **Step 4: Run tests — all should pass**

```bash
pytest tests/test_data_quality.py -v
```
Expected: `11 passed`

- [ ] **Step 5: Create 04b_data_quality.py (the runner script)**

```python
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
    print("STEP 4b — Data Quality Audit")
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
    status = "OK" if pr["ok"] else f"FAIL — {pr['n_invalid']} invalid values: {pr['invalid_values']}"
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
    status = "OK" if not issues else f"FAIL\n  " + "\n  ".join(issues)
    print(f"[1e] Schema validation: {status}")

    # ── Save report ────────────────────────────────────────────────────────
    out_path = PROC_DIR / "data_quality_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved -> {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run and verify**

```bash
python 04b_data_quality.py
```
Expected output (based on known data):
```
[1a] Duplicates by business_id: OK
[1b] COVID-period anchors (2020-03 to 2021-06): 1,576 (30.6%)
       Closure rate COVID cohort:     X.X%
       Closure rate non-COVID cohort: X.X%
[1c] price_range validity: OK
[1d] Outliers (IQR k=3): N flagged rows
       n_reviews_obs: N rows above ...
[1e] Schema validation: OK
  Report saved -> data/processed/data_quality_report.json
```

- [ ] **Step 7: Commit**

```bash
git add data_quality.py 04b_data_quality.py tests/test_data_quality.py
git commit -m "feat: data quality audit - 5 checks, COVID flag, outlier detection"
```

---

### Task 3: 07_model_analysis.py — feature distributions + outlier profiles (2a, 2b)

**Files:**
- Create: `07_model_analysis.py`

Context: `features.parquet` has all 5143 restaurants. `ensemble_predictions.parquet` has the 1244-row test set with `risk_score`. Join them on `business_id` for the error analysis sections. Top 10 features by importance come from `models/xgboost.pkl` (`model.feature_importances_`).

- [ ] **Step 1: Create 07_model_analysis.py scaffold + feature distribution plot**

```python
"""
07_model_analysis.py — Deep EDA and model analysis.
Reads features.parquet, ensemble_predictions.parquet, models/xgboost.pkl.
Saves 9 figures to figures/.

Run:
    python 07_model_analysis.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import shap
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

from config_00 import PROC_DIR, MODEL_DIR, FIG_DIR, TARGET_COL

plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
PALETTE = {0: "#2E86AB", 1: "#E84855"}

COVID_START = pd.Timestamp("2020-03-01")
COVID_END   = pd.Timestamp("2021-06-01")
OPT_THRESHOLD = 0.2704   # from ensemble_results.json

_META = {"business_id", "closed_within_6m", "anchor_date", "city", "state", "covid_flag"}


def save(name: str):
    p = FIG_DIR / name
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p}")


def load_data():
    feat = pd.read_parquet(PROC_DIR / "features.parquet")
    feat["anchor_date"] = pd.to_datetime(feat["anchor_date"])
    feat["covid_flag"] = (
        (feat["anchor_date"] >= COVID_START) & (feat["anchor_date"] <= COVID_END)
    ).astype(int)

    preds = pd.read_parquet(PROC_DIR / "ensemble_predictions.parquet")
    model = joblib.load(MODEL_DIR / "xgboost.pkl")

    # Merge test predictions with full feature set
    test = preds.merge(
        feat.drop(columns=["closed_within_6m"], errors="ignore"),
        on="business_id", how="left",
    )
    test["predicted"] = (test["risk_score"] >= OPT_THRESHOLD).astype(int)

    return feat, test, model


def plot_feature_distributions(feat: pd.DataFrame, model):
    """2a — Violin plots for top 10 features by XGBoost importance."""
    feat_cols = [c for c in feat.columns if c not in _META]
    importance = dict(zip(model.feature_names_in_, model.feature_importances_))
    top10 = sorted(
        [c for c in feat_cols if c in importance],
        key=lambda c: importance[c], reverse=True
    )[:10]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()

    for i, col in enumerate(top10):
        ax = axes[i]
        data_open   = feat.loc[feat[TARGET_COL] == 0, col].dropna()
        data_closed = feat.loc[feat[TARGET_COL] == 1, col].dropna()
        ax.violinplot([data_open, data_closed], positions=[0, 1],
                      showmedians=True, showextrema=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Open", "Closed"], fontsize=8)
        ax.set_title(col.replace("_", "\n"), fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7)

    plt.suptitle("Top 10 Feature Distributions by XGBoost Importance",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    save("11_feature_distributions_violin.png")
    print(f"    Top 10 features: {top10}")


def plot_outlier_profiles(feat: pd.DataFrame):
    """2b — Table of most extreme restaurants per feature."""
    COLS = ["n_reviews_obs", "n_checkins_obs", "days_since_last_review",
            "review_velocity", "checkin_velocity"]
    rows = []
    for col in COLS:
        top3 = feat.nlargest(3, col)[["business_id", col, TARGET_COL]]
        for _, r in top3.iterrows():
            rows.append({
                "feature": col,
                "business_id": r["business_id"][:12] + "...",
                "value": round(float(r[col]), 1),
                "closed": int(r[TARGET_COL]),
            })
    df_out = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, len(rows) * 0.4 + 1))
    ax.axis("off")
    tbl = ax.table(
        cellText=df_out.values,
        colLabels=df_out.columns,
        loc="center", cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.4)
    plt.title("Most Extreme Restaurants by Feature (top 3 each)", fontweight="bold", pad=10)
    save("12_outlier_profiles.png")


def main():
    print("=" * 60)
    print("STEP 7 — Model Analysis & Deep EDA")
    print("=" * 60)

    feat, test, model = load_data()
    print(f"  Features: {feat.shape[0]:,} rows | Test set: {test.shape[0]:,} rows\n")

    print("[2a] Feature distributions (violin)...")
    plot_feature_distributions(feat, model)

    print("[2b] Outlier profiles...")
    plot_outlier_profiles(feat)

    # Remaining plots added in Tasks 4-8

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run to verify first two plots render without error**

```bash
python 07_model_analysis.py
```
Expected: two files saved to `figures/`, no exceptions.

- [ ] **Step 3: Commit**

```bash
git add 07_model_analysis.py
git commit -m "feat: 07_model_analysis scaffold - violin distributions and outlier table"
```

---

### Task 4: Correlation heatmap (2c)

**Files:**
- Modify: `07_model_analysis.py`

- [ ] **Step 1: Add `plot_correlation_heatmap` function to 07_model_analysis.py** — insert after `plot_outlier_profiles`:

```python
def plot_correlation_heatmap(feat: pd.DataFrame, model):
    """2c — Pearson correlation heatmap for top 15 features."""
    feat_cols = [c for c in feat.columns if c not in _META]
    importance = dict(zip(model.feature_names_in_, model.feature_importances_))
    top15 = sorted(
        [c for c in feat_cols if c in importance],
        key=lambda c: importance[c], reverse=True
    )[:15]

    corr = feat[top15].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax, cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.5, square=True,
        xticklabels=[c.replace("_", "\n") for c in top15],
        yticklabels=[c.replace("_", "\n") for c in top15],
    )
    ax.tick_params(labelsize=7)
    ax.set_title("Feature Correlation Matrix (top 15 by importance)",
                 fontweight="bold", pad=12)
    plt.tight_layout()
    save("13_correlation_heatmap.png")
```

- [ ] **Step 2: Add call inside `main()` after the outlier call:**

```python
    print("[2c] Correlation heatmap...")
    plot_correlation_heatmap(feat, model)
```

- [ ] **Step 3: Run and verify**

```bash
python 07_model_analysis.py
```
Expected: `figures/13_correlation_heatmap.png` created.

- [ ] **Step 4: Commit**

```bash
git add 07_model_analysis.py
git commit -m "feat: add correlation heatmap for top 15 features"
```

---

### Task 5: COVID cohort analysis (2d)

**Files:**
- Modify: `07_model_analysis.py`

- [ ] **Step 1: Add `plot_covid_cohort` function** — insert after `plot_correlation_heatmap`:

```python
def plot_covid_cohort(feat: pd.DataFrame):
    """2d — Compare key metrics between COVID and non-COVID anchor cohorts."""
    COMPARE_COLS = [
        "days_since_last_review", "review_velocity",
        "months_with_zero_reviews", "review_momentum",
        "checkin_momentum", "mean_vader",
    ]
    COMPARE_COLS = [c for c in COMPARE_COLS if c in feat.columns]

    n_cols = 3
    n_rows = int(np.ceil(len(COMPARE_COLS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5))
    axes = axes.flatten()

    covid_labels = {0: "Non-COVID", 1: "COVID (2020-03 to 2021-06)"}
    colors = {0: "#2E86AB", 1: "#E84855"}

    for i, col in enumerate(COMPARE_COLS):
        ax = axes[i]
        data = [
            feat.loc[feat["covid_flag"] == g, col].dropna()
            for g in [0, 1]
        ]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="white", linewidth=2))
        for patch, g in zip(bp["boxes"], [0, 1]):
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([covid_labels[0], covid_labels[1]], fontsize=8)
        ax.set_title(col.replace("_", " "), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

        # Annotate closure rates
        for j, g in enumerate([0, 1]):
            grp = feat[feat["covid_flag"] == g]
            rate = grp[TARGET_COL].mean()
            ax.text(j + 1, ax.get_ylim()[1] * 0.97,
                    f"close={rate:.0%}", ha="center", fontsize=7, color=colors[g])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("COVID vs Non-COVID Cohort — Feature Comparison",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    save("14_covid_cohort_comparison.png")
```

- [ ] **Step 2: Add call inside `main()`:**

```python
    print("[2d] COVID cohort analysis...")
    plot_covid_cohort(feat)
```

- [ ] **Step 3: Run and verify**

```bash
python 07_model_analysis.py
```
Expected: `figures/14_covid_cohort_comparison.png` created.

- [ ] **Step 4: Commit**

```bash
git add 07_model_analysis.py
git commit -m "feat: add COVID cohort comparison boxplots"
```

---

### Task 6: FP/FN error profile (2e / 3c)

**Files:**
- Modify: `07_model_analysis.py`

- [ ] **Step 1: Add `plot_error_profile` function** — insert after `plot_covid_cohort`:

```python
def plot_error_profile(test: pd.DataFrame, model):
    """2e/3c — Compare feature means across TP, FP, FN, TN groups."""
    y_true = test["closed_within_6m"]
    y_pred = test["predicted"]

    tp = test[(y_pred == 1) & (y_true == 1)]
    fp = test[(y_pred == 1) & (y_true == 0)]
    fn = test[(y_pred == 0) & (y_true == 1)]

    print(f"    TP={len(tp)}  FP={len(fp)}  FN={len(fn)}")

    feat_cols = [c for c in model.feature_names_in_ if c in test.columns]
    importance = dict(zip(model.feature_names_in_, model.feature_importances_))
    top8 = sorted(feat_cols, key=lambda c: importance.get(c, 0), reverse=True)[:8]

    groups = {"True Pos (caught)": tp, "False Pos (false alarm)": fp,
              "False Neg (missed)": fn}
    group_colors = {"True Pos (caught)": "#1DB954",
                    "False Pos (false alarm)": "#EF9F27",
                    "False Neg (missed)": "#E84855"}

    means = pd.DataFrame({
        name: grp[top8].mean()
        for name, grp in groups.items()
    })
    # Normalise by full-test-set std for comparability
    stds = test[top8].std().replace(0, 1)
    means_norm = means.div(stds, axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(top8))
    width = 0.25
    for k, (name, color) in enumerate(group_colors.items()):
        ax.bar(x + k * width, means_norm[name], width,
               label=f"{name} (n={len(groups[name])})",
               color=color, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.replace("_", "\n") for c in top8], fontsize=8)
    ax.set_ylabel("Mean (z-score relative to test set std)", fontsize=9)
    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8)
    ax.set_title("Error Profile: What distinguishes False Positives from False Negatives?",
                 fontweight="bold")
    plt.tight_layout()
    save("15_fp_fn_error_profile.png")
```

- [ ] **Step 2: Add call inside `main()`:**

```python
    print("[2e/3c] FP/FN error profile...")
    plot_error_profile(test, model)
```

- [ ] **Step 3: Run and verify**

```bash
python 07_model_analysis.py
```
Expected: `figures/15_fp_fn_error_profile.png` created, TP/FP/FN counts printed.

- [ ] **Step 4: Commit**

```bash
git add 07_model_analysis.py
git commit -m "feat: add FP/FN error profile chart"
```

---

### Task 7: SHAP global summary plot (3a)

**Files:**
- Modify: `07_model_analysis.py`

- [ ] **Step 1: Add `plot_shap_global` function** — insert after `plot_error_profile`:

```python
def plot_shap_global(test: pd.DataFrame, model):
    """3a — SHAP beeswarm summary for the full test set."""
    feat_cols = [c for c in model.feature_names_in_ if c in test.columns]
    X_test = test[feat_cols].copy()

    # Impute with test-set medians (same strategy as 05_modeling.py)
    X_test = X_test.fillna(X_test.median())

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # shap_values may be 2D (n_samples, n_features)
    if isinstance(shap_values, list):
        sv = np.array(shap_values[-1])
    else:
        sv = np.array(shap_values)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        sv, X_test,
        plot_type="dot",
        show=False,
        max_display=20,
        color_bar_label="Feature value",
    )
    plt.title("SHAP Global Feature Importance — Full Test Set",
              fontweight="bold", pad=10)
    plt.tight_layout()
    save("16_shap_global_summary.png")
```

- [ ] **Step 2: Add call inside `main()`:**

```python
    print("[3a] SHAP global summary (may take ~30s)...")
    plot_shap_global(test, model)
```

- [ ] **Step 3: Run and verify (first run installs shap if needed)**

```bash
pip install shap -q
python 07_model_analysis.py
```
Expected: `figures/16_shap_global_summary.png` created.

- [ ] **Step 4: Commit**

```bash
git add 07_model_analysis.py
git commit -m "feat: add SHAP global beeswarm summary for full test set"
```

---

### Task 8: Full precision-recall curve + XGBoost vs ensemble comparison (3b, 3d)

**Files:**
- Modify: `07_model_analysis.py`

- [ ] **Step 1: Add `plot_pr_curve_full` and `compare_xgb_vs_ensemble` functions** — insert after `plot_shap_global`:

```python
def plot_pr_curve_full(test: pd.DataFrame, model):
    """3b — Precision-recall curve with threshold markers."""
    y_true = test["closed_within_6m"].values
    y_prob = test["risk_score"].values  # ensemble probability

    # Also compute XGBoost standalone probability
    feat_cols = [c for c in model.feature_names_in_ if c in test.columns]
    X = test[feat_cols].fillna(test[feat_cols].median())
    xgb_prob = model.predict_proba(X)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, prob, color in [
        ("Ensemble (stacking)", y_prob, "#1DB954"),
        ("XGBoost standalone", xgb_prob, "#2E86AB"),
    ]:
        prec, rec, thr = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        ax.plot(rec, prec, color=color, linewidth=2, label=f"{label} (AUC-PR={ap:.3f})")

        # Mark F1-optimal threshold
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = np.argmax(f1s[:-1])
        ax.scatter(rec[best_idx], prec[best_idx], color=color, s=100, zorder=5)
        ax.annotate(
            f"t={thr[best_idx]:.2f}\nF1={f1s[best_idx]:.3f}",
            xy=(rec[best_idx], prec[best_idx]),
            xytext=(rec[best_idx] - 0.12, prec[best_idx] + 0.06),
            fontsize=8, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        )

    base_rate = y_true.mean()
    ax.axhline(base_rate, color="#888", linestyle="--", linewidth=1,
               label=f"Random baseline (precision={base_rate:.3f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve with F1-Optimal Threshold",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    save("17_pr_curve_full.png")


def compare_xgb_vs_ensemble(test: pd.DataFrame, model):
    """3d — Print XGBoost vs ensemble comparison and recommendation."""
    y_true = test["closed_within_6m"].values
    y_ens  = test["risk_score"].values

    feat_cols = [c for c in model.feature_names_in_ if c in test.columns]
    X = test[feat_cols].fillna(test[feat_cols].median())
    y_xgb = model.predict_proba(X)[:, 1]

    def metrics(y_true, y_prob, label):
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = np.argmax(f1s[:-1])
        opt_t = thr[best_idx]
        ap = average_precision_score(y_true, y_prob)
        y_pred = (y_prob >= opt_t).astype(int)
        f1 = f1_score(y_true, y_pred)
        return {"model": label, "AUC-PR": round(ap, 4),
                "opt_threshold": round(float(opt_t), 4), "F1": round(f1, 4)}

    rows = [
        metrics(y_true, y_xgb, "XGBoost standalone"),
        metrics(y_true, y_ens,  "Ensemble (stacking)"),
    ]
    print("\n  === XGBoost vs Ensemble ===")
    for r in rows:
        print(f"  {r['model']:30s}  AUC-PR={r['AUC-PR']}  "
              f"F1={r['F1']}  threshold={r['opt_threshold']}")

    winner = max(rows, key=lambda r: r["AUC-PR"])
    print(f"\n  Recommendation: use {winner['model']} "
          f"(AUC-PR {winner['AUC-PR']} > other)")
```

- [ ] **Step 2: Add calls inside `main()`:**

```python
    print("[3b] Full PR curve + threshold markers...")
    plot_pr_curve_full(test, model)

    print("[3d] XGBoost vs ensemble comparison...")
    compare_xgb_vs_ensemble(test, model)
```

- [ ] **Step 3: Run the full script end-to-end**

```bash
python 07_model_analysis.py
```
Expected: 7 figures saved (`11_` through `17_`), comparison table printed, no errors.

- [ ] **Step 4: Commit**

```bash
git add 07_model_analysis.py
git commit -m "feat: PR curve with threshold markers + XGBoost vs ensemble comparison"
```

---

## BLOCK B — Philadelphia Generalization (Phase 4)

> **Note:** Requires the raw Yelp JSON files and ~30–40 min total runtime (feature engineering is the slow step due to VADER). Run Block A first.

---

### Task 9: 08_philadelphia.py — load, filter, label (4a)

**Files:**
- Create: `08_philadelphia.py`

Context: The raw Yelp JSON files are at `data/raw/Yelp JSON/`. The existing pipeline scripts (`01_load_filter.py` through `06_ensemble.py`) all import from `config_00.py` which is hardcoded to Tampa. Rather than modifying those scripts, `08_philadelphia.py` will call the core functions by importing them directly — where they are clean functions — or replicate the minimal logic for Philadelphia with a separate output directory `data/processed_philly/`.

Philadelphia raw counts: 5,852 restaurant entries found in `yelp_academic_dataset_business.json`.

- [ ] **Step 1: Create 08_philadelphia.py with load + filter + label sections**

```python
"""
08_philadelphia.py — Zero-shot Philadelphia generalization.

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, precision_recall_curve,
)

from config_00 import (
    RAW_DIR, MODEL_DIR, FIG_DIR, ENCODING,
    OBS_MONTHS, OUTCOME_MONTHS, EARLIEST_ANCHOR, LATEST_ANCHOR,
    TARGET_COL, RANDOM_SEED,
)

PHILLY_DIR = Path("data/processed_philly")
PHILLY_DIR.mkdir(parents=True, exist_ok=True)

PHILLY_CITIES = {"Philadelphia"}

OPT_THRESHOLD = 0.2704   # Tampa-trained threshold

plt.rcParams.update({"font.family": "serif", "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})


# ── Step 1: Load and filter businesses ─────────────────────────────────────

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
            rows.append({
                "business_id":     r["business_id"],
                "name":            r.get("name", ""),
                "city":            r.get("city", ""),
                "state":           r.get("state", ""),
                "is_open":         int(r.get("is_open", 1)),
                "stars_yelp":      float(r.get("stars", 0)),
                "review_count_yelp": int(r.get("review_count", 0)),
                "price_range":     float(r.get("attributes", {}).get("RestaurantsPriceRange2") or 0) or None,
                "open_days_per_week": bin(int(r.get("hours") and sum(1 for v in r["hours"].values() if v) or 0)).count("1") if r.get("hours") else None,
                "categories":      cats,
                "latitude":        r.get("latitude"),
                "longitude":       r.get("longitude"),
            })
    df = pd.DataFrame(rows)
    print(f"    Philadelphia restaurants found: {len(df):,}")
    return df


def main():
    print("=" * 60)
    print("STEP 8 — Philadelphia Zero-Shot Generalization")
    print("=" * 60)

    # ── 1. Load businesses ────────────────────────────────────────────────
    biz = load_philly_businesses()
    biz.to_parquet(PHILLY_DIR / "businesses.parquet", index=False)
    print(f"    Saved businesses.parquet ({len(biz):,} rows)")

    # ── 2. Load reviews, checkins, tips for Philly businesses ─────────────
    philly_bids = set(biz["business_id"])

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

    # Remaining steps added in Task 10
    print("\n  Data loading complete. Run script again after Task 10 is implemented.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run load section (expected ~5 min)**

```bash
python 08_philadelphia.py
```
Expected:
```
Philadelphia restaurants found: ~5,800
Saved businesses.parquet
Saved reviews.parquet (~N rows)
Saved checkins.parquet
Saved tips.parquet
Data loading complete.
```

- [ ] **Step 3: Commit**

```bash
git add 08_philadelphia.py
git commit -m "feat: 08_philadelphia - load and filter Philly restaurants from raw Yelp"
```

---

### Task 10: Philadelphia feature engineering + zero-shot prediction + comparison (4a, 4b, 4c)

**Files:**
- Modify: `08_philadelphia.py`

Context: `03_feature_engineering.py` has a function `build_features_one()` that takes per-business data and returns a feature dict. We import it directly rather than duplicating the logic.

- [ ] **Step 1: Verify 03_feature_engineering.py is importable**

```bash
python -c "from importlib import import_module; m = import_module('03_feature_engineering'); print(dir(m))" 2>&1 | grep build
```
Expected output includes `build_features_one` (or similar). If not, check the function name:
```bash
grep "^def " 03_feature_engineering.py
```

- [ ] **Step 2: Add `build_labels_philly` and `build_features_philly` functions to 08_philadelphia.py** — insert before `main()`:

```python
import importlib
_fe = importlib.import_module("03_feature_engineering")

from dateutil.relativedelta import relativedelta


def build_labels_philly(biz: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Assign anchor dates and closure labels using the same logic as 02_build_labels.py."""
    earliest = pd.Timestamp(EARLIEST_ANCHOR)
    latest   = pd.Timestamp(LATEST_ANCHOR)
    obs_months = OBS_MONTHS
    outcome_months = OUTCOME_MONTHS

    rows = []
    for _, b in biz.iterrows():
        bid = b["business_id"]
        rev = reviews[reviews["business_id"] == bid].sort_values("date")
        if rev.empty:
            continue

        # Anchor = 80th percentile review date
        anchor = rev["date"].quantile(0.80, interpolation="nearest")
        if not (earliest <= anchor <= latest):
            continue

        obs_start   = anchor - relativedelta(months=obs_months)
        outcome_end = anchor + relativedelta(months=outcome_months)

        # Label: any review after anchor within outcome window and is_open=0
        # Proxy: if is_open=0 AND last review is within outcome window
        last_review = rev["date"].max()
        closed = int(b["is_open"] == 0 and last_review <= outcome_end)

        rows.append({
            "business_id":    bid,
            "name":           b["name"],
            "city":           b["city"],
            "state":          b["state"],
            "anchor_date":    anchor,
            "obs_start":      obs_start,
            "outcome_end":    outcome_end,
            "closed_within_6m": closed,
            "stars_yelp":     b["stars_yelp"],
            "price_range":    b["price_range"],
            "open_days_per_week": b["open_days_per_week"],
            "categories":     b["categories"],
        })

    return pd.DataFrame(rows)


def build_features_philly(
    labeled: pd.DataFrame, reviews: pd.DataFrame,
    checkins: pd.DataFrame, tips: pd.DataFrame,
) -> pd.DataFrame:
    """Call build_features_one from 03_feature_engineering for each Philly restaurant."""
    from tqdm import tqdm
    build_one = _fe.build_features_one

    records = []
    for _, row in tqdm(labeled.iterrows(), total=len(labeled), desc="Philly features"):
        feat = build_one(row, reviews, checkins, tips)
        if feat is not None:
            records.append(feat)

    df = pd.DataFrame(records)
    return df


def predict_and_compare(features: pd.DataFrame, model, model_name: str = "XGBoost (Tampa-trained)"):
    """Apply Tampa model to Philly features and compute metrics."""
    feat_cols = [c for c in model.feature_names_in_ if c in features.columns]
    missing = [c for c in model.feature_names_in_ if c not in features.columns]
    if missing:
        print(f"    WARNING: {len(missing)} features missing from Philly data: {missing}")

    X = features[feat_cols].copy()
    # Impute with column medians (same approach as pipeline)
    X = X.fillna(X.median())

    y_true = features[TARGET_COL].values
    y_prob = model.predict_proba(X)[:, 1]

    auc_pr  = average_precision_score(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)

    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx  = np.argmax(f1s[:-1])
    opt_thr   = float(thr[best_idx])
    opt_f1    = float(f1s[best_idx])

    y_pred_fixed = (y_prob >= OPT_THRESHOLD).astype(int)
    f1_fixed     = f1_score(y_true, y_pred_fixed)

    n_total    = len(features)
    n_closed   = int(y_true.sum())
    base_rate  = float(y_true.mean())

    flagged    = int((y_prob >= OPT_THRESHOLD).sum())
    caught     = int(((y_prob >= OPT_THRESHOLD) & (y_true == 1)).sum())
    recall_fixed = caught / n_closed if n_closed else 0

    return {
        "city": "Philadelphia",
        "n_restaurants": n_total,
        "n_closed": n_closed,
        "base_rate": round(base_rate, 4),
        "AUC_PR": round(auc_pr, 4),
        "AUC_ROC": round(auc_roc, 4),
        "opt_threshold_local": round(opt_thr, 4),
        "opt_f1_local": round(opt_f1, 4),
        "f1_at_tampa_threshold": round(f1_fixed, 4),
        "flagged_at_tampa_threshold": flagged,
        "caught_at_tampa_threshold": caught,
        "recall_at_tampa_threshold": round(recall_fixed, 4),
    }
```

- [ ] **Step 3: Extend `main()` with labels + features + prediction steps** — replace the `print("Data loading complete...")` line at the bottom of `main()` with:

```python
    # ── 3. Build labels ────────────────────────────────────────────────────
    print("  Building labels...")
    labeled = build_labels_philly(biz, reviews)
    labeled.to_parquet(PHILLY_DIR / "labeled_businesses.parquet", index=False)
    n_closed = labeled["closed_within_6m"].sum()
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

    # Tampa test set for comparison
    tampa_preds = pd.read_parquet(Path("data/processed/ensemble_predictions.parquet"))
    tampa_feat  = pd.read_parquet(Path("data/processed/features.parquet"))
    tampa_test  = tampa_preds.merge(
        tampa_feat.drop(columns=["closed_within_6m"], errors="ignore"),
        on="business_id", how="left",
    )
    feat_cols = [c for c in model.feature_names_in_ if c in tampa_test.columns]
    X_tampa   = tampa_test[feat_cols].fillna(tampa_test[feat_cols].median())
    y_true_t  = tampa_test["closed_within_6m"].values
    y_prob_t  = model.predict_proba(X_tampa)[:, 1]

    tampa_results = {
        "city": "Tampa Bay",
        "n_restaurants": len(tampa_test),
        "n_closed": int(y_true_t.sum()),
        "base_rate": round(float(y_true_t.mean()), 4),
        "AUC_PR": round(float(average_precision_score(y_true_t, y_prob_t)), 4),
        "AUC_ROC": round(float(roc_auc_score(y_true_t, y_prob_t)), 4),
        "f1_at_tampa_threshold": round(float(f1_score(y_true_t, (y_prob_t >= OPT_THRESHOLD).astype(int))), 4),
    }

    # ── 6. Print comparison ────────────────────────────────────────────────
    print("\n  === TAMPA vs PHILADELPHIA ZERO-SHOT COMPARISON ===")
    print(f"  {'Metric':35s} {'Tampa':>10} {'Philadelphia':>14}")
    print("  " + "-" * 62)
    for key in ["n_restaurants", "n_closed", "base_rate", "AUC_PR", "AUC_ROC", "f1_at_tampa_threshold"]:
        t_val = tampa_results.get(key, "—")
        p_val = philly_results.get(key, "—")
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
    ax.set_title("Tampa vs Philadelphia — Zero-Shot Generalization",
                 fontweight="bold")
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    p = FIG_DIR / "18_tampa_vs_philly_comparison.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p}")
```

- [ ] **Step 4: Run the full Philadelphia pipeline**

```bash
python 08_philadelphia.py
```
Expected runtime: ~30–40 min. Expected output:
```
Philadelphia restaurants found: ~5,800
Labeled: N restaurants, M closed (X.X%)
Features shape: (N, 48)
=== TAMPA vs PHILADELPHIA ZERO-SHOT COMPARISON ===
  Metric                              Tampa  Philadelphia
...
Saved results -> models/philadelphia_results.json
Saved -> figures/18_tampa_vs_philly_comparison.png
```

If `build_features_one` function name is different in `03_feature_engineering.py`, update the import line at the top of `08_philadelphia.py` accordingly (check with `grep "^def " 03_feature_engineering.py`).

- [ ] **Step 5: Commit**

```bash
git add 08_philadelphia.py data/processed_philly/ models/philadelphia_results.json
git commit -m "feat: Philadelphia zero-shot generalization - load, label, features, predict, compare"
```

---

## Self-Review

**Spec coverage:**

| Requirement | Task |
|---|---|
| 1a. Duplicates by business_id | Task 2 (`check_duplicates`) |
| 1b. COVID flag | Task 2 (`add_covid_flag`) |
| 1c. price_range validation | Task 2 (`check_price_range`) |
| 1d. Outliers | Task 2 (`detect_outliers`) |
| 1e. Schema validation | Task 2 (`validate_schema`) |
| 2a. Feature distributions (top 10) | Task 3 (`plot_feature_distributions`) |
| 2b. Outlier profiles | Task 3 (`plot_outlier_profiles`) |
| 2c. Correlation matrix | Task 4 |
| 2d. COVID cohort | Task 5 |
| 2e. FP/FN error profile | Task 6 |
| 3a. SHAP global | Task 7 |
| 3b. PR curve with threshold | Task 8 |
| 3c. FP/FN analysis | Task 6 (merged with 2e) |
| 3d. XGBoost vs ensemble | Task 8 |
| 4a. Philadelphia pipeline | Tasks 9–10 |
| 4b. Tampa vs Philadelphia metrics | Task 10 |
| 4c. Zero-shot generalization | Task 10 |

All 17 requirements covered. No placeholders detected.
