# Restaurant Failure Prediction

Binary classification project using the Yelp Academic Dataset to predict whether a restaurant will permanently close within the next 6 months from 12 months of prior behavioral signals.

## Project Layout

```text
.
├── CLAUDE.md
├── config_00.py
├── 01_load_filter.py
├── 02_build_labels.py
├── 03_feature_engineering.py
├── 04_eda.py
├── 05_modeling.py
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── models/
└── requirements.txt
```

## Setup

Create and activate a Python 3.10+ virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Download the Yelp Academic Dataset and place these files in `data/raw/`:

```text
yelp_academic_dataset_business.json
yelp_academic_dataset_review.json
yelp_academic_dataset_checkin.json
yelp_academic_dataset_tip.json
```

The photos file may also be placed in `data/raw/`, but it is reserved for future work and is not used by the current pipeline.

## Run Order

Run the scripts from the repository root in numerical order:

```bash
python 01_load_filter.py
python 02_build_labels.py
python 03_feature_engineering.py
python 04_eda.py
python 05_modeling.py
```

Generated Parquet files, figures, and model artifacts are intentionally ignored by Git.
