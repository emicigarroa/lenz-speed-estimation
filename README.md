# LENZ Speed Estimation

Team LENZ is investigating smart glasses as a platform for real-time run
coaching. This repository supports estimation of treadmill speed from a
head-mounted inertial measurement unit (IMU).

The IMU is sampled at 200 Hz. The planned analysis uses 5-second windows with
a 2.5-second step and evaluates whether inertial features add useful speed
information beyond cadence.

Current baseline results:

- Standard same-subject Random Forest MAE: 0.2311 mph
- Cadence stress-test Random Forest MAE: 0.7192 mph

## Setup

Create and activate a Python virtual environment, then install the project
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Raw data are not included in the repository. Place the source workbooks in the
appropriate folders under `data/raw/` without modifying the original files.
Raw recordings are local-only and ignored by Git.

Run the full reproducible pipeline with:

```bash
python run_pipeline.py
```

Run the automated test suite with:

```bash
pytest
```

## Repository structure

```text
.
├── data/
│   ├── raw/          # Immutable source recordings, grouped by subject/day
│   └── processed/    # Reproducible derived datasets
├── notebooks/        # Ordered exploration and experiment notebooks
├── outputs/
│   ├── figures/      # Generated plots
│   └── tables/       # Generated result tables
├── src/              # Reusable loading, feature, modeling, evaluation code
├── tests/            # Automated pytest suite
├── run_pipeline.py   # One-command pipeline entry point
├── DECISIONS.md      # Project decision log
├── PROJECT_CONTEXT.md
├── TODO.md
└── requirements.txt
```

See the README in each major folder for its intended contents and conventions.

## Workflow

1. Inventory the raw workbooks and record subject, day, speed, trial type,
   exclusions, and trimming requirements in `configs/dataset_manifest.csv`.
2. Validate signal schemas, units, timestamps, sampling rates, and recording
   quality before analysis.
3. Trim non-steady-state intervals before windowing, especially for Subject 1
   Day 2.
4. Create 5-second windows with a 2.5-second step while retaining source-file
   and recording provenance.
5. Extract documented cadence, acceleration, gyroscope, and feature engineering
   v2 IMU features.
6. Train primarily on Subject 1 Day 3 and validate on approved Subject 1 Day 2
   and Day 4 recordings.
7. Report overall, per-speed, and per-recording results before progressing to
   cross-subject and cadence-manipulation experiments.

Overlapping windows from the same recording must never be randomly divided
between training and validation sets. Splits must preserve recording, day, and
subject boundaries as appropriate to the experiment.

## Status

The repository now contains a working end-to-end baseline:

- Data loading from the manifest, with CSV/XLSX support and raw-file
  preservation.
- Windowing, feature extraction, modeling, evaluation, and plotting modules.
- Feature engineering v2, adding low-risk acceleration and gyroscope summary
  features while preserving the original feature set.
- A one-command pipeline that regenerates processed features, result tables,
  and figures.
- Automated pytest coverage for windowing, feature extraction, and evaluation.
- A walkthrough notebook at `notebooks/01_project_walkthrough.ipynb` for
  reviewing pipeline outputs without duplicating core logic.
