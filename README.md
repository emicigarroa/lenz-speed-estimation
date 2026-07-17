# LENZ Speed Estimation

Team LENZ is investigating smart glasses as a platform for real-time run
coaching. This repository supports estimation of treadmill speed from a
head-mounted inertial measurement unit (IMU).

The IMU is sampled at 200 Hz. The planned analysis uses 5-second windows with
a 2.5-second step and evaluates whether inertial features add useful speed
information beyond cadence.

Current frozen Subject 1 benchmark:

- Benchmark artifact: `exported_models/subject1_speed_benchmark_v1/`
- Model: Random Forest with the frozen 19-feature v4 morphology feature order
- Training data: approved Subject 1 Day 3 windows only
- Validation data: frozen Subject 1 standard validation split
- MAE: 0.1634656501460112 mph

The deployment artifact for UI integration is separate:

- Deployment artifact: `exported_models/subject1_speed_deployment_v1/`
- Training data: expanded approved Subject 1 natural-cadence dataset
- Output unit: mph
- Intended use: personalized Subject 1 live inference in the UI

Do not attribute the independent benchmark MAE directly to the deployment
artifact because former validation recordings are included in deployment
training.

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
├── exported_models/  # Frozen benchmark/deployment model packages
├── scripts/          # Reproducible export, validation, and analysis entry points
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
5. Extract documented cadence, acceleration, gyroscope, temporal, and
   morphology IMU features.
6. Train primarily on Subject 1 Day 3 and validate on approved Subject 1 Day 2
   and Day 4 recordings.
7. Report overall, per-speed, and per-recording results before progressing to
   cross-subject and cadence-manipulation experiments.

Overlapping windows from the same recording must never be randomly divided
between training and validation sets. Splits must preserve recording, day, and
subject boundaries as appropriate to the experiment.

## Status

The repository now contains a working end-to-end research and export workflow:

- Data loading from the manifest, with CSV/XLSX support and raw-file
  preservation.
- Windowing, feature extraction, modeling, evaluation, and plotting modules.
- Feature engineering through the frozen v4 morphology feature set used by the
  exported Subject 1 models.
- A one-command pipeline that regenerates processed features, result tables,
  and figures.
- Automated pytest coverage for windowing, feature extraction, evaluation, and
  exported model parity.
- A walkthrough notebook at `notebooks/01_project_walkthrough.ipynb` for
  reviewing pipeline outputs without duplicating core logic.
- Frozen benchmark and deployment model packages under `exported_models/`.

## Reproducing and validating exports

Reproduce the frozen Subject 1 benchmark report:

```bash
python scripts/reproduce_subject1_speed_benchmark_v1.py
```

Regenerate the benchmark export package:

```bash
python scripts/export_subject1_speed_benchmark_v1.py
```

Regenerate the deployment export package:

```bash
python scripts/export_subject1_speed_deployment_v1.py
```

The export packages include golden fixtures and tests that verify feature
parity, prediction parity, checksums, and runtime import isolation.

## Model limitations

The exported models are personalized Subject 1 artifacts. They are useful for
benchmark reproduction and UI integration, but they are not universal
cross-subject estimators and do not provide a statistically calibrated
confidence score.
