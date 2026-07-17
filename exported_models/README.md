# Exported model artifacts

This directory contains frozen, self-contained model packages intended to be
copied or validated without importing the research code in `src/lenz_speed`.

## `subject1_speed_benchmark_v1`

Purpose: reproducible benchmark artifact.

- Model: `RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)`
- Training data: approved Subject 1 Day 3 windows only
- Validation data: the frozen Subject 1 standard validation split
- Historical benchmark MAE: `0.1634656501460112` mph

This artifact is the one to cite for the independent Subject 1 benchmark
result.

## `subject1_speed_deployment_v1`

Purpose: personalized Subject 1 live-UI deployment artifact.

- Model: same estimator class, hyperparameters, and 19-feature order
- Training data: expanded approved Subject 1 natural-cadence data
- Output unit: mph
- Intended consumer: UI integration package

Do not attribute the benchmark MAE directly to this deployment artifact. The
deployment model includes recordings that were held out in the benchmark, so
it no longer has the same independent validation claim.

## Package contents

Each model package contains:

- `model.joblib` — serialized estimator
- `metadata.json` — model identity, feature order, checksum, training metadata
- `feature_extractor.py` and `inference.py` — standalone runtime code
- `requirements-lock.txt` — runtime versions used for parity
- `MODEL_CARD.md` — artifact-specific intended use and limitations
- `fixtures/` — low/mid/high golden windows, expected features, and expected predictions

Golden fixtures validate two separate guarantees:

1. raw canonical window samples produce the expected 19-feature vector;
2. the expected feature vector produces the expected serialized-model prediction.

The exported runtimes must not depend on raw data, manifests, notebooks,
plots, processed tables, or research evaluation modules.
