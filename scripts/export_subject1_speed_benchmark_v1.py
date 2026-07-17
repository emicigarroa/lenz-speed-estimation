"""Build the frozen Subject 1 benchmark model export package."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lenz_speed.data import load_manifest, load_recording  # noqa: E402
from lenz_speed.dataset import _selected_trim_values  # noqa: E402
from lenz_speed.features import extract_window_features  # noqa: E402
from lenz_speed.windowing import apply_trim, make_windows  # noqa: E402


EXPORT_DIR = REPOSITORY_ROOT / "exported_models/subject1_speed_benchmark_v1"
FIXTURE_DIR = EXPORT_DIR / "fixtures"
FEATURE_TABLE_PATH = REPOSITORY_ROOT / "data/processed/windowed_features.csv"
HISTORICAL_METRICS_PATH = (
    REPOSITORY_ROOT / "outputs/tables/v4_morphology_feature_metrics.csv"
)

FEATURES: tuple[str, ...] = (
    "Cadence_spm",
    "RMS_Z",
    "PeakToPeak_Z",
    "Gyro_RMS_X",
    "Gyro_RMS_Y",
    "Gyro_RMS_Z",
    "Accel_Mag_RMS",
    "Dynamic_Accel_Mag_RMS",
    "Accel_Mag_P95_P05",
    "Accel_Mag_Jerk_RMS",
    "Accel_HighFreq_Energy_Ratio",
    "Gyro_Mag_RMS",
    "GyroY_PeakToPeak",
    "Accel_Anisotropy",
    "Vertical_Peak_Sharpness",
    "Impact_Impulse",
    "Peak_Symmetry",
    "Impact_Crest_Factor",
    "Impact_Local_Kurtosis",
)

FIXTURES = {
    "low_speed": ("s1_day2_test9_2mph", 0),
    "mid_speed": ("s1_day4_20260619_6mph_145spm_normal", 0),
    "high_speed": ("s1_day2_test5_8mph", 0),
}


def _git_commit_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_feature_table() -> pd.DataFrame:
    table = pd.read_csv(FEATURE_TABLE_PATH)
    missing = sorted({"recording_id", "subject_id", "session", "condition", "speed_mph", *FEATURES} - set(table.columns))
    if missing:
        raise RuntimeError("windowed feature table is missing columns: " + ", ".join(missing))
    return table


def _standard_split(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    approved_recordings = set(load_manifest()["recording_id"].astype(str))
    approved = table.loc[table["recording_id"].astype(str).isin(approved_recordings)].copy()
    train = approved.loc[
        (approved["subject_id"] == "subject_1") & (approved["session"] == "day3")
    ].copy()
    test = approved.loc[
        (approved["subject_id"] == "subject_1")
        & (
            (approved["session"] == "day2")
            | ((approved["session"] == "day4") & (approved["condition"] == "cadence_normal"))
        )
    ].copy()
    return train, test


def _historical_metrics() -> dict[str, float]:
    rows = pd.read_csv(HISTORICAL_METRICS_PATH)
    row = rows.loc[
        (rows["evaluation"] == "standard")
        & (rows["summary_level"] == "overall")
        & (rows["feature_set"] == "v4_morphology_all")
        & (rows["model"] == "Random Forest")
    ]
    if len(row) != 1:
        raise RuntimeError("expected exactly one historical v4_morphology_all RF row.")
    record = row.iloc[0]
    historical_features = tuple(str(record["features"]).split("|"))
    if historical_features != FEATURES:
        raise RuntimeError("historical feature order does not match frozen export order.")
    return {
        "MAE": float(record["MAE"]),
        "RMSE": float(record["RMSE"]),
        "R2": float(record["R2"]),
    }


def _fit_model(train: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(train.loc[:, FEATURES], train["speed_mph"])
    return model


def _metrics(model: RandomForestRegressor, test: pd.DataFrame) -> dict[str, float]:
    predicted = model.predict(test.loc[:, FEATURES])
    actual = test["speed_mph"].to_numpy(dtype=float)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def _load_exported_feature_extractor() -> Any:
    path = EXPORT_DIR / "feature_extractor.py"
    spec = importlib.util.spec_from_file_location("subject1_export_feature_extractor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load exported feature extractor from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_window(recording_id: str, window_index: int) -> tuple[pd.DataFrame, Any, pd.Series]:
    manifest = load_manifest()
    matches = manifest.loc[manifest["recording_id"].astype(str) == recording_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one manifest row for fixture {recording_id}; found {len(matches)}")
    row = matches.iloc[0]
    recording = load_recording(recording_id)
    trim_start, trim_end = _selected_trim_values(row.to_dict())
    trimmed = apply_trim(recording, trim_start_sec=trim_start, trim_end_sec=trim_end, fs=200)
    windows = make_windows(trimmed, recording_id=recording_id, fs=200)
    if window_index >= len(windows):
        raise RuntimeError(f"{recording_id} has no fixture window index {window_index}.")
    return windows[window_index].signal.loc[:, list(CANONICAL_COLUMNS)].copy(), windows[window_index], row


CANONICAL_COLUMNS = ("ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CANONICAL_COLUMNS))
        writer.writeheader()
        for record in frame.to_dict(orient="records"):
            writer.writerow({column: record[column] for column in CANONICAL_COLUMNS})


def _write_fixtures(model: RandomForestRegressor) -> dict[str, dict[str, Any]]:
    exporter = _load_exported_feature_extractor()
    fixture_summary: dict[str, dict[str, Any]] = {}
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for fixture_name, (recording_id, window_index) in FIXTURES.items():
        signal, research_window, manifest_row = _fixture_window(recording_id, window_index)
        research_features = extract_window_features(research_window, fs=200)
        expected_features = {feature: float(research_features[feature]) for feature in FEATURES}
        production_features = exporter.extract_feature_dict(
            {column: signal[column].to_numpy(dtype=float) for column in CANONICAL_COLUMNS},
            fs=200,
        )
        max_feature_diff = max(
            abs(production_features[feature] - expected_features[feature])
            for feature in FEATURES
        )
        if max_feature_diff > 1e-10:
            raise RuntimeError(
                f"{fixture_name} feature parity failed before export: max diff {max_feature_diff}"
            )
        prediction = float(model.predict(pd.DataFrame([expected_features]).loc[:, FEATURES])[0])

        _write_csv(FIXTURE_DIR / f"{fixture_name}_window.csv", signal)
        (FIXTURE_DIR / f"{fixture_name}_features.json").write_text(
            json.dumps(
                {
                    "feature_names": list(FEATURES),
                    "features": expected_features,
                    "source_recording_id": recording_id,
                    "window_index": window_index,
                    "actual_speed_mph": float(manifest_row["speed_mph"]),
                    "max_research_vs_export_feature_abs_diff": max_feature_diff,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (FIXTURE_DIR / f"{fixture_name}_prediction.json").write_text(
            json.dumps(
                {
                    "prediction_mph": prediction,
                    "source_recording_id": recording_id,
                    "window_index": window_index,
                    "actual_speed_mph": float(manifest_row["speed_mph"]),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        fixture_summary[fixture_name] = {
            "recording_id": recording_id,
            "window_index": window_index,
            "actual_speed_mph": float(manifest_row["speed_mph"]),
            "prediction_mph": prediction,
            "max_feature_abs_diff": max_feature_diff,
        }
    return fixture_summary


def _recording_ids(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["recording_id"].astype(str).unique().tolist())


def _versions() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
    }


def _write_metadata(
    *,
    model_path: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    historical: dict[str, float],
    reproduced: dict[str, float],
) -> dict[str, Any]:
    metadata = {
        "model_version": "subject1_speed_benchmark_v1",
        "model_identity": "Subject 1 calibrated benchmark model",
        "model_role": "benchmark",
        "feature_names": list(FEATURES),
        "window_sec": 5.0,
        "step_sec": 2.5,
        "nominal_fs_hz": 200,
        "accepted_fs_range_hz": [190.0, 210.0],
        "maximum_internal_gap_sec": 0.05,
        "canonical_input_columns": list(CANONICAL_COLUMNS),
        "axis_mapping": {
            "x": "left/right across the face",
            "y": "forward/backward, direction of travel",
            "z": "up/down vertical",
            "gyro_x": "pitch/nodding rate, rotation about left/right axis",
            "gyro_y": "roll/side-tilt rate, rotation about forward axis",
            "gyro_z": "yaw/turning rate, rotation about vertical axis",
        },
        "input_units": {
            "ax_g": "g",
            "ay_g": "g",
            "az_g": "g",
            "gx_dps": "degrees_per_second",
            "gy_dps": "degrees_per_second",
            "gz_dps": "degrees_per_second",
        },
        "output_unit": "mph",
        "estimator_type": "RandomForestRegressor",
        "estimator_params": {
            "n_estimators": 200,
            "max_depth": 5,
            "random_state": 42,
        },
        "training_recording_ids": _recording_ids(train),
        "validation_recording_ids": _recording_ids(test),
        "training_window_count": int(len(train)),
        "validation_window_count": int(len(test)),
        "historical_validation_metrics": historical,
        "reproduced_validation_metrics": reproduced,
        "ml_commit_hash": _git_commit_hash(),
        "feature_code_commit": _git_commit_hash(),
        "model_file_sha256": _sha256(model_path),
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        **_versions(),
    }
    (EXPORT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _write_runtime_requirements() -> None:
    versions = _versions()
    lines = [
        f"joblib=={versions['joblib_version']}",
        f"numpy=={versions['numpy_version']}",
        f"scikit-learn=={versions['scikit_learn_version']}",
        f"scipy=={versions['scipy_version']}",
    ]
    (EXPORT_DIR / "requirements-lock.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_model_card(metadata: dict[str, Any], fixture_summary: dict[str, dict[str, Any]]) -> None:
    fixture_lines = "\n".join(
        f"- `{name}`: `{info['recording_id']}` window {info['window_index']} "
        f"({info['actual_speed_mph']:g} mph), prediction {info['prediction_mph']:.6f} mph"
        for name, info in fixture_summary.items()
    )
    features = "\n".join(f"{index}. `{feature}`" for index, feature in enumerate(FEATURES, start=1))
    metrics = metadata["historical_validation_metrics"]
    card = f"""# Subject 1 speed benchmark v1 model card

## Intended role

This artifact is a personalized Subject 1 benchmark prototype for LENZ
head-mounted IMU speed estimation. It is provided to preserve exact benchmark
parity for the reported Subject 1 experiment. It is not a generalized universal
estimator and is not yet the final deployment artifact.

## Model identity

- Model version: `{metadata['model_version']}`
- Model identity: `{metadata['model_identity']}`
- Model role: `{metadata['model_role']}`
- Estimator: `RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)`
- Output unit: mph
- No scaler, imputer, normalizer, or confidence-calibration layer is used.
- No statistically supported confidence score is emitted.

## Benchmark definition

- Training data: approved Subject 1 Day 3 windows only.
- Validation data: exact Subject 1 same-subject standard validation set:
  approved Subject 1 Day 2 windows plus Subject 1 Day 4 `cadence_normal`.
- Windowing: 5.0 second windows, 2.5 second step, incomplete final windows
  dropped.
- Nominal sampling rate: 200 Hz.

Reported historical validation metrics:

- MAE: `{metrics['MAE']:.15f}` mph
- RMSE: `{metrics['RMSE']:.15f}` mph
- R²: `{metrics['R2']:.15f}`

## Required input schema

Each complete window must contain canonical six-axis samples:

- `ax_g`, `ay_g`, `az_g` in g
- `gx_dps`, `gy_dps`, `gz_dps` in degrees per second

Device frame while worn:

- X axis: left/right across the face
- Y axis: forward/backward, direction of travel
- Z axis: up/down vertical
- Gyro X: pitch/nodding rate
- Gyro Y: roll/side-tilt rate
- Gyro Z: yaw/turning rate

## Runtime behavior

- Batch inference accepts one complete canonical 5-second window.
- Streaming inference needs a 5-second warm-up before the first prediction.
- Intended prediction interval after warm-up is 2.5 seconds.
- Timestamps, when supplied, must be monotonic and roughly compatible with the
  accepted 190--210 Hz effective sampling range.
- The runtime rejects non-finite samples, timestamp reversals, and large
  internal timestamp gaps instead of resampling.

## Feature order

{features}

## Golden fixtures

{fixture_lines}

## Known limitations

- Calibrated on one subject's natural Day 3 training data.
- Validation parity is specific to the historical Subject 1 benchmark split.
- Cross-subject generalization is not guaranteed by this artifact.
- Cadence-manipulation and deployment behavior require separate validation.
- This artifact provides deterministic benchmark reproduction, not medical,
  safety-critical, or production-grade uncertainty estimates.
"""
    (EXPORT_DIR / "MODEL_CARD.md").write_text(card, encoding="utf-8")


def build_export() -> dict[str, Any]:
    table = _load_feature_table()
    train, test = _standard_split(table)
    historical = _historical_metrics()
    model = _fit_model(train)
    reproduced = _metrics(model, test)
    diffs = {key: abs(reproduced[key] - historical[key]) for key in historical}
    if any(value > 1e-10 for value in diffs.values()):
        raise RuntimeError(f"reproduced metrics differ materially: {diffs}")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = EXPORT_DIR / "model.joblib"
    joblib.dump(model, model_path)
    fixture_summary = _write_fixtures(model)
    metadata = _write_metadata(
        model_path=model_path,
        train=train,
        test=test,
        historical=historical,
        reproduced=reproduced,
    )
    _write_runtime_requirements()
    _write_model_card(metadata, fixture_summary)
    return {
        "historical": historical,
        "reproduced": reproduced,
        "metric_differences": diffs,
        "fixture_summary": fixture_summary,
        "model_sha256": metadata["model_file_sha256"],
    }


def main() -> None:
    result = build_export()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
