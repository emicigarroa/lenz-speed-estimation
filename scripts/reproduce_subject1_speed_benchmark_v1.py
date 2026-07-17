"""Reproduce the frozen Subject 1 v4 morphology speed benchmark.

This script is intentionally model-freezing infrastructure, not a deployment
export. It traces the historical ``v4_morphology_all`` benchmark row, rebuilds
the exact deterministic Random Forest evaluation from the processed feature
table, compares the reproduced metrics to the historical CSV, and writes a
small reproducibility report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lenz-speed-matplotlib"),
)

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lenz_speed.data import load_manifest  # noqa: E402
from lenz_speed.modeling import get_models  # noqa: E402


FEATURE_SET_NAME = "v4_morphology_all"
MODEL_NAME = "Random Forest"
EVALUATION_NAME = "standard"
MATERIAL_TOLERANCE = 1e-10

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

FEATURE_TABLE_PATH = REPOSITORY_ROOT / "data/processed/windowed_features.csv"
HISTORICAL_METRICS_PATH = (
    REPOSITORY_ROOT / "outputs/tables/v4_morphology_feature_metrics.csv"
)
REPORT_PATH = REPOSITORY_ROOT / "outputs/reports/subject1_speed_benchmark_v1.md"


@dataclass(frozen=True)
class MetricBundle:
    """Small metric container for comparison and reporting."""

    MAE: float
    RMSE: float
    R2: float


def _git_commit_hash() -> str:
    """Return the current repository commit hash, or ``unknown`` if unavailable."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _load_historical_metrics() -> tuple[MetricBundle, pd.Series]:
    """Load the single historical benchmark row and validate its feature order."""

    if not HISTORICAL_METRICS_PATH.is_file():
        raise FileNotFoundError(f"Historical metrics CSV not found: {HISTORICAL_METRICS_PATH}")

    historical = pd.read_csv(HISTORICAL_METRICS_PATH)
    matches = historical.loc[
        (historical["evaluation"] == EVALUATION_NAME)
        & (historical["summary_level"] == "overall")
        & (historical["feature_set"] == FEATURE_SET_NAME)
        & (historical["model"] == MODEL_NAME)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one historical benchmark row for "
            f"{EVALUATION_NAME}/{FEATURE_SET_NAME}/{MODEL_NAME}; found {len(matches)}."
        )

    row = matches.iloc[0]
    historical_features = tuple(str(row["features"]).split("|"))
    if historical_features != FEATURES:
        raise RuntimeError(
            "Historical feature order differs from required benchmark feature order.\n"
            f"Historical: {historical_features}\nRequired:   {FEATURES}"
        )

    return MetricBundle(
        MAE=float(row["MAE"]),
        RMSE=float(row["RMSE"]),
        R2=float(row["R2"]),
    ), row


def _load_feature_table() -> pd.DataFrame:
    """Load the processed window feature table and validate required columns."""

    if not FEATURE_TABLE_PATH.is_file():
        raise FileNotFoundError(
            f"Processed feature table not found: {FEATURE_TABLE_PATH}. "
            "Run python run_pipeline.py before reproducing the benchmark."
        )
    table = pd.read_csv(FEATURE_TABLE_PATH)
    required = {
        "recording_id",
        "relative_path",
        "subject_id",
        "session",
        "condition",
        "speed_mph",
        *FEATURES,
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise RuntimeError(
            "Processed feature table is missing required benchmark columns: "
            + ", ".join(missing)
        )
    return table


def _standard_subject1_split(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the exact Subject 1 standard train/validation split."""

    approved_recordings = set(load_manifest()["recording_id"].astype(str))
    approved = table.loc[table["recording_id"].astype(str).isin(approved_recordings)].copy()

    train = approved.loc[
        (approved["subject_id"] == "subject_1")
        & (approved["session"] == "day3")
    ].copy()
    test = approved.loc[
        (approved["subject_id"] == "subject_1")
        & (
            (approved["session"] == "day2")
            | (
                (approved["session"] == "day4")
                & (approved["condition"] == "cadence_normal")
            )
        )
    ].copy()

    if train.empty:
        raise RuntimeError("Benchmark training split is empty; expected Subject 1 Day 3.")
    if test.empty:
        raise RuntimeError(
            "Benchmark validation split is empty; expected Subject 1 Day 2 plus "
            "Subject 1 Day 4 cadence_normal."
        )

    numeric_columns = [*FEATURES, "speed_mph"]
    for name, split in (("training", train), ("validation", test)):
        try:
            split[numeric_columns] = split[numeric_columns].apply(
                pd.to_numeric,
                errors="raise",
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"{name} split contains non-numeric values.") from error
        if not np.isfinite(split[numeric_columns].to_numpy(dtype=float)).all():
            raise RuntimeError(f"{name} split contains missing or non-finite values.")

    return train, test


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[Any, np.ndarray]:
    """Fit the deterministic benchmark Random Forest and return predictions."""

    model = clone(get_models()[MODEL_NAME])
    expected_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "random_state": 42,
    }
    actual_params = model.get_params()
    for name, expected in expected_params.items():
        if actual_params.get(name) != expected:
            raise RuntimeError(
                f"Unexpected {MODEL_NAME} parameter {name}: "
                f"{actual_params.get(name)!r} != {expected!r}"
            )

    model.fit(train.loc[:, FEATURES], train["speed_mph"])
    return model, model.predict(test.loc[:, FEATURES])


def _metrics(actual: pd.Series, predicted: np.ndarray) -> MetricBundle:
    """Calculate benchmark metrics."""

    actual_array = actual.to_numpy(dtype=float)
    return MetricBundle(
        MAE=float(mean_absolute_error(actual_array, predicted)),
        RMSE=float(np.sqrt(mean_squared_error(actual_array, predicted))),
        R2=float(r2_score(actual_array, predicted)),
    )


def _recording_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return recording IDs, paths, speeds, and window counts."""

    return (
        frame.groupby(["recording_id", "relative_path", "speed_mph"], sort=True)
        .size()
        .reset_index(name="windows")
    )


def _metric_differences(
    historical: MetricBundle,
    reproduced: MetricBundle,
) -> MetricBundle:
    """Return absolute metric differences."""

    return MetricBundle(
        MAE=abs(reproduced.MAE - historical.MAE),
        RMSE=abs(reproduced.RMSE - historical.RMSE),
        R2=abs(reproduced.R2 - historical.R2),
    )


def _assert_reproduced(differences: MetricBundle) -> None:
    """Fail loudly if metrics differ beyond the material tolerance."""

    bad = {
        name: value
        for name, value in asdict(differences).items()
        if value > MATERIAL_TOLERANCE
    }
    if bad:
        raise RuntimeError(
            "Reproduced benchmark metrics differ materially from the historical CSV "
            f"at tolerance {MATERIAL_TOLERANCE:g}: {bad}"
        )


def _versions() -> dict[str, str]:
    """Return environment versions relevant to benchmark reproduction."""

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _format_metric_table(
    historical: MetricBundle,
    reproduced: MetricBundle,
    differences: MetricBundle,
) -> str:
    rows = [
        ("MAE", historical.MAE, reproduced.MAE, differences.MAE),
        ("RMSE", historical.RMSE, reproduced.RMSE, differences.RMSE),
        ("R2", historical.R2, reproduced.R2, differences.R2),
    ]
    lines = [
        "| metric | historical | reproduced | absolute_difference |",
        "|---|---:|---:|---:|",
    ]
    for name, hist, repro, diff in rows:
        lines.append(f"| {name} | {hist:.15f} | {repro:.15f} | {diff:.3e} |")
    return "\n".join(lines)


def _format_recordings(title: str, frame: pd.DataFrame) -> str:
    lines = [f"### {title}", "", "| recording_id | speed_mph | windows | relative_path |", "|---|---:|---:|---|"]
    for row in frame.to_dict(orient="records"):
        lines.append(
            f"| {row['recording_id']} | {float(row['speed_mph']):g} | "
            f"{int(row['windows'])} | `{row['relative_path']}` |"
        )
    return "\n".join(lines)


def _write_report(
    *,
    historical: MetricBundle,
    reproduced: MetricBundle,
    differences: MetricBundle,
    train: pd.DataFrame,
    test: pd.DataFrame,
    model: Any,
) -> None:
    """Write a concise Markdown benchmark report."""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_summary = _recording_summary(train)
    test_summary = _recording_summary(test)
    versions = _versions()
    estimator_params = {
        key: model.get_params()[key]
        for key in ("n_estimators", "max_depth", "random_state")
    }

    report = f"""# Subject 1 speed benchmark v1 reproduction

## Result

Reproduction status: **PASS**

Historical source: `{HISTORICAL_METRICS_PATH.relative_to(REPOSITORY_ROOT)}`

{_format_metric_table(historical, reproduced, differences)}

Material tolerance: `{MATERIAL_TOLERANCE:g}` absolute difference for each metric.

## Benchmark identity

- Experiment: `{FEATURE_SET_NAME}`
- Evaluation: Subject 1 same-subject standard validation
- Model: `{MODEL_NAME}`
- Estimator class: `{model.__class__.__name__}`
- Estimator parameters: `{estimator_params}`
- Training split: approved Subject 1 Day 3 windows only
- Validation split: approved Subject 1 Day 2 windows plus Subject 1 Day 4 `cadence_normal`
- Windowing: 5.0 s windows, 2.5 s step, incomplete final windows dropped
- Sampling assumption: 200 Hz
- Output unit: mph

## Feature order

{chr(10).join(f'{index}. `{feature}`' for index, feature in enumerate(FEATURES, start=1))}

## Training data

- Training windows: {len(train)}
- Training recordings: {train['recording_id'].nunique()}

{_format_recordings("Training recordings", train_summary)}

## Validation data

- Validation windows: {len(test)}
- Validation recordings: {test['recording_id'].nunique()}

{_format_recordings("Validation recordings", test_summary)}

## Code path traced

- Historical experiment script: `scripts/v4_morphology_experiment.py`
- Split helper mirrored from: `src/lenz_speed/evaluation.py::_same_subject_split`
- Model definition source: `src/lenz_speed/modeling.py::get_models`
- Feature extraction source: `src/lenz_speed/features.py::extract_window_features`
- Windowing source: `src/lenz_speed/windowing.py::make_windows`
- Packet/schema loading source: `src/lenz_speed/data.py`

## Preprocessing and feature-formula summary

- CSV rows are filtered to valid finite IMU packet rows before sample indexing.
- Required signal columns are `ax_g`, `ay_g`, `az_g`, `gx_dps`, `gy_dps`, `gz_dps`.
- Acceleration unit: g.
- Gyroscope unit: degrees per second.
- Axis convention: X lateral, Y forward/backward, Z vertical.
- Cadence, `RMS_Z`, `PeakToPeak_Z`, gait-event, and morphology features use mean-centered Z acceleration filtered by a 4th-order 0.7--5.0 Hz Butterworth bandpass.
- Cadence searches the dominant FFT frequency from 0.8--4.0 Hz and reports steps/min.
- Gyroscope RMS features use raw angular-rate channels.
- Acceleration magnitude features use raw three-axis acceleration magnitude.
- No scaler, encoder, normalizer, imputer, or preprocessing object is used by this benchmark model.

## Environment

- ML git commit: `{_git_commit_hash()}`
{chr(10).join(f'- {name}: `{value}`' for name, value in versions.items())}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def reproduce_benchmark() -> dict[str, Any]:
    """Reproduce the benchmark and write the report."""

    historical_metrics, historical_row = _load_historical_metrics()
    table = _load_feature_table()
    train, test = _standard_subject1_split(table)
    model, predicted = _fit_predict(train, test)
    reproduced_metrics = _metrics(test["speed_mph"], predicted)
    differences = _metric_differences(historical_metrics, reproduced_metrics)
    _assert_reproduced(differences)
    _write_report(
        historical=historical_metrics,
        reproduced=reproduced_metrics,
        differences=differences,
        train=train,
        test=test,
        model=model,
    )
    return {
        "historical": asdict(historical_metrics),
        "reproduced": asdict(reproduced_metrics),
        "differences": asdict(differences),
        "historical_row": historical_row.to_dict(),
        "train_recordings": _recording_summary(train).to_dict(orient="records"),
        "validation_recordings": _recording_summary(test).to_dict(orient="records"),
        "report_path": str(REPORT_PATH),
    }


def main() -> None:
    """CLI entry point."""

    result = reproduce_benchmark()
    print("Subject 1 speed benchmark v1 reproduction: PASS")
    print("Historical metrics:", result["historical"])
    print("Reproduced metrics:", result["reproduced"])
    print("Absolute differences:", result["differences"])
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
