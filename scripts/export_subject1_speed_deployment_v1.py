"""Build the Subject 1 live-UI deployment speed model export package.

This script intentionally creates a deployment artifact that is separate from
the frozen benchmark artifact. It reuses the benchmark runtime feature
extraction code byte-for-byte, but retrains the estimator on the expanded
approved Subject 1 natural-cadence dataset.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
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
from sklearn.ensemble import RandomForestRegressor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lenz_speed.data import load_manifest, load_recording  # noqa: E402
from lenz_speed.dataset import _selected_trim_values  # noqa: E402
from lenz_speed.features import extract_window_features  # noqa: E402
from lenz_speed.windowing import apply_trim, make_windows  # noqa: E402


BENCHMARK_VERSION = "subject1_speed_benchmark_v1"
BENCHMARK_COMMIT = "765ac43"
DEPLOYMENT_VERSION = "subject1_speed_deployment_v1"

BENCHMARK_DIR = REPOSITORY_ROOT / "exported_models" / BENCHMARK_VERSION
EXPORT_DIR = REPOSITORY_ROOT / "exported_models" / DEPLOYMENT_VERSION
FIXTURE_DIR = EXPORT_DIR / "fixtures"
FEATURE_TABLE_PATH = REPOSITORY_ROOT / "data/processed/windowed_features.csv"

CANONICAL_COLUMNS = ("ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps")
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
    "high_speed": ("s1_day3_8p5mph", 0),
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


def _copy_runtime_files() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ("feature_extractor.py", "inference.py", "requirements-lock.txt"):
        shutil.copyfile(BENCHMARK_DIR / filename, EXPORT_DIR / filename)


def _load_exported_feature_extractor() -> Any:
    path = EXPORT_DIR / "feature_extractor.py"
    spec = importlib.util.spec_from_file_location(
        "subject1_deployment_feature_extractor",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load exported feature extractor from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_feature_table() -> pd.DataFrame:
    table = pd.read_csv(FEATURE_TABLE_PATH)
    required = {"recording_id", "subject_id", "session", "condition", "speed_mph", *FEATURES}
    missing = sorted(required - set(table.columns))
    if missing:
        raise RuntimeError("windowed feature table is missing columns: " + ", ".join(missing))
    return table


def _is_trustworthy_subject1_natural_recording(row: pd.Series) -> bool:
    if row["subject_id"] != "subject_1" or not bool(row["include"]):
        return False
    if pd.isna(row["speed_mph"]):
        return False
    session = str(row["session"])
    condition = str(row["condition"])
    if session == "day3" and condition == "steady_state":
        return True
    if session == "day2" and condition == "steady_state":
        return True
    if session == "day4" and condition == "cadence_normal":
        return True
    return False


def _subject1_training_plan(
    manifest: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject1 = manifest.loc[manifest["subject_id"] == "subject_1"].copy()
    windows_by_recording = (
        feature_table.groupby("recording_id", sort=True)
        .size()
        .rename("windows")
        .reset_index()
    )
    subject1 = subject1.merge(windows_by_recording, on="recording_id", how="left")
    subject1["windows"] = subject1["windows"].fillna(0).astype(int)
    subject1["proposed_include"] = [
        _is_trustworthy_subject1_natural_recording(row)
        for _, row in subject1.iterrows()
    ]

    rationales: list[str] = []
    exclusion_reasons: list[str] = []
    for _, row in subject1.iterrows():
        session = str(row["session"])
        condition = str(row["condition"])
        include = bool(row["include"])
        if bool(row["proposed_include"]):
            if session == "day3":
                rationales.append("approved Subject 1 Day 3 natural steady-state training recording")
            elif session == "day2":
                rationales.append("approved Subject 1 Day 2 steady-state benchmark validation recording")
            else:
                rationales.append("approved Subject 1 Day 4 normal-cadence recording")
            exclusion_reasons.append("")
            continue

        rationales.append("")
        if not include:
            reason = str(row.get("exclusion_reason", "")).strip()
            exclusion_reasons.append(reason or "manifest include=false")
        elif condition in {"cadence_decreased", "cadence_elevated"}:
            exclusion_reasons.append("cadence-manipulation recording excluded from deployment training")
        elif row["subject_id"] != "subject_1":
            exclusion_reasons.append("non-Subject-1 recording")
        elif pd.isna(row["speed_mph"]):
            exclusion_reasons.append("uncertain target-speed label")
        else:
            exclusion_reasons.append("not part of approved Subject 1 natural-cadence deployment policy")

    subject1["inclusion_rationale"] = rationales
    subject1["deployment_exclusion_reason"] = exclusion_reasons
    included = subject1.loc[subject1["proposed_include"]].copy()
    excluded = subject1.loc[~subject1["proposed_include"]].copy()
    return included, excluded


def _validate_training_set(included: pd.DataFrame, train: pd.DataFrame) -> None:
    recording_ids = included["recording_id"].astype(str).tolist()
    if len(recording_ids) != len(set(recording_ids)):
        raise RuntimeError("duplicate recording ID in deployment training set.")
    if (included["subject_id"] != "subject_1").any():
        raise RuntimeError("deployment training set contains a non-Subject-1 recording.")
    if included["condition"].isin(["cadence_decreased", "cadence_elevated"]).any():
        raise RuntimeError("deployment training set contains cadence-manipulation recordings.")
    if (~included["include"]).any():
        raise RuntimeError("deployment training set contains manifest-excluded recordings.")
    if train.empty:
        raise RuntimeError("deployment training split is empty.")
    numeric = train.loc[:, [*FEATURES, "speed_mph"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("deployment training split contains missing or non-finite features.")


def _deployment_training_frame(
    included: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    included_ids = set(included["recording_id"].astype(str))
    train = feature_table.loc[
        feature_table["recording_id"].astype(str).isin(included_ids)
    ].copy()
    observed_ids = set(train["recording_id"].astype(str))
    missing = sorted(included_ids - observed_ids)
    if missing:
        raise RuntimeError("included recordings have no feature rows: " + ", ".join(missing))
    return train


def _fit_model(train: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(train.loc[:, FEATURES], train["speed_mph"])
    return model


def _fixture_window(recording_id: str, window_index: int) -> tuple[pd.DataFrame, Any, pd.Series]:
    manifest = load_manifest(include_excluded=True)
    matches = manifest.loc[manifest["recording_id"].astype(str) == recording_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one manifest row for fixture {recording_id}; found {len(matches)}")
    row = matches.iloc[0]
    recording = load_recording(recording_id, include_excluded=True)
    trim_start, trim_end = _selected_trim_values(row.to_dict())
    trimmed = apply_trim(recording, trim_start_sec=trim_start, trim_end_sec=trim_end, fs=200)
    windows = make_windows(trimmed, recording_id=recording_id, fs=200)
    if window_index >= len(windows):
        raise RuntimeError(f"{recording_id} has no fixture window index {window_index}.")
    return windows[window_index].signal.loc[:, list(CANONICAL_COLUMNS)].copy(), windows[window_index], row


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CANONICAL_COLUMNS))
        writer.writeheader()
        for record in frame.to_dict(orient="records"):
            writer.writerow({column: record[column] for column in CANONICAL_COLUMNS})


def _write_fixtures(model: RandomForestRegressor) -> dict[str, dict[str, Any]]:
    exporter = _load_exported_feature_extractor()
    fixture_summary: dict[str, dict[str, Any]] = {}
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


def _speed_distribution(train: pd.DataFrame) -> dict[str, dict[str, int]]:
    summary = (
        train.groupby("speed_mph", sort=True)
        .agg(windows=("recording_id", "size"), recordings=("recording_id", "nunique"))
        .reset_index()
    )
    return {
        f"{float(row.speed_mph):g}": {
            "recordings": int(row.recordings),
            "windows": int(row.windows),
        }
        for row in summary.itertuples(index=False)
    }


def _recording_plan_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "recording_id",
        "subject_id",
        "session",
        "speed_mph",
        "condition",
        "windows",
        "inclusion_rationale",
    ]
    return _json_safe_rows(frame.loc[:, columns].to_dict(orient="records"))


def _excluded_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "recording_id",
        "subject_id",
        "session",
        "speed_mph",
        "condition",
        "include",
        "deployment_exclusion_reason",
    ]
    return _json_safe_rows(frame.loc[:, columns].to_dict(orient="records"))


def _json_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_rows: list[dict[str, Any]] = []
    for row in rows:
        safe_row: dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                safe_row[key] = None
            elif isinstance(value, np.generic):
                safe_row[key] = value.item()
            else:
                safe_row[key] = value
        safe_rows.append(safe_row)
    return safe_rows


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
    included: pd.DataFrame,
    excluded: pd.DataFrame,
    train: pd.DataFrame,
) -> dict[str, Any]:
    metadata = {
        "model_version": DEPLOYMENT_VERSION,
        "model_role": "deployment",
        "model_identity": "Subject 1 calibrated deployment model",
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
        "deployment_training_recording_ids": sorted(included["recording_id"].astype(str).tolist()),
        "deployment_training_recordings": _recording_plan_rows(included.sort_values(["session", "speed_mph", "recording_id"])),
        "excluded_subject1_recordings": _excluded_rows(excluded.sort_values(["session", "speed_mph", "recording_id"])),
        "deployment_training_recording_count": int(included["recording_id"].nunique()),
        "deployment_training_window_count": int(len(train)),
        "deployment_speed_distribution": _speed_distribution(train),
        "originating_benchmark_version": BENCHMARK_VERSION,
        "originating_benchmark_commit": BENCHMARK_COMMIT,
        "benchmark_selection_frozen_before_deployment_retraining": True,
        "independent_benchmark_metrics_do_not_apply_to_deployment_v1": True,
        "ml_commit_hash": _git_commit_hash(),
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_file_sha256": _sha256(model_path),
        **_versions(),
    }
    (EXPORT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _write_model_card(metadata: dict[str, Any], fixture_summary: dict[str, dict[str, Any]]) -> None:
    fixture_lines = "\n".join(
        f"- `{name}`: `{info['recording_id']}` window {info['window_index']} "
        f"({info['actual_speed_mph']:g} mph), prediction {info['prediction_mph']:.6f} mph"
        for name, info in fixture_summary.items()
    )
    training_lines = "\n".join(
        f"- `{row['recording_id']}`: {row['session']}, {row['speed_mph']:g} mph, "
        f"`{row['condition']}`, {row['windows']} windows"
        for row in metadata["deployment_training_recordings"]
    )
    card = f"""# Subject 1 speed deployment v1 model card

## Artifact identity

- Model version: `{metadata['model_version']}`
- Model identity: `{metadata['model_identity']}`
- Model role: `{metadata['model_role']}`
- Estimator: `RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)`
- Feature set: exact 19-feature `v4_morphology_all` order from the frozen benchmark package.
- Output unit: mph.

## Benchmark vs deployment distinction

Benchmark artifact:

- Version: `{BENCHMARK_VERSION}`
- Commit: `{BENCHMARK_COMMIT}`
- Trained on approved Subject 1 Day 3 only.
- Independently validated on Subject 1 Day 2 plus Subject 1 Day 4 normal-cadence recordings.
- Historical validation MAE: `0.1634656501460112` mph.

Deployment artifact:

- Retrained using the expanded approved Subject 1 natural-cadence dataset.
- Includes approved Subject 1 Day 3 steady-state, Subject 1 Day 2 steady-state, and Subject 1 Day 4 `cadence_normal` recordings.
- Intended for personalized Subject 1 live inference in the UI.
- Does **not** carry the benchmark MAE as independent deployment performance, because former validation recordings are now part of deployment training.
- The benchmark selection was frozen before this deployment retraining.

## Deployment training set

- Training recordings: {metadata['deployment_training_recording_count']}
- Training windows: {metadata['deployment_training_window_count']}

{training_lines}

## Runtime input contract

Required columns per sample:

- `ax_g`, `ay_g`, `az_g` in g
- `gx_dps`, `gy_dps`, `gz_dps` in degrees per second

Device frame while worn:

- X axis: left/right across the face
- Y axis: forward/backward, direction of travel
- Z axis: up/down vertical
- Gyro X: pitch/nodding rate
- Gyro Y: roll/side-tilt rate
- Gyro Z: yaw/turning rate

Runtime behavior:

- 5.0-second window at nominal 200 Hz.
- 2.5-second intended prediction interval.
- Streaming mode requires a 5-second warm-up before the first prediction.
- Timestamps, when supplied, must be monotonic and consistent with the accepted 190--210 Hz effective sampling range.
- Large timestamp gaps reset the streaming buffer.
- No statistically supported confidence score is emitted.

## Golden fixtures

{fixture_lines}

## Known limitations

- Personalized to Subject 1.
- Not a universal cross-subject speed estimator.
- Not a medical, safety-critical, or uncertainty-calibrated model.
- Deployment behavior should still be validated in live UI conditions.
"""
    (EXPORT_DIR / "MODEL_CARD.md").write_text(card, encoding="utf-8")


def build_export() -> dict[str, Any]:
    _copy_runtime_files()
    manifest = load_manifest(include_excluded=True)
    feature_table = _load_feature_table()
    included, excluded = _subject1_training_plan(manifest, feature_table)
    train = _deployment_training_frame(included, feature_table)
    _validate_training_set(included, train)
    model = _fit_model(train)
    model_path = EXPORT_DIR / "model.joblib"
    joblib.dump(model, model_path)
    fixture_summary = _write_fixtures(model)
    metadata = _write_metadata(
        model_path=model_path,
        included=included,
        excluded=excluded,
        train=train,
    )
    _write_model_card(metadata, fixture_summary)
    return {
        "model_checksum": metadata["model_file_sha256"],
        "training_recording_count": metadata["deployment_training_recording_count"],
        "training_window_count": metadata["deployment_training_window_count"],
        "speed_distribution": metadata["deployment_speed_distribution"],
        "included_recordings": metadata["deployment_training_recordings"],
        "excluded_subject1_recordings": metadata["excluded_subject1_recordings"],
        "fixture_summary": fixture_summary,
    }


def main() -> None:
    print(json.dumps(build_export(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
