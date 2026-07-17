"""Deterministic same-subject evaluation for LENZ speed models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import load_manifest
from .modeling import DEFAULT_FEATURES, REDUCED_FEATURE_SETS, get_models


_TARGET_COLUMN = "speed_mph"
_SPLIT_COLUMNS = ("recording_id", "subject_id", "session", _TARGET_COLUMN)
_PREDICTION_METADATA = (
    "recording_id",
    "relative_path",
    "subject_id",
    "session",
    "file_type",
    "condition",
    "notes",
    "window_index",
    "window_start_sec",
    "window_end_sec",
    _TARGET_COLUMN,
)
_MORPHOLOGY_FEATURES = (
    "Impact_Impulse",
    "Peak_Symmetry",
    "Impact_Crest_Factor",
    "Impact_Local_Kurtosis",
)
_V2_FEATURES = (
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
)
_V2_PLUS_SHARPNESS_FEATURES = (*_V2_FEATURES, "Vertical_Peak_Sharpness")
_V4_MORPHOLOGY_FEATURES = (*_V2_PLUS_SHARPNESS_FEATURES, *_MORPHOLOGY_FEATURES)
_LATEST_MORPHOLOGY_FEATURES = tuple(
    dict.fromkeys((*DEFAULT_FEATURES, *_MORPHOLOGY_FEATURES))
)
_NORMAL_CONDITIONS = {"steady_state", "cadence_normal", "normal"}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_feature_table(dataframe: pd.DataFrame | None) -> pd.DataFrame:
    if dataframe is not None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame or None.")
        return dataframe.copy()

    path = _repository_root() / "data/processed/windowed_features.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed feature table not found: {path}. Build it before evaluation."
        )
    return pd.read_csv(path)


def _validate_columns(table: pd.DataFrame, features: Sequence[str]) -> None:
    required = set(_SPLIT_COLUMNS) | set(_PREDICTION_METADATA) | set(features)
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(
            "Feature table is missing required columns: " + ", ".join(missing)
        )


def _same_subject_split(
    table: pd.DataFrame,
    features: Sequence[str],
    *,
    mode: Literal["standard", "cadence_stress"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a fixed Subject 1 training and validation split."""

    _validate_columns(table, features)
    approved_recordings = set(load_manifest()["recording_id"].astype(str))
    approved = table.loc[
        table["recording_id"].astype(str).isin(approved_recordings)
    ].copy()

    train = approved.loc[
        (approved["subject_id"] == "subject_1")
        & (approved["session"] == "day3")
    ].copy()
    subject_1 = approved["subject_id"] == "subject_1"
    if mode == "standard":
        test_mask = subject_1 & (
            (approved["session"] == "day2")
            | (
                (approved["session"] == "day4")
                & (approved["condition"] == "cadence_normal")
            )
        )
        expected_test = "approved Subject 1 Day 2 and normal-cadence Day 4 rows"
    elif mode == "cadence_stress":
        test_mask = subject_1 & (approved["session"] == "day4")
        expected_test = "approved Subject 1 Day 4 rows"
    else:
        raise ValueError(f"Unsupported same-subject evaluation mode: {mode!r}")
    test = approved.loc[test_mask].copy()

    if train.empty:
        raise ValueError("Training split is empty; expected Subject 1 Day 3 rows.")
    if test.empty:
        raise ValueError(f"Validation split is empty; expected {expected_test}.")

    numeric_columns = [*features, _TARGET_COLUMN]
    for name, split in (("training", train), ("validation", test)):
        try:
            split[numeric_columns] = split[numeric_columns].apply(
                pd.to_numeric,
                errors="raise",
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"The {name} split contains non-numeric model values.") from error
        if not np.isfinite(split[numeric_columns].to_numpy(dtype=float)).all():
            raise ValueError(f"The {name} split contains missing or non-finite values.")
    return train, test


def _evaluate_specs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    specs: Sequence[tuple[str, Sequence[str], str, RegressorMixin]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for feature_set, features, model_name, model in specs:
        feature_names = list(features)
        estimator = clone(model)
        estimator.fit(train[feature_names], train[_TARGET_COLUMN])
        predicted = estimator.predict(test[feature_names])
        actual = test[_TARGET_COLUMN].to_numpy(dtype=float)

        metrics_rows.append(
            {
                "feature_set": feature_set,
                "features": "|".join(feature_names),
                "model": model_name,
                "n_train_windows": len(train),
                "n_test_windows": len(test),
                "n_train_recordings": train["recording_id"].nunique(),
                "n_test_recordings": test["recording_id"].nunique(),
                "MAE": float(mean_absolute_error(actual, predicted)),
                "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
                "R2": float(r2_score(actual, predicted)),
            }
        )

        predictions = test.loc[:, _PREDICTION_METADATA].reset_index(drop=True).copy()
        predictions = predictions.rename(columns={_TARGET_COLUMN: "actual_speed_mph"})
        predictions.insert(0, "model", model_name)
        predictions.insert(0, "features", "|".join(feature_names))
        predictions.insert(0, "feature_set", feature_set)
        predictions["predicted_speed_mph"] = predicted
        predictions["residual_mph"] = predictions["predicted_speed_mph"] - actual
        predictions["absolute_error_mph"] = np.abs(predictions["residual_mph"])
        prediction_frames.append(predictions)

    return (
        pd.DataFrame(metrics_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def _output_directory(output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser()
    if not destination.is_absolute():
        destination = _repository_root() / destination
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _load_prediction_table(predictions_path: str | Path) -> pd.DataFrame:
    path = Path(predictions_path).expanduser()
    if not path.is_absolute():
        path = _repository_root() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prediction table not found: {path}")

    table = pd.read_csv(path)
    required = {
        "model",
        "recording_id",
        "condition",
        "actual_speed_mph",
        "residual_mph",
        "absolute_error_mph",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(
            f"Prediction table {path} is missing columns: {', '.join(missing)}"
        )
    return table


def _rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values.to_numpy(dtype=float)))))


def cadence_stress_error_analysis(
    predictions_path: str | Path = "outputs/tables/cadence_stress_predictions.csv",
    *,
    output_dir: str | Path = "outputs/tables",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize cadence-stress prediction error by speed, condition, and recording.

    The first returned table groups error by actual speed, cadence condition,
    and model. The second groups by recording and model, sorted from highest
    to lowest recording-level MAE to make the worst stress-test recordings easy
    to inspect.
    """

    table = _load_prediction_table(predictions_path)
    table = table.copy()
    table["speed_mph"] = pd.to_numeric(table["actual_speed_mph"], errors="raise")
    table["residual_mph"] = pd.to_numeric(table["residual_mph"], errors="raise")
    table["absolute_error_mph"] = pd.to_numeric(
        table["absolute_error_mph"],
        errors="raise",
    )

    error_by_speed_condition = (
        table.groupby(["speed_mph", "condition", "model"], dropna=False)
        .agg(
            n_windows=("absolute_error_mph", "size"),
            n_recordings=("recording_id", "nunique"),
            mean_error_mph=("residual_mph", "mean"),
            mean_absolute_error_mph=("absolute_error_mph", "mean"),
            median_absolute_error_mph=("absolute_error_mph", "median"),
            rmse_mph=("residual_mph", _rmse),
            max_absolute_error_mph=("absolute_error_mph", "max"),
        )
        .reset_index()
        .sort_values(["speed_mph", "condition", "model"])
    )

    worst_recordings = (
        table.groupby(
            ["recording_id", "speed_mph", "condition", "model"],
            dropna=False,
        )
        .agg(
            n_windows=("absolute_error_mph", "size"),
            mean_error_mph=("residual_mph", "mean"),
            mean_absolute_error_mph=("absolute_error_mph", "mean"),
            median_absolute_error_mph=("absolute_error_mph", "median"),
            rmse_mph=("residual_mph", _rmse),
            max_absolute_error_mph=("absolute_error_mph", "max"),
        )
        .reset_index()
        .sort_values(
            ["mean_absolute_error_mph", "rmse_mph", "recording_id", "model"],
            ascending=[False, False, True, True],
        )
    )

    destination = _output_directory(output_dir)
    speed_condition_path = destination / "error_by_speed_condition.csv"
    worst_recordings_path = destination / "worst_recordings.csv"
    error_by_speed_condition.to_csv(speed_condition_path, index=False)
    worst_recordings.to_csv(worst_recordings_path, index=False)

    print(
        "Cadence stress error analysis: "
        f"{len(error_by_speed_condition)} speed/condition/model rows; "
        f"{len(worst_recordings)} recording/model rows."
    )
    print(f"Saved speed/condition errors to {speed_condition_path}")
    print(f"Saved worst recordings to {worst_recordings_path}")
    return error_by_speed_condition, worst_recordings


def same_subject_validation(
    dataframe: pd.DataFrame | None = None,
    *,
    output_dir: str | Path = "outputs/tables",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run :func:`same_subject_standard_validation` for compatibility."""

    return same_subject_standard_validation(dataframe, output_dir=output_dir)


def same_subject_standard_validation(
    dataframe: pd.DataFrame | None = None,
    *,
    output_dir: str | Path = "outputs/tables",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate baseline models on standard same-subject validation data.

    Models train on approved Subject 1 Day 3 windows and test on approved
    Subject 1 Day 2 windows plus normal-cadence Subject 1 Day 4 windows. This
    supplies standard validation coverage at 2, 3, 4, 5, 6, 7, and 8 mph while
    keeping cadence-manipulation trials out of the headline result.
    """

    table = _load_feature_table(dataframe)
    train, test = _same_subject_split(
        table,
        DEFAULT_FEATURES,
        mode="standard",
    )
    specs = [
        ("DEFAULT_FEATURES", DEFAULT_FEATURES, model_name, model)
        for model_name, model in get_models().items()
    ]
    metrics, predictions = _evaluate_specs(train, test, specs)

    destination = _output_directory(output_dir)
    metrics_path = destination / "same_subject_standard_metrics.csv"
    predictions_path = destination / "same_subject_standard_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print(
        f"Standard same-subject validation: {len(train)} training windows from "
        f"{train['recording_id'].nunique()} recordings; {len(test)} test windows "
        f"from {test['recording_id'].nunique()} recordings."
    )
    print(metrics.loc[:, ["model", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    return metrics, predictions


def same_subject_cadence_stress_test(
    dataframe: pd.DataFrame | None = None,
    *,
    output_dir: str | Path = "outputs/tables",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate baseline models on every Subject 1 Day 4 cadence condition.

    Models train on approved Subject 1 Day 3 windows. The test split contains
    all approved Subject 1 Day 4 recordings and is reported separately as a
    cadence-manipulation stress test rather than headline validation.
    """

    table = _load_feature_table(dataframe)
    train, test = _same_subject_split(
        table,
        DEFAULT_FEATURES,
        mode="cadence_stress",
    )
    specs = [
        ("DEFAULT_FEATURES", DEFAULT_FEATURES, model_name, model)
        for model_name, model in get_models().items()
    ]
    metrics, predictions = _evaluate_specs(train, test, specs)

    destination = _output_directory(output_dir)
    metrics_path = destination / "cadence_stress_metrics.csv"
    predictions_path = destination / "cadence_stress_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print(
        f"Same-subject cadence stress test: {len(train)} training windows from "
        f"{train['recording_id'].nunique()} recordings; {len(test)} test windows "
        f"from {test['recording_id'].nunique()} recordings."
    )
    print(metrics.loc[:, ["model", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    return metrics, predictions


def subject2_normal_cross_subject_validation(
    dataframe: pd.DataFrame | None = None,
    *,
    output_dir: str | Path = "outputs/tables",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate Subject 1-trained models on Subject 2 normal-cadence data.

    Models train only on approved Subject 1 Day 3 windows and test only on
    approved Subject 2 windows labeled ``condition == "cadence_normal"``. No
    Subject 2 windows are used for training. The feature set is the current
    default model feature set plus the experimental v4 morphology descriptors,
    keeping this cross-subject experiment separate from the existing validation
    behavior.
    """

    table = _load_feature_table(dataframe)
    features = _LATEST_MORPHOLOGY_FEATURES
    _validate_columns(table, features)

    approved_recordings = set(load_manifest()["recording_id"].astype(str))
    approved = table.loc[
        table["recording_id"].astype(str).isin(approved_recordings)
    ].copy()

    train = approved.loc[
        (approved["subject_id"] == "subject_1")
        & (approved["session"] == "day3")
    ].copy()
    test = approved.loc[
        (approved["subject_id"] == "subject_2")
        & (approved["condition"] == "cadence_normal")
    ].copy()

    if train.empty:
        raise ValueError("Training split is empty; expected Subject 1 Day 3 rows.")
    if test.empty:
        raise ValueError(
            "Cross-subject test split is empty; expected included Subject 2 "
            "cadence_normal rows."
        )

    numeric_columns = [*features, _TARGET_COLUMN]
    for name, split in (("training", train), ("cross-subject test", test)):
        try:
            split[numeric_columns] = split[numeric_columns].apply(
                pd.to_numeric,
                errors="raise",
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"The {name} split contains non-numeric values.") from error
        if not np.isfinite(split[numeric_columns].to_numpy(dtype=float)).all():
            raise ValueError(f"The {name} split contains missing or non-finite values.")

    specs = [
        ("LATEST_MORPHOLOGY_FEATURES", features, model_name, model)
        for model_name, model in get_models().items()
    ]
    metrics, predictions = _evaluate_specs(train, test, specs)

    destination = _output_directory(output_dir)
    metrics_path = destination / "subject2_normal_cross_subject_metrics.csv"
    predictions_path = destination / "subject2_normal_cross_subject_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print(
        "Subject 2 normal-cadence cross-subject validation: "
        f"{len(train)} training windows from {train['recording_id'].nunique()} "
        f"Subject 1 recordings; {len(test)} test windows from "
        f"{test['recording_id'].nunique()} Subject 2 normal-cadence recordings."
    )
    print(metrics.loc[:, ["model", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    return metrics, predictions


def subjects_1_3_development_validation(
    dataframe: pd.DataFrame | None = None,
    *,
    output_dir: str | Path = "outputs/tables",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a development cross-subject check using Subjects 1--3 only.

    The split trains on normal/natural Subject 1 and Subject 3 windows, then
    tests on Subject 2 normal-cadence windows. Subject 4 is intentionally
    excluded so it remains untouched for the final frozen test.
    """

    table = _load_feature_table(dataframe)
    features = _V4_MORPHOLOGY_FEATURES
    _validate_columns(table, features)
    approved_recordings = set(load_manifest()["recording_id"].astype(str))
    approved = table.loc[
        table["recording_id"].astype(str).isin(approved_recordings)
    ].copy()
    normal = approved["condition"].isin(_NORMAL_CONDITIONS)
    train = approved.loc[
        approved["subject_id"].isin(["subject_1", "subject_3"]) & normal
    ].copy()
    test = approved.loc[
        (approved["subject_id"] == "subject_2")
        & (approved["condition"] == "cadence_normal")
    ].copy()
    if train.empty or test.empty:
        raise ValueError("Subjects 1--3 development split produced an empty split.")
    numeric_columns = [*features, _TARGET_COLUMN]
    for name, split in (("training", train), ("development test", test)):
        split[numeric_columns] = split[numeric_columns].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(split[numeric_columns].to_numpy(dtype=float)).all():
            raise ValueError(f"The {name} split contains missing or non-finite values.")

    specs = [
        ("V4_MORPHOLOGY", features, model_name, model)
        for model_name, model in get_models().items()
    ]
    metrics, predictions = _evaluate_specs(train, test, specs)
    destination = _output_directory(output_dir)
    metrics_path = destination / "subjects1_3_development_metrics.csv"
    predictions_path = destination / "subjects1_3_development_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    print(
        "Subjects 1--3 development validation: "
        f"{len(train)} train windows, {len(test)} test windows."
    )
    print(metrics.loc[:, ["model", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    return metrics, predictions


def subjects_1_3_loso_validation(
    dataframe: pd.DataFrame | None = None,
    *,
    output_dir: str | Path = "outputs/tables",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave one development subject out using only Subjects 1, 2, and 3."""

    table = _load_feature_table(dataframe)
    features = _V4_MORPHOLOGY_FEATURES
    _validate_columns(table, features)
    approved_recordings = set(load_manifest()["recording_id"].astype(str))
    approved = table.loc[
        table["recording_id"].astype(str).isin(approved_recordings)
    ].copy()
    development = approved.loc[
        approved["subject_id"].isin(["subject_1", "subject_2", "subject_3"])
        & approved["condition"].isin(_NORMAL_CONDITIONS)
    ].copy()
    numeric_columns = [*features, _TARGET_COLUMN]
    development[numeric_columns] = development[numeric_columns].apply(
        pd.to_numeric,
        errors="raise",
    )
    if not np.isfinite(development[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Development LOSO data contains missing or non-finite values.")

    specs = [("V4_MORPHOLOGY", features, "Random Forest", get_models()["Random Forest"])]
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for held_out_subject in ("subject_1", "subject_2", "subject_3"):
        train = development.loc[development["subject_id"] != held_out_subject].copy()
        test = development.loc[development["subject_id"] == held_out_subject].copy()
        if train.empty or test.empty:
            continue
        metrics, predictions = _evaluate_specs(train, test, specs)
        metrics.insert(0, "held_out_subject_id", held_out_subject)
        predictions.insert(0, "held_out_subject_id", held_out_subject)
        metric_frames.append(metrics)
        prediction_frames.append(predictions)

    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    overall_rows: list[dict[str, Any]] = []
    for (feature_set, model_name), group in predictions.groupby(
        ["feature_set", "model"],
        sort=False,
    ):
        actual = group["actual_speed_mph"].to_numpy(dtype=float)
        predicted = group["predicted_speed_mph"].to_numpy(dtype=float)
        overall_rows.append(
            {
                "held_out_subject_id": "overall",
                "feature_set": feature_set,
                "features": str(group["features"].iloc[0]),
                "model": model_name,
                "n_train_windows": np.nan,
                "n_test_windows": len(group),
                "n_train_recordings": np.nan,
                "n_test_recordings": group["recording_id"].nunique(),
                "MAE": float(mean_absolute_error(actual, predicted)),
                "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
                "R2": float(r2_score(actual, predicted)),
            }
        )
    metrics = pd.concat([metrics, pd.DataFrame(overall_rows)], ignore_index=True)

    destination = _output_directory(output_dir)
    metrics_path = destination / "subjects1_3_loso_metrics.csv"
    predictions_path = destination / "subjects1_3_loso_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    print("Subjects 1--3 LOSO development validation:")
    print(metrics.loc[:, ["held_out_subject_id", "model", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    return metrics, predictions


def subject4_final_cross_subject_test(
    dataframe: pd.DataFrame | None = None,
    *,
    output_dir: str | Path = "outputs/tables",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the frozen Subject 4 final cross-subject test.

    Random Forest models train on normal/natural Subjects 1--3 windows and test
    once on included normal Subject 4 windows. The function compares v2,
    v2+Vertical_Peak_Sharpness, and full v4 morphology feature sets.
    """

    table = _load_feature_table(dataframe)
    feature_sets = {
        "v2": _V2_FEATURES,
        "v2_plus_vertical_peak_sharpness": _V2_PLUS_SHARPNESS_FEATURES,
        "v4_morphology": _V4_MORPHOLOGY_FEATURES,
    }
    all_features = tuple(dict.fromkeys(feature for features in feature_sets.values() for feature in features))
    _validate_columns(table, all_features)
    approved_recordings = set(load_manifest()["recording_id"].astype(str))
    approved = table.loc[
        table["recording_id"].astype(str).isin(approved_recordings)
    ].copy()
    normal = approved["condition"].isin(_NORMAL_CONDITIONS)
    train = approved.loc[
        approved["subject_id"].isin(["subject_1", "subject_2", "subject_3"]) & normal
    ].copy()
    test = approved.loc[
        (approved["subject_id"] == "subject_4") & (approved["condition"] == "normal")
    ].copy()
    if train.empty or test.empty:
        raise ValueError("Subject 4 final split produced an empty train or test split.")
    numeric_columns = [*all_features, _TARGET_COLUMN]
    for name, split in (("training", train), ("Subject 4 test", test)):
        split[numeric_columns] = split[numeric_columns].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(split[numeric_columns].to_numpy(dtype=float)).all():
            raise ValueError(f"The {name} split contains missing or non-finite values.")

    rf = get_models()["Random Forest"]
    specs = [
        (feature_set, features, "Random Forest", rf)
        for feature_set, features in feature_sets.items()
    ]
    metrics, predictions = _evaluate_specs(train, test, specs)
    error_by_speed = (
        predictions.groupby(["feature_set", "model", "actual_speed_mph"], as_index=False)
        .agg(
            n_windows=("absolute_error_mph", "size"),
            n_recordings=("recording_id", "nunique"),
            mean_error_mph=("residual_mph", "mean"),
            mean_absolute_error_mph=("absolute_error_mph", "mean"),
            median_absolute_error_mph=("absolute_error_mph", "median"),
            rmse_mph=("residual_mph", _rmse),
            max_absolute_error_mph=("absolute_error_mph", "max"),
        )
        .sort_values(["feature_set", "actual_speed_mph"])
    )

    destination = _output_directory(output_dir)
    metrics_path = destination / "subject4_final_cross_subject_metrics.csv"
    predictions_path = destination / "subject4_final_cross_subject_predictions.csv"
    error_path = destination / "subject4_error_by_speed.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    error_by_speed.to_csv(error_path, index=False)
    print(
        "Subject 4 final cross-subject test: "
        f"{len(train)} train windows, {len(test)} test windows."
    )
    print(metrics.loc[:, ["feature_set", "model", "MAE", "RMSE", "R2"]].to_string(index=False))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved error by speed to {error_path}")
    return metrics, predictions, error_by_speed


def cadence_robustness_loso(
    dataframe: pd.DataFrame | None = None,
    *,
    output_dir: str | Path = "outputs/tables",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run leave-one-speed-out cadence robustness evaluation on Subject 1 Day 4.

    Each fold trains on all approved Subject 1 Day 3 natural-speed windows plus
    approved Subject 1 Day 4 cadence-condition windows from two of the three Day
    4 speeds. The held-out Day 4 speed is used only for testing. This
    experiment is intentionally separate from standard validation and from the
    original cadence stress test.
    """

    table = _load_feature_table(dataframe)
    _validate_columns(table, DEFAULT_FEATURES)

    approved_recordings = set(load_manifest()["recording_id"].astype(str))
    approved = table.loc[
        table["recording_id"].astype(str).isin(approved_recordings)
    ].copy()
    numeric_columns = [*DEFAULT_FEATURES, _TARGET_COLUMN]
    try:
        approved[numeric_columns] = approved[numeric_columns].apply(
            pd.to_numeric,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Feature table contains non-numeric model values.") from error
    if not np.isfinite(approved[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Feature table contains missing or non-finite model values.")

    subject_1 = approved["subject_id"] == "subject_1"
    day3 = subject_1 & (approved["session"] == "day3")
    day4 = subject_1 & (approved["session"] == "day4")
    day3_train = approved.loc[day3].copy()
    day4_table = approved.loc[day4].copy()
    if day3_train.empty:
        raise ValueError("Training split is empty; expected Subject 1 Day 3 rows.")
    if day4_table.empty:
        raise ValueError("Day 4 split is empty; expected Subject 1 Day 4 rows.")

    held_out_speeds = (7.0, 6.0, 5.0)
    available_speeds = set(day4_table[_TARGET_COLUMN].astype(float).unique())
    missing_speeds = sorted(set(held_out_speeds).difference(available_speeds))
    if missing_speeds:
        raise ValueError(
            "Day 4 split is missing required LOSO speeds: "
            + ", ".join(f"{speed:g}" for speed in missing_speeds)
        )

    specs = [
        ("DEFAULT_FEATURES", DEFAULT_FEATURES, model_name, model)
        for model_name, model in get_models().items()
    ]
    fold_metrics: list[pd.DataFrame] = []
    fold_predictions: list[pd.DataFrame] = []

    for fold_index, held_out_speed in enumerate(held_out_speeds, start=1):
        train_day4_speeds = tuple(
            speed for speed in sorted(held_out_speeds) if speed != held_out_speed
        )
        train_day4 = day4_table.loc[
            day4_table[_TARGET_COLUMN].astype(float).isin(train_day4_speeds)
        ].copy()
        test = day4_table.loc[
            day4_table[_TARGET_COLUMN].astype(float) == held_out_speed
        ].copy()
        train = pd.concat([day3_train, train_day4], ignore_index=True)
        if train_day4.empty:
            raise ValueError(
                f"Fold {fold_index} has no Day 4 training rows for speeds "
                + ", ".join(f"{speed:g}" for speed in train_day4_speeds)
            )
        if test.empty:
            raise ValueError(
                f"Fold {fold_index} has no test rows for held-out speed "
                f"{held_out_speed:g} mph."
            )

        metrics, predictions = _evaluate_specs(train, test, specs)
        train_speed_label = "|".join(f"{speed:g}" for speed in train_day4_speeds)
        for frame in (metrics, predictions):
            frame.insert(0, "train_day4_speeds_mph", train_speed_label)
            frame.insert(0, "held_out_speed_mph", held_out_speed)
            frame.insert(0, "fold", fold_index)
        fold_metrics.append(metrics)
        fold_predictions.append(predictions)

    metrics = pd.concat(fold_metrics, ignore_index=True)
    predictions = pd.concat(fold_predictions, ignore_index=True)

    overall_rows: list[dict[str, Any]] = []
    for (feature_set, model_name), group in predictions.groupby(
        ["feature_set", "model"],
        sort=False,
    ):
        actual = group["actual_speed_mph"].to_numpy(dtype=float)
        predicted = group["predicted_speed_mph"].to_numpy(dtype=float)
        overall_rows.append(
            {
                "fold": "overall",
                "held_out_speed_mph": "all",
                "train_day4_speeds_mph": "loso",
                "feature_set": feature_set,
                "features": str(group["features"].iloc[0]),
                "model": model_name,
                "n_train_windows": np.nan,
                "n_test_windows": len(group),
                "n_train_recordings": np.nan,
                "n_test_recordings": group["recording_id"].nunique(),
                "MAE": float(mean_absolute_error(actual, predicted)),
                "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
                "R2": float(r2_score(actual, predicted)),
            }
        )
    metrics = pd.concat([metrics, pd.DataFrame(overall_rows)], ignore_index=True)

    destination = _output_directory(output_dir)
    metrics_path = destination / "cadence_robustness_loso_metrics.csv"
    predictions_path = destination / "cadence_robustness_loso_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print(
        "Cadence robustness LOSO: "
        f"{len(held_out_speeds)} folds, {len(predictions)} predictions."
    )
    print(
        metrics.loc[
            metrics["model"] == "Random Forest",
            ["fold", "held_out_speed_mph", "MAE", "RMSE", "R2"],
        ].to_string(index=False)
    )
    print(f"Saved LOSO metrics to {metrics_path}")
    print(f"Saved LOSO predictions to {predictions_path}")
    return metrics, predictions


def feature_ablation(
    dataframe: pd.DataFrame | None = None,
    *,
    output_dir: str | Path = "outputs/tables",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare reduced feature sets using Linear Regression and Random Forest.

    The function uses the standard validation split: Subject 1 Day 3 for
    training, then approved Subject 1 Day 2 and normal-cadence Day 4 recordings
    for testing. It saves and returns metric and per-window prediction tables.
    """

    table = _load_feature_table(dataframe)
    all_reduced_features = tuple(
        dict.fromkeys(
            feature
            for features in REDUCED_FEATURE_SETS.values()
            for feature in features
        )
    )
    train, test = _same_subject_split(
        table,
        all_reduced_features,
        mode="standard",
    )
    models = get_models()
    specs = [
        (feature_set, features, model_name, models[model_name])
        for feature_set, features in REDUCED_FEATURE_SETS.items()
        for model_name in ("Linear Regression", "Random Forest")
    ]
    metrics, predictions = _evaluate_specs(train, test, specs)

    destination = _output_directory(output_dir)
    metrics_path = destination / "feature_ablation_metrics.csv"
    predictions_path = destination / "feature_ablation_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print(
        f"Feature ablation: {len(REDUCED_FEATURE_SETS)} feature sets, "
        f"{len(metrics)} model comparisons."
    )
    print(
        metrics.loc[:, ["feature_set", "model", "MAE", "RMSE", "R2"]]
        .sort_values(["MAE", "feature_set", "model"])
        .to_string(index=False)
    )
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    return metrics, predictions
