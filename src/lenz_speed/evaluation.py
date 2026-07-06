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
