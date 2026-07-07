"""Selective v3 gait-feature experiment for LENZ Random Forest models."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lenz-speed-matplotlib"),
)
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from lenz_speed.data import load_manifest  # noqa: E402
from lenz_speed.evaluation import _same_subject_split  # noqa: E402
from lenz_speed.modeling import get_models  # noqa: E402


V1_FEATURES = (
    "Cadence_spm",
    "RMS_Z",
    "PeakToPeak_Z",
    "Gyro_RMS_X",
    "Gyro_RMS_Y",
    "Gyro_RMS_Z",
    "Accel_Mag_RMS",
)
V2_FEATURES = (
    *V1_FEATURES,
    "Dynamic_Accel_Mag_RMS",
    "Accel_Mag_P95_P05",
    "Accel_Mag_Jerk_RMS",
    "Accel_HighFreq_Energy_Ratio",
    "Gyro_Mag_RMS",
    "GyroY_PeakToPeak",
    "Accel_Anisotropy",
)
ALL_V3_GAIT_FEATURES = (
    "Impact_Peak_Count",
    "Mean_Impact_Interval_s",
    "Impact_Interval_CV",
    "Mean_Impact_Prominence",
    "Mean_Impact_Width_s",
    "Impact_Duty_Proxy",
    "Vertical_Peak_Sharpness",
)
FEATURE_SETS = {
    "v2_only": V2_FEATURES,
    "v2_plus_mean_impact_interval": (
        *V2_FEATURES,
        "Mean_Impact_Interval_s",
    ),
    "v2_plus_vertical_peak_sharpness": (
        *V2_FEATURES,
        "Vertical_Peak_Sharpness",
    ),
    "v2_plus_interval_and_sharpness": (
        *V2_FEATURES,
        "Mean_Impact_Interval_s",
        "Vertical_Peak_Sharpness",
    ),
    "v2_plus_all_v3_gait_temporal": (
        *V2_FEATURES,
        *ALL_V3_GAIT_FEATURES,
    ),
}
FEATURE_LABELS = {
    "v2_only": "v2 only",
    "v2_plus_mean_impact_interval": "v2 + interval",
    "v2_plus_vertical_peak_sharpness": "v2 + sharpness",
    "v2_plus_interval_and_sharpness": "v2 + both",
    "v2_plus_all_v3_gait_temporal": "v2 + all v3 gait",
}
EVALUATION_LABELS = {
    "standard": "Standard",
    "cadence_stress": "Cadence stress",
    "loso": "LOSO",
}


def _load_feature_table() -> pd.DataFrame:
    path = REPOSITORY_ROOT / "data/processed/windowed_features.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature table not found: {path}. Run python run_pipeline.py first."
        )
    return pd.read_csv(path)


def _random_forest():
    return clone(get_models()["Random Forest"])


def _prediction_frame(
    test: pd.DataFrame,
    *,
    evaluation: str,
    feature_set: str,
    features: tuple[str, ...],
    predicted: np.ndarray,
    fold: str = "overall",
    held_out_speed_mph: str | float = "",
) -> pd.DataFrame:
    frame = test.loc[
        :,
        [
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
            "speed_mph",
        ],
    ].copy()
    frame = frame.rename(columns={"speed_mph": "actual_speed_mph"})
    frame.insert(0, "held_out_speed_mph", held_out_speed_mph)
    frame.insert(0, "fold", fold)
    frame.insert(0, "n_features", len(features))
    frame.insert(0, "features", "|".join(features))
    frame.insert(0, "feature_set", feature_set)
    frame.insert(0, "evaluation", evaluation)
    frame["model"] = "Random Forest"
    frame["predicted_speed_mph"] = predicted
    frame["residual_mph"] = frame["predicted_speed_mph"] - frame["actual_speed_mph"]
    frame["absolute_error_mph"] = frame["residual_mph"].abs()
    return frame


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
) -> np.ndarray:
    model = _random_forest()
    model.fit(train.loc[:, features], train["speed_mph"])
    return model.predict(test.loc[:, features])


def _metrics_row(
    predictions: pd.DataFrame,
    *,
    summary_level: str,
) -> dict[str, float | int | str]:
    actual = predictions["actual_speed_mph"].to_numpy(dtype=float)
    predicted = predictions["predicted_speed_mph"].to_numpy(dtype=float)
    return {
        "evaluation": str(predictions["evaluation"].iloc[0]),
        "fold": str(predictions["fold"].iloc[0]) if summary_level == "fold" else "overall",
        "held_out_speed_mph": (
            str(predictions["held_out_speed_mph"].iloc[0])
            if summary_level == "fold"
            else "all"
        ),
        "summary_level": summary_level,
        "feature_set": str(predictions["feature_set"].iloc[0]),
        "features": str(predictions["features"].iloc[0]),
        "n_features": int(predictions["n_features"].iloc[0]),
        "model": "Random Forest",
        "n_windows": len(predictions),
        "n_recordings": predictions["recording_id"].nunique(),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(np.unique(actual)) > 1 else np.nan,
    }


def _standard_or_stress_predictions(
    table: pd.DataFrame,
    *,
    mode: str,
) -> list[pd.DataFrame]:
    frames = []
    for feature_set, features in FEATURE_SETS.items():
        train, test = _same_subject_split(table, features, mode=mode)
        predicted = _fit_predict(train, test, features)
        frames.append(
            _prediction_frame(
                test,
                evaluation=mode,
                feature_set=feature_set,
                features=features,
                predicted=predicted,
            )
        )
    return frames


def _loso_splits(
    table: pd.DataFrame,
    features: tuple[str, ...],
) -> list[tuple[str, float, pd.DataFrame, pd.DataFrame]]:
    approved_recordings = set(load_manifest()["recording_id"].astype(str))
    approved = table.loc[
        table["recording_id"].astype(str).isin(approved_recordings)
    ].copy()
    numeric_columns = [*features, "speed_mph"]
    approved[numeric_columns] = approved[numeric_columns].apply(
        pd.to_numeric,
        errors="raise",
    )
    subject_1 = approved["subject_id"] == "subject_1"
    day3_train = approved.loc[subject_1 & (approved["session"] == "day3")].copy()
    day4_table = approved.loc[subject_1 & (approved["session"] == "day4")].copy()
    held_out_speeds = (7.0, 6.0, 5.0)
    splits = []
    for fold_index, held_out_speed in enumerate(held_out_speeds, start=1):
        train_day4 = day4_table.loc[
            day4_table["speed_mph"].astype(float) != held_out_speed
        ].copy()
        test = day4_table.loc[
            day4_table["speed_mph"].astype(float) == held_out_speed
        ].copy()
        train = pd.concat([day3_train, train_day4], ignore_index=True)
        splits.append((f"fold_{fold_index}", held_out_speed, train, test))
    return splits


def _loso_predictions(table: pd.DataFrame) -> list[pd.DataFrame]:
    frames = []
    for feature_set, features in FEATURE_SETS.items():
        for fold, held_out_speed, train, test in _loso_splits(table, features):
            predicted = _fit_predict(train, test, features)
            frames.append(
                _prediction_frame(
                    test,
                    evaluation="loso",
                    feature_set=feature_set,
                    features=features,
                    predicted=predicted,
                    fold=fold,
                    held_out_speed_mph=held_out_speed,
                )
            )
    return frames


def _build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in predictions.groupby(["evaluation", "feature_set"], sort=False):
        rows.append(_metrics_row(group, summary_level="overall"))
    for _, group in predictions.loc[predictions["evaluation"] == "loso"].groupby(
        ["feature_set", "fold"],
        sort=False,
    ):
        rows.append(_metrics_row(group, summary_level="fold"))
    return pd.DataFrame(rows)


def _plot_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    overall = metrics.loc[metrics["summary_level"] == "overall"].copy()
    evaluations = ("standard", "cadence_stress", "loso")
    feature_sets = tuple(FEATURE_SETS)
    x = np.arange(len(feature_sets), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7), sharey=True)
    for ax, evaluation in zip(axes, evaluations, strict=True):
        eval_data = overall.loc[overall["evaluation"] == evaluation].set_index(
            "feature_set"
        )
        values = [float(eval_data.loc[feature_set, "MAE"]) for feature_set in feature_sets]
        bars = ax.bar(
            x,
            values,
            color=["#4C78A8", "#59A14F", "#F28E2B", "#B07AA1", "#E15759"],
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_title(EVALUATION_LABELS[evaluation])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [FEATURE_LABELS[feature_set] for feature_set in feature_sets],
            rotation=35,
            ha="right",
            fontsize=8,
        )
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlabel("Feature set")
    axes[0].set_ylabel("Random Forest MAE (mph)")
    fig.suptitle("Selective v3 Feature Experiment", fontsize=15)
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    table = _load_feature_table()
    prediction_frames = [
        *_standard_or_stress_predictions(table, mode="standard"),
        *_standard_or_stress_predictions(table, mode="cadence_stress"),
        *_loso_predictions(table),
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = _build_metrics(predictions)

    tables_dir = REPOSITORY_ROOT / "outputs/tables"
    figures_dir = REPOSITORY_ROOT / "outputs/figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / "selective_v3_feature_metrics.csv"
    predictions_path = tables_dir / "selective_v3_feature_predictions.csv"
    figure_path = figures_dir / "selective_v3_feature_mae.png"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    _plot_metrics(metrics, figure_path)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved figure to {figure_path}")
    print("\nOverall MAE:")
    print(
        metrics.loc[metrics["summary_level"] == "overall"]
        .pivot(index="feature_set", columns="evaluation", values="MAE")
        .loc[list(FEATURE_SETS)]
        .to_string(float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
