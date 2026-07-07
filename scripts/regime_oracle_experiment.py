"""Oracle walking/running regime experiment for LENZ speed estimation."""

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

from lenz_speed.evaluation import _same_subject_split  # noqa: E402
from lenz_speed.modeling import DEFAULT_FEATURES, get_models  # noqa: E402


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
V3_FEATURES = tuple(DEFAULT_FEATURES)
WALKING_THRESHOLD_MPH = 5.0

MODEL_LABELS = {
    "v2_single": "v2 single RF",
    "v3_single": "v3 single RF",
    "oracle_regime": "oracle regime RF",
}
MODEL_COLORS = {
    "v2_single": "#59A14F",
    "v3_single": "#E15759",
    "oracle_regime": "#4C78A8",
}


def _load_feature_table() -> pd.DataFrame:
    path = REPOSITORY_ROOT / "data/processed/windowed_features.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature table not found: {path}. Run python run_pipeline.py first."
        )
    return pd.read_csv(path)


def _regime(speed_mph: pd.Series | np.ndarray) -> np.ndarray:
    return np.where(np.asarray(speed_mph, dtype=float) < WALKING_THRESHOLD_MPH, "walking", "running")


def _random_forest():
    return clone(get_models()["Random Forest"])


def _single_model_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    model_name: str,
) -> pd.DataFrame:
    model = _random_forest()
    model.fit(train.loc[:, features], train["speed_mph"])
    predicted = model.predict(test.loc[:, features])
    return _prediction_frame(
        test,
        model_name=model_name,
        train_scheme="single_model",
        feature_scheme="v2" if model_name == "v2_single" else "v3_full",
        predicted=predicted,
    )


def _oracle_regime_predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    walking_train = train.loc[train["speed_mph"] < WALKING_THRESHOLD_MPH].copy()
    running_train = train.loc[train["speed_mph"] >= WALKING_THRESHOLD_MPH].copy()
    if walking_train.empty or running_train.empty:
        raise ValueError("Oracle regime experiment requires both walking and running training rows.")

    walking_model = _random_forest()
    running_model = _random_forest()
    walking_model.fit(walking_train.loc[:, V2_FEATURES], walking_train["speed_mph"])
    running_model.fit(running_train.loc[:, V3_FEATURES], running_train["speed_mph"])

    predicted = np.empty(len(test), dtype=float)
    test_regime = _regime(test["speed_mph"])
    walking_mask = test_regime == "walking"
    running_mask = ~walking_mask
    predicted[walking_mask] = walking_model.predict(test.loc[walking_mask, V2_FEATURES])
    predicted[running_mask] = running_model.predict(test.loc[running_mask, V3_FEATURES])

    frame = _prediction_frame(
        test,
        model_name="oracle_regime",
        train_scheme="walking_v2_running_v3_oracle_actual_speed",
        feature_scheme="v2_if_speed_lt_5_else_v3",
        predicted=predicted,
    )
    frame["oracle_regime"] = test_regime
    return frame


def _prediction_frame(
    test: pd.DataFrame,
    *,
    model_name: str,
    train_scheme: str,
    feature_scheme: str,
    predicted: np.ndarray,
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
    frame.insert(0, "feature_scheme", feature_scheme)
    frame.insert(0, "train_scheme", train_scheme)
    frame.insert(0, "model", model_name)
    frame["oracle_regime"] = _regime(frame["actual_speed_mph"])
    frame["predicted_speed_mph"] = predicted
    frame["residual_mph"] = frame["predicted_speed_mph"] - frame["actual_speed_mph"]
    frame["absolute_error_mph"] = frame["residual_mph"].abs()
    return frame


def _metrics_for_group(group: pd.DataFrame, *, summary_level: str) -> dict[str, float | int | str]:
    actual = group["actual_speed_mph"].to_numpy(dtype=float)
    predicted = group["predicted_speed_mph"].to_numpy(dtype=float)
    return {
        "summary_level": summary_level,
        "model": str(group["model"].iloc[0]),
        "train_scheme": str(group["train_scheme"].iloc[0]),
        "feature_scheme": str(group["feature_scheme"].iloc[0]),
        "speed_mph": "all" if summary_level == "overall" else float(actual[0]),
        "regime": "all" if summary_level == "overall" else str(group["oracle_regime"].iloc[0]),
        "n_windows": len(group),
        "n_recordings": group["recording_id"].nunique(),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(np.unique(actual)) > 1 else np.nan,
    }


def _build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in predictions.groupby("model", sort=False):
        rows.append(_metrics_for_group(group, summary_level="overall"))
    for _, group in predictions.groupby(["model", "actual_speed_mph"], sort=False):
        rows.append(_metrics_for_group(group, summary_level="speed"))
    return pd.DataFrame(rows)


def _trace_base(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.loc[predictions["model"] == "v2_single"]
        .sort_values(["actual_speed_mph", "recording_id", "window_index"])
        .reset_index(drop=True)
    )


def _apply_recording_ticks(ax: plt.Axes, base: pd.DataFrame) -> None:
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    for _, group in base.groupby("recording_id", sort=False):
        first = int(group.index.min())
        last = int(group.index.max())
        tick_positions.append((first + last) / 2)
        speed = float(group["actual_speed_mph"].iloc[0])
        session = str(group["session"].iloc[0]).replace("day", "Day ")
        tick_labels.append(f"{speed:g} mph\n{session}")
        if first:
            ax.axvline(first - 0.5, color="#BBBBBB", linewidth=0.8)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)


def _plot_trace(predictions: pd.DataFrame, metrics: pd.DataFrame, output_path: Path) -> None:
    base = _trace_base(predictions)
    x = np.arange(len(base))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.step(
        x,
        base["actual_speed_mph"],
        where="mid",
        color="#222222",
        linestyle="--",
        linewidth=1.5,
        label="Actual speed",
    )
    for model_name in ("v2_single", "v3_single", "oracle_regime"):
        model_data = (
            predictions.loc[predictions["model"] == model_name]
            .sort_values(["actual_speed_mph", "recording_id", "window_index"])
            .reset_index(drop=True)
        )
        mae = float(
            metrics.loc[
                (metrics["model"] == model_name)
                & (metrics["summary_level"] == "overall"),
                "MAE",
            ].iloc[0]
        )
        ax.plot(
            x,
            model_data["predicted_speed_mph"],
            marker="o",
            markersize=2.6,
            linewidth=1.15,
            alpha=0.82,
            color=MODEL_COLORS[model_name],
            label=f"{MODEL_LABELS[model_name]} (MAE {mae:.3f})",
        )
    _apply_recording_ticks(ax, base)
    ax.set_xlabel("Validation recording and window sequence")
    ax.set_ylabel("Speed (mph)")
    ax.set_title("Oracle Regime Random Forest Prediction Trace — Standard Validation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    ax.set_ylim(1.6, 8.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_separate_traces(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot separate prediction-trace panels for v2, v3, oracle, and combined."""

    base = _trace_base(predictions)
    x = np.arange(len(base))
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)
    panel_models = (
        ("v2_single", axes[0, 0]),
        ("v3_single", axes[0, 1]),
        ("oracle_regime", axes[1, 0]),
    )

    for model_name, ax in panel_models:
        ax.step(
            x,
            base["actual_speed_mph"],
            where="mid",
            color="#222222",
            linestyle="--",
            linewidth=1.4,
            label="Actual speed",
        )
        model_data = (
            predictions.loc[predictions["model"] == model_name]
            .sort_values(["actual_speed_mph", "recording_id", "window_index"])
            .reset_index(drop=True)
        )
        mae = float(
            metrics.loc[
                (metrics["model"] == model_name)
                & (metrics["summary_level"] == "overall"),
                "MAE",
            ].iloc[0]
        )
        ax.plot(
            x,
            model_data["predicted_speed_mph"],
            marker="o",
            markersize=2.6,
            linewidth=1.15,
            color=MODEL_COLORS[model_name],
            label=f"{MODEL_LABELS[model_name]} (MAE {mae:.3f})",
        )
        _apply_recording_ticks(ax, base)
        ax.set_title(MODEL_LABELS[model_name])
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="upper left")

    combined_ax = axes[1, 1]
    combined_ax.step(
        x,
        base["actual_speed_mph"],
        where="mid",
        color="#222222",
        linestyle="--",
        linewidth=1.4,
        label="Actual speed",
    )
    for model_name in ("v2_single", "v3_single", "oracle_regime"):
        model_data = (
            predictions.loc[predictions["model"] == model_name]
            .sort_values(["actual_speed_mph", "recording_id", "window_index"])
            .reset_index(drop=True)
        )
        mae = float(
            metrics.loc[
                (metrics["model"] == model_name)
                & (metrics["summary_level"] == "overall"),
                "MAE",
            ].iloc[0]
        )
        combined_ax.plot(
            x,
            model_data["predicted_speed_mph"],
            marker="o",
            markersize=2.4,
            linewidth=1.05,
            alpha=0.78,
            color=MODEL_COLORS[model_name],
            label=f"{MODEL_LABELS[model_name]} (MAE {mae:.3f})",
        )
    _apply_recording_ticks(combined_ax, base)
    combined_ax.set_title("All three combined")
    combined_ax.grid(axis="y", alpha=0.25)
    combined_ax.legend(frameon=False, loc="upper left", ncol=2)

    for ax in axes.flat:
        ax.set_ylim(1.6, 8.8)
    fig.suptitle(
        "Oracle Regime Prediction Traces — Separate Model Panels",
        fontsize=16,
    )
    fig.supxlabel("Validation recording and window sequence")
    fig.supylabel("Speed (mph)")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_mae_by_speed(metrics: pd.DataFrame, output_path: Path) -> None:
    speed_metrics = metrics.loc[metrics["summary_level"] == "speed"].copy()
    speeds = sorted(float(speed) for speed in speed_metrics["speed_mph"].unique())
    x = np.arange(len(speeds), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for offset, model_name in zip((-width, 0.0, width), ("v2_single", "v3_single", "oracle_regime"), strict=True):
        model_data = speed_metrics.loc[speed_metrics["model"] == model_name].set_index("speed_mph")
        values = [float(model_data.loc[speed, "MAE"]) for speed in speeds]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.axvline(2.5, color="#777777", linestyle=":", linewidth=1)
    ax.text(
        2.52,
        ax.get_ylim()[1] * 0.95,
        "regime split: 5 mph",
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{speed:g} mph" for speed in speeds])
    ax.set_xlabel("Actual validation speed bin")
    ax.set_ylabel("Random Forest MAE (mph)")
    ax.set_title("Oracle Regime Experiment — MAE by Speed")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    ax.set_ylim(0, float(speed_metrics["MAE"].max()) * 1.24)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    table = _load_feature_table()
    train, test = _same_subject_split(table, V3_FEATURES, mode="standard")

    predictions = pd.concat(
        [
            _single_model_predictions(train, test, V2_FEATURES, "v2_single"),
            _single_model_predictions(train, test, V3_FEATURES, "v3_single"),
            _oracle_regime_predictions(train, test),
        ],
        ignore_index=True,
    )
    metrics = _build_metrics(predictions)

    tables_dir = REPOSITORY_ROOT / "outputs/tables"
    figures_dir = REPOSITORY_ROOT / "outputs/figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / "regime_oracle_metrics.csv"
    predictions_path = tables_dir / "regime_oracle_predictions.csv"
    trace_path = figures_dir / "regime_oracle_prediction_trace.png"
    separate_trace_path = figures_dir / "regime_oracle_prediction_trace_separate.png"
    speed_path = figures_dir / "regime_oracle_mae_by_speed.png"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    _plot_trace(predictions, metrics, trace_path)
    _plot_separate_traces(predictions, metrics, separate_trace_path)
    _plot_mae_by_speed(metrics, speed_path)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved trace figure to {trace_path}")
    print(f"Saved separate trace figure to {separate_trace_path}")
    print(f"Saved speed MAE figure to {speed_path}")
    print("\nOverall standard validation MAE:")
    print(
        metrics.loc[
            metrics["summary_level"] == "overall",
            ["model", "feature_scheme", "MAE", "RMSE", "R2"],
        ].to_string(index=False)
    )
    print("\nMAE by speed:")
    print(
        metrics.loc[metrics["summary_level"] == "speed"]
        .pivot(index="speed_mph", columns="model", values="MAE")
        .to_string(float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
