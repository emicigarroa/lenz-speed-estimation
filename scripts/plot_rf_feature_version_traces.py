"""Plot Random Forest standard-validation traces for v1, v2, and v3 features."""

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
from sklearn.metrics import mean_absolute_error  # noqa: E402

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
FEATURE_SETS = {
    "v1": V1_FEATURES,
    "v2": V2_FEATURES,
    "v3": tuple(DEFAULT_FEATURES),
}
COLORS = {
    "v1": "#4C78A8",
    "v2": "#59A14F",
    "v3": "#E15759",
}
LABELS = {
    "v1": "v1 original features",
    "v2": "v2 low-risk features",
    "v3": "v3 gait/temporal features",
}


def _prediction_table() -> tuple[pd.DataFrame, dict[str, float]]:
    feature_table = pd.read_csv(REPOSITORY_ROOT / "data/processed/windowed_features.csv")
    frames: list[pd.DataFrame] = []
    maes: dict[str, float] = {}

    for version, features in FEATURE_SETS.items():
        train, test = _same_subject_split(feature_table, features, mode="standard")
        model = clone(get_models()["Random Forest"])
        model.fit(train.loc[:, features], train["speed_mph"])
        predictions = model.predict(test.loc[:, features])

        table = test.loc[
            :,
            [
                "recording_id",
                "session",
                "window_index",
                "speed_mph",
                "condition",
            ],
        ].copy()
        table["feature_version"] = version
        table["predicted_speed_mph"] = predictions
        table["absolute_error_mph"] = np.abs(predictions - table["speed_mph"])
        frames.append(table)
        maes[version] = float(mean_absolute_error(table["speed_mph"], predictions))

    combined = pd.concat(frames, ignore_index=True)
    return combined, maes


def _trace_base(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.loc[table["feature_version"] == "v1"]
        .sort_values(["speed_mph", "recording_id", "window_index"])
        .reset_index(drop=True)
    )


def _apply_recording_ticks(ax: plt.Axes, base: pd.DataFrame) -> None:
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    for _, group in base.groupby("recording_id", sort=False):
        first = int(group.index.min())
        last = int(group.index.max())
        tick_positions.append((first + last) / 2)
        speed = float(group["speed_mph"].iloc[0])
        session = str(group["session"].iloc[0]).replace("day", "Day ")
        tick_labels.append(f"{speed:g} mph\n{session}")
        if first:
            ax.axvline(first - 0.5, color="#BBBBBB", linewidth=0.8)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)


def _plot_actual(ax: plt.Axes, base: pd.DataFrame) -> None:
    x = np.arange(len(base))
    ax.step(
        x,
        base["speed_mph"],
        where="mid",
        color="#222222",
        linestyle="--",
        linewidth=1.4,
        label="Actual speed",
    )


def _plot_version(
    ax: plt.Axes,
    predictions: pd.DataFrame,
    base: pd.DataFrame,
    version: str,
    *,
    alpha: float = 0.9,
) -> None:
    version_table = (
        predictions.loc[predictions["feature_version"] == version]
        .sort_values(["speed_mph", "recording_id", "window_index"])
        .reset_index(drop=True)
    )
    x = np.arange(len(version_table))
    ax.plot(
        x,
        version_table["predicted_speed_mph"],
        color=COLORS[version],
        marker="o",
        markersize=2.6,
        linewidth=1.1,
        alpha=alpha,
        label=LABELS[version],
    )
    _apply_recording_ticks(ax, base)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(1.6, 8.7)


def main() -> None:
    predictions, maes = _prediction_table()
    base = _trace_base(predictions)

    output_dir = REPOSITORY_ROOT / "outputs/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = REPOSITORY_ROOT / "outputs/tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "random_forest_prediction_trace_feature_versions_standard.png"
    speed_mae_path = output_dir / "random_forest_mae_by_speed_feature_versions_standard.png"
    speed_vs_reference_path = (
        output_dir / "random_forest_speed_vs_reference_feature_versions_standard.png"
    )
    speed_mae_table_path = table_dir / "random_forest_mae_by_speed_feature_versions_standard.csv"

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)
    panels = (
        ("v1", axes[0, 0]),
        ("v2", axes[0, 1]),
        ("v3", axes[1, 0]),
    )
    for version, ax in panels:
        _plot_actual(ax, base)
        _plot_version(ax, predictions, base, version)
        ax.set_title(f"{LABELS[version]} — RF MAE {maes[version]:.4f} mph")
        ax.legend(frameon=False, loc="upper left")

    overlay_ax = axes[1, 1]
    _plot_actual(overlay_ax, base)
    for version in FEATURE_SETS:
        _plot_version(overlay_ax, predictions, base, version, alpha=0.78)
    overlay_ax.set_title("All feature versions combined")
    overlay_ax.legend(frameon=False, loc="upper left", ncol=2)

    fig.suptitle(
        "Random Forest Prediction Trace by Feature Version — Standard Validation",
        fontsize=16,
    )
    fig.supxlabel("Validation recording and window sequence")
    fig.supylabel("Speed (mph)")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    speed_mae = (
        predictions.groupby(["speed_mph", "feature_version"], as_index=False)
        .agg(
            n_windows=("absolute_error_mph", "size"),
            mae_mph=("absolute_error_mph", "mean"),
        )
        .sort_values(["speed_mph", "feature_version"])
    )
    speed_mae.to_csv(speed_mae_table_path, index=False)

    speeds = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    x = np.arange(len(speeds), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for offset, version in zip((-width, 0.0, width), FEATURE_SETS, strict=True):
        version_data = speed_mae.loc[
            speed_mae["feature_version"] == version
        ].set_index("speed_mph")
        values = [
            float(version_data.loc[speed, "mae_mph"])
            if speed in version_data.index
            else np.nan
            for speed in speeds
        ]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=LABELS[version],
            color=COLORS[version],
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{speed:g} mph" for speed in speeds])
    ax.set_xlabel("Actual validation speed bin")
    ax.set_ylabel("Random Forest MAE (mph)")
    ax.set_title("Random Forest MAE by Speed Bin and Feature Version")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    ax.set_ylim(0, float(speed_mae["mae_mph"].max()) * 1.24)
    fig.tight_layout()
    fig.savefig(speed_mae_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharex=True, sharey=True)
    lower = 1.6
    upper = 8.7
    for ax, version in zip(axes, FEATURE_SETS, strict=True):
        version_table = predictions.loc[
            predictions["feature_version"] == version
        ].copy()
        ax.scatter(
            version_table["speed_mph"],
            version_table["predicted_speed_mph"],
            s=26,
            alpha=0.48,
            color=COLORS[version],
            edgecolors="none",
            label="Window prediction",
        )
        means = version_table.groupby("speed_mph")["predicted_speed_mph"].mean()
        ax.plot(
            means.index,
            means.values,
            "o-",
            color="#222222",
            linewidth=1.4,
            markersize=5,
            label="Mean prediction",
        )
        ax.plot(
            [lower, upper],
            [lower, upper],
            "--",
            color="#777777",
            linewidth=1,
            label="Ideal",
        )
        ax.set_title(f"{LABELS[version]}\nRF MAE {maes[version]:.4f} mph")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle(
        "Random Forest Predicted Speed vs Reference Speed — Standard Validation",
        fontsize=15,
    )
    fig.supxlabel("Reference speed (mph)")
    fig.supylabel("Predicted speed (mph)")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.93))
    fig.savefig(
        speed_vs_reference_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f"Saved figure to {output_path}")
    print(f"Saved figure to {speed_mae_path}")
    print(f"Saved figure to {speed_vs_reference_path}")
    print(f"Saved table to {speed_mae_table_path}")
    for version, mae in maes.items():
        print(f"{version}: RF MAE={mae:.4f} mph")
    print("\nMAE by speed bin:")
    print(
        speed_mae.pivot(
            index="speed_mph",
            columns="feature_version",
            values="mae_mph",
        ).to_string(float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
