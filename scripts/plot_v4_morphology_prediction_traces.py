"""Plot standard-validation traces for v2, v3, sharpness, and v4 morphology."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lenz-speed-matplotlib"),
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


FEATURE_SETS = (
    "v2_only",
    "v3_full",
    "v2_plus_sharpness",
    "v4_morphology_all",
)
COLORS = {
    "v2_only": "#59A14F",
    "v3_full": "#E15759",
    "v2_plus_sharpness": "#4C78A8",
    "v4_morphology_all": "#F28E2B",
}
LABELS = {
    "v2_only": "v2 single RF",
    "v3_full": "v3 full RF",
    "v2_plus_sharpness": "v2 + sharpness RF",
    "v4_morphology_all": "v4 morphology RF",
}


def _load_standard_predictions() -> tuple[pd.DataFrame, dict[str, float]]:
    """Load v4 experiment predictions and return standard-validation traces."""
    path = REPOSITORY_ROOT / "outputs/tables/v4_morphology_feature_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/v4_morphology_experiment.py first."
        )

    predictions = pd.read_csv(path)
    predictions = predictions.loc[
        (predictions["evaluation"] == "standard")
        & predictions["feature_set"].isin(FEATURE_SETS)
    ].copy()
    predictions["speed_mph"] = predictions["actual_speed_mph"].astype(float)

    maes = (
        predictions.groupby("feature_set")["absolute_error_mph"]
        .mean()
        .reindex(FEATURE_SETS)
        .to_dict()
    )
    return predictions, {key: float(value) for key, value in maes.items()}


def _trace_base(predictions: pd.DataFrame) -> pd.DataFrame:
    """Use one feature set to define the shared validation-window order."""
    return (
        predictions.loc[predictions["feature_set"] == FEATURE_SETS[0]]
        .sort_values(["speed_mph", "recording_id", "window_index"])
        .reset_index(drop=True)
    )


def _ordered_predictions(predictions: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Return one feature set in the shared validation-window order."""
    return (
        predictions.loc[predictions["feature_set"] == feature_set]
        .sort_values(["speed_mph", "recording_id", "window_index"])
        .reset_index(drop=True)
    )


def _apply_recording_ticks(ax: plt.Axes, base: pd.DataFrame) -> None:
    """Label the x-axis by validation recording boundaries."""
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
    """Plot the reference speed trace."""
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


def _plot_feature_set(
    ax: plt.Axes,
    predictions: pd.DataFrame,
    base: pd.DataFrame,
    feature_set: str,
    maes: dict[str, float],
    *,
    alpha: float = 0.9,
) -> None:
    """Plot predictions from one feature set."""
    table = _ordered_predictions(predictions, feature_set)
    x = np.arange(len(table))
    ax.plot(
        x,
        table["predicted_speed_mph"],
        color=COLORS[feature_set],
        marker="o",
        markersize=2.6,
        linewidth=1.1,
        alpha=alpha,
        label=f"{LABELS[feature_set]} (MAE {maes[feature_set]:.3f})",
    )
    _apply_recording_ticks(ax, base)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(1.6, 8.7)


def main() -> None:
    """Create separate-panel and combined v4 morphology prediction traces."""
    predictions, maes = _load_standard_predictions()
    base = _trace_base(predictions)

    output_dir = REPOSITORY_ROOT / "outputs/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "v4_morphology_prediction_traces_standard_panels.png"

    fig, axes = plt.subplots(2, 3, figsize=(21, 10), sharex=True, sharey=True)
    panel_axes = {
        "v2_only": axes[0, 0],
        "v3_full": axes[0, 1],
        "v2_plus_sharpness": axes[1, 0],
        "v4_morphology_all": axes[1, 1],
    }

    for feature_set, ax in panel_axes.items():
        _plot_actual(ax, base)
        _plot_feature_set(ax, predictions, base, feature_set, maes)
        ax.set_title(LABELS[feature_set])
        ax.legend(frameon=False, loc="upper left")

    combined_ax = axes[:, 2].reshape(-1)[0]
    combined_ax.remove()
    combined_ax = fig.add_subplot(1, 3, 3)
    _plot_actual(combined_ax, base)
    for feature_set in FEATURE_SETS:
        _plot_feature_set(
            combined_ax,
            predictions,
            base,
            feature_set,
            maes,
            alpha=0.78,
        )
    combined_ax.set_title("All four combined")
    combined_ax.legend(frameon=False, loc="upper left", fontsize=9)

    axes[1, 2].remove()
    fig.suptitle(
        "Random Forest Prediction Traces — v4 Morphology Comparison",
        fontsize=17,
    )
    fig.supxlabel("Validation recording and window sequence")
    fig.supylabel("Speed (mph)")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved figure to {output_path}")
    for feature_set in FEATURE_SETS:
        print(f"{LABELS[feature_set]}: RF MAE={maes[feature_set]:.4f} mph")


if __name__ == "__main__":
    main()
