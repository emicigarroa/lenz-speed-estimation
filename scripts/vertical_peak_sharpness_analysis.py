"""Analyze why Vertical_Peak_Sharpness helps speed estimation."""

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
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


FEATURE = "Vertical_Peak_Sharpness"
CADENCE = "Cadence_spm"
SPEED = "speed_mph"
CONDITION_ORDER = ("normal", "decreased", "elevated")
CONDITION_COLORS = {
    "normal": "#59A14F",
    "decreased": "#4C78A8",
    "elevated": "#E15759",
}


def _load_feature_table() -> pd.DataFrame:
    path = REPOSITORY_ROOT / "data/processed/windowed_features.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature table not found: {path}. Run python run_pipeline.py first."
        )
    table = pd.read_csv(path)
    required = {
        "recording_id",
        "session",
        "condition",
        SPEED,
        CADENCE,
        FEATURE,
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError("Feature table is missing columns: " + ", ".join(missing))
    table = table.copy()
    table["condition_group"] = table["condition"].map(
        {
            "steady_state": "normal",
            "cadence_normal": "normal",
            "cadence_decreased": "decreased",
            "cadence_elevated": "elevated",
        }
    )
    table = table.loc[table["condition_group"].isin(CONDITION_ORDER)].copy()
    numeric_columns = [SPEED, CADENCE, FEATURE]
    table[numeric_columns] = table[numeric_columns].apply(
        pd.to_numeric,
        errors="raise",
    )
    return table


def _corr_value(frame: pd.DataFrame, left: str, right: str, method: str) -> float:
    if len(frame) < 2 or frame[left].nunique() < 2 or frame[right].nunique() < 2:
        return np.nan
    return float(frame[left].corr(frame[right], method=method))


def _standardized_speed_coefficient(frame: pd.DataFrame) -> float:
    """Return standardized speed coefficient for sharpness ~ speed + cadence."""

    if len(frame) < 3 or frame[SPEED].nunique() < 2 or frame[CADENCE].nunique() < 2:
        return np.nan
    x = frame[[SPEED, CADENCE]].to_numpy(dtype=float)
    y = frame[[FEATURE]].to_numpy(dtype=float)
    x_scaled = StandardScaler().fit_transform(x)
    y_scaled = StandardScaler().fit_transform(y).ravel()
    model = LinearRegression().fit(x_scaled, y_scaled)
    return float(model.coef_[0])


def _analysis_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    subsets = [("all", table)]
    for condition in CONDITION_ORDER:
        subsets.append(
            (
                condition,
                table.loc[table["condition_group"] == condition],
            )
        )
    subsets.append(
        (
            "cadence_manipulated_only",
            table.loc[table["condition_group"].isin(["decreased", "elevated"])],
        )
    )
    subsets.append(
        (
            "day4_only",
            table.loc[table["session"] == "day4"],
        )
    )

    for subset_name, frame in subsets:
        rows.append(
            {
                "analysis_type": "correlation",
                "subset": subset_name,
                "condition_group": "mixed" if subset_name in {"all", "cadence_manipulated_only", "day4_only"} else subset_name,
                "speed_mph": "all",
                "n_windows": len(frame),
                "n_recordings": frame["recording_id"].nunique(),
                "mean_vertical_peak_sharpness": frame[FEATURE].mean(),
                "mean_cadence_spm": frame[CADENCE].mean(),
                "pearson_sharpness_speed": _corr_value(frame, FEATURE, SPEED, "pearson"),
                "spearman_sharpness_speed": _corr_value(frame, FEATURE, SPEED, "spearman"),
                "pearson_sharpness_cadence": _corr_value(frame, FEATURE, CADENCE, "pearson"),
                "spearman_sharpness_cadence": _corr_value(frame, FEATURE, CADENCE, "spearman"),
                "pearson_speed_cadence": _corr_value(frame, SPEED, CADENCE, "pearson"),
                "standardized_speed_coef_controlling_cadence": _standardized_speed_coefficient(frame),
            }
        )

    grouped = (
        table.groupby(["condition_group", SPEED], as_index=False)
        .agg(
            n_windows=(FEATURE, "size"),
            n_recordings=("recording_id", "nunique"),
            mean_vertical_peak_sharpness=(FEATURE, "mean"),
            median_vertical_peak_sharpness=(FEATURE, "median"),
            std_vertical_peak_sharpness=(FEATURE, "std"),
            mean_cadence_spm=(CADENCE, "mean"),
            std_cadence_spm=(CADENCE, "std"),
        )
        .sort_values(["condition_group", SPEED])
    )
    for row in grouped.to_dict(orient="records"):
        rows.append(
            {
                "analysis_type": "summary_by_condition_speed",
                "subset": "by_condition_speed",
                "condition_group": row["condition_group"],
                "speed_mph": row[SPEED],
                "n_windows": row["n_windows"],
                "n_recordings": row["n_recordings"],
                "mean_vertical_peak_sharpness": row["mean_vertical_peak_sharpness"],
                "median_vertical_peak_sharpness": row[
                    "median_vertical_peak_sharpness"
                ],
                "std_vertical_peak_sharpness": row["std_vertical_peak_sharpness"],
                "mean_cadence_spm": row["mean_cadence_spm"],
                "std_cadence_spm": row["std_cadence_spm"],
                "pearson_sharpness_speed": np.nan,
                "spearman_sharpness_speed": np.nan,
                "pearson_sharpness_cadence": np.nan,
                "spearman_sharpness_cadence": np.nan,
                "pearson_speed_cadence": np.nan,
                "standardized_speed_coef_controlling_cadence": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _scatter_by_condition(
    table: pd.DataFrame,
    *,
    x_column: str,
    x_label: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for condition in CONDITION_ORDER:
        data = table.loc[table["condition_group"] == condition]
        ax.scatter(
            data[x_column],
            data[FEATURE],
            s=24,
            alpha=0.45,
            color=CONDITION_COLORS[condition],
            edgecolors="none",
            label=condition.title(),
        )
        if data[x_column].nunique() >= 2:
            means = data.groupby(x_column)[FEATURE].mean()
            ax.plot(
                means.index,
                means.values,
                "o-",
                color=CONDITION_COLORS[condition],
                linewidth=1.7,
                markersize=5,
            )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Vertical_Peak_Sharpness")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_by_condition(table: pd.DataFrame, output_path: Path) -> None:
    speeds = sorted(table[SPEED].unique())
    x = np.arange(len(speeds), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 6))

    for offset, condition in zip((-width, 0.0, width), CONDITION_ORDER, strict=True):
        data = (
            table.loc[table["condition_group"] == condition]
            .groupby(SPEED)[FEATURE]
            .mean()
        )
        values = [float(data.loc[speed]) if speed in data.index else np.nan for speed in speeds]
        ax.bar(
            x + offset,
            values,
            width,
            color=CONDITION_COLORS[condition],
            label=condition.title(),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{speed:g} mph" for speed in speeds])
    ax.set_xlabel("Speed (mph)")
    ax.set_ylabel("Mean Vertical_Peak_Sharpness")
    ax.set_title("Vertical_Peak_Sharpness by Speed and Condition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    table = _load_feature_table()
    tables_dir = REPOSITORY_ROOT / "outputs/tables"
    figures_dir = REPOSITORY_ROOT / "outputs/figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    analysis = _analysis_rows(table)
    analysis_path = tables_dir / "vertical_peak_sharpness_analysis.csv"
    speed_path = figures_dir / "vertical_peak_sharpness_vs_speed.png"
    cadence_path = figures_dir / "vertical_peak_sharpness_vs_cadence.png"
    condition_path = figures_dir / "vertical_peak_sharpness_by_condition.png"

    analysis.to_csv(analysis_path, index=False)
    _scatter_by_condition(
        table,
        x_column=SPEED,
        x_label="Speed (mph)",
        title="Vertical_Peak_Sharpness vs Speed",
        output_path=speed_path,
    )
    _scatter_by_condition(
        table,
        x_column=CADENCE,
        x_label="Cadence (spm)",
        title="Vertical_Peak_Sharpness vs Cadence",
        output_path=cadence_path,
    )
    _plot_by_condition(table, condition_path)

    print(f"Saved analysis table to {analysis_path}")
    print(f"Saved speed figure to {speed_path}")
    print(f"Saved cadence figure to {cadence_path}")
    print(f"Saved condition figure to {condition_path}")
    print("\nCorrelation summary:")
    summary = analysis.loc[
        analysis["analysis_type"] == "correlation",
        [
            "subset",
            "n_windows",
            "n_recordings",
            "pearson_sharpness_speed",
            "spearman_sharpness_speed",
            "pearson_sharpness_cadence",
            "spearman_sharpness_cadence",
            "standardized_speed_coef_controlling_cadence",
        ],
    ]
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
