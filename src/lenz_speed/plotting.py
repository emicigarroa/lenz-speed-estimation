"""Matplotlib figures for saved LENZ evaluation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_MODEL_ORDER = (
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Random Forest",
)
_MODEL_COLORS = {
    "Linear Regression": "#4C78A8",
    "Ridge Regression": "#59A14F",
    "Lasso Regression": "#F28E2B",
    "Random Forest": "#E15759",
}
_CONDITION_COLORS = {
    "cadence_decreased": "#4C78A8",
    "cadence_normal": "#59A14F",
    "cadence_elevated": "#E15759",
}
_ABLATION_LABELS = {
    "A_Cadence": "A\nCadence",
    "B_AccelMag": "B\nAccel\nmagnitude",
    "C_GyroY": "C\nGyro Y",
    "D_Cadence_AccelMag": "D\nCadence +\nAccelMag",
    "E_Cadence_GyroY": "E\nCadence +\nGyro Y",
    "F_Cadence_GyroY_AccelMag": "F\nCadence + Gyro Y\n+ AccelMag",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = _repository_root() / resolved
    return resolved.resolve()


def _read_table(path: str | Path, required_columns: set[str]) -> pd.DataFrame:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Evaluation table not found: {resolved}")
    table = pd.read_csv(resolved)
    missing = sorted(required_columns.difference(table.columns))
    if missing:
        raise ValueError(
            f"Evaluation table {resolved} is missing columns: {', '.join(missing)}"
        )
    return table


def _save_figure(fig: plt.Figure, output_dir: str | Path, filename: str) -> Path:
    destination = _resolve_path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / filename
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _prediction_limits(table: pd.DataFrame) -> tuple[float, float]:
    values = table[["actual_speed_mph", "predicted_speed_mph"]].to_numpy(
        dtype=float
    )
    lower = float(np.floor(np.nanmin(values) * 2) / 2 - 0.25)
    upper = float(np.ceil(np.nanmax(values) * 2) / 2 + 0.25)
    return lower, upper


def plot_predicted_vs_actual_standard(
    predictions_path: str | Path = (
        "outputs/tables/same_subject_standard_predictions.csv"
    ),
    *,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Plot standard-validation predictions against actual speed by model."""

    table = _read_table(
        predictions_path,
        {"model", "actual_speed_mph", "predicted_speed_mph"},
    )
    lower, upper = _prediction_limits(table)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)

    for ax, model_name in zip(axes.flat, _MODEL_ORDER, strict=True):
        model_data = table.loc[table["model"] == model_name]
        if model_data.empty:
            raise ValueError(f"Standard prediction table has no rows for {model_name}.")
        color = _MODEL_COLORS[model_name]
        ax.scatter(
            model_data["actual_speed_mph"],
            model_data["predicted_speed_mph"],
            s=23,
            alpha=0.5,
            color=color,
            edgecolors="none",
        )
        means = model_data.groupby("actual_speed_mph")["predicted_speed_mph"].mean()
        ax.plot(means.index, means.values, "o-", color="#222222", linewidth=1.4)
        ax.plot([lower, upper], [lower, upper], "--", color="#777777", linewidth=1)
        ax.set_title(model_name, fontweight="semibold")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.grid(alpha=0.25)

    fig.suptitle("Standard Validation: Predicted vs Actual Speed", fontsize=15)
    fig.supxlabel("Actual speed (mph)")
    fig.supylabel("Predicted speed (mph)")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    return _save_figure(fig, output_dir, "predicted_vs_actual_standard.png")


def plot_residual_by_speed_standard(
    predictions_path: str | Path = (
        "outputs/tables/same_subject_standard_predictions.csv"
    ),
    *,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Plot Random Forest standard-validation residuals at each test speed."""

    table = _read_table(
        predictions_path,
        {"model", "actual_speed_mph", "residual_mph"},
    )
    table = table.loc[table["model"] == "Random Forest"].copy()
    if table.empty:
        raise ValueError("Standard prediction table has no Random Forest rows.")

    speeds = sorted(table["actual_speed_mph"].unique())
    residuals = [
        table.loc[table["actual_speed_mph"] == speed, "residual_mph"].to_numpy()
        for speed in speeds
    ]
    means = [float(np.mean(values)) for values in residuals]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    boxplot = ax.boxplot(
        residuals,
        positions=speeds,
        widths=0.38,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#222222", "linewidth": 1.4},
    )
    for box in boxplot["boxes"]:
        box.set_facecolor(_MODEL_COLORS["Random Forest"])
        box.set_alpha(0.55)
    ax.plot(speeds, means, "D-", color="#222222", label="Mean residual", markersize=5)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(speeds)
    ax.set_xticklabels([f"{speed:g}" for speed in speeds])
    ax.set_xlabel("Actual speed (mph)")
    ax.set_ylabel("Residual: predicted − actual (mph)")
    ax.set_title("Random Forest Residuals by Speed — Standard Validation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "residual_by_speed_standard.png")


def plot_random_forest_prediction_trace_standard(
    predictions_path: str | Path = (
        "outputs/tables/same_subject_standard_predictions.csv"
    ),
    *,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Plot the Random Forest prediction trace across standard recordings."""

    table = _read_table(
        predictions_path,
        {
            "model",
            "recording_id",
            "session",
            "window_index",
            "actual_speed_mph",
            "predicted_speed_mph",
        },
    )
    table = table.loc[table["model"] == "Random Forest"].copy()
    if table.empty:
        raise ValueError("Standard prediction table has no Random Forest rows.")
    table = table.sort_values(
        ["actual_speed_mph", "recording_id", "window_index"]
    ).reset_index(drop=True)
    x = np.arange(len(table))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.step(
        x,
        table["actual_speed_mph"],
        where="mid",
        color="#222222",
        linestyle="--",
        linewidth=1.5,
        label="Actual speed",
    )
    ax.plot(
        x,
        table["predicted_speed_mph"],
        color=_MODEL_COLORS["Random Forest"],
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Random Forest prediction",
    )

    tick_positions: list[float] = []
    tick_labels: list[str] = []
    for _, group in table.groupby("recording_id", sort=False):
        first = int(group.index.min())
        last = int(group.index.max())
        tick_positions.append((first + last) / 2)
        speed = float(group["actual_speed_mph"].iloc[0])
        session = str(group["session"].iloc[0]).replace("day", "Day ")
        tick_labels.append(f"{speed:g} mph\n{session}")
        if first:
            ax.axvline(first - 0.5, color="#BBBBBB", linewidth=0.8)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Validation recording and window sequence")
    ax.set_ylabel("Speed (mph)")
    ax.set_title("Random Forest Prediction Trace — Standard Validation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return _save_figure(
        fig,
        output_dir,
        "random_forest_prediction_trace_standard.png",
    )


def plot_cadence_stress_predicted_vs_actual(
    predictions_path: str | Path = "outputs/tables/cadence_stress_predictions.csv",
    *,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Plot cadence-stress predictions by model and cadence condition."""

    table = _read_table(
        predictions_path,
        {"model", "condition", "actual_speed_mph", "predicted_speed_mph"},
    )
    lower, upper = _prediction_limits(table)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)

    for ax, model_name in zip(axes.flat, _MODEL_ORDER, strict=True):
        model_data = table.loc[table["model"] == model_name]
        if model_data.empty:
            raise ValueError(f"Cadence stress table has no rows for {model_name}.")
        for condition, color in _CONDITION_COLORS.items():
            condition_data = model_data.loc[model_data["condition"] == condition]
            ax.scatter(
                condition_data["actual_speed_mph"],
                condition_data["predicted_speed_mph"],
                s=24,
                alpha=0.55,
                color=color,
                edgecolors="none",
                label=condition.replace("cadence_", "").title(),
            )
        ax.plot([lower, upper], [lower, upper], "--", color="#777777", linewidth=1)
        ax.set_title(model_name, fontweight="semibold")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.grid(alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Cadence-Manipulation Stress Test", fontsize=15)
    fig.supxlabel("Actual speed (mph)", y=0.075)
    fig.supylabel("Predicted speed (mph)")
    fig.tight_layout(rect=(0.02, 0.12, 1, 0.96))
    return _save_figure(fig, output_dir, "cadence_stress_predicted_vs_actual.png")


def plot_feature_ablation_mae(
    metrics_path: str | Path = "outputs/tables/feature_ablation_metrics.csv",
    *,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Plot MAE for each reduced feature set and ablation model."""

    table = _read_table(metrics_path, {"feature_set", "model", "MAE"})
    feature_sets = list(dict.fromkeys(table["feature_set"]))
    models = ("Linear Regression", "Random Forest")
    x = np.arange(len(feature_sets), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for offset, model_name in zip((-width / 2, width / 2), models, strict=True):
        model_data = table.loc[table["model"] == model_name].set_index("feature_set")
        missing = [name for name in feature_sets if name not in model_data.index]
        if missing:
            raise ValueError(
                f"Feature ablation table lacks {model_name} rows for: "
                + ", ".join(missing)
            )
        values = model_data.loc[feature_sets, "MAE"].to_numpy(dtype=float)
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=model_name,
            color=_MODEL_COLORS[model_name],
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    labels = [
        _ABLATION_LABELS.get(name, name.replace("_", "\n", 1))
        for name in feature_sets
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylabel("MAE (mph)")
    ax.set_title("Feature Ablation — Standard Validation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_ylim(0, table["MAE"].max() * 1.18)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "feature_ablation_mae.png")


def plot_cadence_stress_error_by_condition(
    error_table_path: str | Path = "outputs/tables/error_by_speed_condition.csv",
    *,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Plot cadence-stress MAE by cadence condition and model."""

    table = _read_table(
        error_table_path,
        {"condition", "model", "n_windows", "mean_absolute_error_mph"},
    )
    if table.empty:
        raise ValueError("Cadence stress error table is empty.")

    condition_order = [
        condition
        for condition in _CONDITION_COLORS
        if condition in set(table["condition"])
    ]
    condition_order.extend(
        sorted(set(table["condition"]).difference(condition_order))
    )
    model_order = [model for model in _MODEL_ORDER if model in set(table["model"])]
    if not model_order:
        raise ValueError("Cadence stress error table has no recognized model rows.")

    rows: list[dict[str, float | str]] = []
    for condition in condition_order:
        condition_data = table.loc[table["condition"] == condition]
        for model_name in model_order:
            model_data = condition_data.loc[condition_data["model"] == model_name]
            if model_data.empty:
                continue
            weights = model_data["n_windows"].to_numpy(dtype=float)
            mae = np.average(
                model_data["mean_absolute_error_mph"].to_numpy(dtype=float),
                weights=weights,
            )
            rows.append(
                {
                    "condition": condition,
                    "model": model_name,
                    "mae": float(mae),
                }
            )
    plot_data = pd.DataFrame(rows)
    if plot_data.empty:
        raise ValueError("Cadence stress error table produced no plottable rows.")

    x = np.arange(len(condition_order), dtype=float)
    width = min(0.18, 0.75 / len(model_order))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    offsets = (np.arange(len(model_order)) - (len(model_order) - 1) / 2) * width
    for offset, model_name in zip(offsets, model_order, strict=True):
        model_data = plot_data.loc[plot_data["model"] == model_name].set_index(
            "condition"
        )
        values = [
            float(model_data.loc[condition, "mae"])
            if condition in model_data.index
            else np.nan
            for condition in condition_order
        ]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=model_name,
            color=_MODEL_COLORS.get(model_name),
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    labels = [
        str(condition).replace("cadence_", "").replace("_", " ").title()
        for condition in condition_order
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Cadence condition")
    ax.set_ylabel("MAE (mph)")
    ax.set_title("Cadence Stress Error by Condition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    upper = float(np.nanmax(plot_data["mae"])) * 1.18
    ax.set_ylim(0, upper if upper > 0 else 1)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "cadence_stress_error_by_condition.png")


def plot_cadence_robustness_loso_mae(
    metrics_path: str | Path = "outputs/tables/cadence_robustness_loso_metrics.csv",
    *,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Plot LOSO cadence-robustness MAE by held-out Day 4 speed and model."""

    table = _read_table(metrics_path, {"held_out_speed_mph", "model", "MAE"})
    table = table.copy()
    table["held_out_speed_numeric"] = pd.to_numeric(
        table["held_out_speed_mph"],
        errors="coerce",
    )
    table = table.dropna(subset=["held_out_speed_numeric"])
    if table.empty:
        raise ValueError("LOSO metrics table has no held-out speed rows to plot.")

    speeds = sorted(table["held_out_speed_numeric"].unique())
    model_order = [model for model in _MODEL_ORDER if model in set(table["model"])]
    if not model_order:
        raise ValueError("LOSO metrics table has no recognized model rows.")

    x = np.arange(len(speeds), dtype=float)
    width = min(0.18, 0.75 / len(model_order))
    offsets = (np.arange(len(model_order)) - (len(model_order) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for offset, model_name in zip(offsets, model_order, strict=True):
        model_data = table.loc[table["model"] == model_name].set_index(
            "held_out_speed_numeric"
        )
        missing = [speed for speed in speeds if speed not in model_data.index]
        if missing:
            raise ValueError(
                f"LOSO metrics table lacks {model_name} rows for held-out speeds: "
                + ", ".join(f"{speed:g}" for speed in missing)
            )
        values = model_data.loc[speeds, "MAE"].to_numpy(dtype=float)
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=model_name,
            color=_MODEL_COLORS.get(model_name),
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{speed:g} mph" for speed in speeds])
    ax.set_xlabel("Held-out Subject 1 Day 4 speed")
    ax.set_ylabel("MAE (mph)")
    ax.set_title("Cadence Robustness LOSO — MAE by Held-Out Speed")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    ax.set_ylim(0, float(table["MAE"].max()) * 1.18)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "cadence_robustness_loso_mae.png")


def plot_subject2_normal_cross_subject_predicted_vs_actual(
    predictions_path: str | Path = (
        "outputs/tables/subject2_normal_cross_subject_predictions.csv"
    ),
    *,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Plot Subject 2 normal-cadence cross-subject predictions by model."""

    table = _read_table(
        predictions_path,
        {"model", "actual_speed_mph", "predicted_speed_mph"},
    )
    lower, upper = _prediction_limits(table)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)

    for ax, model_name in zip(axes.flat, _MODEL_ORDER, strict=True):
        model_data = table.loc[table["model"] == model_name]
        if model_data.empty:
            raise ValueError(
                "Subject 2 cross-subject prediction table has no rows for "
                f"{model_name}."
            )
        color = _MODEL_COLORS[model_name]
        ax.scatter(
            model_data["actual_speed_mph"],
            model_data["predicted_speed_mph"],
            s=26,
            alpha=0.58,
            color=color,
            edgecolors="none",
        )
        means = model_data.groupby("actual_speed_mph")["predicted_speed_mph"].mean()
        ax.plot(means.index, means.values, "o-", color="#222222", linewidth=1.4)
        ax.plot([lower, upper], [lower, upper], "--", color="#777777", linewidth=1)
        ax.set_title(model_name, fontweight="semibold")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Subject 2 Normal Cadence Cross-Subject Test: Predicted vs Actual",
        fontsize=15,
    )
    fig.supxlabel("Actual speed (mph)")
    fig.supylabel("Predicted speed (mph)")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    return _save_figure(
        fig,
        output_dir,
        "subject2_normal_cross_subject_predicted_vs_actual.png",
    )


def plot_subject2_normal_cross_subject_error_by_speed(
    predictions_path: str | Path = (
        "outputs/tables/subject2_normal_cross_subject_predictions.csv"
    ),
    *,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Plot Subject 2 normal-cadence MAE by speed and model."""

    table = _read_table(
        predictions_path,
        {"model", "actual_speed_mph", "absolute_error_mph"},
    )
    speeds = sorted(table["actual_speed_mph"].unique())
    model_order = [model for model in _MODEL_ORDER if model in set(table["model"])]
    if not model_order:
        raise ValueError(
            "Subject 2 cross-subject prediction table has no recognized model rows."
        )

    speed_mae = (
        table.groupby(["actual_speed_mph", "model"], as_index=False)
        .agg(mae_mph=("absolute_error_mph", "mean"))
        .sort_values(["actual_speed_mph", "model"])
    )
    x = np.arange(len(speeds), dtype=float)
    width = min(0.18, 0.75 / len(model_order))
    offsets = (np.arange(len(model_order)) - (len(model_order) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for offset, model_name in zip(offsets, model_order, strict=True):
        model_data = speed_mae.loc[speed_mae["model"] == model_name].set_index(
            "actual_speed_mph"
        )
        values = [
            float(model_data.loc[speed, "mae_mph"])
            if speed in model_data.index
            else np.nan
            for speed in speeds
        ]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=model_name,
            color=_MODEL_COLORS.get(model_name),
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(speed):g} mph" for speed in speeds])
    ax.set_xlabel("Subject 2 actual speed")
    ax.set_ylabel("MAE (mph)")
    ax.set_title("Subject 2 Normal Cadence Cross-Subject Error by Speed")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    ax.set_ylim(0, float(speed_mae["mae_mph"].max()) * 1.18)
    fig.tight_layout()
    return _save_figure(
        fig,
        output_dir,
        "subject2_normal_cross_subject_error_by_speed.png",
    )


def plot_subject4_predicted_vs_actual(
    predictions_path: str | Path = (
        "outputs/tables/subject4_final_cross_subject_predictions.csv"
    ),
    *,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Plot Subject 4 final-test predictions against actual speed."""

    table = _read_table(
        predictions_path,
        {"feature_set", "actual_speed_mph", "predicted_speed_mph"},
    )
    feature_order = [
        "v2",
        "v2_plus_vertical_peak_sharpness",
        "v4_morphology",
    ]
    labels = {
        "v2": "v2",
        "v2_plus_vertical_peak_sharpness": "v2 + sharpness",
        "v4_morphology": "v4 morphology",
    }
    lower, upper = _prediction_limits(table)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True, sharey=True)
    colors = ("#59A14F", "#4C78A8", "#E15759")
    for ax, feature_set, color in zip(axes, feature_order, colors, strict=True):
        subset = table.loc[table["feature_set"] == feature_set]
        if subset.empty:
            raise ValueError(f"Subject 4 predictions missing feature set {feature_set}.")
        ax.scatter(
            subset["actual_speed_mph"],
            subset["predicted_speed_mph"],
            s=24,
            alpha=0.52,
            color=color,
            edgecolors="none",
        )
        means = subset.groupby("actual_speed_mph")["predicted_speed_mph"].mean()
        ax.plot(means.index, means.values, "o-", color="#222222", linewidth=1.4)
        ax.plot([lower, upper], [lower, upper], "--", color="#777777", linewidth=1)
        ax.set_title(labels[feature_set], fontweight="semibold")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.grid(alpha=0.25)
    fig.suptitle("Subject 4 Final Cross-Subject Test: Predicted vs Actual", fontsize=15)
    fig.supxlabel("Actual speed (mph)")
    fig.supylabel("Predicted speed (mph)")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.94))
    return _save_figure(fig, output_dir, "subject4_predicted_vs_actual.png")


def plot_subject4_error_by_speed(
    error_path: str | Path = "outputs/tables/subject4_error_by_speed.csv",
    *,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Plot Subject 4 final-test MAE by speed and feature set."""

    table = _read_table(
        error_path,
        {"feature_set", "actual_speed_mph", "mean_absolute_error_mph"},
    )
    feature_order = [
        "v2",
        "v2_plus_vertical_peak_sharpness",
        "v4_morphology",
    ]
    labels = {
        "v2": "v2",
        "v2_plus_vertical_peak_sharpness": "v2 + sharpness",
        "v4_morphology": "v4 morphology",
    }
    colors = {
        "v2": "#59A14F",
        "v2_plus_vertical_peak_sharpness": "#4C78A8",
        "v4_morphology": "#E15759",
    }
    speeds = sorted(table["actual_speed_mph"].unique())
    x = np.arange(len(speeds), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 5.8))
    offsets = (-width, 0.0, width)
    for offset, feature_set in zip(offsets, feature_order, strict=True):
        subset = table.loc[table["feature_set"] == feature_set].set_index(
            "actual_speed_mph"
        )
        values = [
            float(subset.loc[speed, "mean_absolute_error_mph"])
            if speed in subset.index
            else np.nan
            for speed in speeds
        ]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=labels[feature_set],
            color=colors[feature_set],
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(speed):g}" for speed in speeds])
    ax.set_xlabel("Subject 4 actual speed (mph)")
    ax.set_ylabel("MAE (mph)")
    ax.set_title("Subject 4 Final Cross-Subject Error by Speed")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    ax.set_ylim(0, float(table["mean_absolute_error_mph"].max()) * 1.22)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "subject4_error_by_speed.png")


def plot_subject3_vs_subject4_cadence_speed(
    feature_table_path: str | Path = "data/processed/windowed_features.csv",
    *,
    output_dir: str | Path = "outputs/figures",
) -> Path:
    """Compare cadence-vs-speed behavior for Subjects 3 and 4."""

    table = _read_table(
        feature_table_path,
        {"subject_id", "condition", "speed_mph", "Cadence_spm"},
    )
    table = table.loc[
        table["subject_id"].isin(["subject_3", "subject_4"])
        & table["condition"].eq("normal")
    ].copy()
    if table.empty:
        raise ValueError("Feature table has no Subject 3/4 normal rows.")
    summary = (
        table.groupby(["subject_id", "speed_mph"], as_index=False)
        .agg(
            cadence_mean=("Cadence_spm", "mean"),
            cadence_std=("Cadence_spm", "std"),
        )
        .sort_values(["subject_id", "speed_mph"])
    )
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for subject_id, color in (("subject_3", "#4C78A8"), ("subject_4", "#E15759")):
        subset = summary.loc[summary["subject_id"] == subject_id]
        ax.errorbar(
            subset["speed_mph"],
            subset["cadence_mean"],
            yerr=subset["cadence_std"].fillna(0),
            marker="o",
            linewidth=1.6,
            capsize=3,
            color=color,
            label=subject_id.replace("_", " ").title(),
        )
    ax.set_xlabel("Speed (mph)")
    ax.set_ylabel("Cadence estimate (spm)")
    ax.set_title("Subject 3 vs Subject 4 Cadence-Speed Behavior")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "subject3_vs_subject4_cadence_speed.png")


def generate_all_plots(
    *,
    output_dir: str | Path = "outputs/figures",
) -> list[Path]:
    """Generate and save all standard LENZ evaluation figures."""

    plot_functions = (
        plot_predicted_vs_actual_standard,
        plot_residual_by_speed_standard,
        plot_random_forest_prediction_trace_standard,
        plot_cadence_stress_predicted_vs_actual,
        plot_feature_ablation_mae,
        plot_cadence_stress_error_by_condition,
        plot_cadence_robustness_loso_mae,
        plot_subject2_normal_cross_subject_predicted_vs_actual,
        plot_subject2_normal_cross_subject_error_by_speed,
        plot_subject4_predicted_vs_actual,
        plot_subject4_error_by_speed,
        plot_subject3_vs_subject4_cadence_speed,
    )
    paths = [plot(output_dir=output_dir) for plot in plot_functions]
    for path in paths:
        print(f"Saved figure to {path}")
    return paths
