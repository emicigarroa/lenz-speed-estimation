"""Rank Random Forest permutation importance for v1--v4 feature sets."""

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
from sklearn.inspection import permutation_importance  # noqa: E402
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
V4_MORPHOLOGY_FEATURES = (
    *V2_FEATURES,
    "Vertical_Peak_Sharpness",
    "Impact_Impulse",
    "Peak_Symmetry",
    "Impact_Crest_Factor",
    "Impact_Local_Kurtosis",
)
FEATURE_SETS = {
    "v1_basic": V1_FEATURES,
    "v2_intensity": V2_FEATURES,
    "v3_temporal": tuple(DEFAULT_FEATURES),
    "v4_morphology": V4_MORPHOLOGY_FEATURES,
}
MODEL_LABELS = {
    "v1_basic": "v1 basic",
    "v2_intensity": "v2 intensity",
    "v3_temporal": "v3 temporal",
    "v4_morphology": "v4 morphology",
}
COLORS = {
    "v1_basic": "#4C78A8",
    "v2_intensity": "#59A14F",
    "v3_temporal": "#E15759",
    "v4_morphology": "#F28E2B",
}


def _load_feature_table() -> pd.DataFrame:
    path = REPOSITORY_ROOT / "data/processed/windowed_features.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run python run_pipeline.py before this script."
        )
    return pd.read_csv(path)


def _rank_importance(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for feature_set, features in FEATURE_SETS.items():
        train, test = _same_subject_split(table, features, mode="standard")
        model = clone(get_models()["Random Forest"])
        model.fit(train.loc[:, features], train["speed_mph"])
        predicted = model.predict(test.loc[:, features])
        baseline_mae = float(mean_absolute_error(test["speed_mph"], predicted))
        result = permutation_importance(
            model,
            test.loc[:, features],
            test["speed_mph"],
            scoring="neg_mean_absolute_error",
            n_repeats=30,
            random_state=42,
            n_jobs=1,
        )
        order = np.argsort(result.importances_mean)[::-1]
        for rank, index in enumerate(order, start=1):
            rows.append(
                {
                    "model_version": feature_set,
                    "model_label": MODEL_LABELS[feature_set],
                    "feature": features[index],
                    "rank": rank,
                    "baseline_mae": baseline_mae,
                    "mae_increase_mean": float(result.importances_mean[index]),
                    "mae_increase_std": float(result.importances_std[index]),
                }
            )
    return pd.DataFrame(rows)


def _plot_ranked_importance(importance: pd.DataFrame) -> Path:
    output_dir = REPOSITORY_ROOT / "outputs/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "feature_importance_ranked_by_model_version.png"

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    for ax, feature_set in zip(axes.flat, FEATURE_SETS, strict=True):
        subset = importance.loc[
            importance["model_version"] == feature_set
        ].sort_values("rank", ascending=False)
        y = np.arange(len(subset))
        values = subset["mae_increase_mean"].to_numpy(dtype=float)
        ax.barh(y, values, color=COLORS[feature_set], alpha=0.88)
        ax.set_yticks(y)
        ax.set_yticklabels(subset["feature"], fontsize=8)
        ax.axvline(0, color="#777777", linewidth=1)
        ax.grid(axis="x", alpha=0.25)
        baseline = float(subset["baseline_mae"].iloc[0])
        ax.set_title(
            f"{MODEL_LABELS[feature_set]} RF — baseline MAE {baseline:.3f} mph",
            fontweight="semibold",
        )
        ax.set_xlabel("MAE increase after permutation (mph)")

    fig.suptitle(
        "Ranked Feature Importance by Speed Model Version\n"
        "Random Forest, standard validation split",
        fontsize=16,
    )
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    table = _load_feature_table()
    importance = _rank_importance(table)
    table_dir = REPOSITORY_ROOT / "outputs/tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "feature_importance_ranked_by_model_version.csv"
    importance.to_csv(table_path, index=False)
    figure_path = _plot_ranked_importance(importance)

    print(f"Saved table to {table_path}")
    print(f"Saved figure to {figure_path}")
    print("\nTop 5 features by model version:")
    for feature_set in FEATURE_SETS:
        subset = importance.loc[importance["model_version"] == feature_set].head(5)
        print(f"\n{MODEL_LABELS[feature_set]}")
        print(
            subset.loc[
                :,
                ["rank", "feature", "mae_increase_mean", "mae_increase_std"],
            ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )


if __name__ == "__main__":
    main()
