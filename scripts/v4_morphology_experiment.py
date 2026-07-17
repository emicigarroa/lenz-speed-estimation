"""Feature Engineering v4 morphology experiment for LENZ speed estimation."""

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from lenz_speed.data import load_manifest  # noqa: E402
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
SHARPNESS = "Vertical_Peak_Sharpness"
V4_FEATURES = (
    "Impact_Impulse",
    "Peak_Symmetry",
    "Impact_Crest_Factor",
    "Impact_Local_Kurtosis",
)
FEATURE_SETS = {
    "v2_only": V2_FEATURES,
    "v3_full": V3_FEATURES,
    "v2_plus_sharpness": (*V2_FEATURES, SHARPNESS),
    "v2_plus_sharpness_impulse": (*V2_FEATURES, SHARPNESS, "Impact_Impulse"),
    "v2_plus_sharpness_symmetry": (*V2_FEATURES, SHARPNESS, "Peak_Symmetry"),
    "v2_plus_sharpness_crest": (*V2_FEATURES, SHARPNESS, "Impact_Crest_Factor"),
    "v2_plus_sharpness_kurtosis": (
        *V2_FEATURES,
        SHARPNESS,
        "Impact_Local_Kurtosis",
    ),
    "v4_morphology_all": (*V2_FEATURES, SHARPNESS, *V4_FEATURES),
}
FEATURE_LABELS = {
    "v2_only": "v2",
    "v3_full": "v3 full",
    "v2_plus_sharpness": "v2 + sharpness",
    "v2_plus_sharpness_impulse": "+ impulse",
    "v2_plus_sharpness_symmetry": "+ symmetry",
    "v2_plus_sharpness_crest": "+ crest",
    "v2_plus_sharpness_kurtosis": "+ kurtosis",
    "v4_morphology_all": "v4 all morph.",
}
EVALUATION_LABELS = {
    "standard": "Standard validation",
    "cadence_stress": "Cadence stress",
    "loso": "LOSO robustness",
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


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[object, np.ndarray]:
    model = _random_forest()
    model.fit(train.loc[:, features], train["speed_mph"])
    return model, model.predict(test.loc[:, features])


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


def _metric_row(predictions: pd.DataFrame, *, summary_level: str) -> dict[str, float | int | str]:
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
        _, predicted = _fit_predict(train, test, features)
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
    splits = []
    for fold_index, held_out_speed in enumerate((7.0, 6.0, 5.0), start=1):
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
            _, predicted = _fit_predict(train, test, features)
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
        rows.append(_metric_row(group, summary_level="overall"))
    for _, group in predictions.loc[predictions["evaluation"] == "loso"].groupby(
        ["feature_set", "fold"],
        sort=False,
    ):
        rows.append(_metric_row(group, summary_level="fold"))
    return pd.DataFrame(rows)


def _permutation_importance_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_set = "v4_morphology_all"
    features = FEATURE_SETS[feature_set]
    for evaluation in ("standard", "cadence_stress"):
        train, test = _same_subject_split(table, features, mode=evaluation)
        model, predicted = _fit_predict(train, test, features)
        baseline_mae = float(mean_absolute_error(test["speed_mph"], predicted))
        result = permutation_importance(
            model,
            test.loc[:, features],
            test["speed_mph"],
            scoring="neg_mean_absolute_error",
            n_repeats=20,
            random_state=42,
            n_jobs=1,
        )
        for feature, mean, std in zip(
            features,
            result.importances_mean,
            result.importances_std,
            strict=True,
        ):
            rows.append(
                {
                    "evaluation": evaluation,
                    "fold": "overall",
                    "held_out_speed_mph": "all",
                    "feature_set": feature_set,
                    "feature": feature,
                    "is_new_v4_feature": feature in V4_FEATURES,
                    "is_sharpness_feature": feature == SHARPNESS,
                    "baseline_mae": baseline_mae,
                    "mae_increase_mean": float(mean),
                    "mae_increase_std": float(std),
                }
            )

    loso_fold_rows = []
    for fold, held_out_speed, train, test in _loso_splits(table, features):
        model, predicted = _fit_predict(train, test, features)
        baseline_mae = float(mean_absolute_error(test["speed_mph"], predicted))
        result = permutation_importance(
            model,
            test.loc[:, features],
            test["speed_mph"],
            scoring="neg_mean_absolute_error",
            n_repeats=20,
            random_state=42,
            n_jobs=1,
        )
        for feature, mean, std in zip(
            features,
            result.importances_mean,
            result.importances_std,
            strict=True,
        ):
            loso_fold_rows.append(
                {
                    "evaluation": "loso",
                    "fold": fold,
                    "held_out_speed_mph": held_out_speed,
                    "feature_set": feature_set,
                    "feature": feature,
                    "is_new_v4_feature": feature in V4_FEATURES,
                    "is_sharpness_feature": feature == SHARPNESS,
                    "baseline_mae": baseline_mae,
                    "mae_increase_mean": float(mean),
                    "mae_increase_std": float(std),
                }
            )
    rows.extend(loso_fold_rows)
    loso = pd.DataFrame(loso_fold_rows)
    for feature, group in loso.groupby("feature", sort=False):
        rows.append(
            {
                "evaluation": "loso",
                "fold": "overall",
                "held_out_speed_mph": "all",
                "feature_set": feature_set,
                "feature": feature,
                "is_new_v4_feature": feature in V4_FEATURES,
                "is_sharpness_feature": feature == SHARPNESS,
                "baseline_mae": float(group["baseline_mae"].mean()),
                "mae_increase_mean": float(group["mae_increase_mean"].mean()),
                "mae_increase_std": float(group["mae_increase_mean"].std(ddof=0)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["evaluation", "fold", "mae_increase_mean"],
        ascending=[True, True, False],
    )


def _plot_metric_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    overall = metrics.loc[metrics["summary_level"] == "overall"].copy()
    evaluations = ("standard", "cadence_stress", "loso")
    feature_sets = tuple(FEATURE_SETS)
    x = np.arange(len(feature_sets), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), sharey=True)
    colors = [
        "#4C78A8",
        "#E15759",
        "#59A14F",
        "#F28E2B",
        "#B07AA1",
        "#76B7B2",
        "#EDC948",
        "#9C755F",
    ]
    for ax, evaluation in zip(axes, evaluations, strict=True):
        eval_data = overall.loc[overall["evaluation"] == evaluation].set_index(
            "feature_set"
        )
        values = [float(eval_data.loc[feature_set, "MAE"]) for feature_set in feature_sets]
        bars = ax.bar(x, values, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_title(EVALUATION_LABELS[evaluation])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [FEATURE_LABELS[feature_set] for feature_set in feature_sets],
            rotation=38,
            ha="right",
            fontsize=8,
        )
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Random Forest MAE (mph)")
    fig.suptitle("Feature Engineering v4 Morphology Comparison", fontsize=15)
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_permutation_importance(importance: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharex=True)
    for ax, evaluation in zip(
        axes,
        ("standard", "cadence_stress", "loso"),
        strict=True,
    ):
        subset = (
            importance.loc[
                (importance["evaluation"] == evaluation)
                & (importance["fold"] == "overall")
            ]
            .sort_values("mae_increase_mean", ascending=False)
            .head(12)
            .sort_values("mae_increase_mean", ascending=True)
        )
        colors = np.where(
            subset["is_new_v4_feature"],
            "#E15759",
            np.where(subset["is_sharpness_feature"], "#59A14F", "#4C78A8"),
        )
        ax.barh(subset["feature"], subset["mae_increase_mean"], color=colors)
        ax.axvline(0, color="#777777", linewidth=1)
        ax.set_title(EVALUATION_LABELS[evaluation])
        ax.set_xlabel("MAE increase after permutation (mph)")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Permutation Importance: v4 Morphology Feature Set", fontsize=15)
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    table = _load_feature_table()
    predictions = pd.concat(
        [
            *_standard_or_stress_predictions(table, mode="standard"),
            *_standard_or_stress_predictions(table, mode="cadence_stress"),
            *_loso_predictions(table),
        ],
        ignore_index=True,
    )
    metrics = _build_metrics(predictions)
    importance = _permutation_importance_rows(table)

    tables_dir = REPOSITORY_ROOT / "outputs/tables"
    figures_dir = REPOSITORY_ROOT / "outputs/figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / "v4_morphology_feature_metrics.csv"
    predictions_path = tables_dir / "v4_morphology_feature_predictions.csv"
    importance_path = tables_dir / "v4_morphology_permutation_importance.csv"
    comparison_path = figures_dir / "v4_morphology_feature_mae.png"
    importance_figure_path = figures_dir / "v4_morphology_permutation_importance.png"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    importance.to_csv(importance_path, index=False)
    _plot_metric_comparison(metrics, comparison_path)
    _plot_permutation_importance(importance, importance_figure_path)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved permutation importance to {importance_path}")
    print(f"Saved comparison figure to {comparison_path}")
    print(f"Saved permutation figure to {importance_figure_path}")
    print("\nOverall MAE:")
    print(
        metrics.loc[metrics["summary_level"] == "overall"]
        .pivot(index="feature_set", columns="evaluation", values="MAE")
        .loc[list(FEATURE_SETS)]
        .to_string(float_format=lambda value: f"{value:.4f}")
    )
    print("\nTop permutation importances:")
    for evaluation in ("standard", "cadence_stress", "loso"):
        top = importance.loc[
            (importance["evaluation"] == evaluation)
            & (importance["fold"] == "overall")
        ].head(8)
        print(f"\n{evaluation}")
        print(
            top.loc[
                :,
                ["feature", "is_new_v4_feature", "mae_increase_mean", "mae_increase_std"],
            ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )


if __name__ == "__main__":
    main()
