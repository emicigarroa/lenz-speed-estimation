"""Permutation-importance comparison for v2 versus v3 LENZ features."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from typing import Literal


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

from lenz_speed.data import load_manifest  # noqa: E402
from lenz_speed.evaluation import _same_subject_split  # noqa: E402
from lenz_speed.modeling import DEFAULT_FEATURES, get_models  # noqa: E402


V2_FEATURES = (
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
V3_FEATURES = tuple(DEFAULT_FEATURES)
V3_ONLY_FEATURES = tuple(feature for feature in V3_FEATURES if feature not in V2_FEATURES)


def _load_feature_table() -> pd.DataFrame:
    path = REPOSITORY_ROOT / "data/processed/windowed_features.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature table not found: {path}. Run python run_pipeline.py first."
        )
    return pd.read_csv(path)


def _random_forest():
    return get_models()["Random Forest"]


def _fit_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[object, float]:
    estimator = clone(_random_forest())
    estimator.fit(train.loc[:, features], train["speed_mph"])
    predictions = estimator.predict(test.loc[:, features])
    mae = float(mean_absolute_error(test["speed_mph"], predictions))
    return estimator, mae


def _permutation_rows(
    *,
    evaluation: str,
    feature_set: Literal["v2", "v3_full"],
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    fold: str = "overall",
    held_out_speed_mph: str | float = "",
    n_repeats: int = 20,
) -> pd.DataFrame:
    estimator, baseline_mae = _fit_and_score(train, test, features)
    result = permutation_importance(
        estimator,
        test.loc[:, features],
        test["speed_mph"],
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=1,
    )
    rows = pd.DataFrame(
        {
            "evaluation": evaluation,
            "fold": fold,
            "held_out_speed_mph": held_out_speed_mph,
            "feature_set": feature_set,
            "model": "Random Forest",
            "feature": features,
            "is_v3_feature": [feature in V3_ONLY_FEATURES for feature in features],
            "baseline_mae": baseline_mae,
            "n_train_windows": len(train),
            "n_test_windows": len(test),
            "mae_increase_mean": result.importances_mean,
            "mae_increase_std": result.importances_std,
        }
    )
    return rows.sort_values(
        ["mae_increase_mean", "mae_increase_std"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _standard_or_stress_importance(
    table: pd.DataFrame,
    *,
    mode: Literal["standard", "cadence_stress"],
) -> pd.DataFrame:
    frames = []
    for feature_set, features in (("v2", V2_FEATURES), ("v3_full", V3_FEATURES)):
        train, test = _same_subject_split(table, features, mode=mode)
        frames.append(
            _permutation_rows(
                evaluation=mode,
                feature_set=feature_set,
                train=train,
                test=test,
                features=features,
            )
        )
    return pd.concat(frames, ignore_index=True)


def _loso_splits(
    table: pd.DataFrame,
    features: tuple[str, ...],
) -> list[tuple[str, float, tuple[float, ...], pd.DataFrame, pd.DataFrame]]:
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
        train_day4_speeds = tuple(
            speed for speed in sorted(held_out_speeds) if speed != held_out_speed
        )
        train_day4 = day4_table.loc[
            day4_table["speed_mph"].astype(float).isin(train_day4_speeds)
        ].copy()
        test = day4_table.loc[
            day4_table["speed_mph"].astype(float) == held_out_speed
        ].copy()
        train = pd.concat([day3_train, train_day4], ignore_index=True)
        splits.append((f"fold_{fold_index}", held_out_speed, train_day4_speeds, train, test))
    return splits


def _loso_importance(table: pd.DataFrame) -> pd.DataFrame:
    fold_frames = []
    for feature_set, features in (("v2", V2_FEATURES), ("v3_full", V3_FEATURES)):
        for fold, held_out_speed, _, train, test in _loso_splits(table, features):
            fold_frames.append(
                _permutation_rows(
                    evaluation="loso",
                    feature_set=feature_set,
                    train=train,
                    test=test,
                    features=features,
                    fold=fold,
                    held_out_speed_mph=held_out_speed,
                    n_repeats=15,
                )
            )
    fold_rows = pd.concat(fold_frames, ignore_index=True)

    overall_rows = []
    for (feature_set, feature), group in fold_rows.groupby(
        ["feature_set", "feature"],
        sort=False,
    ):
        weights = group["n_test_windows"].to_numpy(dtype=float)
        overall_rows.append(
            {
                "evaluation": "loso",
                "fold": "overall",
                "held_out_speed_mph": "all",
                "feature_set": feature_set,
                "model": "Random Forest",
                "feature": feature,
                "is_v3_feature": feature in V3_ONLY_FEATURES,
                "baseline_mae": np.average(
                    group["baseline_mae"].to_numpy(dtype=float),
                    weights=weights,
                ),
                "n_train_windows": np.nan,
                "n_test_windows": int(group["n_test_windows"].sum()),
                "mae_increase_mean": np.average(
                    group["mae_increase_mean"].to_numpy(dtype=float),
                    weights=weights,
                ),
                "mae_increase_std": np.average(
                    group["mae_increase_std"].to_numpy(dtype=float),
                    weights=weights,
                ),
            }
        )
    overall = pd.DataFrame(overall_rows)
    return pd.concat([overall, fold_rows], ignore_index=True).sort_values(
        ["fold", "feature_set", "mae_increase_mean"],
        ascending=[True, True, False],
    )


def _save_importance_plot(
    table: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    top_n: int = 10,
) -> None:
    overall = table.loc[table["fold"] == "overall"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True)

    for ax, feature_set in zip(axes, ("v2", "v3_full"), strict=True):
        subset = (
            overall.loc[overall["feature_set"] == feature_set]
            .sort_values("mae_increase_mean", ascending=False)
            .head(top_n)
            .sort_values("mae_increase_mean", ascending=True)
        )
        colors = np.where(subset["is_v3_feature"], "#E15759", "#4C78A8")
        ax.barh(subset["feature"], subset["mae_increase_mean"], color=colors)
        ax.axvline(0, color="#777777", linewidth=1)
        ax.set_title("v2 features" if feature_set == "v2" else "v3 full features")
        ax.set_xlabel("MAE increase after permutation (mph)")
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    table = _load_feature_table()
    tables_dir = REPOSITORY_ROOT / "outputs/tables"
    figures_dir = REPOSITORY_ROOT / "outputs/figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    standard = _standard_or_stress_importance(table, mode="standard")
    stress = _standard_or_stress_importance(table, mode="cadence_stress")
    loso = _loso_importance(table)

    standard_path = tables_dir / "permutation_importance_standard.csv"
    stress_path = tables_dir / "permutation_importance_cadence_stress.csv"
    loso_path = tables_dir / "permutation_importance_loso.csv"
    standard.to_csv(standard_path, index=False)
    stress.to_csv(stress_path, index=False)
    loso.to_csv(loso_path, index=False)

    _save_importance_plot(
        standard,
        title="Permutation Importance — Standard Validation",
        output_path=figures_dir / "permutation_importance_standard.png",
    )
    _save_importance_plot(
        stress,
        title="Permutation Importance — Cadence Stress",
        output_path=figures_dir / "permutation_importance_cadence_stress.png",
    )

    print(f"Saved {standard_path}")
    print(f"Saved {stress_path}")
    print(f"Saved {loso_path}")
    print(f"Saved {figures_dir / 'permutation_importance_standard.png'}")
    print(f"Saved {figures_dir / 'permutation_importance_cadence_stress.png'}")
    for name, result in (
        ("standard", standard),
        ("cadence_stress", stress),
        ("loso", loso),
    ):
        print(f"\nTop 10 permutation importances: {name}")
        top = (
            result.loc[result["fold"] == "overall"]
            .sort_values(["feature_set", "mae_increase_mean"], ascending=[True, False])
            .groupby("feature_set", group_keys=False)
            .head(10)
        )
        print(
            top.loc[
                :,
                [
                    "feature_set",
                    "feature",
                    "is_v3_feature",
                    "baseline_mae",
                    "mae_increase_mean",
                    "mae_increase_std",
                ],
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
