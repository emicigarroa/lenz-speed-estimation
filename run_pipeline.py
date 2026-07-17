"""Run the complete reproducible LENZ speed-estimation pipeline."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lenz-speed-matplotlib"),
)
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lenz_speed import (  # noqa: E402
    cadence_robustness_loso,
    cadence_stress_error_analysis,
    feature_ablation,
    generate_all_plots,
    same_subject_cadence_stress_test,
    same_subject_standard_validation,
    save_windowed_feature_table,
    subject4_final_cross_subject_test,
    subject2_normal_cross_subject_validation,
    subjects_1_3_development_validation,
    subjects_1_3_loso_validation,
)


def _model_mae(metrics, model_name: str) -> float:
    """Return one model's MAE from an evaluation metrics table."""

    matches = metrics.loc[metrics["model"] == model_name, "MAE"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one MAE row for {model_name!r}.")
    return float(matches.iloc[0])


def main() -> None:
    """Build features, run evaluations, generate plots, and summarize outputs."""

    print("LENZ speed-estimation pipeline")
    print("=" * 30)

    feature_table_path = save_windowed_feature_table()
    standard_metrics, standard_predictions = same_subject_standard_validation()
    stress_metrics, stress_predictions = same_subject_cadence_stress_test()
    error_by_speed_condition, worst_recordings = cadence_stress_error_analysis()
    loso_metrics, loso_predictions = cadence_robustness_loso()
    subject2_metrics, subject2_predictions = subject2_normal_cross_subject_validation()
    development_metrics, development_predictions = subjects_1_3_development_validation()
    development_loso_metrics, development_loso_predictions = subjects_1_3_loso_validation()
    subject4_metrics, subject4_predictions, subject4_error_by_speed = (
        subject4_final_cross_subject_test()
    )
    ablation_metrics, ablation_predictions = feature_ablation()
    figure_paths = generate_all_plots()

    table_paths = [
        REPOSITORY_ROOT / "outputs/tables/same_subject_standard_metrics.csv",
        REPOSITORY_ROOT / "outputs/tables/same_subject_standard_predictions.csv",
        REPOSITORY_ROOT / "outputs/tables/cadence_stress_metrics.csv",
        REPOSITORY_ROOT / "outputs/tables/cadence_stress_predictions.csv",
        REPOSITORY_ROOT / "outputs/tables/error_by_speed_condition.csv",
        REPOSITORY_ROOT / "outputs/tables/worst_recordings.csv",
        REPOSITORY_ROOT / "outputs/tables/cadence_robustness_loso_metrics.csv",
        REPOSITORY_ROOT / "outputs/tables/cadence_robustness_loso_predictions.csv",
        REPOSITORY_ROOT / "outputs/tables/subject2_normal_cross_subject_metrics.csv",
        REPOSITORY_ROOT / "outputs/tables/subject2_normal_cross_subject_predictions.csv",
        REPOSITORY_ROOT / "outputs/tables/subjects1_3_development_metrics.csv",
        REPOSITORY_ROOT / "outputs/tables/subjects1_3_development_predictions.csv",
        REPOSITORY_ROOT / "outputs/tables/subjects1_3_loso_metrics.csv",
        REPOSITORY_ROOT / "outputs/tables/subjects1_3_loso_predictions.csv",
        REPOSITORY_ROOT / "outputs/tables/subject4_final_cross_subject_metrics.csv",
        REPOSITORY_ROOT / "outputs/tables/subject4_final_cross_subject_predictions.csv",
        REPOSITORY_ROOT / "outputs/tables/subject4_error_by_speed.csv",
        REPOSITORY_ROOT / "outputs/tables/feature_ablation_metrics.csv",
        REPOSITORY_ROOT / "outputs/tables/feature_ablation_predictions.csv",
    ]
    generated_paths = [feature_table_path, *table_paths, *figure_paths]
    missing = [path for path in generated_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Pipeline did not create expected files: "
            + ", ".join(str(path) for path in missing)
        )

    standard_rf_mae = _model_mae(standard_metrics, "Random Forest")
    stress_rf_mae = _model_mae(stress_metrics, "Random Forest")
    loso_rf_overall = loso_metrics.loc[
        (loso_metrics["model"] == "Random Forest")
        & (loso_metrics["fold"] == "overall"),
        "MAE",
    ]
    if len(loso_rf_overall) != 1:
        raise ValueError("Expected exactly one overall LOSO Random Forest MAE row.")
    subject2_rf_mae = _model_mae(subject2_metrics, "Random Forest")
    development_rf_mae = _model_mae(development_metrics, "Random Forest")
    subject4_v4_rf = subject4_metrics.loc[
        (subject4_metrics["feature_set"] == "v4_morphology")
        & (subject4_metrics["model"] == "Random Forest"),
        "MAE",
    ]
    if len(subject4_v4_rf) != 1:
        raise ValueError("Expected exactly one Subject 4 v4 Random Forest MAE row.")
    best_ablation = ablation_metrics.sort_values("MAE").iloc[0]
    with feature_table_path.open(encoding="utf-8") as feature_table_file:
        feature_row_count = sum(1 for _ in feature_table_file) - 1

    print("\nPipeline summary")
    print("-" * 30)
    print(f"Windowed feature rows: {feature_row_count}")
    print(
        "Standard validation: "
        f"{len(standard_predictions)} predictions, "
        f"Random Forest MAE={standard_rf_mae:.4f} mph"
    )
    print(
        "Cadence stress test: "
        f"{len(stress_predictions)} predictions, "
        f"Random Forest MAE={stress_rf_mae:.4f} mph"
    )
    print(
        "Cadence stress error analysis: "
        f"{len(error_by_speed_condition)} speed/condition/model rows, "
        f"{len(worst_recordings)} recording/model rows"
    )
    print(
        "Cadence robustness LOSO: "
        f"{len(loso_predictions)} predictions, "
        f"overall Random Forest MAE={float(loso_rf_overall.iloc[0]):.4f} mph"
    )
    print(
        "Subject 2 normal cadence cross-subject: "
        f"{len(subject2_predictions)} predictions, "
        f"Random Forest MAE={subject2_rf_mae:.4f} mph"
    )
    print(
        "Subjects 1--3 development validation: "
        f"{len(development_predictions)} predictions, "
        f"Random Forest MAE={development_rf_mae:.4f} mph"
    )
    print(
        "Subjects 1--3 LOSO development: "
        f"{len(development_loso_predictions)} predictions"
    )
    print(
        "Subject 4 final frozen test: "
        f"{len(subject4_predictions)} predictions, "
        f"v4 Random Forest MAE={float(subject4_v4_rf.iloc[0]):.4f} mph, "
        f"{len(subject4_error_by_speed)} speed-summary rows"
    )
    print(
        "Best feature ablation: "
        f"{best_ablation['feature_set']} / {best_ablation['model']}, "
        f"MAE={best_ablation['MAE']:.4f} mph "
        f"across {len(ablation_predictions)} predictions"
    )
    print("Generated files:")
    for path in generated_paths:
        print(f"  - {path.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
