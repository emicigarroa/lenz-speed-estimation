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
    feature_ablation,
    generate_all_plots,
    same_subject_cadence_stress_test,
    same_subject_standard_validation,
    save_windowed_feature_table,
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
    ablation_metrics, ablation_predictions = feature_ablation()
    figure_paths = generate_all_plots()

    table_paths = [
        REPOSITORY_ROOT / "outputs/tables/same_subject_standard_metrics.csv",
        REPOSITORY_ROOT / "outputs/tables/same_subject_standard_predictions.csv",
        REPOSITORY_ROOT / "outputs/tables/cadence_stress_metrics.csv",
        REPOSITORY_ROOT / "outputs/tables/cadence_stress_predictions.csv",
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
