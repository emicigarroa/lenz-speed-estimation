"""Integration smoke tests for deterministic evaluation workflows."""

from pathlib import Path

from lenz_speed.evaluation import feature_ablation, same_subject_standard_validation


def test_same_subject_standard_validation_runs(tmp_path: Path) -> None:
    metrics, predictions = same_subject_standard_validation(
        output_dir=tmp_path / "standard"
    )

    assert not metrics.empty
    assert not predictions.empty
    assert set(metrics["model"]) == {
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Random Forest",
    }
    assert set(predictions["session"]) == {"day2", "day4"}
    assert (
        predictions.loc[predictions["session"] == "day4", "condition"]
        == "cadence_normal"
    ).all()
    assert (tmp_path / "standard" / "same_subject_standard_metrics.csv").is_file()
    assert (
        tmp_path / "standard" / "same_subject_standard_predictions.csv"
    ).is_file()


def test_feature_ablation_returns_twelve_metric_rows(tmp_path: Path) -> None:
    metrics, predictions = feature_ablation(output_dir=tmp_path / "ablation")

    assert len(metrics) == 12
    assert not predictions.empty
    assert metrics["feature_set"].nunique() == 6
    assert set(metrics["model"]) == {"Linear Regression", "Random Forest"}
    assert (tmp_path / "ablation" / "feature_ablation_metrics.csv").is_file()
    assert (tmp_path / "ablation" / "feature_ablation_predictions.csv").is_file()
