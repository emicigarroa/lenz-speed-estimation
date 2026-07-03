"""Utilities for the Team LENZ IMU speed-estimation project."""

from .data import (
    CANONICAL_SIGNAL_COLUMNS,
    DataQualityWarning,
    ManifestError,
    RecordingLoadError,
    load_dataset,
    load_manifest,
    load_recording,
)
from .dataset import build_windowed_feature_table, save_windowed_feature_table
from .evaluation import (
    feature_ablation,
    same_subject_cadence_stress_test,
    same_subject_standard_validation,
    same_subject_validation,
)
from .features import FeatureExtractionError, extract_window_features
from .modeling import DEFAULT_FEATURES, REDUCED_FEATURE_SETS, get_models
from .plotting import (
    generate_all_plots,
    plot_cadence_stress_predicted_vs_actual,
    plot_feature_ablation_mae,
    plot_predicted_vs_actual_standard,
    plot_random_forest_prediction_trace_standard,
    plot_residual_by_speed_standard,
)
from .windowing import SignalWindow, apply_trim, make_windows

__all__ = [
    "CANONICAL_SIGNAL_COLUMNS",
    "DataQualityWarning",
    "DEFAULT_FEATURES",
    "FeatureExtractionError",
    "ManifestError",
    "RecordingLoadError",
    "REDUCED_FEATURE_SETS",
    "SignalWindow",
    "apply_trim",
    "build_windowed_feature_table",
    "extract_window_features",
    "feature_ablation",
    "generate_all_plots",
    "get_models",
    "load_dataset",
    "load_manifest",
    "load_recording",
    "make_windows",
    "plot_cadence_stress_predicted_vs_actual",
    "plot_feature_ablation_mae",
    "plot_predicted_vs_actual_standard",
    "plot_random_forest_prediction_trace_standard",
    "plot_residual_by_speed_standard",
    "same_subject_cadence_stress_test",
    "same_subject_standard_validation",
    "same_subject_validation",
    "save_windowed_feature_table",
]
