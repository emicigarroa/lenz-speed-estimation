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
from .features import FeatureExtractionError, extract_window_features
from .windowing import SignalWindow, apply_trim, make_windows

__all__ = [
    "CANONICAL_SIGNAL_COLUMNS",
    "DataQualityWarning",
    "FeatureExtractionError",
    "ManifestError",
    "RecordingLoadError",
    "SignalWindow",
    "apply_trim",
    "build_windowed_feature_table",
    "extract_window_features",
    "load_dataset",
    "load_manifest",
    "load_recording",
    "make_windows",
    "save_windowed_feature_table",
]
