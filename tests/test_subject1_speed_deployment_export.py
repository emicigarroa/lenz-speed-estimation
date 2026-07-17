"""Focused tests for the Subject 1 deployment model export package."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = REPOSITORY_ROOT / "exported_models/subject1_speed_deployment_v1"
BENCHMARK_DIR = REPOSITORY_ROOT / "exported_models/subject1_speed_benchmark_v1"
FIXTURE_DIR = DEPLOYMENT_DIR / "fixtures"


def _load_export_modules(export_dir: Path):
    sys.modules.pop("feature_extractor", None)
    sys.modules.pop("inference", None)
    sys.path.insert(0, str(export_dir))
    try:
        feature_spec = importlib.util.spec_from_file_location(
            "feature_extractor",
            export_dir / "feature_extractor.py",
        )
        inference_spec = importlib.util.spec_from_file_location(
            "inference",
            export_dir / "inference.py",
        )
        assert feature_spec is not None and feature_spec.loader is not None
        assert inference_spec is not None and inference_spec.loader is not None
        feature_module = importlib.util.module_from_spec(feature_spec)
        sys.modules["feature_extractor"] = feature_module
        feature_spec.loader.exec_module(feature_module)
        inference_module = importlib.util.module_from_spec(inference_spec)
        sys.modules["inference"] = inference_module
        inference_spec.loader.exec_module(inference_module)
        return feature_module, inference_module
    finally:
        try:
            sys.path.remove(str(export_dir))
        except ValueError:
            pass


def _load_standalone_feature_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


feature_extractor, inference = _load_export_modules(DEPLOYMENT_DIR)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_window(name: str) -> list[dict[str, float]]:
    with (FIXTURE_DIR / f"{name}_window.csv").open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _fixture_names() -> tuple[str, ...]:
    return ("low_speed", "mid_speed", "high_speed")


def _model_sha256() -> str:
    digest = hashlib.sha256()
    with (DEPLOYMENT_DIR / "model.joblib").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_model_load_metadata_checksum_and_feature_order() -> None:
    metadata = inference.load_metadata()
    assert metadata["model_version"] == "subject1_speed_deployment_v1"
    assert metadata["model_identity"] == "Subject 1 calibrated deployment model"
    assert metadata["model_role"] == "deployment"
    assert metadata["originating_benchmark_version"] == "subject1_speed_benchmark_v1"
    assert metadata["originating_benchmark_commit"] == "765ac43"
    assert metadata["deployment_training_recording_count"] == 21
    assert metadata["deployment_training_window_count"] == 407
    assert tuple(metadata["feature_names"]) == feature_extractor.FEATURE_NAMES
    assert metadata["model_file_sha256"] == _model_sha256()
    assert inference.verify_model_checksum() == metadata["model_file_sha256"]
    model = inference.load_model()
    assert model.n_estimators == 200
    assert model.max_depth == 5
    assert model.random_state == 42


@pytest.mark.parametrize("fixture_name", _fixture_names())
def test_golden_feature_parity(fixture_name: str) -> None:
    expected = _load_json(FIXTURE_DIR / f"{fixture_name}_features.json")
    observed = feature_extractor.extract_feature_dict(_load_window(fixture_name))
    diffs = {
        feature: abs(observed[feature] - expected["features"][feature])
        for feature in feature_extractor.FEATURE_NAMES
    }
    assert max(diffs.values()) <= 1e-10


@pytest.mark.parametrize("fixture_name", _fixture_names())
def test_golden_prediction_parity(fixture_name: str) -> None:
    expected = _load_json(FIXTURE_DIR / f"{fixture_name}_prediction.json")
    observed = inference.predict_window_mph(_load_window(fixture_name))
    assert abs(observed - expected["prediction_mph"]) <= 1e-6
    assert math.isfinite(observed)


def test_benchmark_and_deployment_feature_extractors_match_on_same_window() -> None:
    benchmark_feature_extractor = _load_standalone_feature_module(
        BENCHMARK_DIR / "feature_extractor.py",
        "benchmark_feature_extractor_for_deployment_test",
    )
    window = _load_window("mid_speed")
    deployment_features = feature_extractor.extract_feature_dict(window)
    benchmark_features = benchmark_feature_extractor.extract_feature_dict(window)
    diffs = {
        feature: abs(deployment_features[feature] - benchmark_features[feature])
        for feature in feature_extractor.FEATURE_NAMES
    }
    assert max(diffs.values()) <= 1e-12


def test_streaming_no_prediction_before_readiness_then_complete_prediction() -> None:
    estimator = inference.SpeedEstimator()
    samples = _load_window("mid_speed")
    for index, sample in enumerate(samples[:-1]):
        assert estimator.add_sample({**sample, "timestamp_sec": index / 200})
        assert not estimator.ready
    with pytest.raises(inference.InferenceError):
        estimator.predict_mph()
    assert estimator.add_sample({**samples[-1], "timestamp_sec": 999 / 200})
    assert estimator.ready
    prediction = estimator.predict_mph()
    assert isinstance(prediction, float)
    assert math.isfinite(prediction)


def test_streaming_prediction_interval_behavior() -> None:
    estimator = inference.SpeedEstimator()
    samples = _load_window("mid_speed")
    for index, sample in enumerate(samples):
        assert estimator.add_sample({**sample, "timestamp_sec": index / 200})
    assert math.isfinite(estimator.predict_mph())
    for offset, sample in enumerate(samples[:499], start=1000):
        assert estimator.add_sample({**sample, "timestamp_sec": offset / 200})
        assert not estimator.ready
    assert estimator.add_sample({**samples[499], "timestamp_sec": 1499 / 200})
    assert estimator.ready
    assert math.isfinite(estimator.predict_mph())


def test_streaming_malformed_and_nonfinite_sample_rejection() -> None:
    estimator = inference.SpeedEstimator()
    sample = _load_window("low_speed")[0]
    malformed = dict(sample)
    malformed.pop("az_g")
    assert not estimator.add_sample(malformed)
    assert estimator.status()["state"] == "invalid_sample"
    nonfinite = dict(sample)
    nonfinite["az_g"] = np.nan
    assert not estimator.add_sample(nonfinite)
    assert estimator.status()["state"] == "invalid_sample"


def test_streaming_timestamp_reversal_resets() -> None:
    estimator = inference.SpeedEstimator()
    sample = _load_window("low_speed")[0]
    assert estimator.add_sample({**sample, "timestamp_sec": 1.0})
    assert estimator.add_sample({**sample, "timestamp_sec": 1.005})
    assert not estimator.add_sample({**sample, "timestamp_sec": 1.002})
    status = estimator.status()
    assert status["state"] == "timestamp_reversal_reset"
    assert status["buffered_samples"] == 0


def test_streaming_large_gap_resets() -> None:
    estimator = inference.SpeedEstimator()
    sample = _load_window("low_speed")[0]
    assert estimator.add_sample({**sample, "timestamp_sec": 1.0})
    assert not estimator.add_sample({**sample, "timestamp_sec": 1.2})
    status = estimator.status()
    assert status["state"] == "large_gap_reset"
    assert status["buffered_samples"] == 0


def test_output_is_finite_float_after_reset() -> None:
    estimator = inference.SpeedEstimator()
    sample = _load_window("low_speed")[0]
    assert estimator.add_sample(sample)
    estimator.reset()
    assert not estimator.ready
    samples = _load_window("high_speed")
    for row in samples:
        assert estimator.add_sample(row)
    prediction = estimator.predict_mph()
    assert isinstance(prediction, float)
    assert math.isfinite(prediction)
