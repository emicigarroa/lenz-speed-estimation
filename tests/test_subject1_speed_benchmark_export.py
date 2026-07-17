"""Focused tests for the frozen Subject 1 benchmark export package."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = REPOSITORY_ROOT / "exported_models/subject1_speed_benchmark_v1"
FIXTURE_DIR = EXPORT_DIR / "fixtures"

if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

feature_extractor = importlib.import_module("feature_extractor")
inference = importlib.import_module("inference")


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
    with (EXPORT_DIR / "model.joblib").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_model_load_metadata_checksum_and_feature_order() -> None:
    metadata = inference.load_metadata()
    assert metadata["model_identity"] == "Subject 1 calibrated benchmark model"
    assert metadata["model_role"] == "benchmark"
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


def test_streaming_no_prediction_before_readiness_then_complete_prediction() -> None:
    estimator = inference.SpeedEstimator()
    samples = _load_window("mid_speed")
    for index, sample in enumerate(samples[:-1]):
        accepted = estimator.add_sample({**sample, "timestamp_sec": index / 200})
        assert accepted
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
    first_prediction = estimator.predict_mph()
    assert math.isfinite(first_prediction)
    for offset, sample in enumerate(samples[:499], start=1000):
        assert estimator.add_sample({**sample, "timestamp_sec": offset / 200})
        assert not estimator.ready
    assert estimator.add_sample({**samples[499], "timestamp_sec": 1499 / 200})
    assert estimator.ready
    second_prediction = estimator.predict_mph()
    assert math.isfinite(second_prediction)


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


def test_streaming_reset_behavior() -> None:
    estimator = inference.SpeedEstimator()
    sample = _load_window("low_speed")[0]
    assert estimator.add_sample(sample)
    assert estimator.status()["buffered_samples"] == 1
    estimator.reset()
    assert estimator.status()["state"] == "warming_up"
    assert estimator.status()["buffered_samples"] == 0
    assert not estimator.ready


def test_runtime_modules_do_not_import_research_modules() -> None:
    forbidden = (
        "lenz_speed",
        "dataset_manifest",
        "evaluation",
        "plotting",
        "dataset.py",
        "matplotlib",
        "pandas",
    )
    for module_name in ("feature_extractor.py", "inference.py"):
        text = (EXPORT_DIR / module_name).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text
