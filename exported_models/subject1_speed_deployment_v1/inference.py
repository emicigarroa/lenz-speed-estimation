"""Production-style inference helpers for the frozen Subject 1 benchmark model."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from feature_extractor import (
    CANONICAL_INPUT_COLUMNS,
    FEATURE_NAMES,
    FeatureExtractionError,
    extract_feature_dict,
    extract_feature_vector,
)


EXPORT_DIR = Path(__file__).resolve().parent
MODEL_PATH = EXPORT_DIR / "model.joblib"
METADATA_PATH = EXPORT_DIR / "metadata.json"


class InferenceError(RuntimeError):
    """Raised when the exported benchmark model cannot produce a prediction."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(metadata_path: str | Path = METADATA_PATH) -> dict[str, Any]:
    """Load and minimally validate the export metadata."""

    path = Path(metadata_path)
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
        raise InferenceError("metadata feature_names do not match the runtime order.")
    if tuple(metadata.get("canonical_input_columns", ())) != CANONICAL_INPUT_COLUMNS:
        raise InferenceError("metadata canonical_input_columns do not match runtime.")
    return metadata


def verify_model_checksum(
    model_path: str | Path = MODEL_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> str:
    """Return the model SHA-256 after checking it against metadata."""

    metadata = load_metadata(metadata_path)
    observed = _sha256(Path(model_path))
    expected = metadata.get("model_file_sha256")
    if observed != expected:
        raise InferenceError(
            f"model checksum mismatch: observed {observed}, expected {expected}"
        )
    return observed


def load_model(
    model_path: str | Path = MODEL_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> Any:
    """Load the serialized estimator after feature-order and checksum checks."""

    verify_model_checksum(model_path=model_path, metadata_path=metadata_path)
    return joblib.load(model_path)


def predict_window_mph(
    window: Any,
    *,
    model: Any | None = None,
    fs: float = 200,
) -> float:
    """Predict treadmill speed in mph for one complete canonical signal window."""

    estimator = load_model() if model is None else model
    features = extract_feature_vector(window, fs=fs)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        prediction = float(estimator.predict(features)[0])
    if not math.isfinite(prediction):
        raise InferenceError("model produced a non-finite prediction.")
    return prediction


def features_for_window(window: Any, *, fs: float = 200) -> dict[str, float]:
    """Return the benchmark feature dictionary for one complete window."""

    return extract_feature_dict(window, fs=fs)


@dataclass(frozen=True)
class StreamingConfig:
    """Conservative streaming assumptions for the frozen benchmark artifact."""

    window_sec: float = 5.0
    step_sec: float = 2.5
    nominal_fs_hz: float = 200.0
    accepted_fs_range_hz: tuple[float, float] = (190.0, 210.0)
    maximum_internal_gap_sec: float = 0.05

    @property
    def window_samples(self) -> int:
        return int(round(self.window_sec * self.nominal_fs_hz))

    @property
    def step_samples(self) -> int:
        return int(round(self.step_sec * self.nominal_fs_hz))


class SpeedEstimator:
    """Sliding-window speed estimator for canonical six-axis IMU samples.

    The estimator does not resample. If timestamps are supplied via
    ``timestamp_sec`` or ``t_sec``, they must be monotonic and compatible with
    the nominal 200 Hz sampling assumption. A large timestamp gap resets the
    buffer because the 5-second research window is no longer continuous.
    """

    def __init__(
        self,
        *,
        model: Any | None = None,
        config: StreamingConfig | None = None,
    ) -> None:
        self.config = config or StreamingConfig()
        self.model = load_model() if model is None else model
        self._samples: deque[dict[str, float]] = deque(maxlen=self.config.window_samples)
        self._timestamps: deque[float | None] = deque(maxlen=self.config.window_samples)
        self._last_timestamp: float | None = None
        self._samples_since_prediction = 0
        self._has_predicted = False
        self._status = "warming_up"

    def reset(self) -> None:
        """Clear buffered samples and return to warm-up state."""

        self._samples.clear()
        self._timestamps.clear()
        self._last_timestamp = None
        self._samples_since_prediction = 0
        self._has_predicted = False
        self._status = "warming_up"

    def status(self) -> dict[str, Any]:
        """Return current streaming buffer status."""

        return {
            "state": self._status,
            "buffered_samples": len(self._samples),
            "window_samples": self.config.window_samples,
            "samples_until_first_prediction": max(
                self.config.window_samples - len(self._samples),
                0,
            ),
            "samples_since_prediction": self._samples_since_prediction,
        }

    @property
    def ready(self) -> bool:
        """Whether calling ``predict_mph`` can emit a new prediction now."""

        if len(self._samples) < self.config.window_samples:
            return False
        if self._has_predicted and self._samples_since_prediction < self.config.step_samples:
            return False
        return self._timestamps_are_continuous()

    def add_sample(self, sample: Mapping[str, Any]) -> bool:
        """Validate and append one canonical sample.

        Returns true when the sample was accepted. Malformed, non-finite,
        timestamp-reversed, or large-gap samples are rejected. Reversal and
        large-gap cases reset the buffered window.
        """

        try:
            parsed, timestamp = self._parse_sample(sample)
        except (TypeError, ValueError):
            self._status = "invalid_sample"
            return False

        if timestamp is not None and self._last_timestamp is not None:
            dt = timestamp - self._last_timestamp
            if dt <= 0:
                self.reset()
                self._status = "timestamp_reversal_reset"
                return False
            if dt > self.config.maximum_internal_gap_sec:
                self.reset()
                self._status = "large_gap_reset"
                return False

        self._samples.append(parsed)
        self._timestamps.append(timestamp)
        self._last_timestamp = timestamp
        if self._has_predicted:
            self._samples_since_prediction += 1
        self._status = "ready" if self.ready else "warming_up"
        return True

    def predict_mph(self) -> float:
        """Predict speed in mph if a complete valid window is available."""

        if not self.ready:
            raise InferenceError("no complete valid window is ready for prediction.")
        window = list(self._samples)
        try:
            prediction = predict_window_mph(
                window,
                model=self.model,
                fs=self.config.nominal_fs_hz,
            )
        except FeatureExtractionError as error:
            self._status = "feature_extraction_error"
            raise InferenceError(str(error)) from error
        self._has_predicted = True
        self._samples_since_prediction = 0
        self._status = "prediction_emitted"
        return prediction

    def _parse_sample(
        self,
        sample: Mapping[str, Any],
    ) -> tuple[dict[str, float], float | None]:
        if not isinstance(sample, Mapping):
            raise TypeError("sample must be a mapping.")
        missing = [column for column in CANONICAL_INPUT_COLUMNS if column not in sample]
        if missing:
            raise ValueError("sample is missing required columns: " + ", ".join(missing))

        parsed: dict[str, float] = {}
        for column in CANONICAL_INPUT_COLUMNS:
            value = float(sample[column])
            if not math.isfinite(value):
                raise ValueError(f"sample value for {column!r} is not finite.")
            parsed[column] = value

        timestamp_value = sample.get("timestamp_sec", sample.get("t_sec"))
        timestamp = None if timestamp_value is None else float(timestamp_value)
        if timestamp is not None and not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite when supplied.")
        return parsed, timestamp

    def _timestamps_are_continuous(self) -> bool:
        timestamps = [timestamp for timestamp in self._timestamps if timestamp is not None]
        if not timestamps:
            return True
        if len(timestamps) != len(self._timestamps):
            self._status = "mixed_timestamp_state"
            return False
        elapsed = timestamps[-1] - timestamps[0]
        if elapsed <= 0:
            self._status = "bad_timestamp_span"
            return False
        effective_fs = (len(timestamps) - 1) / elapsed
        low, high = self.config.accepted_fs_range_hz
        if not (low <= effective_fs <= high):
            self._status = "bad_effective_sample_rate"
            return False
        max_gap = max(np.diff(np.asarray(timestamps, dtype=float)))
        if max_gap > self.config.maximum_internal_gap_sec:
            self._status = "large_internal_gap"
            return False
        return True
