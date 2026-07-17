"""Standalone feature extraction for the Subject 1 speed benchmark model.

This module is adapted directly from the validated research feature formulas
for the frozen ``subject1_speed_benchmark_v1`` artifact. It intentionally
contains only the runtime math needed for one-window feature extraction.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, find_peaks, peak_widths, sosfiltfilt


CANONICAL_INPUT_COLUMNS: tuple[str, ...] = (
    "ax_g",
    "ay_g",
    "az_g",
    "gx_dps",
    "gy_dps",
    "gz_dps",
)

FEATURE_NAMES: tuple[str, ...] = (
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
    "Vertical_Peak_Sharpness",
    "Impact_Impulse",
    "Peak_Symmetry",
    "Impact_Crest_Factor",
    "Impact_Local_Kurtosis",
)

_BANDPASS_LOW_HZ = 0.7
_BANDPASS_HIGH_HZ = 5.0
_CADENCE_LOW_HZ = 0.8
_CADENCE_HIGH_HZ = 4.0
_SPECTRAL_LOW_HZ = 0.7
_HIGH_FREQUENCY_LOW_HZ = 5.0
_SPECTRAL_HIGH_HZ = 15.0
_FILTER_ORDER = 4
_MIN_IMPACT_INTERVAL_SEC = 0.2
_LOCAL_PEAK_WINDOW_SEC = 0.30


class FeatureExtractionError(ValueError):
    """Raised when a window cannot produce the benchmark feature vector."""


def _validate_sampling_rate(fs: Real) -> float:
    if isinstance(fs, bool) or not isinstance(fs, Real):
        raise TypeError("fs must be a positive number.")

    fs_float = float(fs)
    if not math.isfinite(fs_float) or fs_float <= 0:
        raise ValueError("fs must be finite and greater than zero.")
    if fs_float <= 2 * _SPECTRAL_HIGH_HZ:
        raise ValueError(
            f"fs must be greater than {_SPECTRAL_HIGH_HZ * 2:g} Hz so the "
            f"{_SPECTRAL_HIGH_HZ:g} Hz spectral limit is below Nyquist."
        )
    return fs_float


def _coerce_rows(window: Any) -> dict[str, np.ndarray]:
    """Return canonical arrays from a mapping, row sequence, or 2-D array."""

    if isinstance(window, Mapping):
        signals: dict[str, np.ndarray] = {}
        missing = [column for column in CANONICAL_INPUT_COLUMNS if column not in window]
        if missing:
            raise FeatureExtractionError(
                "window is missing required columns: " + ", ".join(missing)
            )
        for column in CANONICAL_INPUT_COLUMNS:
            signals[column] = np.asarray(window[column], dtype=float)
        return _validate_signals(signals)

    if isinstance(window, np.ndarray):
        array = np.asarray(window, dtype=float)
        if array.ndim != 2 or array.shape[1] != len(CANONICAL_INPUT_COLUMNS):
            raise FeatureExtractionError(
                "array windows must have shape (n_samples, 6) in canonical "
                "column order."
            )
        return _validate_signals(
            {
                column: array[:, index]
                for index, column in enumerate(CANONICAL_INPUT_COLUMNS)
            }
        )

    if isinstance(window, Sequence) and not isinstance(window, (str, bytes)):
        rows = list(window)
        if not rows:
            raise FeatureExtractionError("window must contain at least one sample.")
        signals = {column: [] for column in CANONICAL_INPUT_COLUMNS}
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise FeatureExtractionError(
                    f"window row {row_index} must be a mapping of canonical inputs."
                )
            missing = [column for column in CANONICAL_INPUT_COLUMNS if column not in row]
            if missing:
                raise FeatureExtractionError(
                    f"window row {row_index} is missing required columns: "
                    + ", ".join(missing)
                )
            for column in CANONICAL_INPUT_COLUMNS:
                signals[column].append(row[column])
        return _validate_signals(
            {column: np.asarray(values, dtype=float) for column, values in signals.items()}
        )

    raise TypeError(
        "window must be a mapping of column arrays, a sequence of sample mappings, "
        "or a NumPy array in canonical column order."
    )


def _validate_signals(signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    lengths = {column: values.size for column, values in signals.items()}
    if not lengths or any(length <= 0 for length in lengths.values()):
        raise FeatureExtractionError("window must contain at least one sample.")
    if len(set(lengths.values())) != 1:
        raise FeatureExtractionError(f"canonical columns have unequal lengths: {lengths}")

    arrays: dict[str, np.ndarray] = {}
    for column in CANONICAL_INPUT_COLUMNS:
        values = np.asarray(signals[column], dtype=float)
        if values.ndim != 1:
            raise FeatureExtractionError(f"{column!r} must be one-dimensional.")
        if not np.isfinite(values).all():
            raise FeatureExtractionError(
                f"window contains missing or non-finite values in {column!r}."
            )
        arrays[column] = values
    return arrays


def load_window_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Load a canonical six-axis window CSV into column arrays."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise FeatureExtractionError(f"window CSV has no header: {path}")
        rows = list(reader)
    return _coerce_rows(rows)


def _filter_z_acceleration(az_g: np.ndarray, fs: float) -> np.ndarray:
    centered = az_g - np.mean(az_g)
    sos = butter(
        _FILTER_ORDER,
        [_BANDPASS_LOW_HZ, _BANDPASS_HIGH_HZ],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    try:
        return sosfiltfilt(sos, centered)
    except ValueError as error:
        raise FeatureExtractionError(
            "window is too short for the fourth-order zero-phase Butterworth "
            "bandpass filter."
        ) from error


def _estimate_cadence_spm(filtered_z: np.ndarray, fs: float) -> float:
    frequencies = np.fft.rfftfreq(filtered_z.size, d=1.0 / fs)
    magnitudes = np.abs(np.fft.rfft(filtered_z))
    cadence_band = (frequencies >= _CADENCE_LOW_HZ) & (
        frequencies <= _CADENCE_HIGH_HZ
    )
    if not cadence_band.any():
        raise FeatureExtractionError(
            "window is too short to contain an FFT bin between "
            f"{_CADENCE_LOW_HZ:g} and {_CADENCE_HIGH_HZ:g} Hz."
        )

    band_magnitudes = magnitudes[cadence_band]
    if not np.isfinite(band_magnitudes).all() or np.max(band_magnitudes) <= 0:
        raise FeatureExtractionError(
            "Cadence cannot be estimated because filtered Z acceleration has "
            "no spectral energy in the cadence band."
        )

    dominant_index = int(np.argmax(band_magnitudes))
    dominant_frequency = frequencies[cadence_band][dominant_index]
    return float(dominant_frequency * 60.0)


def _filter_gyro_y(gy_dps: np.ndarray, fs: float) -> np.ndarray:
    centered = gy_dps - np.mean(gy_dps)
    sos = butter(
        _FILTER_ORDER,
        [_BANDPASS_LOW_HZ, _BANDPASS_HIGH_HZ],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    try:
        return sosfiltfilt(sos, centered)
    except ValueError as error:
        raise FeatureExtractionError(
            "window is too short to filter Gyro Y with the fourth-order "
            "zero-phase Butterworth bandpass filter."
        ) from error


def _high_frequency_energy_ratio(
    acceleration_magnitude: np.ndarray,
    fs: float,
) -> float:
    demeaned = acceleration_magnitude - np.mean(acceleration_magnitude)
    frequencies = np.fft.rfftfreq(demeaned.size, d=1.0 / fs)
    power = np.square(np.abs(np.fft.rfft(demeaned)))
    total_band = (frequencies >= _SPECTRAL_LOW_HZ) & (
        frequencies <= _SPECTRAL_HIGH_HZ
    )
    high_band = (frequencies >= _HIGH_FREQUENCY_LOW_HZ) & (
        frequencies <= _SPECTRAL_HIGH_HZ
    )
    total_power = float(np.sum(power[total_band]))
    if not math.isfinite(total_power) or total_power <= 0:
        raise FeatureExtractionError(
            "Acceleration magnitude has no spectral energy between "
            f"{_SPECTRAL_LOW_HZ:g} and {_SPECTRAL_HIGH_HZ:g} Hz."
        )
    return float(np.sum(power[high_band]) / total_power)


def _acceleration_anisotropy(signals: dict[str, np.ndarray]) -> float:
    acceleration = np.column_stack(
        (signals["ax_g"], signals["ay_g"], signals["az_g"])
    )
    demeaned = acceleration - np.mean(acceleration, axis=0, keepdims=True)
    covariance = demeaned.T @ demeaned / (len(demeaned) - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total_variance = float(np.sum(eigenvalues))
    if not math.isfinite(total_variance) or total_variance <= 0:
        raise FeatureExtractionError(
            "Acceleration anisotropy is undefined because the window has no "
            "acceleration variance."
        )
    return float(np.max(eigenvalues) / total_variance)


def _impact_peak_features(filtered_z: np.ndarray, fs: float) -> dict[str, float]:
    amplitude_range = float(np.ptp(filtered_z))
    vertical_std = float(np.std(filtered_z))
    prominence_threshold = max(0.05 * amplitude_range, 0.25 * vertical_std)
    if not math.isfinite(prominence_threshold) or prominence_threshold <= 0:
        return {"Vertical_Peak_Sharpness": 0.0}

    peaks, properties = find_peaks(
        filtered_z,
        distance=max(1, int(round(_MIN_IMPACT_INTERVAL_SEC * fs))),
        prominence=prominence_threshold,
    )
    if peaks.size == 0:
        return {"Vertical_Peak_Sharpness": 0.0}

    prominences = properties["prominences"].astype(float, copy=False)
    widths_samples = peak_widths(
        filtered_z,
        peaks,
        rel_height=0.5,
        prominence_data=(
            properties["prominences"],
            properties["left_bases"],
            properties["right_bases"],
        ),
    )[0]
    widths_sec = widths_samples / fs
    sharpness_values = np.divide(
        prominences,
        widths_sec,
        out=np.zeros_like(prominences, dtype=float),
        where=widths_sec > 0,
    )
    return {"Vertical_Peak_Sharpness": float(np.mean(sharpness_values))}


def _zero_morphology_features() -> dict[str, float]:
    return {
        "Impact_Impulse": 0.0,
        "Peak_Symmetry": 0.0,
        "Impact_Crest_Factor": 0.0,
        "Impact_Local_Kurtosis": 0.0,
    }


def _impact_morphology_features(filtered_z: np.ndarray, fs: float) -> dict[str, float]:
    amplitude_range = float(np.ptp(filtered_z))
    vertical_std = float(np.std(filtered_z))
    prominence_threshold = max(0.05 * amplitude_range, 0.25 * vertical_std)
    filtered_rms = _rms(filtered_z)
    if (
        not math.isfinite(prominence_threshold)
        or prominence_threshold <= 0
        or filtered_rms <= 0
    ):
        return _zero_morphology_features()

    peaks, properties = find_peaks(
        filtered_z,
        distance=max(1, int(round(_MIN_IMPACT_INTERVAL_SEC * fs))),
        prominence=prominence_threshold,
    )
    if peaks.size == 0:
        return _zero_morphology_features()

    widths = peak_widths(
        filtered_z,
        peaks,
        rel_height=0.5,
        prominence_data=(
            properties["prominences"],
            properties["left_bases"],
            properties["right_bases"],
        ),
    )
    half_height = widths[1]
    left_ips = widths[2]
    right_ips = widths[3]

    impulses: list[float] = []
    symmetries: list[float] = []
    local_kurtoses: list[float] = []
    half_window = max(2, int(round(_LOCAL_PEAK_WINDOW_SEC * fs / 2.0)))

    for peak, left_ip, right_ip, baseline in zip(
        peaks,
        left_ips,
        right_ips,
        half_height,
        strict=True,
    ):
        left_index = max(0, int(np.floor(left_ip)))
        right_index = min(filtered_z.size, int(np.ceil(right_ip)) + 1)
        if right_index > left_index:
            positive_lobe = np.maximum(filtered_z[left_index:right_index] - baseline, 0.0)
            impulses.append(float(np.trapezoid(positive_lobe, dx=1.0 / fs)))

        left_width = float(peak - left_ip)
        right_width = float(right_ip - peak)
        wider_side = max(left_width, right_width)
        if wider_side > 0:
            symmetries.append(float(min(left_width, right_width) / wider_side))

        local_left = max(0, int(peak) - half_window)
        local_right = min(filtered_z.size, int(peak) + half_window + 1)
        local = filtered_z[local_left:local_right]
        local_std = float(np.std(local))
        if local.size >= 4 and local_std > 0:
            standardized = (local - np.mean(local)) / local_std
            local_kurtoses.append(float(np.mean(np.power(standardized, 4)) - 3.0))

    crest_factor = float(np.max(filtered_z[peaks]) / filtered_rms)
    return {
        "Impact_Impulse": float(np.mean(impulses)) if impulses else 0.0,
        "Peak_Symmetry": float(np.mean(symmetries)) if symmetries else 0.0,
        "Impact_Crest_Factor": crest_factor,
        "Impact_Local_Kurtosis": float(np.mean(local_kurtoses))
        if local_kurtoses
        else 0.0,
    }


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def extract_feature_dict(window: Any, fs: Real = 200) -> dict[str, float]:
    """Compute the exact 19-feature benchmark vector for one complete window.

    Parameters
    ----------
    window:
        Either a mapping of canonical column arrays, a sequence of sample
        mappings, or a NumPy array with six columns in
        ``CANONICAL_INPUT_COLUMNS`` order.
    fs:
        Sampling rate in hertz. The frozen benchmark uses 200 Hz.
    """

    fs_float = _validate_sampling_rate(fs)
    signals = _coerce_rows(window)
    filtered_z = _filter_z_acceleration(signals["az_g"], fs_float)
    acceleration_magnitude = np.sqrt(
        np.square(signals["ax_g"])
        + np.square(signals["ay_g"])
        + np.square(signals["az_g"])
    )
    dynamic_acceleration_magnitude = acceleration_magnitude - np.mean(
        acceleration_magnitude
    )
    acceleration_magnitude_jerk = np.diff(acceleration_magnitude) * fs_float
    gyroscope_magnitude = np.sqrt(
        np.square(signals["gx_dps"])
        + np.square(signals["gy_dps"])
        + np.square(signals["gz_dps"])
    )
    filtered_gyro_y = _filter_gyro_y(signals["gy_dps"], fs_float)
    impact_features = _impact_peak_features(filtered_z, fs_float)
    morphology_features = _impact_morphology_features(filtered_z, fs_float)

    features = {
        "Cadence_spm": _estimate_cadence_spm(filtered_z, fs_float),
        "RMS_Z": _rms(filtered_z),
        "PeakToPeak_Z": float(np.ptp(filtered_z)),
        "Gyro_RMS_X": _rms(signals["gx_dps"]),
        "Gyro_RMS_Y": _rms(signals["gy_dps"]),
        "Gyro_RMS_Z": _rms(signals["gz_dps"]),
        "Accel_Mag_RMS": _rms(acceleration_magnitude),
        "Dynamic_Accel_Mag_RMS": _rms(dynamic_acceleration_magnitude),
        "Accel_Mag_P95_P05": float(
            np.percentile(acceleration_magnitude, 95)
            - np.percentile(acceleration_magnitude, 5)
        ),
        "Accel_Mag_Jerk_RMS": _rms(acceleration_magnitude_jerk),
        "Accel_HighFreq_Energy_Ratio": _high_frequency_energy_ratio(
            acceleration_magnitude,
            fs_float,
        ),
        "Gyro_Mag_RMS": _rms(gyroscope_magnitude),
        "GyroY_PeakToPeak": float(np.ptp(filtered_gyro_y)),
        "Accel_Anisotropy": _acceleration_anisotropy(signals),
        **impact_features,
        **morphology_features,
    }
    return {name: float(features[name]) for name in FEATURE_NAMES}


def extract_feature_vector(window: Any, fs: Real = 200) -> np.ndarray:
    """Return a 2-D feature vector in the serialized model's required order."""

    features = extract_feature_dict(window, fs=fs)
    return np.asarray([[features[name] for name in FEATURE_NAMES]], dtype=float)
