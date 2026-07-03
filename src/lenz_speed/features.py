"""Feature extraction for individual LENZ IMU signal windows."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .windowing import SignalWindow


REQUIRED_SIGNAL_COLUMNS = (
    "ax_g",
    "ay_g",
    "az_g",
    "gx_dps",
    "gy_dps",
    "gz_dps",
)

_BANDPASS_LOW_HZ = 0.7
_BANDPASS_HIGH_HZ = 5.0
_CADENCE_LOW_HZ = 0.8
_CADENCE_HIGH_HZ = 4.0
_FILTER_ORDER = 4


class FeatureExtractionError(ValueError):
    """Raised when a signal window cannot produce valid features."""


def _validate_sampling_rate(fs: Real) -> float:
    if isinstance(fs, bool) or not isinstance(fs, Real):
        raise TypeError("fs must be a positive number.")

    fs_float = float(fs)
    if not math.isfinite(fs_float) or fs_float <= 0:
        raise ValueError("fs must be finite and greater than zero.")
    if fs_float <= 2 * _BANDPASS_HIGH_HZ:
        raise ValueError(
            f"fs must be greater than {_BANDPASS_HIGH_HZ * 2:g} Hz so the "
            f"{_BANDPASS_HIGH_HZ:g} Hz filter cutoff is below Nyquist."
        )
    return fs_float


def _validated_signals(window: SignalWindow) -> dict[str, np.ndarray]:
    if not isinstance(window, SignalWindow):
        raise TypeError(
            f"window must be a SignalWindow, got {type(window).__name__}."
        )

    signal = window.signal
    if not isinstance(signal, pd.DataFrame):
        raise TypeError("window.signal must be a pandas DataFrame.")
    if signal.empty:
        raise FeatureExtractionError("window.signal must contain at least one sample.")

    missing = [column for column in REQUIRED_SIGNAL_COLUMNS if column not in signal.columns]
    if missing:
        raise FeatureExtractionError(
            "window.signal is missing required columns: " + ", ".join(missing)
        )

    arrays: dict[str, np.ndarray] = {}
    for column in REQUIRED_SIGNAL_COLUMNS:
        try:
            values = pd.to_numeric(signal[column], errors="raise").to_numpy(
                dtype=float,
                copy=False,
            )
        except (TypeError, ValueError) as error:
            raise FeatureExtractionError(
                f"window.signal contains non-numeric values in {column!r}."
            ) from error
        if not np.isfinite(values).all():
            raise FeatureExtractionError(
                f"window.signal contains missing or non-finite values in {column!r}."
            )
        arrays[column] = values
    return arrays


def _filter_z_acceleration(az_g: np.ndarray, fs: float) -> np.ndarray:
    """Mean-center and bandpass-filter vertical acceleration."""

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
            "window.signal is too short for the fourth-order zero-phase "
            "Butterworth bandpass filter."
        ) from error


def _estimate_cadence_spm(filtered_z: np.ndarray, fs: float) -> float:
    """Estimate cadence from the dominant filtered-Z spectral frequency."""

    frequencies = np.fft.rfftfreq(filtered_z.size, d=1.0 / fs)
    magnitudes = np.abs(np.fft.rfft(filtered_z))
    cadence_band = (frequencies >= _CADENCE_LOW_HZ) & (
        frequencies <= _CADENCE_HIGH_HZ
    )
    if not cadence_band.any():
        raise FeatureExtractionError(
            "window.signal is too short to contain an FFT bin between "
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


def _rms(values: np.ndarray) -> float:
    """Return the root mean square of a finite numeric array."""

    return float(np.sqrt(np.mean(np.square(values))))


def extract_window_features(
    window: SignalWindow,
    fs: Real = 200,
) -> dict[str, str | int | float]:
    """Compute the initial speed-estimation features for one signal window.

    Cadence, ``RMS_Z``, and ``PeakToPeak_Z`` use mean-centered Z acceleration
    after a fourth-order 0.7--5.0 Hz Butterworth bandpass filter. Cadence is the
    dominant real-FFT frequency from 0.8--4.0 Hz multiplied by 60. Gyroscope
    RMS features use the unfiltered angular-rate channels, and
    ``Accel_Mag_RMS`` uses the raw three-axis acceleration magnitude.

    Parameters
    ----------
    window:
        One :class:`~lenz_speed.windowing.SignalWindow`.
    fs:
        Sampling rate in hertz. The default is 200 Hz.

    Returns
    -------
    dict
        Window provenance followed by the seven requested feature values.
    """

    fs_float = _validate_sampling_rate(fs)
    signals = _validated_signals(window)
    filtered_z = _filter_z_acceleration(signals["az_g"], fs_float)
    acceleration_magnitude = np.sqrt(
        np.square(signals["ax_g"])
        + np.square(signals["ay_g"])
        + np.square(signals["az_g"])
    )

    return {
        "recording_id": window.recording_id,
        "window_index": window.window_index,
        "window_start_sec": window.window_start_sec,
        "window_end_sec": window.window_end_sec,
        "Cadence_spm": _estimate_cadence_spm(filtered_z, fs_float),
        "RMS_Z": _rms(filtered_z),
        "PeakToPeak_Z": float(np.ptp(filtered_z)),
        "Gyro_RMS_X": _rms(signals["gx_dps"]),
        "Gyro_RMS_Y": _rms(signals["gy_dps"]),
        "Gyro_RMS_Z": _rms(signals["gz_dps"]),
        "Accel_Mag_RMS": _rms(acceleration_magnitude),
    }

