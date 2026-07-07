"""Feature extraction for individual LENZ IMU signal windows."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, peak_widths, sosfiltfilt

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
_SPECTRAL_LOW_HZ = 0.7
_HIGH_FREQUENCY_LOW_HZ = 5.0
_SPECTRAL_HIGH_HZ = 15.0
_FILTER_ORDER = 4
_MIN_IMPACT_INTERVAL_SEC = 0.2


class FeatureExtractionError(ValueError):
    """Raised when a signal window cannot produce valid features."""


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


def _filter_gyro_y(gy_dps: np.ndarray, fs: float) -> np.ndarray:
    """Mean-center and bandpass-filter roll angular velocity."""

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
            "window.signal is too short to filter Gyro Y with the fourth-order "
            "zero-phase Butterworth bandpass filter."
        ) from error


def _high_frequency_energy_ratio(
    acceleration_magnitude: np.ndarray,
    fs: float,
) -> float:
    """Return 5--15 Hz power divided by 0.7--15 Hz magnitude power."""

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
    """Return the dominant share of the demeaned acceleration covariance."""

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
    """Return gait-event summaries from filtered vertical acceleration peaks.

    Peaks are detected on the 0.7--5.0 Hz filtered Z acceleration signal,
    because Z is vertical in the worn glasses frame. The peak detector uses a
    conservative prominence threshold based on within-window vertical motion and
    a 0.2 second minimum spacing to avoid double-counting nearby samples from
    the same impact-like event.

    Smooth or low-amplitude windows can legitimately have no detectable peaks.
    Those windows receive finite zero-valued temporal summaries rather than
    failing feature extraction.
    """

    amplitude_range = float(np.ptp(filtered_z))
    vertical_std = float(np.std(filtered_z))
    prominence_threshold = max(0.05 * amplitude_range, 0.25 * vertical_std)
    if not math.isfinite(prominence_threshold) or prominence_threshold <= 0:
        return {
            "Impact_Peak_Count": 0.0,
            "Mean_Impact_Interval_s": 0.0,
            "Impact_Interval_CV": 0.0,
            "Mean_Impact_Prominence": 0.0,
            "Mean_Impact_Width_s": 0.0,
            "Impact_Duty_Proxy": 0.0,
            "Vertical_Peak_Sharpness": 0.0,
        }

    peaks, properties = find_peaks(
        filtered_z,
        distance=max(1, int(round(_MIN_IMPACT_INTERVAL_SEC * fs))),
        prominence=prominence_threshold,
    )
    peak_count = int(peaks.size)
    if peak_count == 0:
        return {
            "Impact_Peak_Count": 0.0,
            "Mean_Impact_Interval_s": 0.0,
            "Impact_Interval_CV": 0.0,
            "Mean_Impact_Prominence": 0.0,
            "Mean_Impact_Width_s": 0.0,
            "Impact_Duty_Proxy": 0.0,
            "Vertical_Peak_Sharpness": 0.0,
        }

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
    intervals_sec = np.diff(peaks) / fs

    mean_interval = float(np.mean(intervals_sec)) if intervals_sec.size else 0.0
    interval_cv = (
        float(np.std(intervals_sec) / mean_interval)
        if intervals_sec.size and mean_interval > 0
        else 0.0
    )
    mean_width = float(np.mean(widths_sec)) if widths_sec.size else 0.0
    window_duration_sec = filtered_z.size / fs
    duty_proxy = (
        float(np.sum(widths_sec) / window_duration_sec)
        if window_duration_sec > 0
        else 0.0
    )
    mean_prominence = float(np.mean(prominences))
    sharpness_values = np.divide(
        prominences,
        widths_sec,
        out=np.zeros_like(prominences, dtype=float),
        where=widths_sec > 0,
    )

    return {
        "Impact_Peak_Count": float(peak_count),
        "Mean_Impact_Interval_s": mean_interval,
        "Impact_Interval_CV": interval_cv,
        "Mean_Impact_Prominence": mean_prominence,
        "Mean_Impact_Width_s": mean_width,
        "Impact_Duty_Proxy": min(max(duty_proxy, 0.0), 1.0),
        "Vertical_Peak_Sharpness": float(np.mean(sharpness_values)),
    }


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
    ``Accel_Mag_RMS`` uses the raw three-axis acceleration magnitude. Feature
    Engineering v2 adds conservative magnitude, jerk, spectral, roll-amplitude,
    and covariance summaries without changing the original seven formulas.
    Feature Engineering v3 adds gait-event and temporal summaries from the
    same filtered vertical acceleration signal.

    Parameters
    ----------
    window:
        One :class:`~lenz_speed.windowing.SignalWindow`.
    fs:
        Sampling rate in hertz. The default is 200 Hz.

    Returns
    -------
    dict
        Window provenance followed by the original seven, seven v2, and seven
        v3 feature values.
    """

    fs_float = _validate_sampling_rate(fs)
    signals = _validated_signals(window)
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
    }
