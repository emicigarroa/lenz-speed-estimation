"""Tests for window-level IMU feature extraction."""

import numpy as np
import pandas as pd
import pytest

from lenz_speed.features import extract_window_features
from lenz_speed.windowing import SignalWindow


def test_synthetic_two_hz_signal_produces_expected_features() -> None:
    fs = 200
    sample_count = 1_000
    time_sec = np.arange(sample_count) / fs
    signal = pd.DataFrame(
        {
            "ax_g": np.full(sample_count, 0.1),
            "ay_g": np.full(sample_count, -0.2),
            "az_g": np.sin(2 * np.pi * 2 * time_sec),
            "gx_dps": np.full(sample_count, 1.0),
            "gy_dps": np.full(sample_count, 2.0),
            "gz_dps": np.full(sample_count, 3.0),
        }
    )
    window = SignalWindow(
        recording_id="synthetic",
        window_index=0,
        window_start_sec=0.0,
        window_end_sec=5.0,
        signal=signal,
    )

    features = extract_window_features(window, fs=fs)

    assert features["Cadence_spm"] == pytest.approx(120.0)
    feature_names = (
        "RMS_Z",
        "PeakToPeak_Z",
        "Gyro_RMS_X",
        "Gyro_RMS_Y",
        "Gyro_RMS_Z",
        "Accel_Mag_RMS",
    )
    assert all(np.isfinite(features[name]) for name in feature_names)

