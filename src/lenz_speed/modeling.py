"""Model and feature-set definitions for LENZ speed estimation."""

from __future__ import annotations

from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge


DEFAULT_FEATURES = (
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
    "Impact_Peak_Count",
    "Mean_Impact_Interval_s",
    "Impact_Interval_CV",
    "Mean_Impact_Prominence",
    "Mean_Impact_Width_s",
    "Impact_Duty_Proxy",
    "Vertical_Peak_Sharpness",
)

REDUCED_FEATURE_SETS = {
    "A_Cadence": ("Cadence_spm",),
    "B_AccelMag": ("Accel_Mag_RMS",),
    "C_GyroY": ("Gyro_RMS_Y",),
    "D_Cadence_AccelMag": ("Cadence_spm", "Accel_Mag_RMS"),
    "E_Cadence_GyroY": ("Cadence_spm", "Gyro_RMS_Y"),
    "F_Cadence_GyroY_AccelMag": (
        "Cadence_spm",
        "Gyro_RMS_Y",
        "Accel_Mag_RMS",
    ),
}


def get_models() -> dict[str, RegressorMixin]:
    """Return fresh regression models used by the baseline experiments.

    The models are returned in a deterministic display order. Training and
    validation partitioning is deliberately handled elsewhere; this function
    performs no random splitting.
    """

    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(max_iter=10_000),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=42,
        ),
    }
