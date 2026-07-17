# Subject 1 speed benchmark v1 model card

## Intended role

This artifact is a personalized Subject 1 benchmark prototype for LENZ
head-mounted IMU speed estimation. It is provided to preserve exact benchmark
parity for the reported Subject 1 experiment. It is not a generalized universal
estimator and is not yet the final deployment artifact.

## Model identity

- Model version: `subject1_speed_benchmark_v1`
- Model identity: `Subject 1 calibrated benchmark model`
- Model role: `benchmark`
- Estimator: `RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)`
- Output unit: mph
- No scaler, imputer, normalizer, or confidence-calibration layer is used.
- No statistically supported confidence score is emitted.

## Benchmark definition

- Training data: approved Subject 1 Day 3 windows only.
- Validation data: exact Subject 1 same-subject standard validation set:
  approved Subject 1 Day 2 windows plus Subject 1 Day 4 `cadence_normal`.
- Windowing: 5.0 second windows, 2.5 second step, incomplete final windows
  dropped.
- Nominal sampling rate: 200 Hz.

Reported historical validation metrics:

- MAE: `0.163465650146011` mph
- RMSE: `0.192218412883266` mph
- R²: `0.990988130967335`

## Required input schema

Each complete window must contain canonical six-axis samples:

- `ax_g`, `ay_g`, `az_g` in g
- `gx_dps`, `gy_dps`, `gz_dps` in degrees per second

Device frame while worn:

- X axis: left/right across the face
- Y axis: forward/backward, direction of travel
- Z axis: up/down vertical
- Gyro X: pitch/nodding rate
- Gyro Y: roll/side-tilt rate
- Gyro Z: yaw/turning rate

## Runtime behavior

- Batch inference accepts one complete canonical 5-second window.
- Streaming inference needs a 5-second warm-up before the first prediction.
- Intended prediction interval after warm-up is 2.5 seconds.
- Timestamps, when supplied, must be monotonic and roughly compatible with the
  accepted 190--210 Hz effective sampling range.
- The runtime rejects non-finite samples, timestamp reversals, and large
  internal timestamp gaps instead of resampling.

## Feature order

1. `Cadence_spm`
2. `RMS_Z`
3. `PeakToPeak_Z`
4. `Gyro_RMS_X`
5. `Gyro_RMS_Y`
6. `Gyro_RMS_Z`
7. `Accel_Mag_RMS`
8. `Dynamic_Accel_Mag_RMS`
9. `Accel_Mag_P95_P05`
10. `Accel_Mag_Jerk_RMS`
11. `Accel_HighFreq_Energy_Ratio`
12. `Gyro_Mag_RMS`
13. `GyroY_PeakToPeak`
14. `Accel_Anisotropy`
15. `Vertical_Peak_Sharpness`
16. `Impact_Impulse`
17. `Peak_Symmetry`
18. `Impact_Crest_Factor`
19. `Impact_Local_Kurtosis`

## Golden fixtures

- `low_speed`: `s1_day2_test9_2mph` window 0 (2 mph), prediction 2.112500 mph
- `mid_speed`: `s1_day4_20260619_6mph_145spm_normal` window 0 (6 mph), prediction 6.291419 mph
- `high_speed`: `s1_day2_test5_8mph` window 0 (8 mph), prediction 7.753826 mph

## Known limitations

- Calibrated on one subject's natural Day 3 training data.
- Validation parity is specific to the historical Subject 1 benchmark split.
- Cross-subject generalization is not guaranteed by this artifact.
- Cadence-manipulation and deployment behavior require separate validation.
- This artifact provides deterministic benchmark reproduction, not medical,
  safety-critical, or production-grade uncertainty estimates.
