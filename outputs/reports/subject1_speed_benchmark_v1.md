# Subject 1 speed benchmark v1 reproduction

## Result

Reproduction status: **PASS**

Historical source: `outputs/tables/v4_morphology_feature_metrics.csv`

| metric | historical | reproduced | absolute_difference |
|---|---:|---:|---:|
| MAE | 0.163465650146011 | 0.163465650146011 | 0.000e+00 |
| RMSE | 0.192218412883266 | 0.192218412883266 | 2.776e-17 |
| R2 | 0.990988130967335 | 0.990988130967335 | 0.000e+00 |

Material tolerance: `1e-10` absolute difference for each metric.

## Benchmark identity

- Experiment: `v4_morphology_all`
- Evaluation: Subject 1 same-subject standard validation
- Model: `Random Forest`
- Estimator class: `RandomForestRegressor`
- Estimator parameters: `{'n_estimators': 200, 'max_depth': 5, 'random_state': 42}`
- Training split: approved Subject 1 Day 3 windows only
- Validation split: approved Subject 1 Day 2 windows plus Subject 1 Day 4 `cadence_normal`
- Windowing: 5.0 s windows, 2.5 s step, incomplete final windows dropped
- Sampling assumption: 200 Hz
- Output unit: mph

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

## Training data

- Training windows: 297
- Training recordings: 14

### Training recordings

| recording_id | speed_mph | windows | relative_path |
|---|---:|---:|---|
| s1_day3_2p0mph | 2 | 15 | `data/raw/subject_1_day3/s1_2.0mph.csv` |
| s1_day3_2p5mph | 2.5 | 25 | `data/raw/subject_1_day3/s1_2.5mph.csv` |
| s1_day3_3p0mph | 3 | 21 | `data/raw/subject_1_day3/s1_3.0mph.csv` |
| s1_day3_3p5mph | 3.5 | 20 | `data/raw/subject_1_day3/s1_3.5mph.csv` |
| s1_day3_4p0mph | 4 | 18 | `data/raw/subject_1_day3/s1_4.0mph.csv` |
| s1_day3_4p5mph | 4.5 | 21 | `data/raw/subject_1_day3/s1_4.5mph.csv` |
| s1_day3_5p0mph | 5 | 23 | `data/raw/subject_1_day3/s1_5.0mph.csv` |
| s1_day3_5p5mph | 5.5 | 21 | `data/raw/subject_1_day3/s1_5.5mph.csv` |
| s1_day3_6p0mph | 6 | 20 | `data/raw/subject_1_day3/s1_6.0mph.csv` |
| s1_day3_6p5mph | 6.5 | 20 | `data/raw/subject_1_day3/s1_6.5mph.csv` |
| s1_day3_7p0mph | 7 | 22 | `data/raw/subject_1_day3/s1_7.0mph.csv` |
| s1_day3_7p5mph | 7.5 | 28 | `data/raw/subject_1_day3/s1_7.5mph.csv` |
| s1_day3_8p0mph | 8 | 22 | `data/raw/subject_1_day3/s1_8.0mph.csv` |
| s1_day3_8p5mph | 8.5 | 21 | `data/raw/subject_1_day3/s1_8.5mph.csv` |

## Validation data

- Validation windows: 110
- Validation recordings: 7

### Validation recordings

| recording_id | speed_mph | windows | relative_path |
|---|---:|---:|---|
| s1_day2_test5_8mph | 8 | 15 | `data/raw/subject_1_day2/Day2Test5IMU_8-8mph.xlsx` |
| s1_day2_test7_4mph | 4 | 16 | `data/raw/subject_1_day2/Day2Test7IMU_0-4mph.xlsx` |
| s1_day2_test8_3mph | 3 | 15 | `data/raw/subject_1_day2/Day2Test8IMU_0-3mph.xlsx` |
| s1_day2_test9_2mph | 2 | 17 | `data/raw/subject_1_day2/Day2Test9IMU_0-2mph.xlsx` |
| s1_day4_20260619_5mph_140spm_normal | 5 | 14 | `data/raw/subject_1_day4/imu_recording_20260619_5mph_140spm_normal.csv` |
| s1_day4_20260619_6mph_145spm_normal | 6 | 15 | `data/raw/subject_1_day4/imu_recording_20260619_6mph_145spm_normal.csv` |
| s1_day4_20260619_7mph_153spm_normal | 7 | 18 | `data/raw/subject_1_day4/imu_recording_20260619_7mph_153spm_normal.csv` |

## Code path traced

- Historical experiment script: `scripts/v4_morphology_experiment.py`
- Split helper mirrored from: `src/lenz_speed/evaluation.py::_same_subject_split`
- Model definition source: `src/lenz_speed/modeling.py::get_models`
- Feature extraction source: `src/lenz_speed/features.py::extract_window_features`
- Windowing source: `src/lenz_speed/windowing.py::make_windows`
- Packet/schema loading source: `src/lenz_speed/data.py`

## Preprocessing and feature-formula summary

- CSV rows are filtered to valid finite IMU packet rows before sample indexing.
- Required signal columns are `ax_g`, `ay_g`, `az_g`, `gx_dps`, `gy_dps`, `gz_dps`.
- Acceleration unit: g.
- Gyroscope unit: degrees per second.
- Axis convention: X lateral, Y forward/backward, Z vertical.
- Cadence, `RMS_Z`, `PeakToPeak_Z`, gait-event, and morphology features use mean-centered Z acceleration filtered by a 4th-order 0.7--5.0 Hz Butterworth bandpass.
- Cadence searches the dominant FFT frequency from 0.8--4.0 Hz and reports steps/min.
- Gyroscope RMS features use raw angular-rate channels.
- Acceleration magnitude features use raw three-axis acceleration magnitude.
- No scaler, encoder, normalizer, imputer, or preprocessing object is used by this benchmark model.

## Environment

- ML git commit: `9f03a50`
- python: `3.13.12`
- numpy: `2.5.0`
- scipy: `1.18.0`
- scikit_learn: `1.9.0`
- joblib: `1.5.3`
