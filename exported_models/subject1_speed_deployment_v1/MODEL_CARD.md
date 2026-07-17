# Subject 1 speed deployment v1 model card

## Artifact identity

- Model version: `subject1_speed_deployment_v1`
- Model identity: `Subject 1 calibrated deployment model`
- Model role: `deployment`
- Estimator: `RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)`
- Feature set: exact 19-feature `v4_morphology_all` order from the frozen benchmark package.
- Output unit: mph.

## Benchmark vs deployment distinction

Benchmark artifact:

- Version: `subject1_speed_benchmark_v1`
- Commit: `765ac43`
- Trained on approved Subject 1 Day 3 only.
- Independently validated on Subject 1 Day 2 plus Subject 1 Day 4 normal-cadence recordings.
- Historical validation MAE: `0.1634656501460112` mph.

Deployment artifact:

- Retrained using the expanded approved Subject 1 natural-cadence dataset.
- Includes approved Subject 1 Day 3 steady-state, Subject 1 Day 2 steady-state, and Subject 1 Day 4 `cadence_normal` recordings.
- Intended for personalized Subject 1 live inference in the UI.
- Does **not** carry the benchmark MAE as independent deployment performance, because former validation recordings are now part of deployment training.
- The benchmark selection was frozen before this deployment retraining.

## Deployment training set

- Training recordings: 21
- Training windows: 407

- `s1_day2_test9_2mph`: day2, 2 mph, `steady_state`, 17 windows
- `s1_day2_test8_3mph`: day2, 3 mph, `steady_state`, 15 windows
- `s1_day2_test7_4mph`: day2, 4 mph, `steady_state`, 16 windows
- `s1_day2_test5_8mph`: day2, 8 mph, `steady_state`, 15 windows
- `s1_day3_2p0mph`: day3, 2 mph, `steady_state`, 15 windows
- `s1_day3_2p5mph`: day3, 2.5 mph, `steady_state`, 25 windows
- `s1_day3_3p0mph`: day3, 3 mph, `steady_state`, 21 windows
- `s1_day3_3p5mph`: day3, 3.5 mph, `steady_state`, 20 windows
- `s1_day3_4p0mph`: day3, 4 mph, `steady_state`, 18 windows
- `s1_day3_4p5mph`: day3, 4.5 mph, `steady_state`, 21 windows
- `s1_day3_5p0mph`: day3, 5 mph, `steady_state`, 23 windows
- `s1_day3_5p5mph`: day3, 5.5 mph, `steady_state`, 21 windows
- `s1_day3_6p0mph`: day3, 6 mph, `steady_state`, 20 windows
- `s1_day3_6p5mph`: day3, 6.5 mph, `steady_state`, 20 windows
- `s1_day3_7p0mph`: day3, 7 mph, `steady_state`, 22 windows
- `s1_day3_7p5mph`: day3, 7.5 mph, `steady_state`, 28 windows
- `s1_day3_8p0mph`: day3, 8 mph, `steady_state`, 22 windows
- `s1_day3_8p5mph`: day3, 8.5 mph, `steady_state`, 21 windows
- `s1_day4_20260619_5mph_140spm_normal`: day4, 5 mph, `cadence_normal`, 14 windows
- `s1_day4_20260619_6mph_145spm_normal`: day4, 6 mph, `cadence_normal`, 15 windows
- `s1_day4_20260619_7mph_153spm_normal`: day4, 7 mph, `cadence_normal`, 18 windows

## Runtime input contract

Required columns per sample:

- `ax_g`, `ay_g`, `az_g` in g
- `gx_dps`, `gy_dps`, `gz_dps` in degrees per second

Device frame while worn:

- X axis: left/right across the face
- Y axis: forward/backward, direction of travel
- Z axis: up/down vertical
- Gyro X: pitch/nodding rate
- Gyro Y: roll/side-tilt rate
- Gyro Z: yaw/turning rate

Runtime behavior:

- 5.0-second window at nominal 200 Hz.
- 2.5-second intended prediction interval.
- Streaming mode requires a 5-second warm-up before the first prediction.
- Timestamps, when supplied, must be monotonic and consistent with the accepted 190--210 Hz effective sampling range.
- Large timestamp gaps reset the streaming buffer.
- No statistically supported confidence score is emitted.

## Golden fixtures

- `low_speed`: `s1_day2_test9_2mph` window 0 (2 mph), prediction 2.000000 mph
- `mid_speed`: `s1_day4_20260619_6mph_145spm_normal` window 0 (6 mph), prediction 6.093244 mph
- `high_speed`: `s1_day3_8p5mph` window 0 (8.5 mph), prediction 8.261137 mph

## Known limitations

- Personalized to Subject 1.
- Not a universal cross-subject speed estimator.
- Not a medical, safety-critical, or uncertainty-calibrated model.
- Deployment behavior should still be validated in live UI conditions.
