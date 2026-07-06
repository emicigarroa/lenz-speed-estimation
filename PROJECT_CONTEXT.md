# Team LENZ Speed Estimation Project Context

## Project
Team LENZ capstone project: Smart Glasses as a Platform for Real-Time Run Coaching.

Goal: estimate treadmill running speed from a head-mounted IMU mounted on smart glasses.

## Sensor and data
- IMU sampling rate: 200 Hz.
- Window size: 5 seconds.
- Step size: 2.5 seconds.
- Device coordinate frame while worn:
  - X axis: left/right across the face.
  - Y axis: forward/backward, direction of travel.
  - Z axis: up/down vertical.
- Accelerometer axes measure linear acceleration along each axis.
- Gyroscope axes measure angular velocity, or rotation rate, about each axis.
- Axis interpretation:
  - Accel_Z is vertical acceleration.
  - Gyro_X is pitch/nodding rate, rotation about the left/right axis.
  - Gyro_Y is roll/side-tilt rate, rotation about the forward axis.
  - Gyro_Z is yaw/turning rate, rotation about the vertical axis.
- Existing feature columns remain axis-based for now; do not rename them until
  a deliberate schema migration is approved.
- Main features:
  - Cadence_spm
  - RMS_Z
  - PeakToPeak_Z
  - Gyro_RMS_X
  - Gyro_RMS_Y
  - Gyro_RMS_Z
  - Accel_Mag_RMS

## Dataset structure
- subject_1_day3: main training data, speeds 2.0 to 8.5 mph in 0.5 mph increments.
- subject_1_day2: validation data, mostly 2, 3, 4, and 8 mph. Has startup/shutdown artifacts.
- subject_1_day4: validation data, includes 5, 6, and 7 mph plus cadence manipulation tests.
- subject_2: cross-subject / cadence manipulation data.

## Raw file formats
- Subject 1 Day 2 recordings are Excel/XLSX files.
- Subject 1 Day 3 recordings are CSV files.
- Subject 1 Day 4 recordings are CSV files.
- Subject 2 recordings are CSV files.
- The data loader must support both CSV and Excel/XLSX inputs.

## Raw data policy
- Treat all files in `data/raw/` as immutable source data.
- Do not rename or modify raw files.
- Create standardized recording names and corrected metadata in a dataset
  manifest rather than changing raw filenames.
- Ignore macOS metadata such as `.DS_Store`, `__MACOSX/`, and files beginning
  with `._`; these are not data recordings.

## Known data issues
- `data/raw/subject_1_day4/imu_recording_20260619_6mnph_135spm_decreased.csv`
  means 6 mph despite the `6mnph` typo.
- `data/raw/subject_2/imu_recording_20260606_6pmh_elev2.csv` means 6 mph despite
  the `6pmh` typo.
- Some Subject 2 filenames contain cadence or condition notes, including `175`,
  `185`, `leftfoothurt`, and `dropped_to_160`. Preserve these notes in manifest
  metadata rather than discarding them during name standardization.
- `data/raw/subject_1_day2/Day2Test4IMU.xlsx` is suspicious and currently
  excluded.
- `data/raw/subject_1_day2/Test4-8start-8mph.xlsx` must not be used unless it is
  explicitly approved.
- Day 2 files need start/end trimming due to treadmill startup/shutdown and stepping off.

## Current assumptions
- Train primarily on Subject 1 Day 3.
- Validate on Subject 1 Day 2 + Day 4.
- Use one 8 mph validation file unless otherwise stated.
- Do not treat all startup/shutdown windows as valid steady-state speed.

## Current best results
- Random Forest same-subject validation reached approximately 0.25–0.37 mph MAE depending on included files.
- Full-range model performs well across 2–8 mph after trimming/excluding bad Day 2 data.
- Feature ablation suggests Cadence + Gyro_RMS_Y + Accel_Mag_RMS is strongest.
- Accel_Mag_RMS appears more cadence-independent than originally expected.
- Gyro_RMS_Y is useful but not fully cadence-independent.

## Current research questions
1. Can speed be predicted accurately from head-mounted IMU features?
2. Which features add information beyond cadence?
3. Does the model generalize across days?
4. Does the model generalize across subjects?
5. Should walking and running be modeled separately?

## Next recommended task
Create `configs/dataset_manifest.csv` manually or semi-automatically with one
row per real data file and these columns:

```text
recording_id,relative_path,subject_id,session,speed_mph,file_type,condition,include,exclusion_reason,trim_start_sec,trim_end_sec,notes
```
