# Team LENZ Speed Estimation Project Context

## Project
Team LENZ capstone project: Smart Glasses as a Platform for Real-Time Run Coaching.

Goal: estimate treadmill running speed from a head-mounted IMU mounted on smart glasses.

## Sensor and data
- IMU sampling rate: 200 Hz.
- Window size: 5 seconds.
- Step size: 2.5 seconds.
- Gyro Y is ROLL, not pitch.
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

## Known data issues
- Day2Test4IMU.xlsx is suspicious and currently excluded.
- Day 2 files need start/end trimming due to treadmill startup/shutdown and stepping off.
- Some filenames contain typos, such as 6mnph instead of 6mph.

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
