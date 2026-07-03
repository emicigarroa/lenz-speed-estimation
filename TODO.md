# Project TODO

## Phase 1: Setup

- [ ] Confirm the supported Python version.
- [ ] Create and activate a virtual environment.
- [ ] Install and verify the dependencies in `requirements.txt`.
- [ ] Define the installable `src/` package and test structure.
- [ ] Document reproducibility conventions and generated-file policy.

## Phase 2: Data inventory

- [ ] Add the raw workbooks to the appropriate subject/day folders.
- [ ] Record file-level metadata: subject, day, speed, trial type, and cadence
      condition.
- [ ] Inspect workbook sheets, column names, units, timestamps, and durations.
- [ ] Verify the effective 200 Hz sampling rate and identify missing or
      duplicated samples.
- [ ] Record known filename corrections without renaming raw source files.
- [ ] Exclude `Day2Test4IMU.xlsx` with a documented reason.
- [ ] Define and review start/end trimming rules for Day 2 recordings.
- [ ] Select and document the approved 8 mph validation recording.

## Phase 3: Feature extraction

- [ ] Implement deterministic 5-second windows with a 2.5-second step.
- [ ] Preserve subject, day, recording, speed, and window provenance.
- [ ] Implement and test `Cadence_spm`.
- [ ] Implement and test `RMS_Z` and `PeakToPeak_Z`.
- [ ] Implement and test gyroscope RMS for X, Y, and Z.
- [ ] Implement and test `Accel_Mag_RMS`.
- [ ] Validate feature values against representative signal plots.
- [ ] Produce a reproducible processed feature table.

## Phase 4: Modeling

- [ ] Define leakage-safe splits by recording, day, and subject.
- [ ] Establish a cadence-only baseline.
- [ ] Train a Random Forest baseline on Subject 1 Day 3.
- [ ] Compare the full feature set with documented reduced feature sets.
- [ ] Record model parameters, training inputs, and random seeds.

## Phase 5: Evaluation

- [ ] Validate on approved Subject 1 Day 2 and Day 4 recordings.
- [ ] Report MAE, RMSE, signed error, and sample counts.
- [ ] Report results by speed, recording, and day.
- [ ] Generate predicted-versus-actual and residual plots.
- [ ] Confirm that no overlapping windows cross split boundaries.
- [ ] Compare results with the historical 0.25–0.37 mph MAE range.

## Phase 6: Future experiments

- [ ] Evaluate cross-subject generalization on Subject 2.
- [ ] Analyze cadence-manipulation trials as a separate stress test.
- [ ] Measure which features add information beyond cadence.
- [ ] Compare combined and separate walking/running models.
- [ ] Assess robustness to sensor placement and day-to-day variation.
- [ ] Investigate uncertainty estimates and real-time inference constraints.
