# Decision Log

This file records project decisions that affect data handling, experiments, or
interpretation. Add future entries with a date, decision, rationale, and any
conditions that would justify revisiting the decision.

## Initial decisions

### IMU coordinate frame and axis interpretation

**Decision:** Use the worn-device coordinate frame when interpreting IMU axes:
X is left/right across the face, Y is forward/backward in the direction of
travel, and Z is vertical up/down.

Accelerometer axes measure linear acceleration along each axis. Gyroscope axes
measure angular velocity, or rotation rate, about each axis. Therefore:

- `Accel_Z` is vertical acceleration.
- `Gyro_X` is pitch/nodding rate, rotation about the left/right axis.
- `Gyro_Y` is roll/side-tilt rate, rotation about the forward axis.
- `Gyro_Z` is yaw/turning rate, rotation about the vertical axis.

**Rationale:** This distinction prevents confusing linear acceleration axes
with rotational axes. In particular, Gyro Y is roll/side-tilt rate, not pitch.
Existing feature columns may remain axis-based for now; do not rename feature
columns until a deliberate schema migration is approved.

### Subject 1 Day 3 is the canonical training set

**Decision:** Train primary models on Subject 1 Day 3, covering speeds from
2.0 to 8.5 mph in 0.5 mph increments.

**Rationale:** It is the most complete same-subject dataset across the target
speed range.

### Subject 1 Day 2 and Day 4 are validation sets

**Decision:** Reserve approved Subject 1 Day 2 and Day 4 recordings for
same-subject, cross-day validation.

**Rationale:** Keeping these days separate measures generalization across
recording sessions rather than fit to windows from the training session.

### Day2Test4IMU.xlsx is excluded for now

**Decision:** Exclude `Day2Test4IMU.xlsx` from training and evaluation until it
has been investigated and explicitly approved.

**Rationale:** The recording is considered suspicious. The exclusion must be
represented in dataset metadata rather than hidden in analysis code.

### Day 2 requires start/end trimming

**Decision:** Remove documented startup, shutdown, and stepping-off intervals
from Day 2 before generating analysis windows.

**Rationale:** These intervals are not valid steady-state examples of the
recording's labeled treadmill speed. Trimming rules must be explicit and
reviewable for each recording.

### Random window splits are not allowed

**Decision:** Do not randomly divide windows from the same recording between
training and validation data.

**Rationale:** Five-second windows use a 2.5-second step, so adjacent windows
overlap by 50%. Random window splits would leak shared signal samples and
recording-specific patterns into validation. Splits must operate at the
recording, day, or subject level.

### The loader must support CSV and Excel files

**Decision:** Support CSV inputs for Subject 1 Day 3, Subject 1 Day 4, and
Subject 2, and Excel/XLSX inputs for Subject 1 Day 2.

**Rationale:** Both formats are part of the canonical raw inventory. Format
handling should converge on one validated internal schema before trimming,
windowing, or feature extraction.

### Raw data files are immutable

**Decision:** Do not rename or modify files under `data/raw/`. Create
standardized recording IDs and corrected labels in the dataset manifest.

**Rationale:** Preserving source files makes the analysis auditable and avoids
breaking provenance. In particular, interpret `6mnph` in the Subject 1 Day 4
filename and `6pmh` in the Subject 2 filename as 6 mph through manifest
metadata, not filesystem changes.

### Filename condition notes are preserved

**Decision:** Preserve Subject 2 filename notes such as `175`, `185`,
`leftfoothurt`, and `dropped_to_160` in manifest condition or notes fields.

**Rationale:** These annotations may explain cadence manipulation, participant
state, or deviations within a recording and could affect interpretation.

### Test4-8start-8mph.xlsx requires explicit approval

**Decision:** Do not use
`data/raw/subject_1_day2/Test4-8start-8mph.xlsx` unless it is explicitly
approved for a defined experiment.

**Rationale:** Its inclusion status is intentionally unresolved. The manifest
must default it to excluded and state the reason.

### Non-data filesystem metadata is ignored

**Decision:** Ignore `.DS_Store`, `__MACOSX/`, and files beginning with `._`
when inventorying or loading recordings.

**Rationale:** These are macOS metadata artifacts, not sensor data.

### The dataset manifest is the next inventory artifact

**Decision:** Create `configs/dataset_manifest.csv` manually or
semi-automatically, with one row per real data file and the following columns:
`recording_id`, `relative_path`, `subject_id`, `session`, `speed_mph`,
`file_type`, `condition`, `include`, `exclusion_reason`, `trim_start_sec`,
`trim_end_sec`, and `notes`.

**Rationale:** The manifest will centralize corrected names, source paths,
experimental conditions, inclusion decisions, and trimming metadata without
altering raw recordings.
