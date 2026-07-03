# Decision Log

This file records project decisions that affect data handling, experiments, or
interpretation. Add future entries with a date, decision, rationale, and any
conditions that would justify revisiting the decision.

## Initial decisions

### Gyroscope Y represents roll

**Decision:** Interpret Gyro Y as roll, not pitch.

**Rationale:** This is the established sensor-axis convention for the current
dataset. Feature names may remain axis-based, but plots and interpretation must
use the correct motion description.

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
