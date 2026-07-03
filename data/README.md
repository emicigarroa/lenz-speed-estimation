# Data

This directory separates immutable source recordings from reproducible derived
datasets.

## `raw/`

Place original IMU workbooks in the matching subject/day folder:

```text
raw/
├── subject_1_day2/
├── subject_1_day3/
├── subject_1_day4/
└── subject_2/
```

Do not edit, clean, trim, or rename raw files in place. Known filename typos,
speed labels, exclusions, and experimental conditions should be handled through
explicit metadata. `Day2Test4IMU.xlsx` is currently excluded pending review.

Raw recordings may contain sensitive or large experimental data and are not
intended to be committed unless the team adopts an approved data-management
strategy.

## `processed/`

Store generated, reproducible artifacts here, such as canonical signal tables,
window metadata, and feature tables. Every processed row should retain enough
provenance to identify its subject, day, recording, speed, source file, and
window boundaries.

Processed files are ignored by Git and should be rebuildable from raw data,
documented metadata, and analysis code.
