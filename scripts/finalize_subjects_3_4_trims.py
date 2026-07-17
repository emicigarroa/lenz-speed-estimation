"""Promote reviewed Subject 3/4 auto-trim proposals to approved trims.

This script is intentionally narrow: it does not inspect or modify raw sensor
files. It copies the current proposed automatic trim boundaries into the
approved trim columns after manual diagnostic review, while preserving the
automatic proposal values for provenance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "configs/dataset_manifest.csv"
BOUNDARIES_PATH = REPOSITORY_ROOT / "outputs/tables/proposed_trim_boundaries.csv"
REVIEW_NOTE = "trim diagnostic plot manually inspected and approved"
SUBJECTS = {"subject_3", "subject_4"}


def _append_note(existing: object, note: str) -> str:
    """Append note text once to a semicolon-separated notes field."""

    notes = "" if pd.isna(existing) else str(existing).strip()
    if note in notes:
        return notes
    return f"{notes}; {note}" if notes else note


def _format_seconds(value: object) -> str:
    """Format a numeric seconds value compactly for manifest CSV storage."""

    numeric = float(value)
    return f"{numeric:.2f}".rstrip("0").rstrip(".")


def main() -> None:
    """Finalize approved trims in the dataset manifest."""

    manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False, dtype=str)
    boundaries = pd.read_csv(BOUNDARIES_PATH, keep_default_na=False, dtype=str)
    boundary_by_id = boundaries.set_index("recording_id")

    required_boundary_columns = {"trim_from_start_sec", "trim_from_end_sec"}
    missing = required_boundary_columns.difference(boundaries.columns)
    if missing:
        raise ValueError(
            "Proposed trim boundary table is missing columns: "
            + ", ".join(sorted(missing))
        )

    finalized = 0
    kept_excluded = 0
    for index, row in manifest.iterrows():
        subject_id = str(row.get("subject_id", ""))
        if subject_id not in SUBJECTS:
            continue

        recording_id = str(row["recording_id"])
        if recording_id not in boundary_by_id.index:
            raise ValueError(f"No proposed trim boundary found for {recording_id}.")

        boundary = boundary_by_id.loc[recording_id]
        auto_start = _format_seconds(boundary["trim_from_start_sec"])
        auto_end = _format_seconds(boundary["trim_from_end_sec"])
        manifest.at[index, "auto_trim_start_sec"] = auto_start
        manifest.at[index, "auto_trim_end_sec"] = auto_end
        manifest.at[index, "approved_trim_start_sec"] = auto_start
        manifest.at[index, "approved_trim_end_sec"] = auto_end
        manifest.at[index, "trim_review_status"] = "approved"
        manifest.at[index, "trim_method"] = "manual_review_of_auto_proposal"
        manifest.at[index, "notes"] = _append_note(row.get("notes", ""), REVIEW_NOTE)

        exclusion_reason = str(row.get("exclusion_reason", "")).strip()
        still_excluded_reasons = [
            reason.strip()
            for reason in exclusion_reason.split(";")
            if reason.strip()
            and reason.strip() not in {"trim pending manual review"}
        ]
        if still_excluded_reasons:
            manifest.at[index, "include"] = "false"
            manifest.at[index, "exclusion_reason"] = "; ".join(still_excluded_reasons)
            kept_excluded += 1
        else:
            manifest.at[index, "include"] = "true"
            manifest.at[index, "exclusion_reason"] = ""
            finalized += 1

        boundary_index = boundaries.index[boundaries["recording_id"] == recording_id]
        boundaries.loc[boundary_index, "trim_review_status"] = "approved"
        boundaries.loc[boundary_index, "manual_review_reason"] = REVIEW_NOTE

    manifest.to_csv(MANIFEST_PATH, index=False)
    boundaries.to_csv(BOUNDARIES_PATH, index=False)
    print(
        f"Finalized approved trims for {finalized} included recordings; "
        f"kept {kept_excluded} reviewed recordings excluded."
    )


if __name__ == "__main__":
    main()
