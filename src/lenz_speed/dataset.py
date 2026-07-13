"""Build and save the window-level LENZ feature dataset."""

from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from .data import load_manifest, load_recording
from .features import FeatureExtractionError, extract_window_features
from .windowing import apply_trim, make_windows


FEATURE_TABLE_COLUMNS = (
    "recording_id",
    "relative_path",
    "subject_id",
    "session",
    "speed_mph",
    "file_type",
    "condition",
    "notes",
    "window_index",
    "window_start_sec",
    "window_end_sec",
    "Cadence_spm",
    "RMS_Z",
    "PeakToPeak_Z",
    "Gyro_RMS_X",
    "Gyro_RMS_Y",
    "Gyro_RMS_Z",
    "Accel_Mag_RMS",
    "Dynamic_Accel_Mag_RMS",
    "Accel_Mag_P95_P05",
    "Accel_Mag_Jerk_RMS",
    "Accel_HighFreq_Energy_Ratio",
    "Gyro_Mag_RMS",
    "GyroY_PeakToPeak",
    "Accel_Anisotropy",
    "Impact_Peak_Count",
    "Mean_Impact_Interval_s",
    "Impact_Interval_CV",
    "Mean_Impact_Prominence",
    "Mean_Impact_Width_s",
    "Impact_Duty_Proxy",
    "Vertical_Peak_Sharpness",
    "Impact_Impulse",
    "Peak_Symmetry",
    "Impact_Crest_Factor",
    "Impact_Local_Kurtosis",
)

_MANIFEST_METADATA_COLUMNS = (
    "recording_id",
    "relative_path",
    "subject_id",
    "session",
    "speed_mph",
    "file_type",
    "condition",
    "notes",
)


def _trim_value(value: Any) -> float:
    """Interpret blank manifest trim values as zero seconds."""

    return 0.0 if pd.isna(value) else float(value)


def _selected_trim_values(manifest_row: dict[str, Any]) -> tuple[float, float]:
    """Return active trim boundaries for a manifest row.

    Approved manual trim boundaries take priority. Automatic trim proposals are
    used only when ``trim_review_status`` explicitly indicates acceptance. This
    prevents unreviewed ramp/plateau detector proposals from silently entering
    training or evaluation data.
    """

    approved_start = manifest_row.get("approved_trim_start_sec")
    approved_end = manifest_row.get("approved_trim_end_sec")
    if not pd.isna(approved_start) or not pd.isna(approved_end):
        return _trim_value(approved_start), _trim_value(approved_end)

    review_status = str(manifest_row.get("trim_review_status", "")).strip().lower()
    if review_status in {"accepted", "approved", "auto_accepted"}:
        return (
            _trim_value(manifest_row.get("auto_trim_start_sec")),
            _trim_value(manifest_row.get("auto_trim_end_sec")),
        )

    has_unreviewed_auto_trim = (
        not pd.isna(manifest_row.get("auto_trim_start_sec"))
        or not pd.isna(manifest_row.get("auto_trim_end_sec"))
    )
    has_legacy_trim = (
        not pd.isna(manifest_row.get("trim_start_sec"))
        or not pd.isna(manifest_row.get("trim_end_sec"))
    )
    if has_unreviewed_auto_trim and not has_legacy_trim:
        recording_id = manifest_row.get("recording_id", "<unknown>")
        raise ValueError(
            f"{recording_id} has automatic trim proposals but no approved "
            "trim boundaries. Mark trim_review_status as accepted/approved, "
            "or fill approved_trim_start_sec/approved_trim_end_sec before "
            "including this recording."
        )

    return (
        _trim_value(manifest_row.get("trim_start_sec")),
        _trim_value(manifest_row.get("trim_end_sec")),
    )


def _print_build_summary(
    recording_count: int,
    window_count: int,
    windows_by_recording: dict[str, int],
    skipped_window_count: int,
) -> None:
    """Print a compact, deterministic dataset-build summary."""

    print(
        f"Built windowed feature table: {recording_count} recordings, "
        f"{window_count} windows."
    )
    print("Windows by recording:")
    for recording_id, count in windows_by_recording.items():
        print(f"  {recording_id}: {count}")
    if skipped_window_count:
        print(f"Skipped invalid windows: {skipped_window_count}")


def build_windowed_feature_table(
    include_excluded: bool = False,
    fs: Real = 200,
) -> pd.DataFrame:
    """Build one feature row for every complete window in the manifest.

    The function loads each selected recording, applies its manifest trimming
    values, creates the default 5-second windows with 2.5-second steps, and
    extracts the original, v2, v3, and experimental v4 features. Blank trim
    values are treated as zero seconds. Rows marked ``include=false`` are skipped unless
    ``include_excluded=True`` is passed explicitly. Windows containing invalid
    signal values are skipped with a provenance-rich message; values are never
    interpolated silently.

    Parameters
    ----------
    include_excluded:
        Include manifest rows marked for exclusion. The default is false.
    fs:
        Sampling rate in hertz used for trimming, windowing, filtering, and
        cadence estimation. The default is 200 Hz.

    Returns
    -------
    pandas.DataFrame
        One row per complete window, with recording metadata, window
        provenance, and feature values.
    """

    manifest = load_manifest(include_excluded=include_excluded)
    feature_rows: list[dict[str, Any]] = []
    windows_by_recording: dict[str, int] = {}
    skipped_window_count = 0

    for manifest_row in manifest.to_dict(orient="records"):
        recording_id = str(manifest_row["recording_id"])
        recording = load_recording(
            recording_id,
            include_excluded=include_excluded,
        )
        trim_start_sec, trim_end_sec = _selected_trim_values(manifest_row)
        trimmed = apply_trim(
            recording,
            trim_start_sec=trim_start_sec,
            trim_end_sec=trim_end_sec,
            fs=fs,
        )
        windows = make_windows(trimmed, recording_id=recording_id, fs=fs)
        valid_window_count = 0

        metadata = {
            column: manifest_row[column] for column in _MANIFEST_METADATA_COLUMNS
        }
        for window in windows:
            try:
                features = extract_window_features(window, fs=fs)
            except FeatureExtractionError as error:
                skipped_window_count += 1
                print(
                    f"Skipping {recording_id} window {window.window_index} "
                    f"({window.window_start_sec:g}--{window.window_end_sec:g} s): "
                    f"{error}"
                )
                continue
            feature_rows.append(
                {
                    **metadata,
                    "window_index": features["window_index"],
                    "window_start_sec": features["window_start_sec"],
                    "window_end_sec": features["window_end_sec"],
                    "Cadence_spm": features["Cadence_spm"],
                    "RMS_Z": features["RMS_Z"],
                    "PeakToPeak_Z": features["PeakToPeak_Z"],
                    "Gyro_RMS_X": features["Gyro_RMS_X"],
                    "Gyro_RMS_Y": features["Gyro_RMS_Y"],
                    "Gyro_RMS_Z": features["Gyro_RMS_Z"],
                    "Accel_Mag_RMS": features["Accel_Mag_RMS"],
                    "Dynamic_Accel_Mag_RMS": features[
                        "Dynamic_Accel_Mag_RMS"
                    ],
                    "Accel_Mag_P95_P05": features["Accel_Mag_P95_P05"],
                    "Accel_Mag_Jerk_RMS": features["Accel_Mag_Jerk_RMS"],
                    "Accel_HighFreq_Energy_Ratio": features[
                        "Accel_HighFreq_Energy_Ratio"
                    ],
                    "Gyro_Mag_RMS": features["Gyro_Mag_RMS"],
                    "GyroY_PeakToPeak": features["GyroY_PeakToPeak"],
                    "Accel_Anisotropy": features["Accel_Anisotropy"],
                    "Impact_Peak_Count": features["Impact_Peak_Count"],
                    "Mean_Impact_Interval_s": features[
                        "Mean_Impact_Interval_s"
                    ],
                    "Impact_Interval_CV": features["Impact_Interval_CV"],
                    "Mean_Impact_Prominence": features[
                        "Mean_Impact_Prominence"
                    ],
                    "Mean_Impact_Width_s": features["Mean_Impact_Width_s"],
                    "Impact_Duty_Proxy": features["Impact_Duty_Proxy"],
                    "Vertical_Peak_Sharpness": features[
                        "Vertical_Peak_Sharpness"
                    ],
                    "Impact_Impulse": features["Impact_Impulse"],
                    "Peak_Symmetry": features["Peak_Symmetry"],
                    "Impact_Crest_Factor": features["Impact_Crest_Factor"],
                    "Impact_Local_Kurtosis": features[
                        "Impact_Local_Kurtosis"
                    ],
                }
            )
            valid_window_count += 1
        windows_by_recording[recording_id] = valid_window_count

    table = pd.DataFrame(feature_rows, columns=FEATURE_TABLE_COLUMNS)
    _print_build_summary(
        len(manifest),
        len(table),
        windows_by_recording,
        skipped_window_count,
    )
    return table


def save_windowed_feature_table(
    output_path: str | Path = "data/processed/windowed_features.csv",
    *,
    include_excluded: bool = False,
    fs: Real = 200,
) -> Path:
    """Build the windowed feature table and save it as CSV.

    Relative output paths are resolved from the repository root. Parent
    directories are created as needed. The returned path is absolute.
    """

    table = build_windowed_feature_table(
        include_excluded=include_excluded,
        fs=fs,
    )
    destination = Path(output_path).expanduser()
    if not destination.is_absolute():
        destination = Path(__file__).resolve().parents[2] / destination
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    print(f"Saved windowed feature table to {destination}")
    return destination
