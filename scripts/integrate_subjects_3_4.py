"""Inventory, trim-diagnose, and manifest-integrate Subjects 3 and 4."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import tempfile
import warnings


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lenz-speed-matplotlib"),
)
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lenz_speed.data import CANONICAL_SIGNAL_COLUMNS, load_recording  # noqa: E402
from lenz_speed.features import _rms, extract_window_features  # noqa: E402
from lenz_speed.windowing import SignalWindow  # noqa: E402


FS = 200.0
RAW_ROOTS = (
    REPOSITORY_ROOT / "data/raw/subject_3_day2",
    REPOSITORY_ROOT / "data/raw/subject_4_day2",
)
EXPECTED_SPEEDS = {
    "subject_3": {2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0},
    "subject_4": {2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5},
}
BASE_MANIFEST_COLUMNS = (
    "recording_id",
    "relative_path",
    "subject_id",
    "session",
    "speed_mph",
    "file_type",
    "condition",
    "include",
    "exclusion_reason",
    "trim_start_sec",
    "trim_end_sec",
    "auto_trim_start_sec",
    "auto_trim_end_sec",
    "approved_trim_start_sec",
    "approved_trim_end_sec",
    "notes",
    "trim_method",
    "trim_review_status",
)
PLATEAU_FEATURES = (
    "cadence_spm",
    "dynamic_accel_mag_rms",
    "vertical_peak_sharpness",
)
ROLLING_MEDIAN_SECONDS = 10
SUSTAINED_PLATEAU_SECONDS = 15
SAFETY_BUFFER_SECONDS = 5.0
MIN_STEADY_STATE_SECONDS = 30.0
ENTER_FRACTION = 0.88
REMAIN_FRACTION = 0.62
COLLAPSE_FRACTION = 0.55


def _longest_true_run(mask: np.ndarray) -> tuple[int | None, int | None, int]:
    """Return inclusive start/end indices and length for the longest true run."""

    best_start: int | None = None
    best_end: int | None = None
    best_length = 0
    start: int | None = None
    for index, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = index
        if (not value or index == len(mask) - 1) and start is not None:
            end = index if value and index == len(mask) - 1 else index - 1
            length = end - start + 1
            if length > best_length:
                best_start = start
                best_end = end
                best_length = length
            start = None
    return best_start, best_end, best_length


def _is_data_file(path: Path) -> bool:
    parts = set(path.parts)
    return (
        path.is_file()
        and path.suffix.lower() == ".csv"
        and "__MACOSX" not in parts
        and not path.name.startswith("._")
        and path.name != ".DS_Store"
    )


def _subject_from_path(path: Path) -> str:
    if "subject_3_day2" in path.parts:
        return "subject_3"
    if "subject_4_day2" in path.parts:
        return "subject_4"
    raise ValueError(f"Cannot infer subject from {path}")


def _parse_metadata(path: Path) -> dict[str, object]:
    subject_id = _subject_from_path(path)
    session = "day2"
    match = re.search(r"(?P<speed>\d+(?:\.\d+)?)mph", path.name)
    if not match:
        speed = np.nan
    else:
        speed = float(match.group("speed"))
    speed_label = "unknown" if pd.isna(speed) else f"{speed:g}".replace(".", "p")
    subject_number = subject_id.split("_")[1]
    recording_id = f"s{subject_number}_day2_{speed_label}mph"
    duration_label = "2min" if "2min" in path.name else "5min" if "5min" in path.name else "1min"
    return {
        "recording_id": recording_id,
        "relative_path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "subject_id": subject_id,
        "session": session,
        "speed_mph": speed,
        "file_type": "csv",
        "condition": "normal",
        "duration_label": duration_label,
    }


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _valid_imu_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    required = list(CANONICAL_SIGNAL_COLUMNS)
    missing = [column for column in required if column not in raw.columns]
    if missing:
        return pd.DataFrame(columns=required), {
            "missing_signal_columns": "|".join(missing),
            "valid_imu_rows": 0,
            "nonfinite_signal_rows": len(raw),
            "parse_fail_rows": np.nan,
            "non_imu_packet_rows": np.nan,
        }

    signal_numeric = raw.loc[:, required].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(signal_numeric.to_numpy(dtype=float)).all(axis=1)
    parse_ok = np.ones(len(raw), dtype=bool)
    if "parse_ok" in raw.columns:
        parse_ok = raw["parse_ok"].astype("string").str.lower().isin({"true", "1", "yes"}).to_numpy()
    imu_packet = np.ones(len(raw), dtype=bool)
    if "packet_type" in raw.columns:
        imu_packet = pd.to_numeric(raw["packet_type"], errors="coerce").eq(61).to_numpy()
    valid = finite & parse_ok & imu_packet
    frame = signal_numeric.loc[valid].reset_index(drop=True)
    return frame, {
        "missing_signal_columns": "",
        "valid_imu_rows": int(valid.sum()),
        "nonfinite_signal_rows": int((~finite).sum()),
        "parse_fail_rows": int((~parse_ok).sum()) if "parse_ok" in raw.columns else np.nan,
        "non_imu_packet_rows": int((~imu_packet).sum()) if "packet_type" in raw.columns else np.nan,
    }


def _quality_row(path: Path, raw: pd.DataFrame, valid: pd.DataFrame, meta: dict[str, object]) -> dict[str, object]:
    host_time = pd.to_numeric(raw.get("host_time"), errors="coerce") if "host_time" in raw.columns else pd.Series(dtype=float)
    valid_host = host_time.dropna()
    duration_host = float(valid_host.iloc[-1] - valid_host.iloc[0]) if len(valid_host) > 1 else np.nan
    dt = valid_host.diff().dropna()
    median_dt = float(dt.median()) if len(dt) else np.nan
    sample_rate = float(1.0 / median_dt) if median_dt and np.isfinite(median_dt) and median_dt > 0 else np.nan
    duplicate_timestamps = int(host_time.duplicated().sum()) if "host_time" in raw.columns else np.nan
    duplicate_rows = int(raw.duplicated().sum())
    packet_counts = ""
    if "packet_type" in raw.columns:
        packet_counts = ";".join(
            f"{int(k) if pd.notna(k) else 'nan'}:{int(v)}"
            for k, v in pd.to_numeric(raw["packet_type"], errors="coerce").value_counts(dropna=False).sort_index().items()
        )
    issues: list[str] = []
    speed = float(meta["speed_mph"]) if pd.notna(meta["speed_mph"]) else np.nan
    if pd.isna(speed) or speed not in EXPECTED_SPEEDS[str(meta["subject_id"])]:
        issues.append("speed_not_in_protocol")
    if "mphcsv.csv" in path.name:
        issues.append("filename_has_mphcsv_typo")
    if not np.isfinite(sample_rate) or not 180 <= sample_rate <= 220:
        issues.append("sample_rate_outside_expected_range")
    if len(valid) < 1000:
        issues.append("too_few_valid_imu_samples")
    return {
        **meta,
        "source_path": path.as_posix(),
        "raw_rows": len(raw),
        "valid_imu_rows": len(valid),
        "duration_host_sec": duration_host,
        "duration_samples_sec": len(valid) / FS,
        "median_sample_rate_hz": sample_rate,
        "columns": "|".join(map(str, raw.columns)),
        "packet_type_counts": packet_counts,
        "duplicate_timestamps": duplicate_timestamps,
        "duplicate_rows": duplicate_rows,
        "issue_flags": "|".join(issues),
    }


def _window_feature_series(valid: pd.DataFrame, recording_id: str) -> pd.DataFrame:
    window_size = int(5 * FS)
    step = int(1 * FS)
    rows: list[dict[str, float]] = []
    for start in range(0, max(0, len(valid) - window_size + 1), step):
        stop = start + window_size
        signal = valid.iloc[start:stop].reset_index(drop=True)
        window = SignalWindow(
            recording_id=recording_id,
            window_index=len(rows),
            window_start_sec=start / FS,
            window_end_sec=stop / FS,
            signal=signal,
        )
        try:
            features = extract_window_features(window, fs=FS)
        except Exception:
            continue
        accel_mag = np.sqrt(
            signal["ax_g"].to_numpy(dtype=float) ** 2
            + signal["ay_g"].to_numpy(dtype=float) ** 2
            + signal["az_g"].to_numpy(dtype=float) ** 2
        )
        rows.append(
            {
                "time_sec": (start + stop) / (2 * FS),
                "window_start_sec": start / FS,
                "window_end_sec": stop / FS,
                "cadence_spm": features["Cadence_spm"],
                "dynamic_accel_mag_rms": _rms(accel_mag - np.mean(accel_mag)),
                "accel_mag_mean": float(np.mean(accel_mag)),
                "vertical_peak_sharpness": features["Vertical_Peak_Sharpness"],
            }
        )
    return pd.DataFrame(rows)


def _sustained_start(mask: np.ndarray, run_length: int) -> int | None:
    """Return first index where mask stays true for run_length samples."""

    values = mask.astype(bool)
    for index in range(0, max(0, len(values) - run_length + 1)):
        if values[index : index + run_length].all():
            return index
    return None


def _first_sustained_false_after(mask: np.ndarray, start_index: int, run_length: int) -> int | None:
    """Return first false-run index after start_index, ignoring short dips."""

    values = mask.astype(bool)
    for index in range(start_index, max(start_index, len(values) - run_length + 1)):
        if (~values[index : index + run_length]).all():
            return index
    return None


def _propose_trim(feature_series: pd.DataFrame, total_duration: float, subject_id: str) -> dict[str, object]:
    """Propose one continuous target-speed plateau using hysteresis.

    Entry requires dynamic acceleration and vertical impact sharpness to reach a
    high fraction of their running plateau levels for at least 15 seconds. Once
    entered, the detector remains in steady state through modest natural drift
    and exits only on sustained collapse relative to the plateau. Cadence is
    used only as a sanity check, not as the governing reference.
    """

    empty = {
        "original_duration_sec": round(float(total_duration), 2),
        "proposed_trim_start_sec": np.nan,
        "proposed_trim_end_sec": np.nan,
        "trim_from_start_sec": np.nan,
        "trim_from_end_sec": np.nan,
        "steady_state_start_sec": np.nan,
        "steady_state_end_sec": np.nan,
        "steady_state_start_timestamp_sec": np.nan,
        "steady_state_end_timestamp_sec": np.nan,
        "steady_state_duration_sec": 0.0,
        "usable_steady_state_duration_sec": 0.0,
        "apparent_active_running_duration_sec": 0.0,
        "retained_active_fraction": 0.0,
        "trim_confidence": "low",
        "trim_review_status": "review_required",
        "manual_review_reason": "no_feature_windows",
    }
    if feature_series.empty:
        return empty

    data = feature_series.copy()
    for column in PLATEAU_FEATURES:
        data[f"{column}_median"] = (
            data[column]
            .rolling(
                ROLLING_MEDIAN_SECONDS,
                center=True,
                min_periods=max(3, ROLLING_MEDIAN_SECONDS // 2),
            )
            .median()
        )

    first_baseline = data.loc[data["time_sec"].between(0, 10)]
    middle = data.loc[data["time_sec"].between(total_duration * 0.40, total_duration * 0.60)]
    if len(middle) < SUSTAINED_PLATEAU_SECONDS:
        middle = data.loc[data["time_sec"].between(total_duration * 0.30, total_duration * 0.70)]
    if len(middle) < SUSTAINED_PLATEAU_SECONDS:
        result = empty.copy()
        result["manual_review_reason"] = "insufficient_middle_windows_for_plateau_reference"
        return result

    references = {
        "dynamic_accel_mag_rms": float(middle["dynamic_accel_mag_rms_median"].median()),
        "vertical_peak_sharpness": float(middle["vertical_peak_sharpness_median"].median()),
        "cadence_spm": float(middle["cadence_spm_median"].median()),
    }
    baselines = {
        "dynamic_accel_mag_rms": float(first_baseline["dynamic_accel_mag_rms_median"].median()) if len(first_baseline) else np.nan,
        "vertical_peak_sharpness": float(first_baseline["vertical_peak_sharpness_median"].median()) if len(first_baseline) else np.nan,
    }
    if not all(np.isfinite(references[name]) and references[name] > 0 for name in PLATEAU_FEATURES):
        result = empty.copy()
        result["manual_review_reason"] = "invalid_plateau_reference"
        return result

    enter_thresholds = {
        "dynamic_accel_mag_rms": ENTER_FRACTION * references["dynamic_accel_mag_rms"],
        "vertical_peak_sharpness": ENTER_FRACTION * references["vertical_peak_sharpness"],
    }
    remain_thresholds = {
        "dynamic_accel_mag_rms": max(0.05, REMAIN_FRACTION * references["dynamic_accel_mag_rms"]),
        "vertical_peak_sharpness": max(0.75, REMAIN_FRACTION * references["vertical_peak_sharpness"]),
    }
    collapse_thresholds = {
        "dynamic_accel_mag_rms": max(0.04, COLLAPSE_FRACTION * references["dynamic_accel_mag_rms"]),
        "vertical_peak_sharpness": max(0.65, COLLAPSE_FRACTION * references["vertical_peak_sharpness"]),
    }

    cadence_sane = data["cadence_spm_median"].between(45, 260).to_numpy(dtype=bool)
    dyn = data["dynamic_accel_mag_rms_median"].to_numpy(dtype=float)
    sharp = data["vertical_peak_sharpness_median"].to_numpy(dtype=float)
    finite = np.isfinite(dyn) & np.isfinite(sharp)

    entry_mask = (
        finite
        & cadence_sane
        & (dyn >= enter_thresholds["dynamic_accel_mag_rms"])
        & (sharp >= enter_thresholds["vertical_peak_sharpness"])
    )
    remain_mask = (
        finite
        & (dyn >= remain_thresholds["dynamic_accel_mag_rms"])
        & (sharp >= remain_thresholds["vertical_peak_sharpness"])
    )
    collapse_mask = (
        finite
        & (dyn < collapse_thresholds["dynamic_accel_mag_rms"])
        & (sharp < collapse_thresholds["vertical_peak_sharpness"])
    )

    active_mask = remain_mask | entry_mask
    active_start_index = _sustained_start(active_mask, SUSTAINED_PLATEAU_SECONDS)
    active_end_index = None
    if active_start_index is not None:
        collapse_start = _sustained_start(collapse_mask[active_start_index:], 5)
        if collapse_start is None:
            active_end_index = len(data) - 1
        else:
            active_end_index = active_start_index + collapse_start - 1
    apparent_active_duration = 0.0
    if active_start_index is not None and active_end_index is not None and active_end_index >= active_start_index:
        apparent_active_duration = float(
            data.iloc[active_end_index]["window_end_sec"]
            - data.iloc[active_start_index]["window_start_sec"]
        )

    entry_index = _sustained_start(entry_mask, SUSTAINED_PLATEAU_SECONDS)
    if entry_index is None:
        result = empty.copy()
        result.update(
            {
                "apparent_active_running_duration_sec": round(apparent_active_duration, 2),
                **{f"{name}_reference": round(value, 4) for name, value in references.items()},
                **{f"{name}_baseline": round(value, 4) for name, value in baselines.items()},
                **{f"{name}_enter_threshold": round(value, 4) for name, value in enter_thresholds.items()},
                **{f"{name}_remain_threshold": round(value, 4) for name, value in remain_thresholds.items()},
            }
        )
        result["manual_review_reason"] = "no_sustained_entry_to_plateau"
        return result

    exit_index = _first_sustained_false_after(remain_mask, entry_index, 5)
    collapse_exit = _sustained_start(collapse_mask[entry_index:], 5)
    if collapse_exit is not None:
        collapse_index = entry_index + collapse_exit
        if exit_index is None or collapse_index < exit_index:
            exit_index = collapse_index
    if exit_index is None:
        last_index = len(data) - 1
    else:
        last_index = max(entry_index, exit_index - 1)

    detected_start = float(data.iloc[entry_index]["window_start_sec"])
    detected_end = float(data.iloc[last_index]["window_end_sec"])
    steady_start = min(total_duration, detected_start + SAFETY_BUFFER_SECONDS)
    steady_end = max(0.0, detected_end - SAFETY_BUFFER_SECONDS)
    trim_start = max(0.0, steady_start)
    trim_end = max(0.0, total_duration - steady_end)
    steady_duration = max(0.0, steady_end - steady_start)
    retained_fraction = steady_duration / apparent_active_duration if apparent_active_duration > 0 else 0.0

    uncertainty_reasons: list[str] = []
    if steady_duration < MIN_STEADY_STATE_SECONDS:
        uncertainty_reasons.append("usable_steady_state_under_minimum_duration")
    if apparent_active_duration > 0 and retained_fraction < 0.60:
        uncertainty_reasons.append("retains_less_than_60_percent_of_active_running")
    if (
        "5min" in str(feature_series.attrs.get("duration_label", ""))
        and total_duration >= 260
        and steady_duration < 210
    ):
        uncertainty_reasons.append("five_min_recording_under_3p5_min_usable")
    if exit_index is not None and active_end_index is not None and last_index < active_end_index - 15:
        uncertainty_reasons.append("possible_middle_subsection_selected_before_later_running")

    confidence = "low" if uncertainty_reasons else "high"
    reason = "review_required_for_auto_trim" if confidence == "high" else "; ".join(uncertainty_reasons)

    return {
        "original_duration_sec": round(float(total_duration), 2),
        "proposed_trim_start_sec": round(trim_start, 2),
        "proposed_trim_end_sec": round(trim_end, 2),
        "trim_from_start_sec": round(trim_start, 2),
        "trim_from_end_sec": round(trim_end, 2),
        "detected_plateau_start_sec": round(detected_start, 2),
        "detected_plateau_end_sec": round(detected_end, 2),
        "steady_state_start_sec": round(steady_start, 2),
        "steady_state_end_sec": round(steady_end, 2),
        "steady_state_start_timestamp_sec": round(steady_start, 2),
        "steady_state_end_timestamp_sec": round(steady_end, 2),
        "steady_state_duration_sec": round(steady_duration, 2),
        "usable_steady_state_duration_sec": round(steady_duration, 2),
        "apparent_active_running_duration_sec": round(apparent_active_duration, 2),
        "retained_active_fraction": round(retained_fraction, 4),
        "trim_confidence": confidence,
        "trim_review_status": "review_required",
        "manual_review_reason": reason,
        **{f"{name}_reference": round(value, 4) for name, value in references.items()},
        **{f"{name}_baseline": round(value, 4) for name, value in baselines.items()},
        **{f"{name}_enter_threshold": round(value, 4) for name, value in enter_thresholds.items()},
        **{f"{name}_remain_threshold": round(value, 4) for name, value in remain_thresholds.items()},
        **{f"{name}_collapse_threshold": round(value, 4) for name, value in collapse_thresholds.items()},
    }

def _plot_diagnostics(
    recording_id: str,
    features: pd.DataFrame,
    boundary: dict[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    if not features.empty:
        plot_data = features.copy()
        for column in PLATEAU_FEATURES:
            plot_data[f"{column}_median"] = (
                plot_data[column]
                .rolling(
                    ROLLING_MEDIAN_SECONDS,
                    center=True,
                    min_periods=max(3, ROLLING_MEDIAN_SECONDS // 2),
                )
                .median()
            )
        axes[0].plot(features["time_sec"], features["cadence_spm"], color="#4C78A8", alpha=0.28)
        axes[0].plot(plot_data["time_sec"], plot_data["cadence_spm_median"], color="#4C78A8", linewidth=1.6)
        axes[1].plot(features["time_sec"], features["dynamic_accel_mag_rms"], color="#59A14F", alpha=0.28)
        axes[1].plot(plot_data["time_sec"], plot_data["dynamic_accel_mag_rms_median"], color="#59A14F", linewidth=1.6)
        axes[2].plot(features["time_sec"], features["vertical_peak_sharpness"], color="#E15759", alpha=0.28)
        axes[2].plot(plot_data["time_sec"], plot_data["vertical_peak_sharpness_median"], color="#E15759", linewidth=1.6)
        dyn_ref = boundary.get("dynamic_accel_mag_rms_reference")
        dyn_enter = boundary.get("dynamic_accel_mag_rms_enter_threshold")
        dyn_remain = boundary.get("dynamic_accel_mag_rms_remain_threshold")
        sharp_ref = boundary.get("vertical_peak_sharpness_reference")
        sharp_enter = boundary.get("vertical_peak_sharpness_enter_threshold")
        sharp_remain = boundary.get("vertical_peak_sharpness_remain_threshold")
        for value, style, label in (
            (dyn_ref, "-", "plateau reference"),
            (dyn_enter, "--", "entry threshold"),
            (dyn_remain, ":", "remain threshold"),
        ):
            if pd.notna(value):
                axes[1].axhline(float(value), color="#333333", linestyle=style, linewidth=0.9, alpha=0.75, label=label)
        for value, style, label in (
            (sharp_ref, "-", "plateau reference"),
            (sharp_enter, "--", "entry threshold"),
            (sharp_remain, ":", "remain threshold"),
        ):
            if pd.notna(value):
                axes[2].axhline(float(value), color="#333333", linestyle=style, linewidth=0.9, alpha=0.75, label=label)
    labels = ("Cadence (spm)", "Dynamic accel RMS (g)", "Vertical peak sharpness")
    start = boundary.get("steady_state_start_timestamp_sec")
    end = boundary.get("steady_state_end_timestamp_sec")
    for ax, label in zip(axes, labels, strict=True):
        if pd.notna(start):
            ax.axvline(float(start), color="#222222", linestyle="--", label="steady-state start")
        if pd.notna(end):
            ax.axvline(float(end), color="#222222", linestyle=":", label="steady-state end")
        if pd.notna(start) and pd.notna(end):
            ax.axvspan(float(start), float(end), color="#F2CF5B", alpha=0.13)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time from recording start (s)")
    fig.suptitle(f"Trim diagnostic — {recording_id}")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    fig.savefig(output_dir / f"{recording_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_manifest_entries(boundaries: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in boundaries.to_dict(orient="records"):
        q = quality.loc[quality["recording_id"] == row["recording_id"]].iloc[0]
        issue_flags = "" if pd.isna(q["issue_flags"]) else str(q["issue_flags"])
        include = False
        reasons = ["trim pending manual review"]
        if "speed_not_in_protocol" in issue_flags.split("|"):
            reasons.append("speed not in stated protocol")
        if row["trim_confidence"] == "low":
            reasons.append("low-confidence automatic trim")
        exclusion_reason = "; ".join(reasons)
        notes = [
            "new natural-cadence data",
            "raw file values unmodified",
            f"auto trim confidence {row['trim_confidence']}",
        ]
        if issue_flags:
            notes.append(f"quality flags: {issue_flags}")
        rows.append(
            {
                "recording_id": row["recording_id"],
                "relative_path": row["relative_path"],
                "subject_id": row["subject_id"],
                "session": row["session"],
                "speed_mph": row["speed_mph"],
                "file_type": "csv",
                "condition": "normal",
                "include": str(bool(include)).lower(),
                "exclusion_reason": exclusion_reason,
                "trim_start_sec": "",
                "trim_end_sec": "",
                "auto_trim_start_sec": row["trim_from_start_sec"],
                "auto_trim_end_sec": row["trim_from_end_sec"],
                "approved_trim_start_sec": "",
                "approved_trim_end_sec": "",
                "notes": "; ".join(notes),
                "trim_method": "auto_plateau_v2_proposed",
                "trim_review_status": "review_required",
            }
        )
    return pd.DataFrame(rows, columns=BASE_MANIFEST_COLUMNS)


def _update_manifest(new_entries: pd.DataFrame) -> None:
    manifest_path = REPOSITORY_ROOT / "configs/dataset_manifest.csv"
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    for column in BASE_MANIFEST_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = ""
    manifest = manifest.loc[:, BASE_MANIFEST_COLUMNS]
    manifest = manifest.loc[
        ~manifest["recording_id"].astype(str).isin(set(new_entries["recording_id"].astype(str)))
    ].copy()
    manifest = pd.concat([manifest, new_entries], ignore_index=True)
    manifest.to_csv(manifest_path, index=False)


def main() -> None:
    table_dir = REPOSITORY_ROOT / "outputs/tables"
    figure_dir = REPOSITORY_ROOT / "outputs/figures/trim_diagnostics"
    table_dir.mkdir(parents=True, exist_ok=True)

    inventory_rows = []
    quality_rows = []
    boundary_rows = []
    for root in RAW_ROOTS:
        for path in sorted(root.iterdir()):
            if not _is_data_file(path):
                continue
            meta = _parse_metadata(path)
            raw = _read_csv(path)
            valid, valid_report = _valid_imu_frame(raw)
            quality = {
                **_quality_row(path, raw, valid, meta),
                **valid_report,
            }
            features = _window_feature_series(valid, str(meta["recording_id"]))
            features.attrs["duration_label"] = str(meta["duration_label"])
            boundary = _propose_trim(features, float(quality["duration_samples_sec"]), str(meta["subject_id"]))
            inventory_rows.append(
                {
                    **meta,
                    "source_path": path.as_posix(),
                    "filename": path.name,
                    "file_size_bytes": path.stat().st_size,
                }
            )
            quality_rows.append(quality)
            boundary_rows.append({**meta, **boundary})
            _plot_diagnostics(str(meta["recording_id"]), features, boundary, figure_dir)

    inventory = pd.DataFrame(inventory_rows)
    quality = pd.DataFrame(quality_rows)
    boundaries = pd.DataFrame(boundary_rows)
    new_entries = _build_manifest_entries(boundaries, quality)
    _update_manifest(new_entries)

    inventory.to_csv(table_dir / "new_data_inventory.csv", index=False)
    quality.to_csv(table_dir / "new_data_quality_report.csv", index=False)
    boundaries.to_csv(table_dir / "proposed_trim_boundaries.csv", index=False)
    print(f"Inventory rows: {len(inventory)}")
    print(f"Quality rows: {len(quality)}")
    print(f"Proposed trim rows: {len(boundaries)}")
    print(f"Manifest entries written: {len(new_entries)}")
    print("Manual trim review required:")
    print(
        new_entries.loc[
            new_entries["trim_review_status"] == "review_required",
            [
                "recording_id",
                "include",
                "exclusion_reason",
                "auto_trim_start_sec",
                "auto_trim_end_sec",
                "approved_trim_start_sec",
                "approved_trim_end_sec",
            ],
        ].to_string(index=False)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        included = new_entries.loc[new_entries["include"] == "true", "recording_id"]
        failed = []
        for recording_id in included:
            try:
                load_recording(str(recording_id))
            except Exception as error:  # noqa: BLE001
                failed.append((recording_id, str(error)))
        if failed:
            raise RuntimeError(f"Manifest load failures: {failed}")
    print(f"Confirmed {len(included)} new included recordings load through manifest.")


if __name__ == "__main__":
    main()
