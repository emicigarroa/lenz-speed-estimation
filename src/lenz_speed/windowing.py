"""Sample-based trimming and windowing for LENZ IMU recordings.

All boundaries are calculated from row positions and the supplied sampling
rate. Raw timestamps and DataFrame index labels are intentionally ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import pandas as pd


@dataclass(frozen=True)
class SignalWindow:
    """One fixed-length signal window with recording provenance.

    Attributes
    ----------
    recording_id:
        Stable recording identifier from the dataset manifest.
    window_index:
        Zero-based position of this window within the provided recording.
    window_start_sec, window_end_sec:
        Half-open time boundaries relative to the first row of the provided
        DataFrame. The boundaries are derived from sample positions and
        ``fs``, not raw timestamp columns.
    signal:
        Copy of the input rows in this window, with a fresh zero-based index.
    """

    recording_id: str
    window_index: int
    window_start_sec: float
    window_end_sec: float
    signal: pd.DataFrame


def _validate_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")


def _validate_sampling_rate(fs: Real) -> float:
    if isinstance(fs, bool) or not isinstance(fs, Real):
        raise TypeError("fs must be a positive number.")
    fs_float = float(fs)
    if not math.isfinite(fs_float) or fs_float <= 0:
        raise ValueError("fs must be finite and greater than zero.")
    return fs_float


def _seconds_to_samples(
    seconds: Real,
    *,
    fs: float,
    name: str,
    allow_zero: bool,
) -> int:
    if isinstance(seconds, bool) or not isinstance(seconds, Real):
        raise TypeError(f"{name} must be a number.")

    seconds_float = float(seconds)
    minimum_is_valid = seconds_float >= 0 if allow_zero else seconds_float > 0
    if not math.isfinite(seconds_float) or not minimum_is_valid:
        comparison = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"{name} must be finite and {comparison}.")

    exact_samples = seconds_float * fs
    rounded_samples = round(exact_samples)
    if not math.isclose(exact_samples, rounded_samples, rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            f"{name}={seconds_float} does not correspond to a whole number of "
            f"samples at fs={fs}."
        )
    if not allow_zero and rounded_samples == 0:
        raise ValueError(f"{name} must span at least one sample at fs={fs}.")
    return rounded_samples


def apply_trim(
    df: pd.DataFrame,
    trim_start_sec: Real,
    trim_end_sec: Real,
    fs: Real = 200,
) -> pd.DataFrame:
    """Remove fixed durations from the start and end of a recording.

    Trimming uses positional rows only. At 200 Hz, for example,
    ``trim_start_sec=5`` removes the first 1,000 rows. The returned DataFrame is
    a copy with a fresh zero-based index; the input is not modified.

    Parameters
    ----------
    df:
        Recording samples in chronological row order.
    trim_start_sec, trim_end_sec:
        Non-negative durations to remove. Each duration must resolve to a whole
        number of samples at ``fs``.
    fs:
        Sampling rate in hertz.

    Raises
    ------
    ValueError
        If the trim durations are invalid or remove more rows than are present.
    """

    _validate_dataframe(df)
    fs_float = _validate_sampling_rate(fs)
    start_samples = _seconds_to_samples(
        trim_start_sec,
        fs=fs_float,
        name="trim_start_sec",
        allow_zero=True,
    )
    end_samples = _seconds_to_samples(
        trim_end_sec,
        fs=fs_float,
        name="trim_end_sec",
        allow_zero=True,
    )

    if start_samples + end_samples > len(df):
        raise ValueError(
            "Requested trimming removes more samples than the DataFrame "
            f"contains: {start_samples} + {end_samples} > {len(df)}."
        )

    stop = len(df) - end_samples if end_samples else len(df)
    return df.iloc[start_samples:stop].copy().reset_index(drop=True)


def make_windows(
    df: pd.DataFrame,
    recording_id: str,
    window_sec: Real = 5,
    step_sec: Real = 2.5,
    fs: Real = 200,
) -> list[SignalWindow]:
    """Split a recording into complete, fixed-length sample windows.

    At the defaults, each window contains 1,000 samples and consecutive
    windows begin 500 samples apart. Any final segment shorter than a full
    window is dropped. Window times are relative to the first row of ``df`` and
    are calculated from sample positions, not timestamps or index labels.

    Parameters
    ----------
    df:
        Recording samples in chronological row order.
    recording_id:
        Non-empty identifier attached to every returned window.
    window_sec:
        Window duration in seconds.
    step_sec:
        Distance in seconds between consecutive window starts.
    fs:
        Sampling rate in hertz.

    Returns
    -------
    list[SignalWindow]
        Complete windows in chronological order. An input shorter than one
        window returns an empty list.
    """

    _validate_dataframe(df)
    if not isinstance(recording_id, str) or not recording_id.strip():
        raise ValueError("recording_id must be a non-empty string.")

    if "recording_id" in df.columns:
        observed_ids = set(df["recording_id"].dropna().astype(str).unique())
        if observed_ids and observed_ids != {recording_id}:
            raise ValueError(
                f"recording_id={recording_id!r} does not match values in df: "
                f"{sorted(observed_ids)!r}."
            )

    fs_float = _validate_sampling_rate(fs)
    window_samples = _seconds_to_samples(
        window_sec,
        fs=fs_float,
        name="window_sec",
        allow_zero=False,
    )
    step_samples = _seconds_to_samples(
        step_sec,
        fs=fs_float,
        name="step_sec",
        allow_zero=False,
    )

    windows: list[SignalWindow] = []
    complete_stop = len(df) - window_samples + 1
    for window_index, start in enumerate(range(0, max(complete_stop, 0), step_samples)):
        end = start + window_samples
        windows.append(
            SignalWindow(
                recording_id=recording_id,
                window_index=window_index,
                window_start_sec=start / fs_float,
                window_end_sec=end / fs_float,
                signal=df.iloc[start:end].copy().reset_index(drop=True),
            )
        )
    return windows

