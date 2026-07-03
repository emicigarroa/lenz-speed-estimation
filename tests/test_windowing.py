"""Tests for sample-based trimming and window generation."""

import pandas as pd

from lenz_speed.windowing import apply_trim, make_windows


def _signal_frame(sample_count: int) -> pd.DataFrame:
    return pd.DataFrame({"sample": range(sample_count)})


def test_apply_trim_removes_correct_number_of_samples() -> None:
    signal = _signal_frame(2_000)

    trimmed = apply_trim(signal, trim_start_sec=2, trim_end_sec=1, fs=200)

    assert len(trimmed) == 1_400
    assert trimmed.index.tolist() == list(range(1_400))
    assert trimmed["sample"].iloc[0] == 400
    assert trimmed["sample"].iloc[-1] == 1_799
    assert len(signal) == 2_000


def test_make_windows_uses_1000_samples_and_500_sample_step() -> None:
    signal = _signal_frame(2_500)

    windows = make_windows(signal, recording_id="synthetic", fs=200)

    assert len(windows) == 4
    assert all(len(window.signal) == 1_000 for window in windows)
    assert [window.signal["sample"].iloc[0] for window in windows] == [
        0,
        500,
        1_000,
        1_500,
    ]
    assert [window.window_start_sec for window in windows] == [0.0, 2.5, 5.0, 7.5]
    assert [window.window_end_sec for window in windows] == [5.0, 7.5, 10.0, 12.5]


def test_make_windows_drops_incomplete_final_window() -> None:
    signal = _signal_frame(2_499)

    windows = make_windows(signal, recording_id="synthetic", fs=200)

    assert len(windows) == 3
    assert windows[-1].signal["sample"].iloc[0] == 1_000
    assert windows[-1].signal["sample"].iloc[-1] == 1_999

