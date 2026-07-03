"""Load LENZ IMU recordings through the project dataset manifest.

This module is intentionally limited to file discovery, schema normalization,
and metadata attachment. It does not trim signals, create windows, extract
features, or fit models.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd


CANONICAL_SIGNAL_COLUMNS = (
    "ax_g",
    "ay_g",
    "az_g",
    "gx_dps",
    "gy_dps",
    "gz_dps",
)

MANIFEST_COLUMNS = (
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
    "notes",
)

_COLUMN_ALIASES = {
    "ax_g": ("ax_g", "AccX(g)"),
    "ay_g": ("ay_g", "AccY(g)"),
    "az_g": ("az_g", "AccZ(g)"),
    "gx_dps": ("gx_dps", "AsX(¬∞/s)", "AsX(°/s)"),
    "gy_dps": ("gy_dps", "AsY(¬∞/s)", "AsY(°/s)"),
    "gz_dps": ("gz_dps", "AsZ(¬∞/s)", "AsZ(°/s)"),
}
_IMU_PACKET_TYPE = 61


class ManifestError(ValueError):
    """Raised when the dataset manifest is missing or invalid."""


class RecordingLoadError(ValueError):
    """Raised when a raw recording cannot be normalized safely."""


class DataQualityWarning(UserWarning):
    """Warn that non-IMU or invalid source rows were excluded during loading."""


def _default_repository_root() -> Path:
    """Return the repository root for the package's ``src`` layout."""

    return Path(__file__).resolve().parents[2]


def _resolve_repository_root(repository_root: str | Path | None) -> Path:
    root = Path(repository_root) if repository_root is not None else _default_repository_root()
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ManifestError(f"Repository root does not exist or is not a directory: {root}")
    return root


def _resolve_manifest_path(manifest_path: str | Path | None, root: Path) -> Path:
    path = (
        Path(manifest_path)
        if manifest_path is not None
        else Path("configs/dataset_manifest.csv")
    )
    if not path.is_absolute():
        path = root / path
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ManifestError(f"Dataset manifest not found: {path}")
    return path


def _parse_include(value: Any, *, row_number: int) -> bool:
    """Convert a manifest include value to a strict boolean."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ManifestError(
        f"Invalid include value {value!r} in manifest row {row_number}; "
        "expected true or false."
    )


def _resolve_recording_path(relative_path: Any, root: Path, *, recording_id: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ManifestError(f"Recording {recording_id!r} has an empty relative_path.")

    manifest_path = Path(relative_path.strip())
    if manifest_path.is_absolute():
        raise ManifestError(
            f"Recording {recording_id!r} uses an absolute path; relative_path must be "
            "relative to the repository root."
        )

    resolved = (root / manifest_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ManifestError(
            f"Recording {recording_id!r} resolves outside the repository root: "
            f"{relative_path!r}"
        ) from error
    return resolved


def load_manifest(
    manifest_path: str | Path | None = None,
    *,
    include_excluded: bool = False,
    repository_root: str | Path | None = None,
) -> pd.DataFrame:
    """Read and validate the dataset manifest.

    Parameters
    ----------
    manifest_path:
        Manifest path. Relative paths are resolved from ``repository_root``.
        The default is ``configs/dataset_manifest.csv``.
    include_excluded:
        When false (the default), omit rows whose ``include`` value is false.
    repository_root:
        Project root used to resolve manifest and recording paths. It defaults
        to the root inferred from this package's ``src`` layout.

    Returns
    -------
    pandas.DataFrame
        Validated manifest rows with a boolean ``include`` column and a
        ``resolved_path`` column containing absolute :class:`~pathlib.Path`
        objects. Raw files are not opened or modified.
    """

    root = _resolve_repository_root(repository_root)
    path = _resolve_manifest_path(manifest_path, root)
    manifest = pd.read_csv(path, keep_default_na=False)

    missing = [column for column in MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        raise ManifestError(
            f"Manifest {path} is missing required columns: {', '.join(missing)}"
        )

    duplicate_ids = manifest.loc[
        manifest["recording_id"].duplicated(keep=False), "recording_id"
    ].tolist()
    if duplicate_ids:
        raise ManifestError(
            f"Manifest {path} contains duplicate recording_id values: "
            f"{', '.join(map(str, sorted(set(duplicate_ids))))}"
        )

    manifest = manifest.loc[:, MANIFEST_COLUMNS].copy()
    manifest["include"] = [
        _parse_include(value, row_number=index + 2)
        for index, value in enumerate(manifest["include"])
    ]
    for column in ("speed_mph", "trim_start_sec", "trim_end_sec"):
        try:
            manifest[column] = pd.to_numeric(
                manifest[column].replace("", pd.NA), errors="raise"
            )
        except (TypeError, ValueError) as error:
            raise ManifestError(
                f"Manifest {path} contains a non-numeric value in {column!r}."
            ) from error
    manifest["resolved_path"] = [
        _resolve_recording_path(row.relative_path, root, recording_id=str(row.recording_id))
        for row in manifest.itertuples(index=False)
    ]

    if not include_excluded:
        manifest = manifest.loc[manifest["include"]]
    return manifest.reset_index(drop=True)


def _signal_column_mapping(raw: pd.DataFrame, *, source_path: Path) -> dict[Any, str]:
    """Map source signal columns to their canonical names."""

    stripped_to_original: dict[str, Any] = {}
    for column in raw.columns:
        stripped = str(column).strip().lstrip("\ufeff")
        if stripped in stripped_to_original:
            raise RecordingLoadError(
                f"Recording {source_path} contains duplicate columns after whitespace "
                f"normalization: {stripped!r}."
            )
        stripped_to_original[stripped] = column

    rename: dict[Any, str] = {}
    missing: list[str] = []
    for canonical, aliases in _COLUMN_ALIASES.items():
        matches = [
            stripped_to_original[alias]
            for alias in aliases
            if alias in stripped_to_original
        ]
        if not matches:
            missing.append(f"{canonical} (expected one of: {', '.join(aliases)})")
        elif len(matches) > 1:
            raise RecordingLoadError(
                f"Recording {source_path} contains multiple columns for {canonical}: "
                f"{', '.join(map(str, matches))}."
            )
        else:
            rename[matches[0]] = canonical

    if missing:
        available = ", ".join(map(str, raw.columns))
        raise RecordingLoadError(
            f"Recording {source_path} is missing required signal columns: "
            f"{'; '.join(missing)}. Available columns: {available}"
        )
    return rename


def _filter_csv_imu_rows(
    raw: pd.DataFrame,
    *,
    source_path: Path,
    recording_id: str,
) -> pd.DataFrame:
    """Remove non-IMU and invalid rows from a mixed-packet CSV recording."""

    rename = _signal_column_mapping(raw, source_path=source_path)
    normalized_names = {
        str(column).strip().lstrip("\ufeff"): column for column in raw.columns
    }

    numeric_signals: dict[Any, pd.Series] = {}
    finite_signals = np.ones(len(raw), dtype=bool)
    for source_column in rename:
        numeric = pd.to_numeric(raw[source_column], errors="coerce")
        numeric_signals[source_column] = numeric
        finite_signals &= np.isfinite(numeric.to_numpy(dtype=float))

    parse_ok = np.ones(len(raw), dtype=bool)
    parse_column = normalized_names.get("parse_ok")
    if parse_column is not None:
        normalized_parse = (
            raw[parse_column].astype("string").str.strip().str.lower()
        )
        parse_ok = normalized_parse.isin({"true", "1", "yes"}).to_numpy()

    imu_packet = np.ones(len(raw), dtype=bool)
    packet_column = normalized_names.get("packet_type")
    if packet_column is not None:
        packet_type = pd.to_numeric(raw[packet_column], errors="coerce")
        imu_packet = packet_type.eq(_IMU_PACKET_TYPE).to_numpy()

    valid = finite_signals & parse_ok & imu_packet
    dropped_count = int((~valid).sum())
    if dropped_count:
        reason_counts = [
            f"{int((~finite_signals).sum())} with non-numeric/non-finite signals"
        ]
        if parse_column is not None:
            reason_counts.append(f"{int((~parse_ok).sum())} parse_ok failures")
        if packet_column is not None:
            reason_counts.append(f"{int((~imu_packet).sum())} non-IMU packets")
        drop_percent = dropped_count / len(raw) * 100 if len(raw) else 0.0
        warnings.warn(
            f"Recording {recording_id!r}: dropped {dropped_count} of {len(raw)} "
            f"CSV rows ({drop_percent:.3f}%) before sample indexing: "
            f"{', '.join(reason_counts)}. Reason counts may overlap.",
            DataQualityWarning,
            stacklevel=2,
        )

    if len(raw) and not valid.any():
        raise RecordingLoadError(
            f"Recording {recording_id!r} has no valid IMU rows after CSV "
            "packet and signal validation."
        )

    filtered = raw.loc[valid].copy()
    for source_column, numeric in numeric_signals.items():
        filtered[source_column] = numeric.loc[valid].to_numpy()
    return filtered.reset_index(drop=True)


def _normalize_signal_columns(raw: pd.DataFrame, *, source_path: Path) -> pd.DataFrame:
    """Select and rename raw IMU channels to the canonical signal schema."""

    rename = _signal_column_mapping(raw, source_path=source_path)

    normalized = raw.rename(columns=rename).loc[:, CANONICAL_SIGNAL_COLUMNS].copy()
    for column in CANONICAL_SIGNAL_COLUMNS:
        try:
            normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise RecordingLoadError(
                f"Recording {source_path} contains non-numeric values in {column!r}."
            ) from error
    return normalized


def _read_recording_row(row: Mapping[str, Any]) -> pd.DataFrame:
    source_path = Path(row["resolved_path"])
    recording_id = str(row["recording_id"])
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Raw file for recording {recording_id!r} was not found: {source_path}"
        )

    file_type = str(row["file_type"]).strip().lower()
    try:
        if file_type == "csv":
            raw = pd.read_csv(source_path)
            raw = _filter_csv_imu_rows(
                raw,
                source_path=source_path,
                recording_id=recording_id,
            )
        elif file_type == "xlsx":
            raw = pd.read_excel(source_path)
        else:
            raise RecordingLoadError(
                f"Recording {recording_id!r} has unsupported file_type {file_type!r}; "
                "expected 'csv' or 'xlsx'."
            )
    except RecordingLoadError:
        raise
    except Exception as error:
        raise RecordingLoadError(
            f"Failed to read recording {recording_id!r} from {source_path}: {error}"
        ) from error

    normalized = _normalize_signal_columns(raw, source_path=source_path)
    for column in MANIFEST_COLUMNS:
        normalized[column] = row[column]
    return normalized.loc[:, (*CANONICAL_SIGNAL_COLUMNS, *MANIFEST_COLUMNS)]


def load_recording(
    recording_id: str,
    manifest_path: str | Path | None = None,
    *,
    include_excluded: bool = False,
    repository_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load one recording and attach its manifest metadata to every sample.

    Excluded recordings are unavailable by default. Pass
    ``include_excluded=True`` to load one explicitly. Trim values are attached
    as metadata but are not applied by this module.
    """

    manifest = load_manifest(
        manifest_path,
        include_excluded=include_excluded,
        repository_root=repository_root,
    )
    matches = manifest.loc[manifest["recording_id"] == recording_id]
    if matches.empty:
        qualifier = " among included rows" if not include_excluded else ""
        raise ManifestError(f"Recording {recording_id!r} was not found{qualifier}.")
    return _read_recording_row(matches.iloc[0].to_dict())


def load_dataset(
    manifest_path: str | Path | None = None,
    *,
    include_excluded: bool = False,
    repository_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load and concatenate recordings selected by the dataset manifest.

    By default, rows with ``include=false`` are ignored. The returned frame has
    the six canonical IMU columns followed by all manifest metadata columns.
    No trimming, windowing, feature extraction, or modeling is performed.
    """

    manifest = load_manifest(
        manifest_path,
        include_excluded=include_excluded,
        repository_root=repository_root,
    )
    frames = [_read_recording_row(row) for row in manifest.to_dict(orient="records")]
    if not frames:
        return pd.DataFrame(columns=(*CANONICAL_SIGNAL_COLUMNS, *MANIFEST_COLUMNS))
    return pd.concat(frames, ignore_index=True)
