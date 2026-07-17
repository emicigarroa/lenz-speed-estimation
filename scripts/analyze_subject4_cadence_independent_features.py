"""Analyze Subject 4 features that track speed despite near-fixed cadence.

The analysis is intentionally read-only with respect to raw data and production
model code. It uses the already-built approved steady-state window table and
generates interpretation artifacts for Subject 4 normal-cadence windows from
5.0 to 8.5 mph.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import textwrap
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lenz-speed-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FEATURE_TABLE_PATH = REPOSITORY_ROOT / "data/processed/windowed_features.csv"
MANIFEST_PATH = REPOSITORY_ROOT / "configs/dataset_manifest.csv"
TABLE_DIR = REPOSITORY_ROOT / "outputs/tables"
FIGURE_DIR = REPOSITORY_ROOT / "outputs/figures"
FEATURE_FIGURE_DIR = FIGURE_DIR / "subject4_feature_vs_speed"
MECHANICS_PATH = TABLE_DIR / "subject4_feature_speed_mechanics.csv"
RANKINGS_PATH = TABLE_DIR / "subject4_feature_rankings.csv"
SUMMARY_PATH = TABLE_DIR / "subject4_feature_summary_by_speed.csv"
REPORT_PATH = TABLE_DIR / "subject4_cadence_independent_feature_report.md"

METADATA_COLUMNS = {
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
}
PRIORITY_FEATURES = [
    "Cadence_spm",
    "Dynamic_Accel_Mag_RMS",
    "Accel_Mag_RMS",
    "RMS_Z",
    "PeakToPeak_Z",
    "Vertical_Peak_Sharpness",
    "Impact_Impulse",
    "Peak_Symmetry",
    "Impact_Crest_Factor",
    "Impact_Local_Kurtosis",
    "Accel_Mag_Jerk_RMS",
    "Accel_HighFreq_Energy_Ratio",
    "Gyro_Mag_RMS",
    "Gyro_RMS_X",
    "Gyro_RMS_Y",
    "Gyro_RMS_Z",
    "GyroY_PeakToPeak",
    "Accel_Anisotropy",
    "Z_Displacement_Proxy",
]
SCORE_FORMULA = (
    "0.30*abs_spearman_speed + 0.25*linear_r2 + 0.20*monotonicity_score "
    "+ 0.15*(1-abs_cadence_corr) + 0.10*within_speed_stability_score"
)


@dataclass(frozen=True)
class FeatureStats:
    """Container for feature-level mechanics and ranking values."""

    feature: str
    n_windows: int
    n_speed_bins: int
    speed_min_mph: float
    speed_max_mph: float
    feature_mean: float
    feature_std: float
    cadence_mean_spm: float
    cadence_std_spm: float
    cadence_range_spm: float
    pearson_r_speed: float
    pearson_p_speed: float
    spearman_rho_speed: float
    spearman_p_speed: float
    pearson_r_cadence: float
    pearson_p_cadence: float
    spearman_rho_cadence: float
    spearman_p_cadence: float
    partial_corr_speed_control_cadence: float
    partial_corr_p_value: float
    partial_corr_caution: str
    linear_slope_per_mph: float
    linear_intercept: float
    linear_r2: float
    linear_p_value: float
    monotonicity_score: float
    monotonic_direction: str
    within_speed_cv_median: float
    within_speed_cv_mean: float
    within_speed_stability_score: float
    within_speed_noise_flag: str
    between_speed_separation: float
    speed_mean_range: float
    high_speed_saturation_flag: str
    abs_spearman_speed: float
    abs_cadence_corr: float
    cadence_independent_score: float
    priority_feature: bool


def _safe_name(feature: str) -> str:
    """Return a filesystem-safe feature name."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", feature).strip("_")


def _numeric_features(table: pd.DataFrame) -> list[str]:
    """Return all numeric feature columns in a stable priority-first order."""

    numeric = [
        column
        for column in table.columns
        if column not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(table[column])
    ]
    priority = [feature for feature in PRIORITY_FEATURES if feature in numeric]
    remaining = sorted(column for column in numeric if column not in priority)
    return [*priority, *remaining]


def _subject4_windows() -> pd.DataFrame:
    """Load approved Subject 4 normal-cadence steady-state windows."""

    table = pd.read_csv(FEATURE_TABLE_PATH)
    manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    approved_subject4 = manifest.loc[
        (manifest["subject_id"] == "subject_4")
        & (manifest["include"].astype(str).str.lower() == "true")
        & (manifest["trim_review_status"].astype(str).str.lower() == "approved"),
        "recording_id",
    ].astype(str)
    subset = table.loc[
        table["recording_id"].astype(str).isin(set(approved_subject4))
        & (table["subject_id"] == "subject_4")
        & (table["condition"] == "normal")
        & table["speed_mph"].between(5.0, 8.5)
    ].copy()
    if subset.empty:
        raise ValueError("No approved Subject 4 windows found for 5.0--8.5 mph.")
    return subset


def _corr(x: np.ndarray, y: np.ndarray, *, method: str) -> tuple[float, float]:
    """Return correlation and p-value, handling constant arrays."""

    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan
    if method == "pearson":
        result = stats.pearsonr(x, y)
    elif method == "spearman":
        result = stats.spearmanr(x, y)
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    return float(result.statistic), float(result.pvalue)


def _partial_corr(feature: np.ndarray, speed: np.ndarray, cadence: np.ndarray) -> tuple[float, float]:
    """Partial correlation between feature and speed after removing cadence."""

    if len(feature) < 4 or np.nanstd(cadence) == 0:
        return np.nan, np.nan
    cadence_2d = cadence.reshape(-1, 1)
    feature_residual = feature - LinearRegression().fit(cadence_2d, feature).predict(cadence_2d)
    speed_residual = speed - LinearRegression().fit(cadence_2d, speed).predict(cadence_2d)
    return _corr(feature_residual, speed_residual, method="pearson")


def _linear_regression(feature: np.ndarray, speed: np.ndarray) -> tuple[float, float, float, float]:
    """Fit feature = intercept + slope * speed."""

    if len(feature) < 3 or np.nanstd(speed) == 0 or np.nanstd(feature) == 0:
        return np.nan, np.nan, np.nan, np.nan
    result = stats.linregress(speed, feature)
    return (
        float(result.slope),
        float(result.intercept),
        float(result.rvalue**2),
        float(result.pvalue),
    )


def _monotonicity(speed_summary: pd.DataFrame, slope: float) -> tuple[float, str]:
    """Score how consistently feature medians move across ordered speed bins."""

    ordered = speed_summary.sort_values("speed_mph")
    values = ordered["median"].to_numpy(dtype=float)
    diffs = np.diff(values)
    valid = diffs[np.isfinite(diffs) & (np.abs(diffs) > 1e-12)]
    if len(valid) == 0 or not np.isfinite(slope):
        return np.nan, "flat_or_unstable"
    direction = "increasing" if slope >= 0 else "decreasing"
    score = float(np.mean(valid > 0)) if slope >= 0 else float(np.mean(valid < 0))
    return score, direction


def _high_speed_saturation(speed_summary: pd.DataFrame) -> str:
    """Flag features that flatten in the upper half of Subject 4 speeds."""

    ordered = speed_summary.sort_values("speed_mph")
    speeds = ordered["speed_mph"].to_numpy(dtype=float)
    medians = ordered["median"].to_numpy(dtype=float)
    if len(medians) < 6:
        return "insufficient_speed_bins"
    split = np.median(speeds)
    low = ordered.loc[ordered["speed_mph"] <= split]
    high = ordered.loc[ordered["speed_mph"] > split]
    low_range = float(low["median"].max() - low["median"].min())
    high_range = float(high["median"].max() - high["median"].min())
    total_range = float(ordered["median"].max() - ordered["median"].min())
    if total_range <= 0:
        return "flat"
    if high_range < 0.25 * low_range and high_range < 0.20 * total_range:
        return "possible_high_speed_saturation"
    return "none"


def _summary_by_speed(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Calculate per-speed summary rows for every feature."""

    rows: list[dict[str, float | str | int]] = []
    for feature in features:
        for speed, group in table.groupby("speed_mph", sort=True):
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            rows.append(
                {
                    "feature": feature,
                    "speed_mph": float(speed),
                    "n_windows": int(len(values)),
                    "mean": mean,
                    "median": float(values.median()),
                    "std": std,
                    "cv": float(std / abs(mean)) if mean != 0 and np.isfinite(std) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _feature_stats(table: pd.DataFrame, summary: pd.DataFrame, feature: str) -> FeatureStats:
    """Calculate all feature-level metrics."""

    columns = list(dict.fromkeys(["speed_mph", "Cadence_spm", feature]))
    clean = table.loc[:, columns].apply(pd.to_numeric, errors="coerce").dropna()
    speed = clean["speed_mph"].to_numpy(dtype=float)
    cadence = clean["Cadence_spm"].to_numpy(dtype=float)
    values = clean[feature].to_numpy(dtype=float)
    feature_summary = summary.loc[summary["feature"] == feature].copy()

    pearson_speed, pearson_speed_p = _corr(values, speed, method="pearson")
    spearman_speed, spearman_speed_p = _corr(values, speed, method="spearman")
    pearson_cadence, pearson_cadence_p = _corr(values, cadence, method="pearson")
    spearman_cadence, spearman_cadence_p = _corr(values, cadence, method="spearman")
    partial_r, partial_p = _partial_corr(values, speed, cadence)
    slope, intercept, r2, p_value = _linear_regression(values, speed)
    monotonic_score, monotonic_direction = _monotonicity(feature_summary, slope)

    cv_values = feature_summary["cv"].replace([np.inf, -np.inf], np.nan).dropna().abs()
    cv_median = float(cv_values.median()) if len(cv_values) else np.nan
    cv_mean = float(cv_values.mean()) if len(cv_values) else np.nan
    stability_score = float(1.0 / (1.0 + cv_median)) if np.isfinite(cv_median) else np.nan
    mean_within_std = float(feature_summary["std"].dropna().mean())
    speed_mean_range = float(feature_summary["mean"].max() - feature_summary["mean"].min())
    separation = (
        float(speed_mean_range / mean_within_std)
        if np.isfinite(mean_within_std) and mean_within_std > 0
        else np.nan
    )

    abs_spearman = abs(float(spearman_speed)) if np.isfinite(spearman_speed) else 0.0
    abs_cadence = abs(float(pearson_cadence)) if np.isfinite(pearson_cadence) else 1.0
    r2_component = float(r2) if np.isfinite(r2) else 0.0
    monotonic_component = float(monotonic_score) if np.isfinite(monotonic_score) else 0.0
    stability_component = float(stability_score) if np.isfinite(stability_score) else 0.0
    score = (
        0.30 * abs_spearman
        + 0.25 * r2_component
        + 0.20 * monotonic_component
        + 0.15 * (1.0 - min(abs_cadence, 1.0))
        + 0.10 * stability_component
    )

    cadence_range = float(np.nanmax(cadence) - np.nanmin(cadence))
    partial_caution = (
        "unstable: Subject 4 cadence range is narrow"
        if cadence_range < 10
        else "interpret normally"
    )
    noise_flag = "high_within_speed_noise" if np.isfinite(cv_median) and cv_median > 0.25 else "none"

    return FeatureStats(
        feature=feature,
        n_windows=int(len(clean)),
        n_speed_bins=int(clean["speed_mph"].nunique()),
        speed_min_mph=float(clean["speed_mph"].min()),
        speed_max_mph=float(clean["speed_mph"].max()),
        feature_mean=float(np.nanmean(values)),
        feature_std=float(np.nanstd(values, ddof=1)),
        cadence_mean_spm=float(np.nanmean(cadence)),
        cadence_std_spm=float(np.nanstd(cadence, ddof=1)),
        cadence_range_spm=cadence_range,
        pearson_r_speed=pearson_speed,
        pearson_p_speed=pearson_speed_p,
        spearman_rho_speed=spearman_speed,
        spearman_p_speed=spearman_speed_p,
        pearson_r_cadence=pearson_cadence,
        pearson_p_cadence=pearson_cadence_p,
        spearman_rho_cadence=spearman_cadence,
        spearman_p_cadence=spearman_cadence_p,
        partial_corr_speed_control_cadence=partial_r,
        partial_corr_p_value=partial_p,
        partial_corr_caution=partial_caution,
        linear_slope_per_mph=slope,
        linear_intercept=intercept,
        linear_r2=r2,
        linear_p_value=p_value,
        monotonicity_score=monotonic_score,
        monotonic_direction=monotonic_direction,
        within_speed_cv_median=cv_median,
        within_speed_cv_mean=cv_mean,
        within_speed_stability_score=stability_score,
        within_speed_noise_flag=noise_flag,
        between_speed_separation=separation,
        speed_mean_range=speed_mean_range,
        high_speed_saturation_flag=_high_speed_saturation(feature_summary),
        abs_spearman_speed=abs_spearman,
        abs_cadence_corr=abs_cadence,
        cadence_independent_score=score,
        priority_feature=feature in PRIORITY_FEATURES,
    )


def _plot_feature(table: pd.DataFrame, summary: pd.DataFrame, stats_row: pd.Series) -> Path:
    """Create one feature-vs-speed plot with trend statistics."""

    feature = str(stats_row["feature"])
    feature_summary = summary.loc[summary["feature"] == feature].sort_values("speed_mph")
    columns = list(dict.fromkeys(["speed_mph", "Cadence_spm", feature]))
    clean = table[columns].apply(pd.to_numeric, errors="coerce").dropna()
    x = clean["speed_mph"].to_numpy(dtype=float)
    y = clean[feature].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, s=16, alpha=0.35, color="#4C78A8", label="window")
    ax.errorbar(
        feature_summary["speed_mph"],
        feature_summary["mean"],
        yerr=feature_summary["std"],
        fmt="o-",
        color="#E15759",
        ecolor="#E15759",
        elinewidth=1,
        capsize=3,
        label="mean ± SD",
    )
    if np.isfinite(stats_row["linear_slope_per_mph"]):
        xfit = np.linspace(float(x.min()), float(x.max()), 100)
        yfit = float(stats_row["linear_intercept"]) + float(stats_row["linear_slope_per_mph"]) * xfit
        ax.plot(xfit, yfit, color="#222222", linewidth=1.5, label="linear fit")
    text = (
        f"Pearson r={stats_row['pearson_r_speed']:.3f}\n"
        f"Spearman ρ={stats_row['spearman_rho_speed']:.3f}\n"
        f"R²={stats_row['linear_r2']:.3f}\n"
        f"cadence r={stats_row['pearson_r_cadence']:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.9},
    )
    ax.set_title(f"Subject 4: {feature} vs speed")
    ax.set_xlabel("Treadmill speed (mph)")
    ax.set_ylabel(feature)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    path = FEATURE_FIGURE_DIR / f"{_safe_name(feature)}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_ranking(rankings: pd.DataFrame) -> Path:
    """Plot cadence-independent score for all features."""

    ordered = rankings.sort_values("cadence_independent_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(ordered))))
    ax.barh(ordered["feature"], ordered["cadence_independent_score"], color="#59A14F")
    ax.set_xlabel("Cadence-independent score")
    ax.set_title("Subject 4 cadence-independent feature ranking")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = FIGURE_DIR / "subject4_cadence_independent_feature_ranking.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_top_trends(summary: pd.DataFrame, rankings: pd.DataFrame, top_n: int = 8) -> Path:
    """Plot normalized speed-bin trends for top-ranked features."""

    top = rankings.head(top_n)["feature"].tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    for feature in top:
        feature_summary = summary.loc[summary["feature"] == feature].sort_values("speed_mph")
        values = feature_summary["median"].to_numpy(dtype=float)
        spread = np.nanmax(values) - np.nanmin(values)
        if not np.isfinite(spread) or spread == 0:
            normalized = np.zeros_like(values)
        else:
            normalized = (values - np.nanmin(values)) / spread
        ax.plot(feature_summary["speed_mph"], normalized, marker="o", linewidth=1.5, label=feature)
    ax.set_xlabel("Treadmill speed (mph)")
    ax.set_ylabel("Normalized median feature value")
    ax.set_title("Subject 4 top cadence-independent feature trends")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    path = FIGURE_DIR / "subject4_top_feature_normalized_trends.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_heatmap(table: pd.DataFrame, features: list[str]) -> Path:
    """Plot a feature correlation heatmap including speed and cadence."""

    columns = ["speed_mph", "Cadence_spm", *features]
    corr = table[columns].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Subject 4 Spearman feature correlation heatmap")
    fig.colorbar(image, ax=ax, shrink=0.75, label="Spearman correlation")
    fig.tight_layout()
    path = FIGURE_DIR / "subject4_feature_correlation_heatmap.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_report(rankings: pd.DataFrame, mechanics: pd.DataFrame, table: pd.DataFrame) -> None:
    """Write a concise interpretation report."""

    top = rankings.head(8)
    cadence_proxies = mechanics.loc[
        mechanics["abs_cadence_corr"] >= 0.60,
        ["feature", "abs_cadence_corr", "spearman_rho_speed", "cadence_independent_score"],
    ].sort_values("abs_cadence_corr", ascending=False)
    noisy = mechanics.loc[
        mechanics["within_speed_noise_flag"] == "high_within_speed_noise",
        ["feature", "within_speed_cv_median", "cadence_independent_score"],
    ].sort_values("within_speed_cv_median", ascending=False)
    saturated = mechanics.loc[
        mechanics["high_speed_saturation_flag"] == "possible_high_speed_saturation",
        ["feature", "high_speed_saturation_flag", "cadence_independent_score"],
    ].sort_values("cadence_independent_score", ascending=False)
    take_forward = top.loc[
        (top["abs_cadence_corr"] < 0.60)
        & (top["monotonicity_score"] >= 0.70)
        & (top["within_speed_cv_median"] < 0.25)
    ].head(6)

    def markdown_rows(frame: pd.DataFrame, columns: list[str], max_rows: int = 8) -> str:
        if frame.empty:
            return "_None flagged._"
        subset = frame.loc[:, columns].head(max_rows).copy()
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join("---" for _ in columns) + " |"
        rows = []
        for _, row in subset.iterrows():
            values = []
            for column in columns:
                value = row[column]
                if isinstance(value, float):
                    values.append(f"{value:.4f}" if np.isfinite(value) else "nan")
                else:
                    values.append(str(value))
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join([header, divider, *rows])

    cadence_range = table["Cadence_spm"].max() - table["Cadence_spm"].min()
    report = f"""# Subject 4 cadence-independent feature analysis

Subject 4 is useful here because cadence is comparatively narrow while speed
changes from 5.0 to 8.5 mph. In the analyzed windows, cadence spans
{table['Cadence_spm'].min():.2f}--{table['Cadence_spm'].max():.2f} spm
(range {cadence_range:.2f} spm), so partial correlations that control for
cadence are reported but should be treated as unstable rather than definitive.

Scoring formula:

`{SCORE_FORMULA}`

The score is a ranking heuristic, not a statistical proof. Component metrics are
saved separately in `subject4_feature_speed_mechanics.csv`.

## 1. Best cadence-independent speed features

{markdown_rows(top, ['feature', 'cadence_independent_score', 'spearman_rho_speed', 'linear_r2', 'monotonicity_score', 'abs_cadence_corr', 'within_speed_cv_median'])}

## 2. Features that are mostly cadence proxies

These features have high absolute correlation with cadence in this Subject 4
subset. Because cadence range is narrow, this flag is conservative rather than
final.

{markdown_rows(cadence_proxies, ['feature', 'abs_cadence_corr', 'spearman_rho_speed', 'cadence_independent_score'])}

## 3. Features that are monotonic but may be subject-specific

Features with strong monotonicity and high speed correlation are promising, but
Subject 4 has unusually stable high-speed cadence. These should be checked
against Subjects 1--3 before being treated as universal biomechanics.

{markdown_rows(mechanics.sort_values(['monotonicity_score', 'abs_spearman_speed'], ascending=False), ['feature', 'monotonicity_score', 'spearman_rho_speed', 'linear_r2', 'abs_cadence_corr'])}

## 4. Features with high within-speed noise

{markdown_rows(noisy, ['feature', 'within_speed_cv_median', 'cadence_independent_score'])}

## 5. Features that appear saturated at higher speeds

{markdown_rows(saturated, ['feature', 'high_speed_saturation_flag', 'cadence_independent_score'])}

## 6. Features to take forward into Phase 2

{markdown_rows(take_forward, ['feature', 'cadence_independent_score', 'spearman_rho_speed', 'linear_r2', 'monotonicity_score', 'abs_cadence_corr'])}

## 7. Statistical cautions

- Subject 4 has only eight speed bins in the primary 5.0--8.5 mph range.
- Cadence varies narrowly, so partial correlation can become numerically
  unstable or misleading.
- These results identify candidates that track speed under Subject 4's
  cadence-controlled biomechanics; they should not be used alone to select
  production features or tune trims.
"""
    REPORT_PATH.write_text(textwrap.dedent(report), encoding="utf-8")


def main() -> None:
    """Run the Subject 4 cadence-independent feature analysis."""

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    table = _subject4_windows()
    features = _numeric_features(table)
    if "Z_Displacement_Proxy" not in features:
        print("Z_Displacement_Proxy is not available in the current feature table; skipping it.")

    summary = _summary_by_speed(table, features)
    mechanics = pd.DataFrame([_feature_stats(table, summary, feature).__dict__ for feature in features])
    rankings = mechanics.sort_values(
        ["cadence_independent_score", "abs_spearman_speed", "linear_r2"],
        ascending=False,
    ).reset_index(drop=True)
    rankings.insert(0, "rank", np.arange(1, len(rankings) + 1))
    rankings["scoring_formula"] = SCORE_FORMULA

    summary.to_csv(SUMMARY_PATH, index=False)
    mechanics.to_csv(MECHANICS_PATH, index=False)
    rankings.to_csv(RANKINGS_PATH, index=False)

    for _, row in mechanics.iterrows():
        _plot_feature(table, summary, row)
    _plot_ranking(rankings)
    _plot_top_trends(summary, rankings)
    _plot_heatmap(table, features)
    _write_report(rankings, mechanics, table)

    print(f"Analyzed {len(features)} features from {len(table)} Subject 4 windows.")
    print(f"Saved {SUMMARY_PATH.relative_to(REPOSITORY_ROOT)}")
    print(f"Saved {MECHANICS_PATH.relative_to(REPOSITORY_ROOT)}")
    print(f"Saved {RANKINGS_PATH.relative_to(REPOSITORY_ROOT)}")
    print(f"Saved {REPORT_PATH.relative_to(REPOSITORY_ROOT)}")
    print("Top 10 cadence-independent feature candidates:")
    print(
        rankings.loc[
            :9,
            [
                "rank",
                "feature",
                "cadence_independent_score",
                "spearman_rho_speed",
                "linear_r2",
                "monotonicity_score",
                "abs_cadence_corr",
                "within_speed_cv_median",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
