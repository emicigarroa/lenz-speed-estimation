# Outputs

This directory contains generated analysis results and is not a source of
record. Results should be reproducible from raw data, documented metadata, and
versioned analysis code.

## `figures/`

Store generated plots such as signal-quality views, predicted-versus-actual
plots, residual plots, per-speed comparisons, and feature-ablation summaries.

## `tables/`

Store machine-readable result tables, including overall metrics, per-speed and
per-recording metrics, model comparisons, and feature-ablation results.

Use descriptive filenames that identify the experiment and dataset split. Do
not place manually edited results here, and do not rely on generated outputs as
the only record of model settings or data inclusion decisions.
