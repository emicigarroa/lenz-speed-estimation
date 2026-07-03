# Notebooks

Notebooks are for ordered exploration, quality review, and presentation of
experiments. Reusable data loading, windowing, feature extraction, splitting,
modeling, and evaluation logic belongs in `src/`, not only in notebooks.

Planned sequence:

1. `00_data_inventory.ipynb`
2. `01_signal_quality_and_trimming.ipynb`
3. `02_window_and_feature_validation.ipynb`
4. `03_same_subject_baseline.ipynb`
5. `04_feature_ablation.ipynb`
6. `05_cross_day_validation.ipynb`
7. `06_cross_subject_validation.ipynb`
8. `07_cadence_manipulation_analysis.ipynb`

Each notebook should run from a clean kernel, state its input dataset and split,
avoid hidden manual corrections, and save finalized figures or tables under
`outputs/`.
