# Declared analysis (fill in and commit BEFORE any test split is opened)

- **Primary comparison:** sensitivity for early DR (grades 1-2) vs No DR on the DDR test set, at the declared specificity.
- **Declared specificity:** 0.90 (also set in `configs/base.yaml` -> `declared.specificity`; the threshold is fitted on validation and frozen).
- **Secondary:** AUROC and sensitivity for late DR (3-4) among DR-positive images.
- **Robustness:** the same metrics on held-out EyePACS patients, IDRiD and APTOS.
- **Also reported:** grade 1 vs No DR, grade 2+ vs the rest, PR-AUC (with test prevalence), ECE, Brier, per-grade recall.
- **Statistics:** patient-level bootstrap 95% CIs (1000 resamples), 3-5 seeds for the shortlist, Holm correction across comparisons.
- **Arm comparison:** B vs A and C vs B on DDR test; C is the negative control (see README).
- **Date declared:** ____
