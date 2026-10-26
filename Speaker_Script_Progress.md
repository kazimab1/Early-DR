# Progress update (12 slides, about 11 min)

| # | Slide | Time | Say |
|---|---|---|---|
| 1 | Title | 0:15 | "Progress update on early detection of diabetic retinopathy, using DDR, EyePACS and APTOS." |
| 2 | Progress at a glance | 1:00 | "Done: data audit, the full pipeline on Kaggle, arms A, B and C with three seeds each, internal and APTOS evaluation, twelve imbalance runs, three extensions and the Grad-CAM check. Prepared but not yet run: a duplicate-leakage check and an image-property shortcut check. To do: an official or patient-level DDR split, missing metrics, re-evaluating APTOS. Main finding so far: the original merge teaches the model the source." |
| 3 | Data | 1:00 | "Grade mixes differ: early DR is 41% of DDR, 22% of EyePACS, 37% of APTOS. DDR has no official split or patient IDs in this copy, so I used a random split. EyePACS is split by patient. APTOS is external only." |
| 4 | EyePACS merge cost | 1:30 | "Topping up only the short grades makes the dataset name predict the grade: 79% of grades 1 and 3 come from EyePACS. A classifier tells DDR from EyePACS with 0.996 AUROC. So the merge is tested with three arms." |
| 5 | Methods | 0:45 | "Arms A, B and C, one change at a time, same model, image size and epochs." |
| 6 | Training | 0:45 | "EfficientNet-B0, 512 pixels, 10 epochs, three seeds, about six hours on a Kaggle GPU. Raw outputs are overconfident, so calibration is fitted on validation." |
| 7 | Key results | 2:00 | "On DDR all arms are alike. On held-out EyePACS, arm C has the highest sensitivity, 0.997, but the lowest PR-AUC, 0.43, below DDR-only A. It learned the source. Source-balanced B gets PR-AUC 0.80. These are preliminary: the DDR split is random, duplicates are unchecked, and APTOS scored 0.994 AUROC, which I do not claim yet." |
| 8 | Imbalance methods | 1:00 | "Twelve single-seed runs: no sampler, loss or combination beat plain cross-entropy. Strong augmentation did not train, so that test is inconclusive." |
| 9 | Extensions and checks | 0:45 | "Early-versus-late separates well, validation AUROC 0.951. Three-class and colour normalisation give no gain. The Grad-CAM sanity check passes." |
| 10 | What results do not show | 1:15 | "Random DDR split; 834 candidate duplicate groups, 375 across splits and 39 across datasets; APTOS looks too good; specificity, grade-1 sensitivity, calibration error and intervals are not shown yet." |
| 11 | Conclusions and next steps | 1:00 | "Three conclusions, then five next steps: duplicate check, an official or patient-level split and retraining, diagnose strong augmentation, report the missing metrics, and re-evaluate APTOS." |
| 12 | Thank you | 0:05 | "Thank you. Questions?" |
