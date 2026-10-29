# Progress update with three-seed group-split results: speaking script (about 6 minutes)

Slide 9 holds the three-seed group-split table (mean ± sd over seeds 42, 43, 44).

| # | Slide | Time | Say |
|---|---|---|---|
| 1 | Title | 0:15 | "This is a progress update on my early diabetic retinopathy project, now with the group-split models evaluated over three seeds." |
| 2 | Progress at a glance | 1:00 | "Done: the audit, the Kaggle pipeline, arms A, B and C from the original three-seed run and the group-split run with three seeds, internal and APTOS evaluation, fourteen imbalance runs and three extensions, and four checks. Ruled out: near-duplicate leakage does not explain the high DDR scores, and image size alone does not explain the APTOS score, which stays 0.984 to 0.989 within same-size groups, a post-hoc result. To do: own-source operating points, a matched-subset check, the literature search and the thesis." |
| 3 | Data | 0:30 | "Three datasets with different grade mixes: early DR is 41% of DDR, 22% of EyePACS and 37% of APTOS. EyePACS is split by patient; APTOS is external only." |
| 4 | EyePACS merge cost | 0:40 | "The dataset name alone carries information about the label, so a naive merge can teach the model the source." |
| 5 | Methods | 0:30 | "Three data arms, one change at a time: DDR only, source-balanced, and the old merge as a negative control." |
| 6 | Training | 0:20 | "EfficientNet-B0 at 512 pixels, the same recipe for every arm." |
| 7 | Key results so far | 0:50 | "In the original three-seed run all arms reach sensitivity 0.93 at specificity 0.91 on DDR. On held-out EyePACS patients the same frozen threshold gives specificity 0.39, 0.89 and 0.013: the old merge flags 99% of healthy images." |
| 8 | Group-split results | 0:50 | "Over three seeds with 95% intervals: on held-out EyePACS the source-balanced arm B has AUROC 0.884 (0.870 to 0.897), against 0.780 for DDR only and 0.717 for the old merge, and the intervals do not overlap. Its probabilities are also far better calibrated: ECE 0.056 against 0.326 and 0.717, and Brier 0.091 against 0.300 and 0.707. On DDR all arms overlap, and on APTOS AUROC is 0.990 to 0.994. Calibration, like the threshold, does not transfer for arms A and C." |
| 9 | What the checks found | 0:40 | "Duplicates: a group split removed them and results did not change. Image size alone predicts APTOS at 0.96 but not EyePACS, yet within same-size APTOS groups the models still score 0.984 to 0.989. Strong augmentation failed from a BatchNorm mismatch." |
| 10 | Imbalance methods | 0:30 | "No imbalance method beat plain cross-entropy within seed noise; minority-only augmentation leaked the label." |
| 11 | What these results do not show | 0:40 | "The frozen threshold does not transfer, APTOS may carry other capture cues, same-patient leakage in DDR is untestable, and per-grade and size-strata results are post-hoc." |
| 12 | Conclusions and next steps | 0:30 | "The old merge teaches the source, source-balancing helps off-domain, and EyePACS is the clean benchmark. Next: own-source operating points, a matched-subset check, the literature search and the thesis." |
| 13 | Thank you | 0:05 | "Thank you." |


**New slide "The Three Data Arms" (after Methods; later slide numbers shift by one), about 0:20:** "A is DDR only, the clean baseline. B is DDR plus all EyePACS grades, with a sampler that makes each source equally likely within every grade. C is the old merge: DDR plus only EyePACS grades 1, 3 and 4, so EyePACS supplies mostly disease images and the dataset name predicts disease. It is a deliberately flawed control. Everything else is identical."
