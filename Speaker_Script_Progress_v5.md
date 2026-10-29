# Progress update, version 4 (12 slides, about 12 min)

| # | Slide | Time | Say |
|---|---|---|---|
| 1 | Title | 0:10 | "Progress update on early detection of diabetic retinopathy." |
| 2 | Progress at a glance | 1:00 | "Done: audit, the full Kaggle pipeline, arms A, B and C with three seeds, internal and APTOS evaluation, fourteen imbalance runs including geometric augmentation, three extensions and three diagnostic checks, and a group-aware DDR split with the arms retrained on it and evaluated over three seeds. AUROC changed by at most 0.003 on DDR and APTOS, so near-duplicate leakage does not explain the high DDR scores. A post-hoc APTOS size-stratified check on saved predictions: inside groups of images of the same size the AUROC stays 0.984 to 0.989, so image size alone does not explain the 0.99, though other capture cues remain untested. To do: AUROC and per-grade reporting, capture-artifact mitigation, five seeds with intervals, and the thesis. Main findings: the original merge teaches the source; APTOS is camera-confounded; EyePACS is the clean benchmark." |
| 3 | Data | 1:00 | "Grade mixes differ. DDR is a random split with no patient IDs; EyePACS is split by patient; APTOS is external only." |
| 4 | Merge cost | 1:20 | "The merge makes the dataset name predict the grade, and a classifier tells DDR from EyePACS with AUROC 0.996. So three arms test it." |
| 5 | Methods | 0:40 | "A DDR only, B source-balanced, C the original merge as a control." |
| 6 | Training | 0:35 | "EfficientNet-B0, ten epochs, three seeds, about six hours on a Kaggle GPU." |
| 7 | Key results | 2:00 | "On DDR the operating point works: specificity 0.91, sensitivity about 0.93. On EyePACS with the same threshold, specificity is 0.39 for A, 0.89 for B and 0.013 for C: C flags 99% of healthy images. The frozen threshold does not transfer, so I compare AUROC across datasets." |
| 8 | What the checks found | 1:50 | "Duplicates: 834 candidate groups, only 480 same-grade, removing them changes no metric by more than 0.001. Image properties: image size alone predicts early DR 0.58 on DDR, 0.53 on EyePACS and 0.96 inside APTOS, so APTOS is camera-confounded and EyePACS is clean. Duplicates: 4% of DDR test had a near-duplicate in train; a group split removed it and the test results did not change. Strong augmentation: the failure is a BatchNorm mismatch from the noise and blur part. Geometric augmentation on all classes scores 0.977, but on minority classes only it scores 0.834: the augmentation leaks the label." |
| 9 | Imbalance | 0:50 | "No sampler, loss or combination beat plain cross-entropy; geometric augmentation on all classes is as good as baseline with better raw calibration; on minority classes only it drops to 0.834, a label cue." |
| 10 | Limits | 1:10 | "Threshold does not transfer; capture artifacts in DDR and APTOS; random DDR split; intervals and five seeds still to do; the new analyses are post-hoc." |
| 11 | Conclusions, next steps | 1:20 | "Three conclusions and six next steps, starting with five seeds on the group split." |
| 12 | Thanks | 0:05 | "Thank you. Questions?" |


**New slide "The Three Data Arms" (after Methods; later slide numbers shift by one), about 0:20:** "A is DDR only, the clean baseline. B is DDR plus all EyePACS grades, with a sampler that makes each source equally likely within every grade. C is the old merge: DDR plus only EyePACS grades 1, 3 and 4, so EyePACS supplies mostly disease images and the dataset name predicts disease. It is a deliberately flawed control. Everything else is identical."
