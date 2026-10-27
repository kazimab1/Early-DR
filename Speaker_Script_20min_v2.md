# 20-minute speaking script, version 2 (21 slides, about 19 min 20 s)

Frame: the pipeline works and the data audit was right, but the checks show that the headline in-domain numbers are inflated and the frozen threshold does not transfer. The new per-grade, specificity and AUROC analyses are post-hoc (not in the declared analysis); say so.

| # | Slide | Time | Say |
|---|---|---|---|
| 1 | Title | 0:15 | "Early detection of diabetic retinopathy, grades 1 and 2, with results and what the checks found." |
| 2 | About DR | 1:00 | "Adults with diabetes: 151 million in 2000, 589 million in 2024. More patients means more images and a need for fast triage." |
| 3 | Aim | 1:00 | "Target the earliest grades. Primary: sensitivity at a fixed specificity. Secondary: early vs late. Robustness: held-out EyePACS patients, then APTOS." |
| 4 | Data | 1:00 | "Grade mixes differ. DDR is split randomly because this copy has no official split or patient IDs; EyePACS is split by patient; APTOS is external only." |
| 5 | Merge cost | 1:20 | "Version 1 topped up only some grades from EyePACS, so the dataset name predicts the grade. A classifier tells DDR from EyePACS with AUROC 0.996. So the merge is tested with three arms." |
| 6 | Methods | 0:50 | "A: DDR only. B: DDR plus EyePACS, source-balanced. C: the original merge as a negative control." |
| 7 | Architecture | 0:50 | "Preprocess, sample, EfficientNet-B0, head, loss. Best epoch, temperature, prior correction and the 90% specificity threshold are fitted on validation only." |
| 8 | Training | 0:35 | "Ten epochs, three seeds, about six hours on a Kaggle GPU. Raw outputs are overconfident, so I calibrate." |
| 9 | Result 1: DDR | 1:10 | "In-domain the operating point works: specificity 0.91 against a 0.90 target, sensitivity about 0.93 for every arm. Adding EyePACS does not help DDR. But treat this as optimistic: the split is random and image properties alone predict DDR labels." |
| 10 | Result 2: EyePACS | 1:40 | "Same threshold on unseen EyePACS patients. Arm C has sensitivity 0.997 but specificity 0.013: it flags 99% of healthy EyePACS images as DR. That is the source shortcut, measured directly. Arm A, which never saw EyePACS, has specificity 0.39: the threshold does not transfer. Arm B keeps specificity 0.89." |
| 11 | Result 3: APTOS | 1:20 | "APTOS ranking is excellent, AUROC 0.994 for every arm. But sensitivity near 1.0 comes with specificity 0.24 to 0.83, and image properties alone predict APTOS labels with AUROC 0.976. So I report AUROC and do not claim robustness." |
| 12 | Grade by grade | 1:10 | "Grade 2 is easy everywhere. Grade 1 is harder off-domain: on EyePACS its AUROC is 0.61 for A, 0.77 for B, 0.63 for C. DDR 'early' is mostly grade 2, which flatters the headline." |
| 13 | Duplicates | 1:00 | "834 candidate duplicate groups, only 480 with matching grades. Removing them changes no metric by more than 0.001. So exact duplicates are not the explanation. What this cannot catch is the same patient photographed twice." |
| 14 | Capture properties | 1:20 | "A small model using only image size, file size, colour and sharpness predicts early DR with AUROC 0.87 on DDR and 0.976 inside APTOS, but only 0.5 to 0.6 elsewhere. The CNN gets 0.976 on DDR, so it learns more, but part of that score is the camera." |
| 15 | Imbalance | 0:50 | "Twelve single-seed runs. No sampler, loss or combination beat plain cross-entropy; differences are inside seed noise. Strong augmentation failed, and I found out why." |
| 16 | Strong augmentation | 1:00 | "The images are fine. The saved model scores 0.58 normally but 0.96 once BatchNorm is re-estimated: a train/test mismatch. A short retrain shows geometric augmentation trains fine and colour, blur and noise breaks it. Applying it to minority classes only also leaks the label. Geometric-only runs are prepared, not yet run." |
| 17 | Extensions | 0:35 | "Early-vs-late separates well, validation AUROC 0.951; three-class and colour normalisation give no gain; the Grad-CAM sanity check passes." |
| 18 | Limits | 1:00 | "The frozen threshold does not transfer; DDR and APTOS labels are entangled with capture properties; the DDR split is random; intervals, calibration error and five seeds are still to do." |
| 19 | Conclusions, next steps | 1:20 | "The merge as written teaches the source. Source-balancing helps off-domain. In-domain scores are inflated. Next: rerun augmentation geometric-only, a group-aware DDR split, report AUROC and per-grade results with own-source operating points, mitigate capture artifacts, five seeds with intervals, and re-evaluate once." |
| 20-21 | References, thanks | 0:10 | "Thank you. Questions?" |

Likely questions
- "Is the arm C result real if sensitivity is 0.997?" Yes: specificity is 0.013, so it simply flags almost everything.
- "Why not just retune the threshold per dataset?" That is a separate, own-source analysis and is on the next-steps list; it answers a different question from transfer.
- "Is APTOS 0.994 robustness?" Not claimed: the label is tied to image properties there.
