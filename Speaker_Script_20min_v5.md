# 20-minute speaking script, version 4 (22 slides, about 20 min)

Frame: the pipeline works and the data audit was right. The checks show the frozen threshold does not transfer, APTOS is confounded by capture, and DDR near-duplicate leakage is small; EyePACS is the cleanest benchmark. The new per-grade, specificity and AUROC analyses are post-hoc (not in the declared analysis); say so.

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
| 9 | Result 1: DDR | 1:10 | "In-domain the operating point works: specificity 0.91 against a 0.90 target, sensitivity about 0.93 for every arm. Adding EyePACS does not help DDR. Treat it as optimistic: DDR has no patient IDs, so a patient's other eye can be in train." |
| 10 | Result 2: EyePACS | 1:40 | "Same threshold on unseen EyePACS patients. Arm C has sensitivity 0.997 but specificity 0.013: it flags 99% of healthy EyePACS images as DR. That is the source shortcut, measured directly. Arm A, which never saw EyePACS, has specificity 0.39: the threshold does not transfer. Arm B keeps specificity 0.89." |
| 11 | Result 3: APTOS | 1:20 | "APTOS ranking is excellent, AUROC 0.994 for every arm. But sensitivity near 1.0 comes with specificity 0.24 to 0.83, and image size alone predicts APTOS labels with AUROC 0.96. So I report AUROC and do not claim robustness." |
| 12 | Grade by grade | 1:10 | "Grade 2 is easy everywhere. Grade 1 is harder off-domain: on EyePACS its AUROC is 0.61 for A, 0.77 for B, 0.63 for C. DDR 'early' is mostly grade 2, which flatters the headline." |
| 13 | Duplicates | 1:00 | "834 candidate duplicate groups, only 480 with matching grades. Removing them changes no metric by more than 0.001. So exact duplicates are not the explanation. What this cannot catch is the same patient photographed twice." |
| 14 | Group split | 1:00 | "I built a group-aware split that keeps near-duplicate images together. Under the random split 4.2% of test images had a near-duplicate in train; under the group split none. I retrained the three arms on it and evaluated once. APTOS AUROC fell only 0.002 to 0.008, DDR grade-1 sensitivity moved in both directions, and arm C's EyePACS drop from 0.75 to 0.71 is seed noise. On the APTOS side, a post-hoc check within groups of same-size images still gives AUROC 0.984 to 0.989, so size alone does not explain the 0.99; other capture cues are untested. What it cannot fix: a patient's other eye looks similar, about 0.92 to 0.96, and stays on both sides, so same-patient leakage is untested." |
| 15 | Image properties | 1:10 | "A small model that sees only image properties. Image size alone: 0.58 on DDR, 0.53 on EyePACS, but 0.96 inside APTOS. So APTOS has a real camera or hospital confound, EyePACS is clean, and DDR is mixed: its colour and file-size signals are probably lesion content, not camera." |
| 16 | Imbalance | 0:50 | "Fourteen single-seed runs. No sampler, loss or combination beat plain cross-entropy; differences are inside seed noise. The augmentation runs are the interesting part: geometric augmentation on all classes scores 0.977, the same augmentation on minority classes only scores 0.834. That gap is the augmentation acting as a cue for the label." |
| 17 | Strong augmentation | 1:00 | "Strong augmentation failed outright, around 0.57. The images are fine. The saved model scores 0.58 normally but 0.96 once BatchNorm is re-estimated: a train/test mismatch, and a short retrain shows the colour, blur and noise part causes it while the geometric part trains fine. I confirmed it by rerunning: geometric-only on all classes scores 0.977 and has the best raw calibration, temperature 1.65 against 2.39 for baseline." |
| 18 | Extensions | 0:35 | "Early-vs-late separates well, validation AUROC 0.951; three-class and colour normalisation give no gain; the Grad-CAM sanity check passes." |
| 19 | Limits | 1:00 | "The frozen threshold does not transfer; APTOS labels are tied to image size; same-patient leakage in DDR cannot be removed; intervals, calibration error and five seeds are still to do. EyePACS is the most trustworthy benchmark." |
| 20 | Conclusions, next steps | 1:20 | "The merge as written teaches the source. Source-balancing helps off-domain. EyePACS is the clean benchmark. Next: report AUROC and per-grade results with own-source operating points, mitigate capture artifacts, five seeds on the group split with intervals, use geometric augmentation on all classes with minority-only kept as the cue control, and label every post-hoc analysis." |
| 21-22 | References, thanks | 0:10 | "Thank you. Questions?" |

Likely questions
- "Is the arm C result real if sensitivity is 0.997?" Yes: specificity is 0.013, so it simply flags almost everything.
- "Why not just retune the threshold per dataset?" That is a separate, own-source analysis and is on the next-steps list; it answers a different question from transfer.
- "Is APTOS 0.994 robustness?" Not claimed: the label is tied to image properties there.


**New slide "The Three Data Arms" (after Methods; later slide numbers shift by one), about 0:20:** "A is DDR only, the clean baseline. B is DDR plus all EyePACS grades, with a sampler that makes each source equally likely within every grade. C is the old merge: DDR plus only EyePACS grades 1, 3 and 4, so EyePACS supplies mostly disease images and the dataset name predicts disease. It is a deliberately flawed control. Everything else is identical."
