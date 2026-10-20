# 20-minute speaking script (17 slides, about 20 min)

Numbers come from the Kaggle runs (3 seeds per arm, threshold frozen on validation). Say the caveats out loud:
the DDR split is random, duplicate checks are pending, and the APTOS numbers are preliminary.

| # | Slide | Time | Say |
|---|---|---|---|
| 1 | Title | 0:20 | "Early detection of diabetic retinopathy: grades 1 and 2, tested on DDR, EyePACS and APTOS. Today I show the first full set of results." |
| 2 | About DR | 1:00 | "Diabetes grew from 151 million adults in 2000 to 589 million in 2024. More patients means more fundus images to grade, so we need fast, consistent triage." |
| 3 | Research aim | 1:30 | "Many studies target referable DR, moderate or worse. I target the earliest grades. Primary question: how sensitive is the model at a fixed specificity. Secondary: separate early from late. Robustness: held-out EyePACS patients, then APTOS." |
| 4 | Data | 1:30 | "Three datasets with different grade mixes. Early DR is 41% of DDR, 22% of EyePACS, 37% of APTOS. DDR has no official split or patient IDs in this copy, so I used a random 60/20/20 split; I come back to that. EyePACS is split by patient. APTOS is external only." |
| 5 | EyePACS merge | 1:30 | "In version 1, EyePACS topped up only the short grades, so about 79% of grades 1 and 3 come from EyePACS and grades 0 and 2 from none. The dataset name alone carries 0.512 bits of the label. A classifier also tells DDR from EyePACS images with 0.996 AUROC. So the merge could teach the camera instead of the disease." |
| 6 | Methods | 1:15 | "Three arms, one change at a time: A DDR only, B DDR plus EyePACS source-balanced, C the original merge as a negative control. Same model, image size and epochs; imbalance methods tested one at a time with an augmentation control." |
| 7 | Architecture | 1:00 | "Top row is training: preprocess, sampler and augmentation, EfficientNet-B0, a head, a loss. Bottom row is fitted on validation only: best epoch, temperature scaling, prior-shift correction, and a threshold at 90% specificity. Test sets are opened after that." |
| 8 | Training | 1:00 | "Ten epochs, three seeds per arm, about six hours on a Kaggle GPU. Raw outputs are overconfident: the fitted temperature is 1.6 to 2.4. These curves show convergence, not which arm is better, because the validation sets differ." |
| 9 | Result 1: DDR | 1:30 | "On the DDR test set sensitivity is 0.937 for A and C, 0.925 for B; the B-versus-A gap is not significant after Holm correction. B is slightly higher in PR-AUC. Adding EyePACS did not improve DDR sensitivity." |
| 10 | Result 2: EyePACS | 2:00 | "This is the key slide. Arm C has the highest sensitivity, 0.997, but the lowest PR-AUC, 0.43, below DDR-only A. In C every EyePACS training image has disease, so the camera predicts the label. That is the shortcut the audit predicted. Arm B, source-balanced, gets PR-AUC 0.80, but its sensitivity at the frozen threshold is 0.73." |
| 11 | Result 3: APTOS | 1:30 | "On APTOS all three arms look alike: sensitivity near 1.0 and AUROC 0.994. That is higher than I expected from a model trained on DDR only, so I do not claim it as robustness yet; I am checking for overlap between the datasets." |
| 12 | Result 4: Imbalance | 1:30 | "Twelve single-seed runs. No sampler, loss or combination beat plain cross-entropy; differences are inside seed noise. The two strong-augmentation runs did not train properly, so that test is inconclusive. The score only covers early-versus-none, not the rare grades separately." |
| 13 | Extensions | 1:00 | "Early-versus-late separates well, validation AUROC 0.951. A three-class label and colour normalisation give no gain, and colour normalisation is 3.4 times slower. The Grad-CAM sanity check passes: trained and randomised-weights maps correlate near zero." |
| 14 | Limitations | 1:30 | "What these results do not show. The DDR split is random, so images from one patient can sit in both train and test. The audit found 834 candidate duplicate groups, 375 crossing splits and 39 crossing datasets; this is under review. APTOS looks too good to claim. Specificity reached, grade-1-only sensitivity, calibration error and intervals are in the result files but not shown here." |
| 15 | Conclusions | 1:15 | "Three conclusions: the merge as written teaches the source; source-balancing helps EyePACS but not DDR; imbalance methods show no clear gain. Next: duplicate check, an official or patient-level DDR split with retraining, diagnose strong augmentation, report the missing metrics, then re-evaluate APTOS." |
| 16-17 | References, Thank you | 0:10 | "Thank you. Questions?" |

Likely questions
- "Why is arm C's sensitivity so high yet it is worse?" Sensitivity is measured at one frozen threshold; PR-AUC shows it also flags many healthy EyePACS images (specificity is in the result files).
- "Is APTOS 0.994 real?" Not claimed yet; overlap and grade-1 checks pending.
- "Why a random DDR split?" The Kaggle copy has no official split or patient IDs; the plan is an official or patient-level split and retraining.
