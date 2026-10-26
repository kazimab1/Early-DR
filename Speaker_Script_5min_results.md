# 5-minute speaking script with results (11 slides, about 4 min 40 s)

Say the caveats out loud: the DDR split is random, duplicate checks are pending, APTOS is not claimed.

| # | Slide | Time | Say |
|---|---|---|---|
| 1 | Title | 0:10 | "Early detection of diabetic retinopathy: grades 1 and 2, on DDR, EyePACS and APTOS. I will show the first results." |
| 2 | About DR | 0:25 | "Adults with diabetes grew from 151 million in 2000 to 589 million in 2024. More patients means more fundus images to grade and a need for fast triage." |
| 3 | Research aim | 0:35 | "Many studies target referable DR. I target the earliest grades. Primary question: how sensitive is the model at a fixed specificity. Secondary: separate early from late. Robustness: held-out EyePACS, then APTOS." |
| 4 | EyePACS merge cost | 0:50 | "Version 1 topped up only the short grades from EyePACS, so about 79% of grades 1 and 3 come from EyePACS and grades 0 and 2 from none. The dataset name alone carries 0.512 bits of the label, and a classifier tells DDR from EyePACS images with 0.996 AUROC. So the merge could teach the camera, not the disease." |
| 5 | Methods | 0:30 | "Three arms, one change at a time: A DDR only, B DDR plus EyePACS source-balanced, C the original merge as a negative control." |
| 6 | Architecture | 0:30 | "Preprocess, sample, EfficientNet-B0, a head, a loss. Everything below is fitted on validation only: best epoch, temperature, prior-shift correction and a 90% specificity threshold. Test sets come after." |
| 7 | Key results | 1:00 | "On DDR all arms are alike: sensitivity about 0.93. On held-out EyePACS, arm C has the highest sensitivity, 0.997, but the lowest PR-AUC, 0.43, below DDR-only A. It learned the source, as the audit predicted. Arm B, source-balanced, reaches PR-AUC 0.80. These are preliminary." |
| 8 | What results do not show | 0:30 | "The DDR split is random, so one patient can be in train and test. 834 candidate duplicate groups are under review. APTOS scored AUROC 0.994, which looks too good, so I do not claim it." |
| 9 | Conclusions | 0:35 | "The merge as written teaches the source; source-balancing helps EyePACS but not DDR; imbalance methods show no clear gain. Next: duplicate check, an official or patient-level DDR split with retraining, then APTOS again." |
| 10-11 | References, Thank you | 0:05 | "Thank you. Questions?" |

Likely questions
- "Why is arm C's sensitivity highest yet it is worse?" Sensitivity is at one frozen threshold; PR-AUC shows it also flags many healthy EyePACS images.
- "Is APTOS real?" Not claimed until the overlap and grade-1 checks are done.
