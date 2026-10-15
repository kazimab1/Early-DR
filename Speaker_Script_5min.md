# 5-minute speaking script (about 4 min 40 s)

| # | Slide | Time | Say |
|---|---|---|---|
| 1 | Title | 0:15 | "Early detection of diabetic retinopathy. I focus on grades 1 and 2, the earliest disease, using DDR, EyePACS and APTOS fundus images." |
| 2 | About DR | 0:35 | "People living with diabetes grew from 151 million in 2000 to 589 million in 2024. More patients means more images to grade and more pressure for fast triage." |
| 3 | Research aim | 0:40 | "Many studies target referable DR, moderate or worse. I target the earliest grades. Primary question: how sensitive is the model at a fixed specificity. Secondary: separate early from late. Robustness: test on held-out EyePACS patients and on APTOS." |
| 4 | EyePACS merge cost | 1:00 | "I measured something from the label files alone. When EyePACS tops up only the short grades, the dataset name predicts the severity: about 79% of grades 1 and 3 come from EyePACS, and grades 0 and 2 come from none. 0.512 bits of the label can be recovered from the dataset name. A model could learn the camera instead of the disease, and this shortcut is strongest for any-disease versus none, the decision I am targeting. So the merge has to be tested, not assumed safe." |
| 5 | Methods | 0:55 | "Three arms, one change at a time. A is DDR only. B adds EyePACS with a source-balanced sampler. C is the original merge, kept as a negative control: if C wins on its own test split but not on APTOS, it learned the source. Same model, EfficientNet-B0, same image size and epochs, frozen patient-level splits. I test the imbalance methods one at a time, including an augmentation control. This is not a new network; the value is the design and the evidence." |
| 6 | Contribution | 0:40 | "I promise no accuracy gain in advance. I will show early-DR sensitivity at a fixed specificity with confidence intervals, whether EyePACS helps or only teaches the source, which imbalance method works, and calibrated probabilities on unseen sources. APTOS is opened last." |
| 7 | Next steps | 0:25 | "The pipeline is built and tested on synthetic data. Next: train the three arms on a Kaggle GPU, compare ablations on validation only, test on held-out EyePACS then APTOS, then write up." |
| 8-9 | References, Thank you | 0:10 | "Thank you. Questions?" |

Do not claim model results: none exist yet. Slide 4 numbers come from your own label-file audit.
