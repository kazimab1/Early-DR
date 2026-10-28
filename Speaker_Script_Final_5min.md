# Final 5-minute presentation: speaking script

The same text is in the speaker notes of DR_Final_5min.pptx.

| # | Time | Say |
|---|---|---|
| 1 | 0:08 | Hi, my project is early detection of diabetic retinopathy from retinal photographs. |
| 2 | 0:22 | Let me start with why I chose this. The number of adults living with diabetes grew from 151 million in 2000 to 589 million in 2024. Each of them needs regular retinal screening, so the number of fundus images to grade keeps growing. |
| 3 | 0:25 | The screening pathway breaks down before an image is graded. About 60 percent of patients do not get an annual eye exam, grading takes specialist time, and early disease has no symptoms, so late detection lets preventable vision loss progress. |
| 4 | 0:25 | So the goal is a triage aid for the ophthalmologist, not a replacement. The model reads a fundus image and gives a calibrated probability of early disease, grades 1 and 2, at a fixed specificity, so high-risk images are seen first. |
| 5 | 0:35 | My research question is: does merging datasets help early detection, or does the model only learn which dataset an image came from? RQ1: does adding EyePACS to DDR improve sensitivity at a fixed specificity on sources the model has not seen? RQ2: do calibrated probabilities and a frozen threshold still hold on unseen data? |
| 6 | 0:30 | This work builds on a few key pieces. Gulshan and colleagues showed deep learning for referable DR. DDR, EyePACS and APTOS are my data sources, and temperature scaling gives calibrated outputs. What I still need to do is a literature search for early-DR and merged-data studies. |
| 7 | 0:35 | How is this different? The common target is referable DR; mine is early DR at a fixed specificity. I test the merge instead of assuming it helps, I compare imbalance methods one at a time with a control, I test on held-out patients and an external set, and I report calibrated probabilities. It is not a new network: the value is the design and the evidence. |
| 8 | 0:30 | The method has three arms with everything else the same. A is DDR only, B is DDR plus EyePACS source-balanced, and C is an earlier merge kept as a negative control. I also test a weighted sampler, class-balanced and focal loss, and augmentation, one at a time. |
| 9 | 0:30 | The model is simple. A fundus image is cropped and resized, passed through an EfficientNet-B0, then calibrated with temperature scaling and a prior-shift correction. The output is a probability of early disease and a flag at a fixed specificity of 0.90, all set on validation data only. |
| 10 | 0:25 | For data I use DDR, EyePACS split by patient, and APTOS as an external test only. The splits are frozen files, the model is EfficientNet-B0 at 512 pixels with three seeds on a Kaggle GPU, and test sets are opened only through a declared analysis. |
| 11 | 0:25 | Next steps: train the three arms with several seeds, compare the imbalance methods one at a time, test on held-out EyePACS patients and APTOS with confidence intervals, calibrate and report sensitivity at a fixed specificity, and finish the literature search and thesis. |
| 12 | 0:03 | These are my references. |
| 13 | 0:03 | Thank you for listening. |
