# Final 5-minute presentation: speaking script

The same text is in the speaker notes of DR_Final_5min.pptx.

| # | Time | Say |
|---|---|---|
| 1 | 0:08 | Hi, my project is early detection of diabetic retinopathy from retinal photographs. |
| 2 | 0:22 | Let me start with why. The number of adults living with diabetes grew from 151 million in 2000 to 589 million in 2024. Each of them needs regular retinal screening, so the number of fundus images to grade keeps growing. |
| 3 | 0:20 | But the screening pathway breaks down before an image is graded. About 60 percent of patients do not get an annual eye exam, grading takes specialist time, and early disease has no symptoms, so late detection lets preventable vision loss progress. |
| 4 | 0:20 | So the goal is a triage aid for the ophthalmologist, not a replacement. The model reads a fundus image and gives a calibrated probability of early disease, grades 1 and 2, at a fixed specificity, so high-risk images are seen first. |
| 5 | 0:30 | Here is an early result from my own runs. I trained on DDR and on a merge with EyePACS, and tested on held-out EyePACS patients with one frozen threshold. The original merge has specificity 0.013: it flags 99 percent of healthy images, because it learned which dataset an image came from. A source-balanced merge reaches 0.89, close to the 0.90 target. DDR alone gives 0.39. |
| 6 | 0:25 | So my research question is: does merging datasets help early detection, or does the model only learn the source? RQ1: does adding EyePACS improve sensitivity at a fixed specificity on unseen sources? RQ2: do calibrated probabilities and a frozen threshold still hold on unseen data? |
| 7 | 0:25 | This work builds on a few key pieces. Gulshan and colleagues showed deep learning for referable DR. DDR, EyePACS and APTOS are my data sources, and temperature scaling gives calibrated outputs. What I still need to do is a literature search for early-DR and merged-data studies. |
| 8 | 0:25 | How is this different? The common target is referable DR; mine is early DR at a fixed specificity. I test the merge instead of assuming it helps, I compare imbalance methods one at a time with a control, I test on held-out patients and an external set, and I report calibrated probabilities. It is not a new network: the value is the design and the evidence. |
| 9 | 0:20 | The method has three arms with everything else the same. A is DDR only, B is DDR plus EyePACS source-balanced, and C is my earlier merge as a negative control. I also test a weighted sampler, class-balanced and focal loss, and augmentation, one at a time. |
| 10 | 0:20 | This is the full pipeline: preprocess, EfficientNet-B0, then on validation only I pick the best epoch, fit temperature scaling and prior-shift correction, and freeze a threshold at specificity 0.90. Test sets are opened after that. |
| 11 | 0:25 | For data I use DDR, EyePACS split by patient, and APTOS as an external test only. The splits are frozen files, the model is EfficientNet-B0 at 512 pixels with three seeds on a Kaggle GPU, and test sets are opened only through a declared analysis. |
| 12 | 0:20 | Six rules keep the evidence trustworthy: freeze splits first, declare the analysis, change one thing per run, select on validation only, measure the noise with seeds and bootstrap intervals, and open the test once with external data last. |
| 13 | 0:15 | Next steps: five seeds on the group-aware split, full reporting with intervals, calibration and per-grade results, own-source operating points, and a matched-subset check on APTOS. Then the literature search and the thesis writing. |
| 14 | 0:03 | These are my references. |
| 15 | 0:03 | Thank you for listening. |
