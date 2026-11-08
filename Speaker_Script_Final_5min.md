# Final 5-minute presentation: speaking script

The same text is in the speaker notes of DR_Final_5min.pptx.

| # | Time | Say |
|---|---|---|
| 1 | 0:08 | Hi, my project is early detection of diabetic retinopathy from retinal photographs. |
| 2 | 0:22 | Let me start with why I chose this. The number of adults living with diabetes grew from 151 million in 2000 to 589 million in 2024. Each of them needs regular retinal screening, so the number of fundus images to grade keeps growing. |
| 3 | 0:25 | The screening pathway breaks down before an image is graded. About 60 percent of patients do not get an annual eye exam, grading takes specialist time, and early disease has no symptoms, so late detection lets preventable vision loss progress. |
| 4 | 0:25 | So the goal is a triage aid for the ophthalmologist, not a replacement. The model reads a fundus image and gives a calibrated probability of early disease, grades 1 and 2, at a fixed specificity, so high-risk images are seen first. |
| 5 | 0:40 | My main question is: when datasets are merged for early-DR detection, does the model learn the disease or the source? Merging is common, yet I found no study testing it. RQ1 asks whether adding EyePACS really helps, using a negative-control merge. RQ2 asks whether a threshold frozen on validation survives a change of source. RQ3 asks whether capture cues like image size are predicting the label. RQ4 asks what we need to take into account when merging datasets, such as image quality, camera type and lighting, which factors help us, which harm us, and what findings come out of that. This is the second main focus of the project. |
| 6 | 0:35 | How is this different? The common target is referable DR; mine is early DR at a fixed specificity. I test the merge instead of assuming it helps, I compare imbalance methods one at a time with a control, I test on held-out patients and an external set, and I report calibrated probabilities. It is not a new network: the value is the design and the evidence. |
| 7 | 0:35 | The model is simple. Here is one example photo. It is 2588 by 1958 pixels with a black border. We crop the border, pad it to a square and resize it to 512 by 512. The EfficientNet-B0 network finds edges first, then vessels, then lesions, and turns 1,280 features into two scores: No DR and DR. Calibration turns the raw score of 0.88 into a probability of 0.62, which is above the 0.30 threshold set on validation, so the image is flagged as likely early DR and sent for grading. The numbers are illustrative. |
| 8 | 0:25 | For data I use DDR, EyePACS split by patient, and APTOS as an external test only. The splits are frozen files, the model is EfficientNet-B0 at 512 pixels with three seeds on a Kaggle GPU, and test sets are opened only through a declared analysis. |
| 9 | 0:35 | Next steps: train three arms with several seeds. A is DDR only. B is DDR plus all EyePACS grades, source-balanced. C is DDR plus only EyePACS grades 1, 3 and 4, an old merge kept as a negative control. Then handle imbalance one strategy at a time, test on unseen sources with confidence intervals, and report sensitivity at a fixed specificity. For RQ4 I will study image quality, camera type, lighting, image size, sharpness and colour, to see which help and which harm a merged model. |
| 10 | 0:03 | These are my references. |
| 11 | 0:03 | Thank you for listening. |
