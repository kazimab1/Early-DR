# Final 5-minute presentation: speaking script

The same text is in the speaker notes of DR_Final_5min.pptx.

| # | Time | Say |
|---|---|---|
| 1 | 0:08 | Hi, my project is early detection of diabetic retinopathy from retinal photographs. |
| 2 | 0:22 | Let me start with why. The number of adults living with diabetes grew from 151 million in 2000 to 589 million in 2024. Each of them needs regular retinal screening, so the number of fundus images to grade keeps growing. |
| 3 | 0:20 | The screening pathway breaks down before an image is graded. About 60 percent of patients skip the annual exam, grading takes specialist time, and early disease has no symptoms. We need rapid, consistent triage right after the image is captured. |
| 4 | 0:30 | Early disease, grades 1 and 2, is where AI could help most. It is silent, so patients do not seek care. Much well-known work, such as Gulshan and colleagues, targets referable DR, and I found no study reporting early-DR sensitivity at a pre-set specificity. My tool is a triage aid, not a replacement. |
| 5 | 0:35 | Here is the gap. Public datasets like DDR, EyePACS and APTOS are often pooled to get more data. A recent review notes that multi-dataset training can mask poor external generalization, and I found no study testing whether a merged model learns the disease or the source. Accuracy can look excellent while the model over-flags or uses camera cues. An over-flagging tool overloads graders; an under-flagging one misses early disease. |
| 6 | 0:40 | So my main question is: when datasets are merged for early-DR detection, does the model learn the disease or the source? Merging is common, yet I found no study testing it. RQ1 asks whether adding EyePACS really helps, using a negative-control merge. RQ2 asks whether a threshold frozen on validation survives a change of source, because AUROC can hide over-flagging or missed early cases. RQ3 asks whether capture cues like image size are predicting the label. The contribution is a protocol to test DR models before trusting them, not a new network. |
| 7 | 0:25 | This builds on a few pieces. Gulshan and colleagues showed deep learning for referable DR. DDR, EyePACS and APTOS are my data sources, and temperature scaling gives calibrated outputs. A recent review and a cross-camera study describe dataset and camera effects, but for referable DR. I still need a systematic literature search. |
| 8 | 0:25 | How is this different? My target is early DR at a fixed specificity. I test the merge itself, compare imbalance methods one at a time with a control, test on held-out patients and an external set, and report shortcut audits and calibrated probabilities. It is not a new network: the value is the design and the evidence. |
| 9 | 0:25 | The method uses three arms: A is DDR only, B is DDR plus all EyePACS grades, source-balanced, and C is DDR plus only EyePACS grades 1, 3 and 4, an old merge kept as a negative control. I add honest splits, a shortcut audit, and a frozen operating point at specificity 0.90 set on validation. |
| 10 | 0:25 | The model is simple: a fundus image is cropped and resized, passed through an EfficientNet-B0, then calibrated with temperature scaling and a prior-shift correction. The output is a probability of early disease and a flag, all set on validation data only. |
| 11 | 0:20 | For data I use DDR, EyePACS split by patient, and APTOS as an external test only. The splits are frozen files, and test sets are opened only through a declared analysis. |
| 12 | 0:15 | Next steps: train the arms with several seeds, compare the imbalance methods, test on unseen sources with confidence intervals, and finish the literature search and thesis. |
| 13 | 0:03 | These are my references. |
| 14 | 0:03 | Thank you for listening. |
