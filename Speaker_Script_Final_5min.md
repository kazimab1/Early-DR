# Final 5-minute presentation: speaking script

The same text is in the speaker notes of DR_Final_5min.pptx.

| # | Time | Say |
|---|---|---|
| 1 | 0:08 | Hi, my project is early detection of diabetic retinopathy from retinal photographs. |
| 2 | 0:22 | Let me start with why. The number of adults living with diabetes grew from 151 million in 2000 to 589 million in 2024. Each of them needs regular retinal screening, so the number of fundus images to grade keeps growing. |
| 3 | 0:20 | The screening pathway breaks down before an image is graded. About 60 percent of patients skip the annual exam, grading takes specialist time, and early disease has no symptoms. We need rapid, consistent triage right after the image is captured. |
| 4 | 0:30 | Early disease, grades 1 and 2, is where AI could help most. It is silent, so patients do not seek care. Much well-known work, such as Gulshan and colleagues, targets referable DR, and I found no study reporting early-DR sensitivity at a pre-set specificity. My tool is a triage aid, not a replacement. |
| 5 | 0:30 | Here is the gap. Public medical-image datasets are often pooled; here DDR, EyePACS and APTOS. A recent review notes that multi-dataset training can mask poor external generalization. In other fields, models have learned hospital or scanner site instead of disease, in chest X-rays, and dataset identity in general. I found no study testing this for early DR, or which image properties carry it. An over-flagging tool overloads graders; an under-flagging one misses early disease. |
| 6 | 0:35 | My main question is general: when medical-image datasets are merged, does a model learn the condition or the source? Early DR is my case study. RQ1: does a second source really help on unseen sources? A negative-control merge tests it. RQ2: does a threshold frozen on validation survive a change of source? RQ3: which image properties differ between sources, such as size, quality, brightness and colour, and how do they affect what a merged model learns? The contribution is a protocol, not a new network. |
| 7 | 0:25 | This builds on a few pieces. Gulshan and colleagues for referable DR, my three datasets, and temperature scaling for calibration. Torralba and Efros, Geirhos, Zech and DeGrave show that models learn dataset or site cues. A DR review and a cross-camera study describe dataset and camera effects. I still need a systematic literature search. |
| 8 | 0:25 | How is this different? My target is early DR at a fixed specificity. I test the merge itself, compare imbalance methods one at a time, test on held-out patients and an external set, and ask which image properties carry the source. DR is the case study; the protocol should apply to any merged medical-image task. It is not a new network. |
| 9 | 0:30 | The method uses three arms. A is DDR only, B is DDR plus all EyePACS grades, source-balanced, and C is DDR plus only EyePACS grades 1, 3 and 4, an old merge kept as a negative control. I add honest splits, a property analysis of size, brightness, colour and sharpness, and a frozen operating point at specificity 0.90 set on validation. |
| 10 | 0:20 | The model is simple: a fundus image is cropped and resized, passed through an EfficientNet-B0, then calibrated with temperature scaling and a prior-shift correction. The output is a probability of early disease and a flag, all set on validation data only. |
| 11 | 0:15 | For data I use DDR, EyePACS split by patient, and APTOS as an external test only. The splits are frozen files, and test sets are opened only through a declared analysis. |
| 12 | 0:20 | Next steps: train the three arms with several seeds, compare the imbalance methods, analyse image properties, test on unseen sources with confidence intervals, report sensitivity at a fixed specificity, and finish the literature search and thesis. |
| 13 | 0:03 | These are my references. |
| 14 | 0:03 | Thank you for listening. |
