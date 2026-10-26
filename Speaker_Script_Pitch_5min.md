# 5-minute pitch: why it matters, what it can do, how I will do it, how it is different (10 slides, about 4 min 15 s)

| # | Slide | Time | Say |
|---|---|---|---|
| 1 | Title | 0:10 | "Early detection of diabetic retinopathy: why it matters, what it can do, and how I will build it." |
| 2 | About DR | 0:35 | "Adults living with diabetes grew from 151 million in 2000 to 589 million in 2024. Every one of them needs retinal screening, so there are more images to grade and more pressure for fast triage." |
| 3 | Why screening needs help | 0:40 | "The pathway is patient, annual exam, grading queue, referral. 60% do not get annual exams. More images need more grading time. Early disease often has no symptoms, so late detection lets preventable vision loss progress. The need is rapid, consistent triage right after the image is taken." |
| 4 | What it can do | 0:45 | "A model that reads a fundus image and returns three things: an early-DR flag for grades 1 and 2 at a fixed specificity, whether the disease is early or late to set priority, and a calibrated probability a clinician can trust. It supports the ophthalmologist; it does not replace them." |
| 5 | How it is different | 0:55 | "Common practice, as I understand it and will verify in the literature review, targets referable DR, merges datasets without testing the merge, compares one new method against a few baselines, tests on one dataset, and reports accuracy or AUC. I target early DR, test the merge with a control, compare imbalance methods one at a time, test on held-out patients plus a second dataset per source, and report calibrated probabilities." |
| 6 | Methods | 0:30 | "Three data arms: DDR only, DDR plus EyePACS source-balanced, and the original merge as a negative control. Same model, image size and epochs." |
| 7 | Architecture | 0:35 | "Preprocess, sample, EfficientNet-B0, a head, a loss. Everything below is fitted on validation only: best epoch, temperature, prior-shift correction and the specificity threshold. Test sets come last." |
| 8 | Six rules | 0:30 | "Six rules: freeze the splits, declare the analysis, change one thing per run, select on validation, measure the noise with several seeds, and open the test sets once with the external one last." |
| 9 | Contributions | 0:30 | "Four contributions: early-first evidence, a tested merge, a controlled comparison, and calibrated probabilities for screening. It is not a new network, and no accuracy gain is promised in advance." |
| 10-11 | References, Thank you | 0:10 | "Thank you. Questions?" |

Say if asked: the "common practice" column is my understanding and is checked in the literature review, so novelty claims are not final.
