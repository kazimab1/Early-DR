---
title: "Disease or Source? Testing Merged Datasets for Early Diabetic Retinopathy Detection"
subtitle: "Progress report"
author: "Advisor: Navid Amini, PhD, Department of Computer Science, Cal State LA"
---

# 1. Summary

This project asks whether models trained on merged diabetic retinopathy (DR) datasets learn the disease or the dataset they came from, for early DR (grades 1-2 versus No DR) at a fixed specificity. The full pipeline runs on Kaggle with three public datasets (DDR, EyePACS, APTOS). Results so far show that a naive merge teaches the source, that a source-balanced merge helps on unseen patients, and that in-domain scores are optimistic. Items marked **[FILL]** depend on the five-seed run and must be completed before submitting.

# 2. Problem and gap

- Public medical-image datasets are often pooled to get more training data; here DDR, EyePACS and APTOS. A 2025 review notes that multi-dataset training can mask poor external generalization.
- In other fields, models have learned hospital or scanner site, or dataset identity, instead of disease (Torralba and Efros 2011; Geirhos et al. 2020; Zech et al. 2018; DeGrave et al. 2021).
- Early DR (grades 1-2) is symptomless and under-studied; much well-known work targets referable DR (moderate or worse).
- In three quick searches I found no study that tests whether a merged early-DR model learns the disease or the source, which image properties carry the source, or that reports early-DR sensitivity at a pre-set specificity. A systematic literature search is still to do, so this is not a novelty claim.

# 3. Research questions

Dataset merging is treated in general, with early DR as the case study.

1. **Main:** when medical-image datasets are merged, does a model learn the condition or the source?
2. **RQ1 (the merge):** does adding a second source (EyePACS to DDR) improve early-DR sensitivity at a fixed specificity on unseen sources?
3. **RQ2 (the threshold):** does a threshold frozen on validation survive a change of source?
4. **RQ3 (image properties):** which properties differ between sources (size, file size and quality, brightness, colour, sharpness; camera proxies, since the camera model is not recorded) and how do they affect what a merged model learns?

# 4. Method

**Data arms** (everything else identical: EfficientNet-B0, 512 px, ten epochs, frozen splits, seeds):

| Arm | Training data | Role |
|---|---|---|
| A | DDR only | Clean baseline |
| B | DDR + all EyePACS grades, source-balanced within each grade | Does more data help without teaching the source? |
| C | DDR + EyePACS grades 1, 3 and 4 only (old merge) | Deliberately flawed negative control |

**Splits.** DDR: 7,454 / 2,552 / 2,516 (random 60/20/20; no patient IDs; a group-aware version was also built). EyePACS: 24,572 / 5,266 / 5,270, split by patient. APTOS (3,662 images): external test only.

**Calibration.** Temperature scaling and prior-shift correction fitted on validation; threshold frozen at specificity 0.90; test sets opened only through a declared analysis. Post-hoc analyses are labelled.

# 5. Progress to date

## 5.1 Completed

- Data audit and Kaggle pipeline (prepare, train, calibrate, evaluate, plot).
- Arms A, B and C: original run with three seeds; group-split run with five seeds **[FILL: confirm completed]**.
- Internal (DDR, held-out EyePACS patients) and external (APTOS) evaluation.
- Fourteen imbalance runs (sampler, loss, augmentation) and three extensions.
- Four diagnostic checks (Section 5.3).

## 5.2 Results (original run, three seeds)

| Source | Finding |
|---|---|
| DDR (in-domain) | Sensitivity 0.93 at specificity 0.91 for all arms; adding EyePACS did not improve DDR sensitivity |
| Held-out EyePACS | Same frozen threshold gives specificity 0.39 (A), 0.89 (B), 0.013 (C): arm C flags 99% of healthy images |
| APTOS (external) | AUROC 0.994-0.995 for all arms, but specificity at the frozen threshold is 0.46 (A), 0.83 (B), 0.24 (C) |
| Per grade | Grade 2 is easy (AUROC 0.98-0.999); grade 1 is harder off-domain (EyePACS 0.61 A, 0.77 B, 0.63 C) |
| Imbalance methods | No method beat plain cross-entropy beyond seed noise (best 0.9772 vs baseline 0.9761) |

**Five-seed results** (mean and 95% patient-level bootstrap interval, group-split models) **[FILL from the evaluation output]**:

| Metric | A: DDR only | B: source-balanced | C: old merge |
|---|---|---|---|
| DDR: sensitivity at specificity 0.90 | [  ] | [  ] | [  ] |
| EyePACS: AUROC | [  ] | [  ] | [  ] |
| EyePACS: specificity at frozen threshold | [  ] | [  ] | [  ] |
| APTOS: AUROC | [  ] | [  ] | [  ] |
| EyePACS: ECE and Brier | [  ] | [  ] | [  ] |

## 5.3 Diagnostic checks

- **Duplicates.** 834 candidate groups; 4% of DDR test images had a near-duplicate in train. A group-aware split removed it and test results did not change beyond seed noise.
- **Image properties (first step of RQ3).** Image size alone predicts early DR at AUROC 0.96 in APTOS, 0.58 in DDR and 0.53 in EyePACS, so APTOS is camera-confounded and EyePACS is the cleanest benchmark. Within groups of APTOS images of the same size the models still score AUROC 0.984-0.989 (post-hoc), so size alone does not explain the APTOS score.
- **Strong augmentation.** The failure was a BatchNorm train/eval mismatch from the noise and blur part. Geometric augmentation on all classes scores 0.977; on minority classes only it scores 0.834, because it leaks the label.

# 6. Limitations

- The frozen threshold does not transfer across datasets, so sensitivity alone is misleading off-domain; compare AUROC.
- APTOS may carry other capture cues beyond image size.
- DDR has no patient IDs, so same-patient leakage cannot be tested.
- Per-grade, specificity and image-size analyses were added after seeing the data and are post-hoc.
- Novelty claims depend on a literature search that is not yet done.

# 7. Remaining work

| Task | Status |
|---|---|
| Own-source operating points (post-hoc) | To do |
| Property analysis for RQ3: source prediction from properties, scores within brightness, quality and sharpness strata, normalisation tests | To do (size strata done) |
| Matched-subset check on APTOS (other capture cues) | To do |
| Use geometric augmentation on all classes; keep minority-only as the cue control | To do |
| Systematic literature search | To do |
| Thesis writing | To do |

# References

Key sources: International Diabetes Federation, *IDF Diabetes Atlas*, 11th ed., 2025. Gulshan, V., et al., *JAMA*, 2016. Li, T., et al., DDR dataset, *Information Sciences*, 2019. Kaggle EyePACS (2015) and APTOS 2019. Guo, C., et al., On calibration of modern neural networks, ICML, 2017. Tan, M., & Le, Q., EfficientNet, ICML, 2019. Torralba, A., & Efros, A., Unbiased look at dataset bias, CVPR, 2011. Geirhos, R., et al., Shortcut learning in deep neural networks, *Nature Machine Intelligence*, 2020. Zech, J. R., et al., *PLOS Medicine*, 2018. DeGrave, A. J., et al., *Nature Machine Intelligence*, 2021. *From Retinal Pixels to Patients: Evolution of Deep Learning Research in Diabetic Retinopathy Screening*, arXiv:2511.11065, 2025.
