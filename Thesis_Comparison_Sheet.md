---
title: "Thesis comparison sheet"
subtitle: "Disease or Source? Testing Merged Datasets for Early Diabetic Retinopathy Detection"
---

Use the **This thesis** column as a reference and fill the other columns from the theses you compare. Numbers are from my group-split runs, three seeds (42, 43, 44), unless stated. "Post-hoc" means the analysis was added after seeing results.

# 1. Comparison table

| Attribute | This thesis | Other thesis A | Other thesis B |
|---|---|---|---|
| Domain and task | Retinal fundus images; early diabetic retinopathy (grades 1-2) vs No DR | | |
| Main question | When datasets are merged, does the model learn the condition or the source? | | |
| Contribution type | Evidence and protocol (not a new network) | | |
| Datasets | DDR (12,522 images), EyePACS (35,108), APTOS (3,662) | | |
| How datasets are combined | Three arms: A DDR only; B DDR + all EyePACS, source-balanced; C DDR + EyePACS grades 1, 3, 4 (negative control) | | |
| Model | EfficientNet-B0, 512 px, ImageNet weights, 10 epochs | | |
| Splits | EyePACS by patient 70/15/15; DDR random 60/20/20 (no patient IDs) plus a group-aware version; frozen files | | |
| External test | APTOS, never used for training or selection | | |
| Metrics | Sensitivity at a frozen threshold (specificity 0.90 target), AUROC, PR-AUC, per-grade AUROC, ECE, Brier | | |
| Calibration | Temperature scaling and prior-shift correction, fitted on validation | | |
| Uncertainty | 3 seeds; patient-level bootstrap 95% intervals | | |
| Leakage checks | Near-duplicate detection; group-aware DDR split | | |
| Shortcut checks | Image size, file size, colour, sharpness; size-stratified AUROC (post-hoc) | | |
| Key result | Old merge: EyePACS AUROC 0.717 but specificity 0.008 (flags almost every healthy image). Source-balanced: AUROC 0.884 (0.870-0.897), specificity 0.898, ECE 0.056 | | |
| In-domain vs external | DDR AUROC about 0.98 for every arm; EyePACS 0.717 to 0.884 | | |
| Novelty claim | Not yet verified; no study found in three quick searches; systematic search pending | | |
| Reproducibility | Code and Kaggle notebooks, frozen splits, declaration files | | |
| Main limitations | One architecture; three datasets; DDR lacks patient IDs; camera model unknown (proxies only); arm C is a deliberately flawed control | | |
| Status | Core experiments done; RQ3 analysis, natural-merge arm and literature search pending | | |

# 2. Rigour checklist (this thesis)

| Item | This thesis | Other A | Other B |
|---|---|---|---|
| Patient-level split | Partly (EyePACS yes; DDR no IDs) | | |
| External test set never used for training or selection | Yes (APTOS) | | |
| Threshold and calibration fitted on validation only | Yes | | |
| Several seeds | Yes (3) | | |
| Confidence intervals | Yes (patient-level bootstrap) | | |
| Calibration reported (ECE, Brier) | Yes | | |
| Negative control | Yes (arm C) | | |
| Realistic baseline merge (natural mix of all grades) | No (arm D planned) | | |
| Leakage checked | Yes (near-duplicates; same-patient untestable in DDR) | | |
| Shortcut / confounder analysis | Yes (image properties; RQ3 extension pending) | | |
| More than one architecture | No | | |
| More than one external dataset | No (IDRiD planned) | | |
| Pre-declared analysis, post-hoc labelled | Yes | | |
| Systematic literature review | Not yet | | |
| Comparison with clinician grading | No | | |

# 3. Questions to ask of any other thesis

1. Was the test set truly unseen: for patients, for images, and for the source dataset?
2. Was the threshold chosen on validation data, or on the test set?
3. Are intervals and several seeds reported, or a single run?
4. Is performance reported per source, or only on one pooled test set?
5. Could image properties (size, file size, brightness) predict the label without the disease? Was it checked?
6. If datasets are merged, is there a control showing the model did not learn the source?
7. Are probabilities calibrated, and does calibration hold on a new source?

# 4. What this thesis claims, and what it does not

**Claims (supported by my runs):** a naive merge can teach the source, source-balancing helps on unseen patients, in-domain scores are inflated, and calibration and the threshold do not transfer across datasets.

**Does not claim:** a new network, better accuracy than published systems, clinical readiness, or first-of-its-kind novelty (not yet verified).
