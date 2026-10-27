# Early detection of diabetic retinopathy (grading only)

Code for the thesis plan *"Early detection of diabetic retinopathy, version 2"*: grades 1-2 against No DR
at a pre-declared specificity, calibrated probabilities, and tests on unseen data.
No segmentation model is trained anywhere. Written to run on **Kaggle** (`kaggle_notebook.ipynb`).

## How to run on Kaggle
See `RUN_INSTRUCTIONS.md` for Kaggle setup and timings. Run the numbered files in `steps/` in order (`01_prepare_data.py` ... `11_figures.py`); dataset paths and seeds are in `steps/settings.py`. `kaggle_notebook.ipynb` just calls them one per cell.

1. New notebook, upload `kaggle_notebook.ipynb`, turn on **GPU** and **Internet**.
2. Add the datasets: DDR, EyePACS (diabetic-retinopathy-detection), APTOS 2019. (IDRiD is optional: set `paths.idrid`.)
3. Edit the `PATHS` cell, then run the cells top to bottom.

Everything the notebook does is a `python -m early_dr.<module>` command, so you can also run them one by one.

## Plan -> code
| Plan item | Where |
|---|---|
| Data audit (class counts, image statistics, duplicates, merge composition) | `early_dr/audit.py` |
| Source classifier (does the image reveal its dataset?) | `early_dr/source_check.py` |
| Patient-level frozen splits (rule 1) | `early_dr/data.py` -> `splits.csv`, never recomputed |
| Preprocess: crop border, pad to square, resize (512 px) | `early_dr/preprocess.py` |
| Merge arms A (DDR), B (source-balanced), C (version 1 merge, negative control) | `data.select_arm`, `dataset.make_sampler` |
| Labels: 0 vs 1-4, 1-2 vs 3-4, 0/Early/Late, five-grade | `data.task_label`, `task:` in the config |
| EfficientNet-B0, ViT with retinal weights | `early_dr/model.py` |
| Imbalance: samplers, losses, augmentation (+ augmentation control) | `dataset.py`, `losses.py`, `configs/imbalance/` |
| Temperature scaling, prior-shift correction, frozen threshold (rule 4) | `early_dr/calibration.py`, `evaluate.fit_calibration` |
| PR-AUC, AUROC, sensitivity at specificity, ECE, Brier, QWK, per-grade recall, per source | `early_dr/metrics.py`, `evaluate.py` |
| Patient-level bootstrap 95% CI, Holm correction, 3-5 seeds | `metrics.py`, `summarise.py`, `--seeds` |
| Test sets opened once, external last (rule 6) | `evaluate.py` (`--external`, `test_opened.log`) |
| Colour normalisation, AdaBN | `dataset.colour_normalise`, `model.adabn` |
| Saliency + randomised-weights sanity check + optional DDR lesion overlap | `early_dr/saliency.py` |

## The six protocol rules in practice
1. **Freeze the splits first** - `prepare_data` writes `splits.csv` once. EyePACS is split by patient (file name `10_left` -> patient 10), 70/15/15. DDR uses its official train/valid/test split (it has no patient IDs); `audit` checks for duplicates across splits.
2. **Declare the analysis** - fill in `DECLARED_ANALYSIS.md` and commit it before running the evaluate stage.
3. **One change per run** - one config file per experiment; each file lists only what it changes from its parent (`base: ...`). Extra changes: `--set key=value`. Use `--set name=...` if you change something without a new file, otherwise the run folder is reused.
4. **Select on validation only** - the best epoch, temperature, prior correction and threshold all come from validation data.
5. **Measure the noise** - `--seeds 42 43 44`, then `summarise.py`.
6. **Open the test once, external last** - the evaluate stage opens the internal sets only; the external set (APTOS) is opened separately with `--external`.

## Definitions used (so the numbers can be checked)
* **Early-vs-none (primary):** images of grade 0, 1, 2 only; positive = grade 1-2. Score = P(any DR) = 1 - P(grade 0).
  The threshold is the validation threshold at the declared specificity (`declared.specificity`, default **0.90** - change it before opening any test set). Sensitivity is reported at that frozen threshold, together with the specificity actually reached.
* **Also reported:** grade 1 vs No DR; grade 2+ vs grades 0-1; early vs late (DR-positive images, AUROC and sensitivity for late).
* **ECE / Brier:** for "any DR" on all images of a test set. PR-AUC is always printed with the test prevalence.
* **Prior-shift correction:** probabilities are re-weighted from the class mix the model effectively trained on (sampler x loss weights) to the natural prevalence of the validation set. The EM estimate of each source's own prevalence is saved as `estimated_prior_em` for comparison. For focal loss no weight correction is applied.
* **Arm B** draws images with equal chance from each source inside every grade (grade frequencies stay natural). **Arm C** is DDR plus EyePACS grades 1, 3, 4 only, exactly as in version 1.
* **`aug_scope: minority`** means every class except the largest; **`all`** is the control.

## Layout
```
early_dr/        the code (one small file per topic)
configs/         base.yaml, arms/, imbalance/, extensions/, tasks/
kaggle_notebook.ipynb
DECLARED_ANALYSIS.md
```
Outputs (in `/kaggle/working`): `splits.csv`, `manifest.csv`, `processed_512/`, `audit/`,
`runs/<experiment>/seed<N>/{best.pt, train_info.json, calibration.json, results.json, preds_<source>.csv}`.

## Honest notes
* Tested end to end only on small **synthetic** images (all arms, all four tasks, sampler/loss/augmentation options, AdaBN, saliency, summarise). It has **not** been run on the real datasets, so the file readers in `data.py` assume these layouts: DDR `train/valid/test.txt` (`name label`) or a single `DR_grading.csv` (name + grade [+ split]), EyePACS `trainLabels.csv` (`image,level`), IDRiD `*Training Labels*.csv` / `*Testing Labels*.csv`, APTOS `train.csv` (`id_code,diagnosis`). If a Kaggle copy differs, only the `load_*` functions need changing.
* The ViT/RETFound path and the DDR lesion-overlap check need files I could not test (RETFound checkpoint, lesion masks).
* Saliency (Grad-CAM) supports the CNN backbone only.
* The novelty claims in the plan are left to the literature search, as the plan says.

## Figures (`python -m early_dr.plots`)
Written to `<work>/figures/`: `01_class_distribution`, `02_training_<run>`, `03_roc_pr_<run>`, `04_reliability_<run>`,
`05_flagged_by_grade_<run>`, `06_confusion_<run>` (five-grade runs), `07_comparison_sensitivity` / `_auroc`
(arms side by side, per source, 95% CI), `08_validation_scores`. Plus the Grad-CAM pictures from `saliency.py`.
Figures are drawn from the saved `results.json` / `preds_*.csv`; no model is loaded.

## Checks for results that look too good
`steps/12_check_duplicates.py` (duplicates, per-grade AUROC; uses saved predictions) and `steps/13_shortcut_check.py` (can image size, file size, colour or sharpness predict early DR alone?).

`steps/14_diagnose_strong_aug.py` diagnoses why strong augmentation failed; `steps/run_all_checks.py` runs steps 12-14.
