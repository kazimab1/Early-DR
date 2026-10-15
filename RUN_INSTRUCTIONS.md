# How to run this on Kaggle (and how long it takes)

## 1. Set up (5 minutes)
1. **Upload the code:** Kaggle -> Datasets -> New Dataset -> drag in `Early-DR.zip` -> name it `early-dr-code` -> Create.
   (Kaggle unpacks the zip itself.)
2. **New Notebook** -> upload `kaggle_notebook.ipynb` (it is also inside the zip, in the dataset).
3. **Add data** (right-hand panel): `early-dr-code`, plus the three image datasets:
   * **DDR** - either the official `train.txt`/`valid.txt`/`test.txt` split files, or the Kaggle copy with a single
     `DR_grading.csv` (both work). If the CSV has no split column, a seeded 60/20/20 split is made once and frozen.
   * **EyePACS** - use a copy that is already unzipped and contains `trainLabels.csv`
     Recommended: **"Diabetic Retinopathy (resized)" by tanlikesmath** (`trainLabels.csv` + `resized_train_cropped/`;
     the code uses the cropped folder automatically). Avoid the "Resized 2015 & 2019" copy: it mixes in APTOS images.
   * **APTOS 2019** (`train.csv` + `train_images/`)
4. **Settings:** Accelerator **GPU T4 x2 or P100** (the code uses one GPU), **Internet ON**
   (needs a phone-verified Kaggle account; it downloads the ImageNet weights - if it cannot, training starts from
   scratch and prints a WARNING), Persistence: Files only.
5. Run the cells top to bottom. The second cell prints the dataset folders it found; if one is `None`, write the folder
   in `steps/settings.py`.
6. Knobs in `steps/settings.py`: `EPOCHS` (10), `SEEDS` ([42, 43, 44]).

## 2. GPU or CPU?
**GPU for everything that trains** (steps 3, 4, 5, 10, and the inference in 6 and 8). On CPU, EfficientNet-B0 at 512 px
handles only a few images per second, so one epoch of arm A would take hours and the full run weeks; do not try it.
Steps 1, 2, 7 and 11 are CPU work (the GPU sits idle, but Kaggle still counts the hours).

## 3. How long? (estimates, not measurements)
Not yet run on the real data. Assumptions: T4 GPU, about 100 training images per second at 512 px with 4 CPU
workers, 10 epochs. The training sets are about 6.3k images (arm A), 31k (arm B), 9k (arm C), from the plan's counts and my split sizes.

| Step | What | Estimated time |
|---|---|---|
| 1 | Prepare data (52k images, CPU) | 20-60 min (full-size EyePACS originals: up to 1.5 h) |
| 2 | Audit + class figure | 5-10 min |
| 3 | Source check | 2-5 min |
| 4 | Train arms A, B, C x 3 seeds | **4-6 h** (A ~13 min, B ~55 min, C ~17 min per seed) |
| 5 | Imbalance ablations (12 runs, 1 seed) | 2-3.5 h (set `EPOCHS = 5` for ~1.3 h) |
| 6 | Internal test evaluation (9 runs) | 30-60 min (bootstrap on CPU dominates) |
| 7 | Arm comparisons | 10-25 min |
| 8 | APTOS evaluation | 10-20 min |
| 9 | Saliency | 2-5 min |
| 10 | Optional extensions | 45-75 min |
| 11 | Figures | 1-2 min |

* **Core without ablations (1-4, 6-9, 11): about 6-8 h.**
* **With ablations: about 9-11.5 h.** Plus about 1 h for step 10.
* **Quick first pass** (`SEEDS = [42]`, `EPOCHS = 5`): about 2.5-3 h in total. Do this first to find problems cheaply.

**Measure it yourself:** step 4 prints the seconds per epoch (`epoch 1: ... 74 s`). Arm B's epoch takes about 5x arm A's,
arm C's about 1.4x. Multiply by epochs x seeds to correct the table.

## 4. Kaggle limits to plan around
* One session runs at most about 12 hours; the weekly GPU quota (about 30 h at the time of writing, check your account)
  is shared with everything else you run.
* Files in `/kaggle/working` disappear when the session ends unless you **Save Version -> Save & Run All (Commit)**
  and let it finish. Expect about 4 GB of outputs (limit 20 GB).
* **Splitting across sessions:** run steps 1-4 in one session and Save Version. In a new notebook add that first
  notebook's output as input and start with a cell: `!cp -r /kaggle/input/<first-notebook>/* /kaggle/working/`
  (it restores `splits.csv`, `manifest.csv`, `processed_512/` and `runs/`), then continue with step 5 onwards.
  Add the code dataset and the image datasets again. Never delete `splits.csv`: it keeps the splits frozen.

## 5. Order and what to keep from each step
See the table in the notebook and in `README.md`. Fill in the declaration cell (it stamps a date on
`DECLARED_ANALYSIS.md`) only after step 5, before step 6. Results are in `/kaggle/working/runs/...` and
`/kaggle/working/figures/`.

## 6. If the GPU quota or the session runs out
* **Do not rely on a half-finished "Save & Run All".** I cannot promise Kaggle keeps outputs of a run that was stopped.
  Work in chunks that each finish inside your quota (e.g. steps 1-3, then arm A, then arm B ...; set `ARMS` and
  `SEEDS` in `steps/settings.py`) and **Save Version -> Save & Run All** for each chunk, so each one is saved whole.
* **To continue:** open a new notebook, add (1) the code dataset, (2) the image datasets and (3) the previous
  notebook's output as input, then run the **Resume** cell. It restores `splits.csv`, `processed_512/` and
  `runs/`.
* Step 4 skips any arm/seed that is already trained and calibrated, so just run it again. A run that was
  interrupted part-way restarts from epoch 1 (no mid-run checkpoint). Use `--force` to retrain a finished one.
* Step 1 also skips images that are already processed. Keep `splits.csv`: it is what freezes the splits.
