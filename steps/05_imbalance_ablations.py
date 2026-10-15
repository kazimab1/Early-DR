"""Step 5 - imbalance ablations, one change at a time (five-grade head, DDR, natural prevalence):
samplers, losses (CE variants, focal gamma 1-3), augmentation + the augmentation control.
Single seed (42). Look at: the table of best validation scores at the end (choose on validation only),
then edit configs/imbalance/combined.yaml with the best sampler + loss + augmentation."""
import glob
import json
import os

from common import REPO, WORK, config, overrides, run

failed = []
for path in sorted(glob.glob(os.path.join(REPO, "configs", "imbalance", "*.yaml"))):
    try:
        run("run_experiment", config(f"imbalance/{os.path.basename(path)}"), *overrides(),
            "--stage", "train", "--seeds", "42")
    except Exception as err:                         # one failed run must not stop the others
        print("FAILED:", os.path.basename(path), err)
        failed.append(os.path.basename(path))

print("\nBest validation score per run (higher is better):")
names = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(REPO, "configs", "imbalance", "*.yaml"))]
rows = []
for name in names:
    info = os.path.join(WORK, "runs", name, "seed42", "train_info.json")
    if os.path.exists(info):
        rows.append((json.load(open(info))["best_val_score"], name))
for score, name in sorted(rows, reverse=True):
    print(f"  {score:.4f}  {name}")
if failed:
    print("\nThese runs failed and are missing from the table:", failed)
