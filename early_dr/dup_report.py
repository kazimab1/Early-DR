"""Are the test results inflated by duplicate images?  Uses only files already on disk.

    python -m early_dr.dup_report --work /kaggle/working
1. Summarises audit/duplicates.csv (hash matches are CANDIDATES for duplicates).
2. For each arm: finds val/test/external images that have a duplicate among that arm's TRAINING images.
3. Recomputes the early-DR (grades 1-2 vs No DR) metrics from the saved predictions, on all images and on
   the images without a training duplicate. Models are not run, so no test set is opened again.
4. Saves a picture of a few duplicate pairs to look at: <work>/figures/duplicate_pairs.jpg
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd

from . import metrics as M
from .data import select_arm


def leaky_images(manifest, dups, arm):
    """Ids of non-training images that share a hash with one of this arm's training images."""
    train_ids = set(select_arm(manifest, arm, "train").image_id)
    groups = dups[dups.image_id.isin(train_ids)].group.unique()
    hit = dups[dups.group.isin(groups) & ~dups.image_id.isin(train_ids)]
    return set(hit.image_id)


def early_metrics(preds, threshold):
    """Early DR (grades 1-2) vs No DR at the frozen threshold, plus grade-wise detection."""
    sub = preds[preds.grade <= 2]
    y, score = (sub.grade > 0).astype(int).values, sub.dr_score.values
    if len(np.unique(y)) < 2:
        return {}
    flagged = score > threshold
    g = sub.grade.values
    return {"n": len(sub), "specificity": (~flagged)[g == 0].mean(),
            "sens_grade1": flagged[g == 1].mean() if (g == 1).any() else np.nan,
            # how well is each grade separated from No DR? "early" is mostly grade 2 in DDR, which is easy
            "auroc_g1_vs_0": M.auroc((g[g <= 1] > 0).astype(int), score[g <= 1]),
            "auroc_g2_vs_0": M.auroc((g[g != 1] > 0).astype(int), score[g != 1]),
            "sens_grade2": flagged[g == 2].mean() if (g == 2).any() else np.nan,
            "sens_early": flagged[y == 1].mean(), "pr_auc": M.pr_auc(y, score), "auroc": M.auroc(y, score)}


def picture(dups, work, size=160, pairs=6):
    both = dups[dups.crosses_splits | dups.crosses_sources]
    rows = []
    for _, g in both.groupby("group"):
        if len(g) >= 2:
            rows.append(g.iloc[:2])
        if len(rows) == pairs:
            break
    if not rows:
        return
    folder = sorted(glob.glob(os.path.join(work, "processed_*")))[0]
    tiles = []
    for pair in rows:
        pic = []
        for r in pair.itertuples():
            img = cv2.imread(os.path.join(folder, r.image_id + ".jpg"))
            img = cv2.resize(img if img is not None else np.zeros((8, 8, 3), np.uint8), (size, size))
            cv2.putText(img, f"{r.source} {r.split} g{r.grade}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            pic.append(img)
        tiles.append(np.concatenate(pic, axis=1))
    os.makedirs(os.path.join(work, "figures"), exist_ok=True)
    cv2.imwrite(os.path.join(work, "figures", "duplicate_pairs.jpg"), np.concatenate(tiles, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/kaggle/working")
    parser.add_argument("--arms", nargs="+", default=["arm_A", "arm_B", "arm_C"])
    args = parser.parse_args()
    work = args.work
    manifest = pd.read_csv(os.path.join(work, "manifest.csv"))
    dups = pd.read_csv(os.path.join(work, "audit", "duplicates.csv"))
    if dups.empty:
        print("No duplicate candidates.")
        return

    print(f"{dups.group.nunique()} candidate groups, {len(dups)} images")
    # do the images in a group even agree on the grade? If not, it is probably a hash collision
    same_grade = dups.groupby("group").grade.nunique().eq(1)
    print(f"  groups whose images all have the SAME grade: {same_grade.sum()} of {len(same_grade)} "
          "(different grades usually means a hash collision, not a true duplicate)")
    cross = dups[dups.crosses_sources].groupby("group").source.apply(lambda s: " + ".join(sorted(set(s))))
    print("  groups across sources:\n" + cross.value_counts().to_string())
    within = dups[dups.crosses_splits & ~dups.crosses_sources].groupby("group").source.first()
    print("  groups across splits inside one source:\n" + within.value_counts().to_string())
    picture(dups, work)

    arm_of = {"arm_A": "A", "arm_B": "B", "arm_C": "C", "arm_D": "D"}
    rows = []
    for name in args.arms:
        leaky = leaky_images(manifest, dups, arm_of[name])
        print(f"\n{name}: {len(leaky)} val/test/external images have a candidate duplicate in its training set")
        print(manifest[manifest.image_id.isin(leaky)].groupby(["source", "split"]).size().to_string())
        for source in ["DDR", "EyePACS", "APTOS"]:
            for variant in ["all", "without duplicates"]:
                per_seed = []
                for f in sorted(glob.glob(os.path.join(work, "runs", name, "seed*", f"preds_{source}.csv"))):
                    preds = pd.read_csv(f)
                    if variant != "all":
                        preds = preds[~preds.image_id.isin(leaky)]
                    thr = json.load(open(os.path.join(os.path.dirname(f), "calibration.json")))["threshold_dr"]
                    res = early_metrics(preds, thr)
                    if res:
                        per_seed.append(res)
                if per_seed:
                    mean = pd.DataFrame(per_seed).mean().round(3).to_dict()
                    rows.append({"arm": name, "source": source, "images": variant, **mean})
    table = pd.DataFrame(rows)
    out = os.path.join(work, "audit", "duplicate_sensitivity.csv")
    table.to_csv(out, index=False)
    print("\nEarly DR (grades 1-2) vs No DR at the frozen threshold, mean over seeds\n")
    print(table.to_string(index=False))
    print("\nsaved", out)


if __name__ == "__main__":
    main()
