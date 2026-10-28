"""Is a model's APTOS score coming from image SIZE?  Compare its AUROC overall with its AUROC inside groups of
images that have the SAME size (inside such a group, size cannot help tell early DR from No DR).

Uses saved predictions and image sizes only: no model is run and no test set is opened again.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score


def image_sizes(table):
    """Width and height of every original image (cheap: only the file header is read)."""
    sizes = []
    for path in table.path:
        try:
            with Image.open(path) as im:
                sizes.append(im.size)
        except OSError:
            sizes.append((0, 0))
    return pd.DataFrame({"width": [s[0] for s in sizes], "height": [s[1] for s in sizes]}, index=table.index)


def aptos_table(work, sizes_cache=None):
    """APTOS early-DR subset (grades 0-2) with width and height."""
    manifest = pd.read_csv(os.path.join(work, "manifest.csv"))
    aptos = manifest[(manifest.source == "APTOS") & (manifest.grade <= 2)].reset_index(drop=True)
    if sizes_cache and os.path.exists(sizes_cache):
        feats = pd.read_csv(sizes_cache)[["image_id", "width", "height"]]
        aptos = aptos.merge(feats, on="image_id", how="left")
        if aptos.width.notna().all():
            print("image sizes taken from", sizes_cache)
            return aptos
    print("reading image sizes from the original APTOS files")
    return pd.concat([aptos, image_sizes(aptos)], axis=1)


def pooled_auroc(y, score, group):
    """AUROC using only pairs of images from the same group (weights = positives x negatives per group)."""
    total, weight = 0.0, 0.0
    for g in np.unique(group):
        m = group == g
        pos, neg = y[m].sum(), (1 - y[m]).sum()
        if pos >= 1 and neg >= 1:
            total += roc_auc_score(y[m], score[m]) * pos * neg
            weight += pos * neg
    return total / weight if weight else np.nan


def usable_groups(table, min_images=30, min_each=8):
    """Size groups big enough, and with enough of both classes, to say anything."""
    key = table.width.astype(str) + "x" + table.height.astype(str)
    y = (table.grade > 0).astype(int)
    stats = pd.DataFrame({"size": key, "y": y}).groupby("size").y.agg(["count", "sum"])
    stats.columns = ["images", "early_dr"]
    stats["prevalence"] = (stats.early_dr / stats.images).round(3)
    stats["usable"] = (stats.images >= min_images) & (stats.early_dr >= min_each) & (stats.images - stats.early_dr >= min_each)
    return key, stats


def bootstrap_pooled(y, score, group, n=300, seed=0):
    rng = np.random.RandomState(seed)
    idx_by_group = [np.where(group == g)[0] for g in np.unique(group)]
    values = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in idx_by_group])
        values.append(pooled_auroc(y[idx], score[idx], group[idx]))
    return tuple(np.nanpercentile(values, [2.5, 97.5]))


def analyse(work, arms, sizes_cache=None, min_images=30, min_each=8):
    aptos = aptos_table(work, sizes_cache)
    key, stats = usable_groups(aptos, min_images, min_each)
    aptos["size"] = key
    keep = aptos["size"].isin(stats.index[stats.usable])
    print(f"{len(aptos)} APTOS images (grades 0-2): {len(stats)} image sizes, {int(stats.usable.sum())} usable groups "
          f"covering {int(keep.sum())} images ({keep.mean():.0%})")
    rows = []
    for arm in arms:
        per_seed = []
        for f in sorted(glob.glob(os.path.join(work, "runs", arm, "seed*", "preds_APTOS.csv"))):
            preds = pd.read_csv(f)[["image_id", "dr_score"]]
            t = aptos.merge(preds, on="image_id")
            y, s, g = (t.grade > 0).astype(int).values, t.dr_score.values, t["size"].values
            usable = t["size"].isin(stats.index[stats.usable]).values
            per_seed.append({"overall": roc_auc_score(y, s),
                             "overall (usable sizes only)": roc_auc_score(y[usable], s[usable]),
                             "within same size": pooled_auroc(y[usable], s[usable], g[usable]),
                             "ci": bootstrap_pooled(y[usable], s[usable], g[usable])})
        if per_seed:
            mean = lambda k: float(np.mean([p[k] for p in per_seed]))
            rows.append({"arm": arm, "seeds": len(per_seed), "overall": round(mean("overall"), 3),
                         "overall, usable sizes": round(mean("overall (usable sizes only)"), 3),
                         "within same size": round(mean("within same size"), 3),
                         "within-size 95% CI (seed 1)": tuple(round(v, 3) for v in per_seed[0]["ci"])})
    return pd.DataFrame(rows), stats.sort_values("images", ascending=False)


def verdict(table):
    lines = []
    for _, r in table.iterrows():
        drop = r["overall, usable sizes"] - r["within same size"]
        if drop < 0.02:
            lines.append(f"- {r['arm']}: AUROC within same-size groups is {r['within same size']} (overall {r['overall, usable sizes']}): "
                         "the score does NOT come from image size.")
        else:
            lines.append(f"- {r['arm']}: AUROC falls from {r['overall, usable sizes']} to {r['within same size']} within same-size groups: "
                         "part of the score may come from image size.")
    return "\n".join(lines)
