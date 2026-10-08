"""Data audit: class distribution, merge composition, image statistics per source, duplicates.

    python -m early_dr.audit configs/base.yaml
Everything is written to <work>/audit/.
"""
import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from .config import load_config
from .data import select_arm
from .preprocess import processed_path


def class_distribution(manifest):
    counts = manifest.groupby(["source", "grade"]).size().unstack(fill_value=0)
    return counts.join((counts.div(counts.sum(axis=1), axis=0) * 100).round(1), rsuffix="_%")


def merge_composition(manifest):
    """Version 1 merge (arm C): how much of each grade comes from EyePACS?
    A large share for some grades and none for others means the source predicts the grade."""
    merged = pd.concat([select_arm(manifest, "C", s) for s in ["train", "val", "test"]])
    table = merged.groupby(["grade", "source"]).size().unstack(fill_value=0)
    table["merged"] = table.sum(axis=1)
    table["EyePACS_share_%"] = (100 * table.get("EyePACS", 0) / table["merged"]).round(0)
    return table


def _image_stats(args):
    original, processed = args
    raw = cv2.imread(original)
    img = cv2.imread(processed)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retina = gray > 10
    return {"height": raw.shape[0], "width": raw.shape[1],
            "mean_b": img[..., 0][retina].mean(), "mean_g": img[..., 1][retina].mean(),
            "mean_r": img[..., 2][retina].mean(),
            "brightness": gray[retina].mean(),
            "sharpness": cv2.Laplacian(gray, cv2.CV_64F).var(),     # low = blurry
            "retina_fraction": retina.mean()}


def source_statistics(cfg, manifest, per_source=300, workers=4):
    """Resolution, field of view, mean colour and sharpness on a random sample per source."""
    rows = []
    for source, part in manifest.groupby("source"):
        sample = part.sample(min(per_source, len(part)), random_state=0)
        jobs = [(r.path, processed_path(cfg, r.image_id)) for r in sample.itertuples()]
        with Pool(workers) as pool:
            stats = pool.map(_image_stats, jobs)
        table = pd.DataFrame(stats)
        row = table.mean().round(2).to_dict()
        row.update(source=source, dark_fraction=float((table.brightness < 40).mean()),
                   blurry_fraction=float((table.sharpness < 20).mean()))
        rows.append(row)
    return pd.DataFrame(rows).set_index("source")


def _average_hash(path):
    img = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    small = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).tobytes()


def find_duplicates(cfg, manifest, workers=4):
    """Images with the same 16x16 average hash. Duplicates ACROSS splits are leakage."""
    paths = [processed_path(cfg, i) for i in manifest.image_id]
    with Pool(workers) as pool:
        hashes = pool.map(_average_hash, paths, chunksize=64)
    manifest = manifest.assign(hash=hashes)
    groups = manifest.groupby("hash").filter(lambda g: len(g) > 1)
    if groups.empty:
        return groups
    groups = groups.assign(
        group=groups.groupby("hash").ngroup(),
        crosses_splits=groups.groupby("hash").split.transform("nunique") > 1,
        crosses_sources=groups.groupby("hash").source.transform("nunique") > 1)
    return groups.drop(columns="hash").sort_values("group")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/base.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    manifest = pd.read_csv(os.path.join(cfg.paths.work, "manifest.csv"))
    out = os.path.join(cfg.paths.work, "audit")
    os.makedirs(out, exist_ok=True)

    tables = {"class_distribution": class_distribution(manifest),
              "merge_composition": merge_composition(manifest),
              "source_statistics": source_statistics(cfg, manifest, workers=cfg.train.workers)}
    for name, table in tables.items():
        print(f"\n== {name} ==\n{table}")
        table.to_csv(os.path.join(out, name + ".csv"))

    dup = find_duplicates(cfg, manifest, workers=cfg.train.workers)
    dup.to_csv(os.path.join(out, "duplicates.csv"), index=False)
    if dup.empty:
        print("\nNo duplicate images found.")
    else:
        print(f"\n{dup.group.nunique()} duplicate groups; "
              f"{dup[dup.crosses_splits].group.nunique()} cross splits (leakage), "
              f"{dup[dup.crosses_sources].group.nunique()} cross sources. See duplicates.csv")


if __name__ == "__main__":
    main()
