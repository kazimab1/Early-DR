"""Step 1: read the datasets, freeze the patient-level splits, cache preprocessed images.

    python -m early_dr.prepare_data configs/base.yaml
Writes  splits.csv (frozen, never changed),  processed_512/*.jpg,  manifest.csv
"""
import argparse
import os

from .config import load_config
from .data import build_manifest
from .preprocess import preprocess_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/base.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    manifest = build_manifest(cfg)
    print(manifest.groupby(["source", "split"]).size().unstack(fill_value=0))
    manifest = preprocess_all(cfg, manifest, workers=cfg.train.workers)
    manifest.to_csv(os.path.join(cfg.paths.work, "manifest.csv"), index=False)
    print("manifest.csv written:", len(manifest), "images")


if __name__ == "__main__":
    main()
