"""Command line: train -> calibrate on validation -> evaluate.

    python -m early_dr.run_experiment configs/arms/arm_A.yaml --seeds 42 43 44
    python -m early_dr.run_experiment configs/arms/arm_A.yaml --stage evaluate --external
"""
import argparse
import os

import pandas as pd

from .config import load_config
from .evaluate import evaluate, fit_calibration
from .train import run_dir, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="repeat with these seeds")
    parser.add_argument("--set", nargs="*", default=[], help="override, e.g. train.epochs=3")
    parser.add_argument("--stage", choices=["all", "train", "evaluate"], default="all")
    parser.add_argument("--external", action="store_true",
                        help="also open IDRiD and APTOS (do this last, after model selection)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    manifest = pd.read_csv(os.path.join(cfg.paths.work, "manifest.csv"))
    for seed in args.seeds or [cfg.seed]:
        cfg["seed"] = seed
        if args.stage in ("all", "train"):
            path = train(cfg, manifest)
            fit_calibration(cfg, manifest, path)         # validation only, then frozen
        if args.stage in ("all", "evaluate"):
            evaluate(cfg, manifest, run_dir(cfg, seed), external=args.external)


if __name__ == "__main__":
    main()
