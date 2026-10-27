"""Step 14 - why did strong augmentation fail to train?  (train/validation only; no test set is opened)
Prints augmented-image statistics, scores the saved strong-augmentation models normally / with AdaBN / on
augmented validation images, and retrains briefly on a subset with each part of the augmentation.
Needs a GPU for the retrain part (a few minutes per variant). Add --skip-retrain to skip it."""
import sys

from common import config, overrides, run

run("aug_diagnose", config("imbalance/aug_strong_all.yaml"), *overrides(), *sys.argv[1:])
