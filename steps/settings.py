"""The ONLY file you may need to edit. Every numbered step reads it."""
import glob
import os

INPUT_ROOT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")


def _parent_of(pattern):
    """Folder holding the first file/folder matching pattern anywhere under the Kaggle input folder."""
    hits = sorted(glob.glob(os.path.join(INPUT_ROOT, "**", pattern), recursive=True))
    return os.path.dirname(hits[0]) if hits else None


# "auto" = find the dataset by its files. Or write the folder yourself, e.g. "/kaggle/input/my-ddr".
DDR = "auto"        # looks for DR_grading/train.txt
EYEPACS = "auto"    # looks for trainLabels.csv
APTOS = "auto"      # looks for train.csv next to a train_images folder

PATHS = {
    "paths.ddr": _parent_of("train.txt") if DDR == "auto" else DDR,
    "paths.eyepacs": _parent_of("trainLabels.csv") if EYEPACS == "auto" else EYEPACS,
    "paths.aptos": _parent_of("train_images") if APTOS == "auto" else APTOS,
    "paths.work": "/kaggle/working",
}

EPOCHS = 10                   # lower (e.g. 5) to save time
SEEDS = [42, 43, 44]          # 3 seeds for the shortlist (use [42] for a quick first pass)
ARMS = ["arm_A", "arm_B", "arm_C"]

# Quick test mode: EARLYDR_TEST="train.epochs=1 data.prep_size=128 ..." overrides anything.
EXTRA = os.environ.get("EARLYDR_TEST", "").split()
