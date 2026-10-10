"""The ONLY file you need to edit. Every numbered step reads it."""
import os

# Kaggle dataset folders (see the Input panel). IDRiD is not used.
PATHS = {
    "paths.ddr": "/kaggle/input/ddr-dataset",
    "paths.eyepacs": "/kaggle/input/diabetic-retinopathy-detection",
    "paths.aptos": "/kaggle/input/aptos2019-blindness-detection",
    "paths.work": "/kaggle/working",
}
SEEDS = [42, 43, 44]          # 3 seeds for the shortlist (use 5 if time allows)
ARMS = ["arm_A", "arm_B", "arm_C"]

# Quick test mode: set EARLYDR_TEST="train.epochs=1 data.prep_size=128 ..." to override anything.
EXTRA = os.environ.get("EARLYDR_TEST", "").split()
