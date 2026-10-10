"""Step 4 - train merge arms A (DDR), B (DDR+EyePACS source-balanced), C (version 1 merge, negative control)
with the early-DR head (0 vs 1-4). Also fits calibration on VALIDATION data and freezes it.
No test set is opened here.  Look at: validation score per epoch."""
import settings
from common import config, overrides, run

for arm in settings.ARMS:
    run("run_experiment", config(f"arms/{arm}.yaml"), *overrides("task=stage1"),
        "--stage", "train", "--seeds", *map(str, settings.SEEDS))
