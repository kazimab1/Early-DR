"""Step 8 - LAST: open the external test set (APTOS). Do this only after model selection is finished.
Look at: the same metrics as step 6. If arm C beats B on its own test split but not on APTOS,
the model learned the source."""
import settings
from common import WORK, config, overrides, require_declared_analysis, run

require_declared_analysis()
for arm in settings.ARMS:
    run("run_experiment", config(f"arms/{arm}.yaml"), *overrides("task=stage1"),
        "--stage", "evaluate", "--external", "--seeds", *map(str, settings.SEEDS))
for metric in ["sensitivity", "auroc"]:
    run("summarise", "--work", WORK, "--baseline", settings.ARMS[0], "--experiments", *settings.ARMS[1:],
        "--source", "APTOS", "--metric", metric)
