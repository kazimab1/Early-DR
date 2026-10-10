"""Step 6 - open the INTERNAL test sets: DDR test and held-out EyePACS patients (plus each arm's pooled test).
Requires DECLARED_ANALYSIS.md to be filled in.  Look at: results.json per run
(sensitivity at the declared specificity, PR-AUC, AUROC, ECE, per-grade recall, 95% intervals)."""
import settings
from common import config, overrides, require_declared_analysis, run

require_declared_analysis()
for arm in settings.ARMS:
    run("run_experiment", config(f"arms/{arm}.yaml"), *overrides("task=stage1"),
        "--stage", "evaluate", "--seeds", *map(str, settings.SEEDS))
