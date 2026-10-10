"""Step 2 - data audit + class-distribution figure.
Look at: class_distribution, merge_composition (EyePACS share per grade), image statistics per source,
and how many duplicate groups cross splits. Files: <work>/audit/*.csv, figures/01_class_distribution.png"""
from common import WORK, config, overrides, run

run("audit", config("base.yaml"), *overrides())
run("plots", "--work", WORK, "--runs")
