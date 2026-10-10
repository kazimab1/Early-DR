"""Step 1 - read DDR, EyePACS, APTOS; freeze patient-level splits; cache 512 px images.
Look at: the table of image counts per source and split.
Expected (from the plan): DDR 12,522 gradable images, EyePACS ~35k, APTOS 3,662."""
from common import config, overrides, run

run("prepare_data", config("base.yaml"), *overrides())
