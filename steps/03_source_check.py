"""Step 3 - can a classifier tell DDR from EyePACS by the image alone?
Look at: the printed AUROC (0.5 = cannot tell, close to 1 = the source is easy to detect)."""
from common import config, overrides, run

run("source_check", config("base.yaml"), "--a", "DDR", "--b", "EyePACS", *overrides())
