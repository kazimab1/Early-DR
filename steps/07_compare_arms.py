"""Step 7 - Holm-corrected, patient-level bootstrap comparison: B vs A and C vs A on DDR and EyePACS.
Look at: difference_vs_baseline and p_holm. (C is the negative control.)"""
import settings
from common import WORK, run

base, others = settings.ARMS[0], settings.ARMS[1:]
for source, metric in [("DDR", "sensitivity"), ("DDR", "pr_auc"), ("EyePACS", "sensitivity"), ("EyePACS", "pr_auc")]:
    run("summarise", "--work", WORK, "--baseline", base, "--experiments", *others,
        "--source", source, "--metric", metric)
