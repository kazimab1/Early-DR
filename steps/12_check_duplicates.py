"""Step 12 - do duplicate images inflate the results? (uses saved predictions only; opens no test set again)
Look at: how many val/test/external images have a duplicate in the training set, specificity and
per-grade sensitivity, and whether the metrics change on 'without duplicates'.
Also check <work>/figures/duplicate_pairs.jpg by eye: true duplicates look identical."""
import settings
from common import WORK, run

run("dup_report", "--work", WORK, "--arms", *settings.ARMS)
