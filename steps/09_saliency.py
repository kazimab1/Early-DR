"""Step 9 - Grad-CAM on early-DR DDR test images with the randomised-weights sanity check.
Look at: spearman_trained_vs_random_weights (close to 0 = the maps depend on the model) and the pictures in
<work>/runs/arm_A/seed42/saliency/ (image | trained map | random-weights map)."""
from common import config, overrides, run

run("saliency", config("arms/arm_A.yaml"), "--n", "20", *overrides("task=stage1"))
