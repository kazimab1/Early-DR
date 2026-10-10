"""Step 10 - OPTIONAL, if time allows (drop first if short): early-vs-late head (stage 2), three-class label,
colour normalisation. ViT/RETFound and 768 px need extra setup (see configs/extensions/).
Each is trained, then evaluated on the internal test sets.  Look at: early_vs_late AUROC and sensitivity for
late DR (stage 2), and per-source results for the others."""
from common import config, overrides, require_declared_analysis, run

require_declared_analysis()
for path in ["tasks/stage2.yaml", "extensions/three_class.yaml", "extensions/colour_graham.yaml"]:
    run("run_experiment", config(path), *overrides(), "--stage", "all", "--seeds", "42")
