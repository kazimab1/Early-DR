"""Step 11 - all figures -> <work>/figures/ (training curves, ROC/PR, reliability, per-grade, comparison,
validation scores).  Also shows them in the notebook if run there."""
import glob
import os

import settings
from common import WORK, run

run("plots", "--work", WORK, "--runs", *settings.ARMS)
try:
    from IPython.display import Image, display
    for f in sorted(glob.glob(os.path.join(WORK, "figures", "*.png"))):
        print(f)
        display(Image(f))
except ImportError:
    pass
