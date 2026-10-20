"""Step 13 - can simple image properties (size, file size, colour, sharpness) predict early DR on their own?
CPU only, a few minutes. If AUROC is well above 0.5 on a dataset, high CNN scores there may come from
capture artefacts, not the disease. Compare with the CNN AUROCs from steps 6 and 8."""
import settings
from common import WORK, run

run("shortcut_check", "--work", WORK)
