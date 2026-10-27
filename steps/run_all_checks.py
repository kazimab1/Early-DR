"""Run the three 'results look too good' checks in order:
  step 12  duplicate leakage (needs steps 6 and 8 done; uses saved predictions only)
  step 13  image-property shortcut check (CPU)
  step 14  strong-augmentation diagnosis (GPU recommended)
A failing check does not stop the others."""
import os
import subprocess
import sys

import settings
from common import REPO, WORK

here = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(WORK, "audit", "duplicates.csv")):
    print("audit/duplicates.csv not found: running step 2 first")
    subprocess.run([sys.executable, os.path.join(here, "02_audit.py")], cwd=REPO)
for step in ["12_check_duplicates.py", "13_shortcut_check.py", "14_diagnose_strong_aug.py"]:
    print("\n" + "=" * 78 + f"\n{step}\n" + "=" * 78, flush=True)
    result = subprocess.run([sys.executable, os.path.join(here, step)], cwd=REPO)
    if result.returncode:
        print(f"!! {step} failed (exit code {result.returncode}); continuing with the next check")
