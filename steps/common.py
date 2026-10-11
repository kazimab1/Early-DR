"""Small helpers shared by the steps."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import settings  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = dict(i.split('=', 1) for i in settings.EXTRA if i.startswith('paths.work=')).get('paths.work', settings.PATHS['paths.work'])


def overrides(*more):
    """--set arguments: dataset paths, any test overrides, then step-specific ones."""
    missing = [k for k, v in settings.PATHS.items() if not v]
    if missing and not settings.EXTRA:
        sys.exit(f"STOP: could not find a dataset folder for {missing}. Add the dataset to the notebook "
                 "(Input panel) or write its folder in steps/settings.py.")
    items = [f"{k}={v}" for k, v in settings.PATHS.items()] + [f"train.epochs={settings.EPOCHS}"] \
        + settings.EXTRA + list(more)
    return ["--set", *items]


def run(module, *args):
    """python -m early_dr.<module> ... ; stops the step if the command fails."""
    command = [sys.executable, "-m", f"early_dr.{module}", *args]
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def config(path):
    return os.path.join("configs", path)


def require_declared_analysis():
    """Rule 2: the analysis is written down BEFORE any test split is opened."""
    text = open(os.path.join(REPO, "DECLARED_ANALYSIS.md")).read()
    if "____" in text:
        sys.exit("STOP: fill in the date in DECLARED_ANALYSIS.md (and commit it) before opening any test set.")
