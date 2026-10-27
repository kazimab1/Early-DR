"""ONE FILE for a NEW Kaggle notebook: restores your finished run from the old notebook's output and runs
the three checks. Nothing is retrained and no test set is opened again.

Before running, in the new notebook:
  * Accelerator: GPU on, Internet: On
  * Add as inputs: (1) your OLD notebook's output (the one with the finished runs), (2) the code dataset
    (the Early-DR.zip dataset), and, for step 13 only, the DDR, EyePACS and APTOS datasets.
Then paste this file into a cell and run it.

  step 12  duplicate leakage          (saved predictions only)
  step 13  image-property shortcut    (CPU; needs the three image datasets)
  step 14  strong-augmentation diagnosis (GPU, roughly 15 minutes)
Results: /kaggle/working/audit/*.csv and /kaggle/working/figures/
"""
import base64, glob, json, os, shutil, subprocess, sys, zlib
INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("EARLYDR_WORK", "/kaggle/working")
CODE = os.environ.get("EARLYDR_CODE", os.path.join(WORK, "code"))
PAYLOAD = (
    "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"
)

# 1. code folder: copy it from the code dataset if this notebook does not have it yet
if not os.path.exists(os.path.join(CODE, "steps", "settings.py")):
    found = glob.glob(os.path.join(INPUT, "**", "steps", "settings.py"), recursive=True)
    if not found:
        sys.exit("STOP: add the code dataset (the Early-DR.zip you uploaded) as an input of this notebook.")
    shutil.copytree(os.path.dirname(os.path.dirname(found[0])), CODE, dirs_exist_ok=True)
    print("code folder copied from", os.path.dirname(os.path.dirname(found[0])))
try:
    import timm  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"])

# 2. install the check scripts (also refreshes older copies)
for name, text in json.loads(zlib.decompress(base64.b64decode(PAYLOAD))).items():
    path = os.path.join(CODE, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text)
    print("installed", name)

# 3. restore the finished run from the old notebook's output (the big image folder is linked, not copied)
if not os.path.exists(os.path.join(WORK, "splits.csv")):
    candidates = [os.path.dirname(f) for f in glob.glob(os.path.join(INPUT, "**", "splits.csv"), recursive=True)]
    candidates = [c for c in candidates if os.path.abspath(c) != os.path.abspath(WORK)]
    if not candidates:
        sys.exit("STOP: add your old notebook's output as an input of this notebook.")
    # prefer the output that contains the imbalance runs
    best = [c for c in candidates if os.path.exists(os.path.join(c, "runs", "aug_strong_all", "seed42", "best.pt"))]
    previous = (best or candidates)[0]
    print("restoring from", previous)
    for name in ["splits.csv", "manifest.csv"]:
        shutil.copy(os.path.join(previous, name), os.path.join(WORK, name))
    for name in ["audit", "runs", "figures"]:
        if os.path.exists(os.path.join(previous, name)):
            shutil.copytree(os.path.join(previous, name), os.path.join(WORK, name), dirs_exist_ok=True)
    for name in os.listdir(previous):
        if name.startswith("processed_") and not os.path.exists(os.path.join(WORK, name)):
            os.symlink(os.path.join(previous, name), os.path.join(WORK, name))
else:
    print("run files already in", WORK)

# 4. run the checks
subprocess.run([sys.executable, "steps/run_all_checks.py"], cwd=CODE)
