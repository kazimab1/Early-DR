"""ONE FILE for a NEW Kaggle notebook: restores your finished run from the old notebook's output and trains the
two geometric-augmentation runs (aug_geo_all, aug_geo_minority). Everything already trained is skipped.
Train and validation only: no test set is opened. About 40 minutes on a GPU.

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
    "1kqoKhttO/f/C59GmXM="
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

# 4. train the two new runs (the other imbalance runs are skipped because they already finished)
subprocess.run([sys.executable, "steps/05_imbalance_ablations.py"], cwd=CODE)
