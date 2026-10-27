"""ONE FILE for a NEW Kaggle notebook: restores your finished run from the old notebook's output and runs
the image-property check again, now split by property type (image size / file size / colour / sharpness).
Nothing is retrained and no test set is opened. Needs the DDR, EyePACS and APTOS datasets as inputs. The first
run reads the original images (slow, it took about 50 minutes before) and saves the result to audit/image_features.csv.

Before running, in the new notebook:
  * Accelerator: GPU on, Internet: On
  * Add as inputs: (1) your OLD notebook's output (the one with the finished runs), (2) the code dataset
    (the Early-DR.zip dataset), and, for step 13 only, the DDR, EyePACS and APTOS datasets.
Then paste this file into a cell and run it.

Result: /kaggle/working/audit/shortcut_check.csv (printed at the end)
"""
import base64, glob, json, os, shutil, subprocess, sys, zlib
INPUT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("EARLYDR_WORK", "/kaggle/working")
CODE = os.environ.get("EARLYDR_CODE", os.path.join(WORK, "code"))
PAYLOAD = (
    "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"
    "P21rSSqzo2Ly+H+/hcYO"
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

# 4. run the check
subprocess.run([sys.executable, "steps/13_shortcut_check.py"], cwd=CODE)
