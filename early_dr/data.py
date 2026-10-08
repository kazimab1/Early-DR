"""Read the four datasets into ONE table (the manifest) and freeze the splits.

Manifest columns
    image_id    unique name, e.g. "EyePACS_10_left"
    source      DDR | EyePACS | IDRiD | APTOS
    path        original image file
    grade       0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 PDR
    patient_id  EyePACS file names carry it ("10_left" -> patient 10); others use the image id
    split       train | val | test  (DDR, EyePACS)   or   external  (IDRiD, APTOS)
"""
import glob
import os

import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
SOURCES = ["DDR", "EyePACS", "IDRiD", "APTOS"]


def index_images(root):
    """Map file name (without extension) -> full path for every image under root."""
    found = {}
    for folder, _, files in os.walk(root):
        for name in files:
            stem, ext = os.path.splitext(name)
            if ext.lower() in IMAGE_EXTENSIONS:
                found[stem] = os.path.join(folder, name)
    return found


def _find_file(root, pattern):
    hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f"No file matching '{pattern}' under {root}")
    return hits[0]


def _rows(source, stems, grades, images, patients=None):
    """Build manifest rows; images missing on disk are skipped (and counted)."""
    rows, missing = [], 0
    for i, (stem, grade) in enumerate(zip(stems, grades)):
        stem = os.path.splitext(str(stem))[0]
        if stem not in images:
            missing += 1
            continue
        rows.append(dict(image_id=f"{source}_{stem}", source=source, path=images[stem],
                         grade=int(grade),
                         patient_id=f"{source}_{patients[i] if patients is not None else stem}"))
    if missing:
        print(f"[{source}] {missing} listed images were not found on disk")
    return rows


def load_ddr(root):
    """DDR grading folder: train.txt / valid.txt / test.txt with lines 'name.jpg grade'.
    Grade 5 means ungradable and is dropped (gradable images only)."""
    images = index_images(root)
    rows = []
    for split_file, split in [("train", "train"), ("valid", "val"), ("test", "test")]:
        table = pd.read_csv(_find_file(root, f"{split_file}.txt"), sep=r"\s+", header=None,
                            names=["name", "grade"])
        table = table[table.grade <= 4]
        part = pd.DataFrame(_rows("DDR", table.name, table.grade, images))
        part["split"] = split
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def load_eyepacs(root):
    """Kaggle EyePACS: trainLabels.csv with columns image, level.  Split later, by patient."""
    table = pd.read_csv(_find_file(root, "trainLabels.csv"))
    images = index_images(root)
    patients = [str(name).split("_")[0] for name in table["image"]]
    return pd.DataFrame(_rows("EyePACS", table["image"], table["level"], images, patients))


def load_idrid(root):
    """IDRiD disease grading: training + testing label CSVs (516 images), all used as external."""
    images = index_images(root)
    parts = []
    for pattern in ["*Training Labels*.csv", "*Testing Labels*.csv"]:
        table = pd.read_csv(_find_file(root, pattern))
        table.columns = [c.strip() for c in table.columns]
        parts.append(pd.DataFrame(_rows("IDRiD", table["Image name"], table["Retinopathy grade"], images)))
    out = pd.concat(parts, ignore_index=True)
    out["split"] = "external"
    return out


def load_aptos(root):
    """APTOS 2019: train.csv with columns id_code, diagnosis (3,662 images), all external."""
    table = pd.read_csv(_find_file(root, "train.csv"))
    out = pd.DataFrame(_rows("APTOS", table["id_code"], table["diagnosis"], index_images(root)))
    out["split"] = "external"
    return out


def split_eyepacs_by_patient(df, fractions, seed):
    """Both eyes of a patient always land in the same split."""
    patients = np.array(sorted(df.patient_id.unique()))
    np.random.RandomState(seed).shuffle(patients)
    n_train = int(len(patients) * fractions[0])
    n_val = int(len(patients) * fractions[1])
    split_of = {}
    for i, p in enumerate(patients):
        split_of[p] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    df = df.copy()
    df["split"] = df.patient_id.map(split_of)
    return df


def build_manifest(cfg):
    """Read all datasets and freeze the splits. If splits.csv exists it is reused, never redone."""
    work = cfg.paths.work
    os.makedirs(work, exist_ok=True)
    frozen = os.path.join(work, "splits.csv")
    if os.path.exists(frozen):
        print("Using the frozen splits:", frozen)
        return pd.read_csv(frozen)

    ddr = load_ddr(cfg.paths.ddr)
    eyepacs = split_eyepacs_by_patient(load_eyepacs(cfg.paths.eyepacs),
                                       cfg.data.eyepacs_fractions, cfg.data.split_seed)
    manifest = pd.concat([ddr, eyepacs, load_idrid(cfg.paths.idrid), load_aptos(cfg.paths.aptos)],
                         ignore_index=True)
    manifest.to_csv(frozen, index=False)
    print("Splits frozen ->", frozen)
    return manifest


# ----- labels and training arms -----------------------------------------------------------------

def num_classes(task):
    return {"stage1": 2, "stage2": 2, "three_class": 3, "five_grade": 5}[task]


def task_label(grade, task):
    """Turn a 0-4 grade into the label used by a task."""
    grade = np.asarray(grade)
    if task == "stage1":
        return (grade > 0).astype(int)                       # 0 = No DR, 1 = any DR
    if task == "stage2":
        return (grade >= 3).astype(int)                      # 0 = early (1-2), 1 = late (3-4)
    if task == "three_class":
        return np.where(grade == 0, 0, np.where(grade <= 2, 1, 2))
    return grade.astype(int)


def rows_for_task(df, task):
    """stage2 only uses DR-positive images."""
    return df[df.grade > 0] if task == "stage2" else df


def select_arm(manifest, arm, split):
    """The images an arm uses for one split.
    A: DDR only.  B: DDR + all EyePACS.  C: the version 1 merge (DDR + EyePACS grades 1, 3, 4)."""
    part = manifest[manifest.split == split]
    ddr = part[part.source == "DDR"]
    eye = part[part.source == "EyePACS"]
    if arm == "A":
        return ddr.reset_index(drop=True)
    if arm == "C":
        eye = eye[eye.grade.isin([1, 3, 4])]
    return pd.concat([ddr, eye], ignore_index=True)
