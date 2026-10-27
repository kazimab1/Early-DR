"""Can the label be guessed WITHOUT looking at the retina? (a test for results that look too good)

Trains a small tree model on simple image properties only: original width and height, file size,
mean colour, brightness and sharpness. Then reports its AUROC for early DR (grades 1-2) vs No DR.

    python -m early_dr.shortcut_check --work /kaggle/working

Reading it: 0.5 = these properties say nothing. Well above 0.5 on a dataset means the label is entangled
with how images were captured or saved, so a CNN could score well without learning the disease.
"""
import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .preprocess import processed_path


def features(args):
    original, processed = args
    with Image.open(original) as im:
        width, height = im.size
    img = cv2.imread(processed, cv2.IMREAD_REDUCED_COLOR_4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retina = gray > 10
    return {"width": width, "height": height, "aspect": width / height, "file_kb": os.path.getsize(original) / 1024,
            "mean_b": img[..., 0][retina].mean(), "mean_g": img[..., 1][retina].mean(), "mean_r": img[..., 2][retina].mean(),
            "brightness": gray[retina].mean(), "sharpness": cv2.Laplacian(gray, cv2.CV_64F).var(),
            "retina_fraction": retina.mean()}


def auc(model, table, columns):
    y = (table.grade > 0).astype(int)
    return roc_auc_score(y, model.predict_proba(table[columns])[:, 1]) if y.nunique() == 2 else np.nan


FEATURE_SETS = {
    "image size only (width, height, aspect)": ["width", "height", "aspect"],          # a camera / hospital proxy
    "file size only": ["file_kb"],                                                      # also grows with lesion detail
    "colour and brightness only": ["mean_b", "mean_g", "mean_r", "brightness"],
    "sharpness and field of view only": ["sharpness", "retina_fraction"],
    "all properties": ["width", "height", "aspect", "file_kb", "mean_b", "mean_g", "mean_r", "brightness",
                       "sharpness", "retina_fraction"],
}


def fit(train, columns):
    return HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(train[columns], (train.grade > 0).astype(int))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/kaggle/working")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = pd.read_csv(os.path.join(args.work, "manifest.csv"))
    manifest = manifest[manifest.grade <= 2].reset_index(drop=True)          # early DR vs No DR only
    cache = os.path.join(args.work, "audit", "image_features.csv")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.exists(cache):                                               # reading the originals is slow: do it once
        table = manifest.merge(pd.read_csv(cache), on="image_id", how="left")
        print("image features loaded from", cache)
    else:
        class Cfg:                                                          # processed_path needs a config-like object
            class paths: work = args.work
            class data: prep_size = 512
        Cfg.data.prep_size = int([d for d in os.listdir(args.work) if d.startswith("processed_")][0].split("_")[1])
        jobs = [(r.path, processed_path(Cfg, r.image_id)) for r in manifest.itertuples()]
        with Pool(args.workers) as pool:
            feats = pd.DataFrame(pool.map(features, jobs, chunksize=64))
        feats.insert(0, "image_id", manifest.image_id.values)
        feats.to_csv(cache, index=False)
        table = manifest.merge(feats, on="image_id", how="left")

    rows = []
    aptos = table[table.source == "APTOS"].sample(frac=1, random_state=0)
    half = len(aptos) // 2
    scenarios = [("DDR train to DDR test", table[(table.source == "DDR") & (table.split == "train")], table[(table.source == "DDR") & (table.split == "test")]),
                 ("DDR train to APTOS", table[(table.source == "DDR") & (table.split == "train")], aptos),
                 ("EyePACS train to EyePACS test", table[(table.source == "EyePACS") & (table.split == "train")], table[(table.source == "EyePACS") & (table.split == "test")]),
                 ("APTOS half to other half", aptos.iloc[:half], aptos.iloc[half:])]
    for feature_set, columns in FEATURE_SETS.items():
        row = {"image properties used": feature_set}
        for name, train, test in scenarios:
            row[name] = round(auc(fit(train, columns), test, columns), 3)
        rows.append(row)
    result = pd.DataFrame(rows)
    print("\nAUROC for early DR vs No DR from image properties only (no retina content)\n")
    print(result.to_string(index=False))
    print("\nImage size is the cleanest camera proxy. File size and sharpness also rise with lesion detail, "
          "so a high value there is not proof of a camera artefact.")
    result.to_csv(os.path.join(args.work, "audit", "shortcut_check.csv"), index=False)


if __name__ == "__main__":
    main()
