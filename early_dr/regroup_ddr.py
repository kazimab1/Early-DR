"""Group-aware splits for DDR (it has no patient IDs, so the random 60/20/20 split can leak).

Two alternatives to the random split:
  group   images that look like the same eye (near-duplicates) always land in the same split
  camera  whole image sizes (a proxy for the hospital / camera) are held out, so the test set contains
          captures the model never saw during training

Everything here uses train/validation-independent information only (images and sizes, no model results).
"""
import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .dataset import colour_normalise


def processed_file(work, image_id):
    folder = sorted(glob.glob(os.path.join(work, "processed_*")))[0]
    return os.path.join(folder, image_id + ".jpg")


# ----- near-duplicate groups --------------------------------------------------------------------

def embeddings(ddr, work, device, size=224, batch=128, pretrained=True):
    """One unit-length vector per image: ImageNet features of the colour-normalised picture, which
    emphasises the vessel pattern (the same eye gives almost the same vector)."""
    import timm
    model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    ids = list(ddr.image_id)
    for start in range(0, len(ids), batch):
        images = []
        for image_id in ids[start:start + batch]:
            img = cv2.cvtColor(cv2.imread(processed_file(work, image_id)), cv2.COLOR_BGR2RGB)
            img = cv2.resize(colour_normalise(img), (size, size), interpolation=cv2.INTER_AREA)
            images.append(img)
        x = torch.tensor(np.stack(images), device=device).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            out.append(torch.nn.functional.normalize(model((x - mean) / std), dim=1).cpu())
    return torch.cat(out)


def neighbours(emb, device, k=10, chunk=2048):
    """For every image, its k most similar OTHER images: (indices, cosine similarities)."""
    emb = emb.to(device)
    idx, sim = [], []
    for start in range(0, len(emb), chunk):
        s = emb[start:start + chunk] @ emb.T
        s[torch.arange(s.shape[0]), torch.arange(start, start + s.shape[0])] = -1      # not itself
        top = s.topk(k, dim=1)
        idx.append(top.indices.cpu())
        sim.append(top.values.cpu())
    return torch.cat(idx).numpy(), torch.cat(sim).numpy()


def auto_threshold(ddr, emb, audit_csv):
    """Pick the similarity above which two images count as the same eye. Taken from pairs the hash audit
    already found with identical grades (very likely true duplicates): their 5th-percentile similarity."""
    if not os.path.exists(audit_csv):
        return 0.95
    dups = pd.read_csv(audit_csv)
    dups = dups[dups.source == "DDR"]
    row = {image_id: i for i, image_id in enumerate(ddr.image_id)}
    sims = []
    for _, g in dups.groupby("group"):
        g = g[g.image_id.isin(row)]
        if len(g) >= 2 and g.grade.nunique() == 1:
            first = row[g.image_id.iloc[0]]
            sims += [float(emb[first] @ emb[row[i]]) for i in g.image_id.iloc[1:]]
    if len(sims) < 10:
        return 0.95
    return float(np.clip(np.percentile(sims, 5), 0.88, 0.995))


def pairs_above(idx, sim, threshold):
    keep = sim >= threshold
    rows = np.repeat(np.arange(len(idx)), idx.shape[1]).reshape(idx.shape)
    a, b = rows[keep], idx[keep]
    return sorted({(min(i, j), max(i, j)) for i, j in zip(a, b)})


def build_groups(n, idx, sim, threshold, max_share=0.02):
    """Connected groups of near-duplicates. If chaining makes one giant group (more than max_share of all
    images), the threshold is raised in small steps until it no longer does. Returns (group ids, threshold used)."""
    thr = threshold
    while True:
        groups = connected_groups(n, pairs_above(idx, sim, thr))
        largest = pd.Series(groups).value_counts().iloc[0]
        if largest <= max_share * n or thr >= 0.995:
            return groups, thr, int(largest)
        thr = round(thr + 0.005, 3)


def connected_groups(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        parent[find(a)] = find(b)
    return np.array([find(i) for i in range(n)])


def montage(ddr, work, idx, sim, bins=((0.88, 0.93), (0.93, 0.97), (0.97, 1.01)), per_bin=4, size=120, seed=0):
    """Pairs at different similarity levels, side by side, so you can see where 'same eye' starts."""
    rng = np.random.RandomState(seed)
    rows = []
    for lo, hi in bins:
        i_s = np.where((sim[:, 0] >= lo) & (sim[:, 0] < hi))[0]
        if len(i_s) == 0:
            continue
        picks = rng.choice(len(i_s), min(per_bin, len(i_s)), replace=False)
        tiles = []
        for p in picks:
            i, j = i_s[p], idx[i_s[p], 0]
            pair = []
            for k in (i, j):
                img = cv2.imread(processed_file(work, ddr.image_id.iloc[k]))
                pair.append(cv2.resize(img, (size, size)))
            cv2.putText(pair[0], f"{sim[i, 0]:.3f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(np.concatenate(pair, axis=1))
        while len(tiles) < per_bin:
            tiles.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(rows, axis=0) if rows else None


# ----- splits -----------------------------------------------------------------------------------

def assign_groups(group_ids, fractions, seed):
    """Whole groups go to one split. Each group is placed where the split is furthest below its target."""
    names = ["train", "val", "test"]
    rng = np.random.RandomState(seed)
    unique = np.unique(group_ids)
    rng.shuffle(unique)
    sizes = pd.Series(group_ids).value_counts()
    target = np.array(fractions) * len(group_ids)
    filled = np.zeros(3)
    where = {}
    for g in unique:
        k = int(np.argmin(filled / target))
        where[g] = names[k]
        filled[k] += sizes[g]
    return np.array([where[g] for g in group_ids])


def image_sizes(ddr):
    sizes = []
    for path in ddr.path:
        try:
            with Image.open(path) as im:
                sizes.append(im.size)
        except OSError:
            sizes.append((0, 0))
    return sizes


def leakage(ddr, split, idx, sim, threshold):
    """Share of val/test images with a near-duplicate (similarity >= threshold) in the TRAIN split."""
    split = np.asarray(split)
    out = {}
    for name in ["val", "test"]:
        rows = np.where(split == name)[0]
        hit = [(split[idx[r]] == "train")[sim[r] >= threshold].any() for r in rows]
        out[name] = round(float(np.mean(hit)), 4) if len(rows) else np.nan
    return out


def metadata_auroc(ddr, split, sizes):
    """Can width / height / file size alone predict early DR (grades 1-2) vs No DR on the held-out test split?"""
    table = pd.DataFrame({"w": [s[0] for s in sizes], "h": [s[1] for s in sizes],
                          "kb": [os.path.getsize(p) / 1024 if os.path.exists(p) else 0 for p in ddr.path],
                          "grade": ddr.grade.values, "split": np.asarray(split)})
    table = table[table.grade <= 2]
    train, test = table[table.split == "train"], table[table.split == "test"]
    if test.grade.gt(0).nunique() < 2:
        return float("nan")
    model = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(train[["w", "h", "kb"]], train.grade > 0)
    return round(float(roc_auc_score(test.grade > 0, model.predict_proba(test[["w", "h", "kb"]])[:, 1])), 3)


def balance(ddr, split):
    t = pd.crosstab(np.asarray(split), ddr.grade.values, normalize="index").round(3) * 100
    t.insert(0, "images", pd.Series(np.asarray(split)).value_counts())
    return t.reindex(["train", "val", "test"])


def write_variant(manifest, ddr, split, out_dir, processed_folder):
    """A new work folder that uses `split` for DDR (EyePACS and APTOS splits unchanged)."""
    os.makedirs(out_dir, exist_ok=True)
    new = manifest.copy()
    new.loc[new.source == "DDR", "split"] = pd.Series(np.asarray(split), index=ddr.index)
    new.to_csv(os.path.join(out_dir, "splits.csv"), index=False)
    new.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    link = os.path.join(out_dir, os.path.basename(processed_folder))
    if not os.path.exists(link):
        os.symlink(processed_folder, link)
    return new
