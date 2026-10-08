"""Explainability checks on early-DR images (no segmentation model is trained anywhere).

1. Grad-CAM maps for early-DR test images.
2. Sanity check (Adebayo et al. 2018): re-draw all weights at random. A trustworthy map must
   CHANGE; if it does not, it only shows image structure and tells us nothing about the model.
3. Optional yardstick: the share of saliency that falls inside DDR lesion annotations.
   The annotations only measure the maps; nothing is trained on them.

    python -m early_dr.saliency configs/arms/arm_A.yaml --n 20
CNN backbones only (EfficientNet); the ViT would need attention rollout instead.
"""
import argparse
import copy
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .config import load_config
from .data import task_label
from .dataset import DRDataset
from .preprocess import crop_pad_resize, find_crop_box
from .train import get_device, load_trained, run_dir

LESIONS = ("MA", "HE", "EX", "SE")


def grad_cam(model, image, device):
    """Heat map (size x size, 0..1) for the highest-scoring DR class of one image."""
    model.eval()
    x = image[None].to(device).requires_grad_(False)
    features = model.forward_features(x)
    features.retain_grad()
    logits = model.forward_head(features)
    target = int(logits[0, 1:].argmax()) + 1                      # best non-"No DR" class
    model.zero_grad()
    logits[0, target].backward()
    weights = features.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * features).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[1:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def randomise_weights(model):
    """Return a copy with every layer re-initialised (the sanity-check model)."""
    clone = copy.deepcopy(model)
    for module in clone.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return clone


def lesion_masks(root):
    """(lesion type, image stem) -> mask file, for the DDR lesion annotation folders."""
    found = {}
    for folder, _, files in os.walk(root):
        kind = os.path.basename(folder).upper()
        if kind in LESIONS:
            for name in files:
                found[(kind, os.path.splitext(name)[0])] = os.path.join(folder, name)
    return found


def lesion_overlap(cam, df_row, masks, size):
    """Fraction of the saliency mass inside the annotated lesions, and the lesion area fraction
    (what a map spread evenly over the retina would score)."""
    stem = df_row.image_id.replace("DDR_", "")
    original = cv2.imread(df_row.path)
    box = find_crop_box(original)
    union = np.zeros((size, size), bool)
    for kind in LESIONS:
        if (kind, stem) in masks:
            raw = cv2.imread(masks[(kind, stem)], cv2.IMREAD_GRAYSCALE)
            if raw is not None and raw.shape == original.shape[:2]:
                union |= crop_pad_resize(raw, size, box, cv2.INTER_NEAREST) > 0
    if not union.any():
        return None
    return float(cam[union].sum() / cam.sum()), float(union.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--n", type=int, default=20, help="number of early-DR images")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    device = get_device()
    run_path = run_dir(cfg, cfg.seed)
    out = os.path.join(run_path, "saliency")
    os.makedirs(out, exist_ok=True)

    manifest = pd.read_csv(os.path.join(cfg.paths.work, "manifest.csv"))
    early = manifest[(manifest.source == "DDR") & (manifest.split == "test") & manifest.grade.isin([1, 2])]
    early = early.sample(min(args.n, len(early)), random_state=0).reset_index(drop=True)
    data = DRDataset(early, cfg, task_label(early.grade.values, cfg.task), train=False)

    model = load_trained(cfg, run_path, device)
    random_model = randomise_weights(model).to(device)
    masks = lesion_masks(cfg.paths.ddr_lesions) if cfg.paths.ddr_lesions else {}

    correlations, overlaps = [], []
    for i, row in enumerate(early.itertuples()):
        image = data[i][0]
        cam, cam_random = grad_cam(model, image, device), grad_cam(random_model, image, device)
        correlations.append(spearmanr(cam.ravel(), cam_random.ravel())[0])
        if masks:
            hit = lesion_overlap(cam, row, masks, cam.shape[0])
            if hit:
                overlaps.append(hit)
        if i < 8:                                       # a few pictures to look at
            picture = cv2.cvtColor(cv2.imread(data.paths[i]), cv2.COLOR_BGR2RGB)
            picture = cv2.resize(picture, cam.shape[::-1])
            rows = [picture]
            for c in (cam, cam_random):
                heat = cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
                rows.append(cv2.addWeighted(picture, 0.5, heat, 0.5, 0))
            cv2.imwrite(os.path.join(out, f"{row.image_id}.jpg"),
                        np.concatenate(rows, axis=1)[..., ::-1])   # image | trained map | random-weights map

    summary = {"n_images": len(early),
               "spearman_trained_vs_random_weights": float(np.nanmean(correlations)),
               "note": "close to 0 = the map depends on the model (good); close to 1 = not trustworthy"}
    if overlaps:
        summary["lesion_saliency_share"] = float(np.mean([o[0] for o in overlaps]))
        summary["lesion_area_share"] = float(np.mean([o[1] for o in overlaps]))
        summary["n_images_with_annotations"] = len(overlaps)
    print(json.dumps(summary, indent=2))
    with open(os.path.join(out, "saliency_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
