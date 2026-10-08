"""Can a classifier tell WHICH DATASET an image came from?  (domain-shift measurement)

    python -m early_dr.source_check configs/base.yaml --a DDR --b EyePACS
A high AUROC means the images carry a strong source signature the grading model could exploit.
Only images are used (no DR labels). IDRiD/APTOS may be used here as a measurement; they are
never used to train or select a grading model.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .calibration import softmax
from .config import load_config
from .dataset import DRDataset
from .model import build_model
from .scoring import predict_logits
from .train import get_device, seed_everything


def source_auc(cfg, manifest, a, b, per_source=2000, epochs=3, size=256):
    seed_everything(cfg.seed)
    device = get_device()
    cfg = small_image_config(cfg, size)
    parts = [manifest[manifest.source == s].sample(frac=1, random_state=0) for s in (a, b)]
    n = min(per_source, *(len(p) for p in parts))
    df = pd.concat([p.iloc[:n] for p in parts], ignore_index=True)     # equal numbers from each source
    labels = (df.source == b).astype(int).values

    # hold out by patient so both eyes of one patient stay on the same side
    patients = np.array(df.patient_id.unique())
    np.random.RandomState(0).shuffle(patients)
    test_patients = set(patients[: int(0.3 * len(patients))])
    is_test = df.patient_id.isin(test_patients).values

    def loader(mask, train):
        data = DRDataset(df[mask].reset_index(drop=True), cfg, labels[mask], train=train)
        return DataLoader(data, batch_size=32, shuffle=train, num_workers=cfg.train.workers)

    train_loader, test_loader = loader(~is_test, True), loader(is_test, False)
    model = build_model(cfg, 2).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(epochs):
        model.train()
        for images, y, _ in train_loader:
            loss = torch.nn.functional.cross_entropy(model(images.to(device)), y.to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    prob = softmax(predict_logits(model, test_loader, device, amp=False))[:, 1]
    return roc_auc_score(labels[is_test], prob)


def small_image_config(cfg, size):
    cfg = type(cfg)(cfg)
    cfg["model"] = dict(cfg["model"], image_size=size)
    cfg["imbalance"] = dict(cfg["imbalance"], aug="mild", aug_scope="all")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/base.yaml")
    parser.add_argument("--a", default="DDR")
    parser.add_argument("--b", default="EyePACS")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    manifest = pd.read_csv(os.path.join(cfg.paths.work, "manifest.csv"))
    print(f"Source classifier {args.a} vs {args.b}: held-out AUROC = "
          f"{source_auc(cfg, manifest, args.a, args.b):.3f}  (0.5 = cannot tell, 1.0 = trivial)")


if __name__ == "__main__":
    main()
