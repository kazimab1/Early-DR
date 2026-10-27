"""Why did strong augmentation fail to train?  Four checks, each answering one question.

    python -m early_dr.aug_diagnose configs/imbalance/aug_strong_all.yaml --set ...

A  What do augmented images look like?   picture + pixel statistics for each augmentation
B  Is the saved model broken at test time or only mismatched?  scores the trained model normally,
   after AdaBN (re-estimated BatchNorm statistics), and with the strong augmentation applied to validation
C  Which part of "strong" is the culprit?  short retrain on a subset: mild / geometric only /
   colour+blur+noise only / strong / strong at a lower learning rate
Only train and validation images are used. No test set is opened.
"""
import argparse
import copy
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .calibration import softmax
from .config import Config, load_config
from .data import rows_for_task, select_arm, task_label
from .dataset import DRDataset, MEAN, STD
from .model import adabn, build_model
from .scoring import predict_logits, selection_score
from .train import get_device, load_trained, run_dir, seed_everything

KINDS = ["plain", "mild", "strong", "geo", "photo"]


def clone(cfg, **imbalance):
    c = Config(copy.deepcopy(dict(cfg)))
    c["imbalance"] = dict(c["imbalance"], **imbalance)
    return c


def dataset_for(df, cfg, kind):
    labels = task_label(df.grade.values, cfg.task)
    if kind == "plain":
        return DRDataset(df, cfg, labels, train=False)
    return DRDataset(df, clone(cfg, aug=kind, aug_scope="all"), labels, train=True)


def unnormalise(tensor):
    x = tensor.numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)
    return x.clip(0, 1)


# ----- A: what do the augmented images look like? ----------------------------------------------

def check_images(cfg, train_df, work, n_stats=96):
    rows = []
    sample = train_df.sample(min(n_stats, len(train_df)), random_state=0).reset_index(drop=True)
    for kind in KINDS:
        data = dataset_for(sample, cfg, kind)
        tensors = torch.stack([data[i][0] for i in range(len(data))])
        pixels = np.stack([unnormalise(t) for t in tensors])
        visible = (pixels.mean(axis=3) > 0.04).mean()                  # share of pixels that are not black
        rows.append({"augmentation": kind, "non_finite": int((~torch.isfinite(tensors)).sum()),
                     "mean": round(float(tensors.mean()), 3), "std": round(float(tensors.std()), 3),
                     "abs_max": round(float(tensors.abs().max()), 2), "visible_pixels": round(float(visible), 3)})
    table = pd.DataFrame(rows)
    # picture: 5 images x every augmentation
    size, pics = 150, sample.iloc[:5]
    grid = []
    for kind in KINDS:
        data = dataset_for(pics, cfg, kind)
        row = [cv2.resize((unnormalise(data[i][0]) * 255).astype(np.uint8)[..., ::-1], (size, size)) for i in range(len(pics))]
        row[0] = row[0].copy()
        cv2.putText(row[0], kind, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        grid.append(np.concatenate(row, axis=1))
    os.makedirs(os.path.join(work, "figures"), exist_ok=True)
    cv2.imwrite(os.path.join(work, "figures", "strong_aug_samples.jpg"), np.concatenate(grid, axis=0))
    return table


# ----- B: is the trained model broken, or only mismatched? -------------------------------------

def score_with(model, df, cfg, device, mode):
    net = model
    if mode == "adabn":
        net = copy.deepcopy(model)
        adabn(net, DataLoader(dataset_for(df, cfg, "plain"), batch_size=cfg.train.batch_size * 2,
                              num_workers=cfg.train.workers), device)
    kind = "strong" if mode == "strong augmentation on validation" else "plain"
    loader = DataLoader(dataset_for(df, cfg, kind), batch_size=cfg.train.batch_size * 2, shuffle=False,
                        num_workers=cfg.train.workers)
    logits = predict_logits(net, loader, device, amp=False)
    if not np.isfinite(logits).all():
        return float("nan")
    return float(selection_score(softmax(logits), df.grade.values, cfg.task))


def check_models(args, manifest, device):
    rows = []
    for name in ["aug_strong_all", "aug_strong_minority", "baseline"]:
        cfg = load_config(os.path.join("configs", "imbalance", name + ".yaml"), args.set)
        path = run_dir(cfg, 42)
        if not os.path.exists(os.path.join(path, "best.pt")):
            print(f"  (no trained run called {name} - skipped)")
            continue
        model = load_trained(cfg, path, device)
        val = rows_for_task(select_arm(manifest, cfg.data.arm, "val"), cfg.task).reset_index(drop=True)
        for mode in ["normal", "adabn", "strong augmentation on validation"]:
            rows.append({"trained run": name, "validation scored": mode, "score": round(score_with(model, val, cfg, device, mode), 4)})
    return pd.DataFrame(rows)


# ----- C: which part of "strong" is the culprit? -----------------------------------------------

def quick_train(cfg, train_df, val_df, kind, lr, epochs, device):
    seed_everything(0)
    c = clone(cfg, aug=kind, aug_scope="all")
    c["task"] = "stage1"
    c["train"] = dict(c["train"], lr=lr)
    labels = task_label(train_df.grade.values, "stage1")
    loader = DataLoader(DRDataset(train_df, c, labels, train=True), batch_size=c.train.batch_size, shuffle=True,
                        drop_last=True, num_workers=c.train.workers)
    val_loader = DataLoader(DRDataset(val_df, c, task_label(val_df.grade.values, "stage1"), train=False),
                            batch_size=c.train.batch_size * 2, num_workers=c.train.workers)
    model = build_model(c, 2).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=c.train.weight_decay)
    scaler = torch.amp.GradScaler(enabled=c.train.amp and device.type == "cuda")
    scores, skipped = [], 0
    for _ in range(epochs):
        model.train()
        for images, y, _ in loader:
            images, y = images.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=c.train.amp and device.type == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(images).float(), y)
            optimiser.zero_grad()
            if not torch.isfinite(loss):
                skipped += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimiser)
            scaler.update()
        logits = predict_logits(model, val_loader, device, amp=False)
        scores.append(round(float(selection_score(softmax(logits), val_df.grade.values, "stage1")), 4)
                      if np.isfinite(logits).all() else float("nan"))
    return scores, skipped


def check_retrain(cfg, manifest, device, subset, epochs):
    train_all, val_all = select_arm(manifest, "A", "train"), select_arm(manifest, "A", "val")
    train = train_all.sample(min(subset, len(train_all)), random_state=0).reset_index(drop=True)
    val = val_all.sample(min(1500, subset, len(val_all)), random_state=0).reset_index(drop=True)
    rows = []
    for kind, lr in [("mild", 3e-4), ("geo", 3e-4), ("photo", 3e-4), ("strong", 3e-4), ("strong", 1e-4)]:
        started = time.time()
        scores, skipped = quick_train(cfg, train, val, kind, lr, epochs, device)
        rows.append({"augmentation": kind, "learning rate": lr, **{f"epoch {i + 1}": s for i, s in enumerate(scores)},
                     "bad batches": skipped, "seconds": round(time.time() - started)})
        print("  done", kind, lr, scores, flush=True)
    return pd.DataFrame(rows)


def explain(models, retrain):
    print("\nHow to read this")
    if not models.empty:
        for name, g in models.groupby("trained run"):
            s = dict(zip(g["validation scored"], g.score))
            if "adabn" in s and s["adabn"] > s["normal"] + 0.1:
                print(f"- {name}: AdaBN recovers the score ({s['normal']} -> {s['adabn']}): a BatchNorm train/test mismatch.")
            if s.get("strong augmentation on validation", 0) > s["normal"] + 0.1:
                print(f"- {name}: scores much better on AUGMENTED validation images: the model only works on the augmented look.")
    if not retrain.empty:
        last = [c for c in retrain.columns if c.startswith("epoch")][-1]
        by = {(r["augmentation"], r["learning rate"]): r[last] for _, r in retrain.iterrows()}
        base = by.get(("mild", 3e-4), np.nan)
        for key, label in [(("geo", 3e-4), "geometric part (flips, 180-degree rotation, crop)"),
                           (("photo", 3e-4), "colour / blur / noise part")]:
            if by.get(key, base) < base - 0.1:
                print(f"- The {label} breaks training ({by[key]} vs {base} for mild).")
        if by.get(("strong", 1e-4), 0) > by.get(("strong", 3e-4), 0) + 0.1:
            print("- Strong augmentation trains with a lower learning rate: the 3e-4 rate is too high for it.")
    print("- Look at figures/strong_aug_samples.jpg: black corners, blank images or heavy noise explain a lot.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--subset", type=int, default=3000, help="training images for the quick retrain")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--skip-retrain", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    work, device = cfg.paths.work, get_device()
    manifest = pd.read_csv(os.path.join(work, "manifest.csv"))
    train_df = rows_for_task(select_arm(manifest, "A", "train"), cfg.task).reset_index(drop=True)

    print("\nA. Augmented images (picture saved to figures/strong_aug_samples.jpg)\n")
    stats = check_images(cfg, train_df, work)
    print(stats.to_string(index=False))
    print("\nB. The trained strong-augmentation models, scored on VALIDATION (early DR PR-AUC)\n")
    models = check_models(args, manifest, device)
    print(models.to_string(index=False) if not models.empty else "no trained runs found")
    retrain = pd.DataFrame()
    if not args.skip_retrain:
        print(f"\nC. Quick retrain: {args.subset} training images, {args.epochs} epochs, validation PR-AUC per epoch\n")
        retrain = check_retrain(cfg, manifest, device, args.subset, args.epochs)
        print("\n" + retrain.to_string(index=False))
    os.makedirs(os.path.join(work, "audit"), exist_ok=True)
    for name, table in [("strong_aug_stats", stats), ("strong_aug_models", models), ("strong_aug_retrain", retrain)]:
        if not table.empty:
            table.to_csv(os.path.join(work, "audit", name + ".csv"), index=False)
    explain(models, retrain)


if __name__ == "__main__":
    main()
