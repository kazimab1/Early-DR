"""Train one model (one arm, one task, one seed) and keep the best epoch by validation score."""
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .calibration import softmax, training_prior
from .data import num_classes, rows_for_task, select_arm, task_label
from .dataset import DRDataset, make_sampler
from .losses import build_loss
from .model import build_model
from .scoring import predict_logits, selection_score


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_dir(cfg, seed):
    path = os.path.join(cfg.paths.work, "runs", cfg.name, f"seed{seed}")
    os.makedirs(path, exist_ok=True)
    return path


def make_eval_loader(df, cfg, task):
    labels = task_label(df.grade.values, task)
    return DataLoader(DRDataset(df, cfg, labels, train=False), batch_size=cfg.train.batch_size * 2,
                      shuffle=False, num_workers=cfg.train.workers)


def train(cfg, manifest):
    seed_everything(cfg.seed)
    device = get_device()
    task, n_classes = cfg.task, num_classes(cfg.task)
    out = run_dir(cfg, cfg.seed)

    train_df = rows_for_task(select_arm(manifest, cfg.data.arm, "train"), task).reset_index(drop=True)
    val_df = rows_for_task(select_arm(manifest, cfg.data.arm, "val"), task).reset_index(drop=True)
    train_labels = task_label(train_df.grade.values, task)
    print(f"[{cfg.name}] arm {cfg.data.arm}, task {task}: "
          f"{len(train_df)} train images, {len(val_df)} validation images")

    sampler_args, sampler_prior = make_sampler(train_df, train_labels, n_classes, cfg)
    train_loader = DataLoader(DRDataset(train_df, cfg, train_labels, train=True),
                              num_workers=cfg.train.workers, pin_memory=True, **sampler_args)
    val_loader = make_eval_loader(val_df, cfg, task)

    counts = np.bincount(train_labels, minlength=n_classes)
    loss_fn, loss_weights = build_loss(cfg, counts, device)
    model = build_model(cfg, n_classes).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, cfg.train.epochs)
    scaler = torch.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")

    best, history = -1.0, []
    for epoch in range(1, cfg.train.epochs + 1):
        started = time.time()
        model.train()
        total, n_batches = 0.0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=cfg.train.amp and device.type == "cuda"):
                loss = loss_fn(model(images).float(), labels)
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            total, n_batches = total + loss.item(), n_batches + 1
        scheduler.step()

        # choose the epoch on VALIDATION data only
        val_prob = softmax(predict_logits(model, val_loader, device, cfg.train.amp))
        score = selection_score(val_prob, val_df.grade.values, task)
        history.append({"epoch": epoch, "train_loss": total / max(n_batches, 1), "val_score": score})
        print(f"  epoch {epoch}: train loss {total / max(n_batches, 1):.4f}, validation score {score:.4f}, "
              f"{time.time() - started:.0f} s")
        if score > best:
            best = score
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))

    info = {"sampler_prior": sampler_prior.tolist(),
            "loss_weights": None if loss_weights is None else list(map(float, loss_weights)),
            "training_prior": training_prior(sampler_prior, loss_weights).tolist(),
            "class_counts": counts.tolist(), "best_val_score": best, "history": history}
    with open(os.path.join(out, "train_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    return out


def load_trained(cfg, run_path, device):
    model = build_model(cfg, num_classes(cfg.task))
    model.load_state_dict(torch.load(os.path.join(run_path, "best.pt"), map_location="cpu"))
    return model.to(device).eval()
