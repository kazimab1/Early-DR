"""Loss functions: plain CE, class-balanced CE (effective number), smoothed weights, focal."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Cross-entropy that down-weights easy examples; gamma=0 is plain CE."""

    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        log_p = F.log_softmax(logits, dim=1).gather(1, target[:, None])[:, 0]
        return (-(1 - log_p.exp()) ** self.gamma * log_p).mean()


def class_weights(name, counts):
    """Per-class loss weights (or None) from the training class counts."""
    counts = np.asarray(counts, dtype=float).clip(min=1)
    if name == "cb_ce":                                  # effective number of samples
        beta = 0.999
        w = (1 - beta) / (1 - beta ** counts)
    elif name == "smoothed_ce":                          # inverse frequency, square-rooted
        w = (1.0 / counts) ** 0.5
    else:
        return None
    return w / w.sum() * len(counts)                      # average weight = 1


def build_loss(cfg, counts, device):
    """Returns (loss module, class weights or None)."""
    w = class_weights(cfg.imbalance.loss, counts)
    if cfg.imbalance.loss == "focal":
        return FocalLoss(cfg.imbalance.focal_gamma), None
    weight = None if w is None else torch.tensor(w, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight), w
