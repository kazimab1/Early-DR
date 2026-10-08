"""Turn model outputs into the scores the plan evaluates.

dr_score    probability of "any DR" (1 - P(grade 0)); used for early DR vs No DR
late_score  probability of "late" among DR-positive images; used for early vs late
"""
import numpy as np
import torch

from .metrics import pr_auc, auroc


@torch.no_grad()
def predict_logits(model, loader, device, amp=True):
    model.eval()
    out = []
    for images, *_ in loader:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out.append(model(images.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def dr_score(prob, task):
    if task == "stage1":
        return prob[:, 1]
    if task in ("three_class", "five_grade"):
        return 1.0 - prob[:, 0]
    return None                                        # stage2 has no No-DR output


def late_score(prob, task):
    if task == "stage2":
        return prob[:, 1]
    if task == "three_class":
        return prob[:, 2] / (prob[:, 1] + prob[:, 2])
    if task == "five_grade":
        return prob[:, 3:].sum(axis=1) / prob[:, 1:].sum(axis=1)
    return None                                        # stage1 has no early/late output


def selection_score(prob, grades, task):
    """The number used on VALIDATION data to choose the best epoch.
    stage2: AUROC of early vs late.  other tasks: PR-AUC of early DR (1-2) vs No DR."""
    grades = np.asarray(grades)
    if task == "stage2":
        return auroc((grades >= 3).astype(int), late_score(prob, task))
    keep = grades <= 2
    return pr_auc((grades[keep] > 0).astype(int), dr_score(prob, task)[keep])
