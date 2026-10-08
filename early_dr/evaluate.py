"""Calibrate on validation, freeze, then evaluate on the test sets and save per-image predictions.

Rules from the plan this file enforces:
  * thresholds, temperature and prior correction come from VALIDATION data only (fit_calibration)
  * internal test sets are opened first; IDRiD and APTOS only with external=True, and every
    opening is written to test_opened.log
  * every metric is reported per source, with patient-level bootstrap 95% intervals
"""
import copy
import datetime
import json
import os

import numpy as np
import pandas as pd

from . import metrics as M
from .calibration import calibrate, estimate_prior_em, fit_temperature, softmax
from .data import num_classes, rows_for_task, select_arm, task_label
from .model import adabn
from .scoring import dr_score, late_score, predict_logits
from .train import get_device, load_trained, make_eval_loader


# ----- step 1: fit calibration on validation, then freeze ---------------------------------------

def fit_calibration(cfg, manifest, run_path):
    device = get_device()
    task, n_classes = cfg.task, num_classes(cfg.task)
    model = load_trained(cfg, run_path, device)
    val_df = rows_for_task(select_arm(manifest, cfg.data.arm, "val"), task).reset_index(drop=True)
    logits = predict_logits(model, make_eval_loader(val_df, cfg, task), device, cfg.train.amp)
    labels = task_label(val_df.grade.values, task)
    grades = val_df.grade.values

    with open(os.path.join(run_path, "train_info.json")) as f:
        training_prior = np.array(json.load(f)["training_prior"])
    natural_prior = np.bincount(labels, minlength=n_classes) / len(labels)

    temperature = fit_temperature(logits, labels)
    use_prior = cfg.eval.prior_correction
    prob = calibrate(logits, temperature, training_prior if use_prior else None,
                     natural_prior if use_prior else None)

    spec = cfg.declared.specificity
    cal = {"temperature": temperature, "training_prior": training_prior.tolist(),
           "natural_prior": natural_prior.tolist(), "prior_correction": bool(use_prior),
           "specificity_target": spec, "threshold_dr": None, "threshold_late": None}
    dr, late = dr_score(prob, task), late_score(prob, task)
    if dr is not None:                                 # early DR (1-2) vs No DR
        keep = grades <= 2
        cal["threshold_dr"] = float(M.threshold_at_specificity((grades[keep] > 0).astype(int), dr[keep], spec))
    if late is not None:                               # late vs early, DR-positive images only
        keep = grades >= 1
        cal["threshold_late"] = float(M.threshold_at_specificity((grades[keep] >= 3).astype(int), late[keep], spec))
    with open(os.path.join(run_path, "calibration.json"), "w") as f:
        json.dump(cal, f, indent=2)
    print(f"Calibration frozen: T={temperature:.3f}, thresholds {cal['threshold_dr']}, {cal['threshold_late']}")
    return cal


# ----- step 2: metrics --------------------------------------------------------------------------

def _readout(y, score, patients, threshold, n_boot):
    """Sensitivity / specificity at the frozen threshold, PR-AUC and AUROC, each with a 95% CI."""
    def sens(idx):
        return M.sensitivity_specificity(y[idx], score[idx], threshold)[0]

    everything = np.arange(len(y))
    sens_value, spec_value = M.sensitivity_specificity(y, score, threshold)
    return {"n": int(len(y)), "prevalence": float(y.mean()),          # PR-AUC depends on prevalence
            "sensitivity": sens_value, "sensitivity_ci": M.bootstrap_ci(patients, sens, n_boot),
            "specificity": spec_value,
            "pr_auc": M.pr_auc(y, score),
            "pr_auc_ci": M.bootstrap_ci(patients, lambda i: M.pr_auc(y[i], score[i]), n_boot),
            "auroc": M.auroc(y, score),
            "auroc_ci": M.bootstrap_ci(patients, lambda i: M.auroc(y[i], score[i]), n_boot)}


def compute_metrics(df, prob, cal, cfg):
    task, n_boot = cfg.task, cfg.eval.bootstrap
    grades, patients = df.grade.values, df.patient_id.values
    dr, late = dr_score(prob, task), late_score(prob, task)
    out = {"n_images": int(len(df)),
           "grade_counts": np.bincount(grades, minlength=5).tolist()}

    if dr is not None:
        thr = cal["threshold_dr"]
        for name, keep, positive in [
                ("early_vs_none", grades <= 2, lambda g: g > 0),           # PRIMARY
                ("grade1_vs_none", grades <= 1, lambda g: g > 0),
                ("grade2plus_vs_rest", np.ones(len(df), bool), lambda g: g >= 2)]:
            if keep.sum() and len(np.unique(positive(grades[keep]))) == 2:
                out[name] = _readout(positive(grades[keep]).astype(int), dr[keep], patients[keep], thr, n_boot)
        # recall of each grade at the frozen threshold (grade 0 gives the false-positive rate)
        out["flagged_fraction_by_grade"] = [float((dr[grades == g] > thr).mean()) if (grades == g).any()
                                            else None for g in range(5)]
        # calibration of "any DR" on all images
        has_dr = (grades > 0).astype(int)
        out["calibration"] = {"ece": M.ece(has_dr, dr), "brier": M.brier(has_dr, dr),
                              "reliability": M.reliability_curve(has_dr, dr)}

    if late is not None:                                # secondary: early vs late, DR-positive only
        keep = grades >= 1
        if keep.any() and len(np.unique(grades[keep] >= 3)) == 2:
            out["early_vs_late"] = _readout((grades[keep] >= 3).astype(int), late[keep],
                                            patients[keep], cal["threshold_late"], n_boot)

    if task == "five_grade":
        out["qwk"] = M.qwk(grades, prob.argmax(axis=1))

    if task != "stage2":                                # how well the prevalence is recovered
        out["estimated_prior_em"] = estimate_prior_em(prob, cal["natural_prior"]).tolist()
    return out


# ----- step 3: run on the test sets -------------------------------------------------------------

def test_sets(manifest, cfg, external):
    sets = {"DDR": select_arm(manifest, "A", "test"),
            "EyePACS": manifest[(manifest.source == "EyePACS") & (manifest.split == "test")]}
    if cfg.data.arm != "A":                             # the arm's own pooled test split
        sets["arm_test"] = select_arm(manifest, cfg.data.arm, "test")
    if external:
        sets["IDRiD"] = manifest[manifest.source == "IDRiD"]
        sets["APTOS"] = manifest[manifest.source == "APTOS"]
    return {k: v.reset_index(drop=True) for k, v in sets.items()}


def evaluate(cfg, manifest, run_path, external=False):
    device = get_device()
    with open(os.path.join(run_path, "calibration.json")) as f:
        cal = json.load(f)
    model = load_trained(cfg, run_path, device)

    results_file = os.path.join(run_path, "results.json")
    results = json.load(open(results_file)) if os.path.exists(results_file) else {}

    sets = test_sets(manifest, cfg, external)
    with open(os.path.join(cfg.paths.work, "test_opened.log"), "a") as log:
        log.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M} {cfg.name} seed{cfg.seed}: {list(sets)}\n")

    for name, df in sets.items():
        loader = make_eval_loader(df, cfg, cfg.task)
        net = model
        if cfg.eval.adabn:                              # unlabelled images from the new source
            net = copy.deepcopy(model)
            adabn(net, loader, device)
        logits = predict_logits(net, loader, device, cfg.train.amp)
        prob = calibrate(logits, cal["temperature"],
                         cal["training_prior"] if cal["prior_correction"] else None,
                         cal["natural_prior"] if cal["prior_correction"] else None)
        results[name] = compute_metrics(df, prob, cal, cfg)

        preds = df[["image_id", "source", "patient_id", "grade", "split"]].copy()
        for c in range(prob.shape[1]):
            preds[f"p{c}"] = prob[:, c]
        if dr_score(prob, cfg.task) is not None:
            preds["dr_score"] = dr_score(prob, cfg.task)
        if late_score(prob, cfg.task) is not None:
            preds["late_score"] = late_score(prob, cfg.task)
        preds.to_csv(os.path.join(run_path, f"preds_{name}.csv"), index=False)   # for later re-analysis
        print(f"  {name}: {len(df)} images evaluated")

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results
