"""Combine seeds and compare experiments with Holm-corrected, patient-level bootstrap tests.

    python -m early_dr.summarise --work /kaggle/working --baseline arm_A \
        --experiments arm_B arm_C --source DDR --metric sensitivity
Metrics are for early DR (grades 1-2) vs No DR: sensitivity (at the frozen threshold), pr_auc, auroc.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from . import metrics as M


def load_runs(work, name, source):
    """One experiment, all seeds -> (patient ids, [(labels, scores, frozen threshold) per seed]).
    Early DR (grades 1-2) vs No DR, so grades 3-4 are left out."""
    base = os.path.join(work, "runs", name)
    patients, runs = None, []
    for seed_dir in sorted(os.listdir(base)):
        path = os.path.join(base, seed_dir)
        if not os.path.exists(os.path.join(path, "results.json")):
            continue
        preds = pd.read_csv(os.path.join(path, f"preds_{source}.csv"))
        preds = preds[preds.grade <= 2]
        threshold = json.load(open(os.path.join(path, "calibration.json")))["threshold_dr"]
        runs.append(((preds.grade > 0).astype(int).values, preds.dr_score.values, threshold))
        patients = preds.patient_id.values
    return patients, runs


def metric_value(name, y, score, threshold):
    if name == "sensitivity":
        return M.sensitivity_specificity(y, score, threshold)[0]
    return {"pr_auc": M.pr_auc, "auroc": M.auroc}[name](y, score)


def seed_average(runs, metric, idx=slice(None)):
    """Mean over seeds of the metric (every seed is evaluated on the same images, in the same order)."""
    return float(np.mean([metric_value(metric, y[idx], s[idx], t) for y, s, t in runs]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/kaggle/working")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--source", default="DDR")
    parser.add_argument("--metric", default="sensitivity", choices=["sensitivity", "pr_auc", "auroc"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    patients, base_runs = load_runs(args.work, args.baseline, args.source)
    base_value = seed_average(base_runs, args.metric)
    rows = []
    for name in args.experiments:
        _, runs = load_runs(args.work, name, args.source)
        per_seed = [metric_value(args.metric, *run) for run in runs]
        # paired bootstrap: the same patient resample is used for the experiment and the baseline
        p = M.paired_bootstrap_p(patients,
                                 lambda idx: seed_average(runs, args.metric, idx),
                                 lambda idx: seed_average(base_runs, args.metric, idx),
                                 n_resamples=args.bootstrap)
        rows.append({"experiment": name, "seeds": len(runs),
                     f"{args.metric}_mean": np.mean(per_seed), f"{args.metric}_sd": np.std(per_seed),
                     "difference_vs_baseline": np.mean(per_seed) - base_value, "p_raw": p})
    table = pd.DataFrame(rows)
    table["p_holm"] = M.holm(table.p_raw.values)
    print(f"\nEarly DR vs No DR on {args.source}, baseline = {args.baseline} "
          f"({args.metric} = {base_value:.4f})\n")
    print(table.round(4).to_string(index=False))
    table.to_csv(os.path.join(args.work, f"comparison_{args.source}_{args.metric}.csv"), index=False)


if __name__ == "__main__":
    main()
