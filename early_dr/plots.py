"""Figures from the saved results (nothing is recomputed on the test sets, no model is loaded).

    python -m early_dr.plots --work /kaggle/working --runs arm_A arm_B arm_C
Writes PNG files to <work>/figures/.

Colours: the first three slots of a validated categorical palette (blue, orange, aqua).
Aqua is light, so every chart also has a legend and distinct line styles/markers.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from . import metrics as M

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLOURS = [BLUE, ORANGE, AQUA]
MARKERS = ["o", "s", "^"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GRADE_NAMES = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 PDR"]
SOURCES = ["DDR", "EyePACS", "APTOS"]

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False})


def run_path(work, name, seed):
    return os.path.join(work, "runs", name, f"seed{seed}")


def save(fig, work, name):
    folder = os.path.join(work, "figures")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def early_subset(preds):
    """Early DR (grades 1-2) vs No DR: the primary comparison."""
    keep = preds[preds.grade <= 2]
    return (keep.grade > 0).astype(int).values, keep.dr_score.values


def available_sources(path):
    return [s for s in SOURCES if os.path.exists(os.path.join(path, f"preds_{s}.csv"))]


# ----- data audit -------------------------------------------------------------------------------

def plot_class_distribution(work):
    file = os.path.join(work, "audit", "class_distribution.csv")
    if not os.path.exists(file):
        return
    table = pd.read_csv(file, index_col=0)
    counts = table[[str(g) for g in range(5)]].reindex([s for s in SOURCES if s in table.index])
    fig, axes = plt.subplots(1, len(counts), figsize=(4.2 * len(counts), 3.6), sharey=False)
    for ax, (source, row) in zip(np.atleast_1d(axes), counts.iterrows()):
        colours = [ORANGE if g in (1, 2) else BLUE for g in range(5)]   # orange = early grades
        bars = ax.bar(range(5), row.values, color=colours, width=0.65)
        ax.bar_label(bars, labels=[f"{v / row.sum():.0%}" for v in row.values], color=INK, fontsize=9)
        ax.set_xticks(range(5), ["0", "1", "2", "3", "4"])
        ax.set_title(f"{source} (n = {int(row.sum()):,})")
        ax.set_xlabel("DR grade")
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("images")
    fig.suptitle("Class distribution (orange = early grades 1-2, the primary target)", x=0.01, y=1.06,
                 ha="left", color=INK, fontweight="bold")
    save(fig, work, "01_class_distribution.png")


# ----- per run ----------------------------------------------------------------------------------

def plot_training(work, name, seed):
    info = json.load(open(os.path.join(run_path(work, name, seed), "train_info.json")))
    history = pd.DataFrame(info["history"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    a.plot(history.epoch, history.train_loss, color=BLUE, marker="o")
    a.set_title("Training loss")
    a.set_xlabel("epoch")
    b.plot(history.epoch, history.val_score, color=ORANGE, marker="s")
    best = history.val_score.idxmax()
    b.annotate(f"best: epoch {history.epoch[best]}", (history.epoch[best], history.val_score[best]),
               textcoords="offset points", xytext=(6, -14), color=INK)
    b.set_title("Validation score (used to pick the epoch)")
    b.set_xlabel("epoch")
    save(fig, work, f"02_training_{name}_seed{seed}.png")


def plot_roc_pr(work, name, seed):
    path = run_path(work, name, seed)
    fig, (roc, pr) = plt.subplots(1, 2, figsize=(9.5, 4))
    for i, source in enumerate(available_sources(path)):
        y, score = early_subset(pd.read_csv(os.path.join(path, f"preds_{source}.csv")))
        if len(np.unique(y)) < 2:
            continue
        style = dict(color=COLOURS[i], linestyle=["-", "--", ":"][i], linewidth=2)
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        roc.plot(fpr, tpr, label=f"{source}  AUROC {M.auroc(y, score):.2f}", **style)
        pr.plot(recall, precision, label=f"{source}  PR-AUC {M.pr_auc(y, score):.2f} (prev. {y.mean():.2f})", **style)
    roc.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    spec = json.load(open(os.path.join(path, "calibration.json")))["specificity_target"]
    roc.axvline(1 - spec, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
    roc.text(1 - spec + 0.015, 0.02, f"declared specificity {spec:.2f}", rotation=90, color=MUTED, fontsize=8)
    roc.set(title="ROC: early DR (1-2) vs No DR", xlabel="1 - specificity", ylabel="sensitivity")
    pr.set(title="Precision-recall (prevalence differs by source)", xlabel="recall", ylabel="precision", ylim=(0, 1.02))
    below = dict(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)   # legends under the plots
    roc.legend(**below)
    pr.legend(**below)
    save(fig, work, f"03_roc_pr_{name}_seed{seed}.png")


def plot_reliability(work, name, seed):
    path = run_path(work, name, seed)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="perfect")
    results = json.load(open(os.path.join(path, "results.json")))
    for i, source in enumerate(available_sources(path)):
        preds = pd.read_csv(os.path.join(path, f"preds_{source}.csv"))
        curve = M.reliability_curve((preds.grade > 0).astype(int).values, preds.dr_score.values)
        ece = results[source]["calibration"]["ece"]
        ax.plot([c[0] for c in curve], [c[1] for c in curve], color=COLOURS[i], marker=MARKERS[i],
                linewidth=2, label=f"{source}  ECE {ece:.3f}")
    ax.set(title="Reliability: any DR vs No DR", xlabel="predicted probability", ylabel="observed fraction",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=8)
    save(fig, work, f"04_reliability_{name}_seed{seed}.png")


def plot_flagged_by_grade(work, name, seed):
    path = run_path(work, name, seed)
    results = json.load(open(os.path.join(path, "results.json")))
    sources = [s for s in available_sources(path) if "flagged_fraction_by_grade" in results[s]]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    width = 0.8 / len(sources)
    for i, source in enumerate(sources):
        values = [v if v is not None else 0 for v in results[source]["flagged_fraction_by_grade"]]
        bars = ax.bar(np.arange(5) + i * width, values, width * 0.9, color=COLOURS[i], label=source)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], fontsize=7, color=INK)
    ax.set_xticks(np.arange(5) + width * (len(sources) - 1) / 2, GRADE_NAMES)
    ax.set(title="Share of each grade flagged as DR (grade 0 = false-positive rate)", ylabel="flagged fraction",
           ylim=(0, 1.1))
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(sources), loc="upper left")
    save(fig, work, f"05_flagged_by_grade_{name}_seed{seed}.png")


def plot_confusion(work, name, seed):
    path = run_path(work, name, seed)
    sources = available_sources(path)
    fig, axes = plt.subplots(1, len(sources), figsize=(4.4 * len(sources), 4.2))
    for ax, source in zip(np.atleast_1d(axes), sources):
        preds = pd.read_csv(os.path.join(path, f"preds_{source}.csv"))
        probs = preds[[c for c in preds.columns if c.startswith("p") and c[1:].isdigit()]].values
        matrix = confusion_matrix(preds.grade, probs.argmax(axis=1), labels=range(5), normalize="true")
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        for r in range(5):
            for c in range(5):
                ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center",
                        color="white" if matrix[r, c] > 0.55 else INK, fontsize=8)
        ax.set_xticks(range(5), range(5))
        ax.set_yticks(range(5), range(5))
        ax.grid(False)
        ax.set(title=source, xlabel="predicted grade", ylabel="true grade")
    save(fig, work, f"06_confusion_{name}_seed{seed}.png")


# ----- across experiments -----------------------------------------------------------------------

def mean_over_seeds(work, name, source, metric):
    """Mean of a metric (and its CI bounds) over every seed of an experiment."""
    values = []
    for f in glob.glob(os.path.join(work, "runs", name, "seed*", "results.json")):
        r = json.load(open(f)).get(source, {}).get("early_vs_none")
        if r:
            values.append([r[metric], *r[metric + "_ci"]])
    return np.mean(values, axis=0) if values else None


def plot_comparison(work, names, metric="sensitivity"):
    """Grouped bars: one group per test source, one bar per experiment, with 95% CI."""
    sources = [s for s in SOURCES if any(mean_over_seeds(work, n, s, metric) is not None for n in names)]
    fig, ax = plt.subplots(figsize=(2.6 * len(sources) + 2, 4.2))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        for j, source in enumerate(sources):
            v = mean_over_seeds(work, name, source, metric)
            if v is None:
                continue
            x = j + i * width
            ax.bar(x, v[0], width * 0.9, color=COLOURS[i % 3], label=name if j == 0 else None)
            ax.errorbar(x, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], color=INK, linewidth=1, capsize=3)
            ax.text(x, v[2] + 0.015, f"{v[0]:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(np.arange(len(sources)) + width * (len(names) - 1) / 2, sources)
    ax.set(ylim=(0, 1.1), ylabel=metric.replace("_", "-"),
           title=f"Early DR vs No DR: {metric.replace('_', '-')} by test source (95% CI)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(names), loc="upper right")
    save(fig, work, f"07_comparison_{metric}.png")


def plot_ablation(work):
    """Best validation score of every run (the table used to choose the imbalance method)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(work, "runs", "*", "seed*", "train_info.json"))):
        rows.append((f.split(os.sep)[-3], json.load(open(f))["best_val_score"]))
    if not rows:
        return
    table = pd.DataFrame(rows, columns=["run", "score"]).groupby("run").score.mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(table) + 1.5))
    bars = ax.barh(table.index, table.values, color=BLUE, height=0.6)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in table.values], fontsize=8, color=INK, padding=3)
    ax.set(title="Best validation score per run (choose on validation only)", xlim=(0, min(1.05, table.max() + 0.1)))
    ax.grid(axis="y", visible=False)
    save(fig, work, "08_validation_scores.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/kaggle/working")
    parser.add_argument("--runs", nargs="*", default=["arm_A", "arm_B", "arm_C"],
                        help="experiments to plot (none = only the data figures)")
    parser.add_argument("--seed", type=int, default=42, help="seed used for the single-run figures")
    args = parser.parse_args()

    plot_class_distribution(args.work)
    for name in args.runs:
        if not os.path.exists(os.path.join(run_path(args.work, name, args.seed), "results.json")):
            print("skipping", name, "(no results.json yet - run the evaluate stage first)")
            continue
        plot_training(args.work, name, args.seed)
        plot_roc_pr(args.work, name, args.seed)
        plot_reliability(args.work, name, args.seed)
        plot_flagged_by_grade(args.work, name, args.seed)
        if f"p4" in pd.read_csv(os.path.join(run_path(args.work, name, args.seed), "preds_DDR.csv")).columns:
            plot_confusion(args.work, name, args.seed)
    if args.runs:
        plot_comparison(args.work, args.runs, "sensitivity")
        plot_comparison(args.work, args.runs, "auroc")
    plot_ablation(args.work)


if __name__ == "__main__":
    main()
