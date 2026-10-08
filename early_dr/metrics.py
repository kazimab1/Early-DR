"""Metrics, patient-level bootstrap intervals and Holm correction."""
import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score


def auroc(y, score):
    return roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan


def pr_auc(y, score):
    return average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan


def threshold_at_specificity(y, score, specificity):
    """Smallest threshold whose specificity on (y, score) is at least `specificity`."""
    negatives = np.sort(score[y == 0])
    return np.quantile(negatives, specificity)


def sensitivity_specificity(y, score, threshold):
    """Positive when score > threshold."""
    flagged = score > threshold
    sens = flagged[y == 1].mean() if (y == 1).any() else np.nan
    spec = (~flagged)[y == 0].mean() if (y == 0).any() else np.nan
    return sens, spec


def ece(y, prob, bins=15):
    """Expected calibration error for a binary probability."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        in_bin = which == b
        if in_bin.any():
            total += in_bin.mean() * abs(prob[in_bin].mean() - y[in_bin].mean())
    return total


def brier(y, prob):
    return float(np.mean((prob - y) ** 2))


def reliability_curve(y, prob, bins=10):
    """(mean predicted, observed fraction, count) per bin, for plotting."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    return [(prob[which == b].mean(), y[which == b].mean(), int((which == b).sum()))
            for b in range(bins) if (which == b).any()]


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


# ----- bootstrap and Holm -----------------------------------------------------------------------

def bootstrap_indices(patient_ids, n_resamples, seed=0):
    """Yield row indices of a bootstrap sample drawn by PATIENT (both eyes stay together)."""
    rng = np.random.RandomState(seed)
    codes, first = np.unique(patient_ids, return_inverse=True)
    rows_of = [[] for _ in codes]
    for row, code in enumerate(first):
        rows_of[code].append(row)
    rows_of = [np.array(r) for r in rows_of]
    for _ in range(n_resamples):
        picked = rng.randint(0, len(codes), len(codes))
        yield np.concatenate([rows_of[p] for p in picked])


def bootstrap_ci(patient_ids, metric_fn, n_resamples=1000, seed=0):
    """95% interval of metric_fn(row_indices). Resamples where the metric is undefined are skipped."""
    values = np.array([metric_fn(idx) for idx in bootstrap_indices(patient_ids, n_resamples, seed)],
                      dtype=float)
    if np.isnan(values).all():
        return np.nan, np.nan
    return tuple(np.nanpercentile(values, [2.5, 97.5]))


def paired_bootstrap_p(patient_ids, metric_a, metric_b, n_resamples=1000, seed=0):
    """Two-sided p-value for 'metric_a differs from metric_b' on the SAME images (paired)."""
    diffs = np.array([metric_a(idx) - metric_b(idx)
                      for idx in bootstrap_indices(patient_ids, n_resamples, seed)], dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return np.nan
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(min(1.0, max(p, 1.0 / (len(diffs) + 1))))


def holm(p_values):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted
