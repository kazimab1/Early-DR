"""Calibration, fitted on VALIDATION data only and then frozen.

1. Temperature scaling (Guo et al. 2017): one number T that divides the logits.
2. Prior-shift correction (Saerens et al. 2002; Sipka et al. 2022): the model saw the class mix
   produced by the sampler and loss weights; probabilities are re-weighted to the natural
   prevalence. `estimate_prior_em` can also estimate the prevalence of an unlabelled new source.
"""
import numpy as np
import torch
import torch.nn.functional as F


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits, labels):
    """Find T > 0 minimising the validation negative log-likelihood."""
    logits = torch.tensor(logits, dtype=torch.float32)
    labels = torch.tensor(labels)
    log_t = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimiser.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_t.exp().detach().clamp(0.05, 20.0))   # guard against a degenerate fit


def training_prior(sampler_prior, loss_weights):
    """Class mix the model effectively learned: what the sampler showed x what the loss weighted."""
    prior = np.asarray(sampler_prior, dtype=float)
    if loss_weights is not None:
        prior = prior * loss_weights
    return prior / prior.sum()


def prior_correct(prob, source_prior, target_prior):
    """Re-weight probabilities from one class prior to another and renormalise."""
    adjusted = prob * (np.asarray(target_prior) / np.asarray(source_prior))
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def estimate_prior_em(prob, source_prior, iterations=100):
    """Estimate the class prior of an UNLABELLED set from the model's probabilities (EM)."""
    prior = np.asarray(source_prior, dtype=float).copy()
    for _ in range(iterations):
        prior = prior_correct(prob, source_prior, prior).mean(axis=0)
    return prior


def calibrate(logits, temperature, source_prior=None, target_prior=None):
    """logits -> calibrated probabilities (temperature, then optional prior correction)."""
    prob = softmax(logits / temperature)
    if source_prior is not None and target_prior is not None:
        prob = prior_correct(prob, source_prior, target_prior)
    return prob
