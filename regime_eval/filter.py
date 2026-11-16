"""Causal state probabilities for a Gaussian hidden Markov model.

`forward_filter` returns P(state on day t | observations on days 1..t).
It never uses observations after day t. This is the difference from
hmmlearn's `predict_proba`, which returns smoothed probabilities
P(state on day t | all observations) and therefore uses the future.
"""
from __future__ import annotations

import numpy as np


def gaussian_log_emission(X: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Log density of each observation under each state, diagonal covariance.

    X: (T, d); means, variances: (K, d). Returns (T, K).
    """
    X = np.asarray(X, dtype=float)
    diff = X[:, None, :] - means[None, :, :]                       # (T, K, d)
    return -0.5 * (np.log(2 * np.pi * variances)[None, :, :] + diff**2 / variances[None, :, :]).sum(axis=2)


def forward_filter(X, startprob, transmat, means, variances):
    """Forward recursion with per-day normalisation.

    Parameters
    ----------
    X : (T, d) observations, already scaled.
    startprob : (K,) state probabilities on day 1.
    transmat : (K, K), transmat[i, j] = P(state j tomorrow | state i today).
    means, variances : (K, d) Gaussian emission parameters.

    Returns
    -------
    filtered : (T, K), row t = P(state | X[0..t]); each row sums to 1.
    loglik : float, log P(X) under the model.
    """
    log_b = gaussian_log_emission(X, np.asarray(means, float), np.asarray(variances, float))
    return forward_filter_from_log_emission(log_b, startprob, transmat)


def forward_filter_from_log_emission(log_b, startprob, transmat):
    """Same recursion, taking the (T, K) log emission matrix directly."""
    log_b = np.asarray(log_b, dtype=float)
    startprob = np.asarray(startprob, dtype=float)
    transmat = np.asarray(transmat, dtype=float)
    T, K = log_b.shape
    filtered = np.empty((T, K))
    loglik = 0.0
    predicted = startprob                      # day 1: no yesterday, use the start distribution
    for t in range(T):
        if t > 0:
            predicted = filtered[t - 1] @ transmat          # predict
        shift = log_b[t].max()                 # keeps the best state at weight 1: no underflow
        unnorm = predicted * np.exp(log_b[t] - shift)       # update
        c = unnorm.sum()
        if not np.isfinite(c) or c <= 0:
            raise FloatingPointError(f"Filter failed on row {t}: normaliser {c}")
        filtered[t] = unnorm / c
        loglik += np.log(c) + shift
    return filtered, float(loglik)
