"""Most likely regime path (Viterbi) for both models.

These use the whole sample, so they are for in-sample description only
(the survival curves). Nothing in the trading path calls them.
"""
from __future__ import annotations

import numpy as np

from regime_eval.filter import gaussian_log_emission
from regime_eval.hsmm import hazards


def _log(a):
    with np.errstate(divide="ignore"):
        return np.log(a)


def viterbi_hmm(model: dict, X: np.ndarray) -> np.ndarray:
    """Most likely state sequence under the plain HMM. Returns (T,) integers."""
    Z = (X - model["scaler_mean"]) / model["scaler_std"]
    log_b = gaussian_log_emission(Z, model["means"], model["variances"])
    log_a = _log(model["transmat"])
    T, K = log_b.shape
    delta = _log(model["startprob"]) + log_b[0]
    back = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        cand = delta[:, None] + log_a                    # from i (rows) to j (columns)
        back[t] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + log_b[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_hsmm(model: dict, X: np.ndarray) -> np.ndarray:
    """Most likely regime sequence under the semi-Markov model. Returns (T,) integers.

    Works on (regime, days-in-regime). A regime either continues, one day
    older, or ends and a different regime starts at day 1.
    """
    Z = (X - model["scaler_mean"]) / model["scaler_std"]
    log_b = gaussian_log_emission(Z, model["means"], model["variances"])
    h = hazards(model["dur_r"], model["dur_p"], model["max_duration"])
    log_h, log_stay, log_a = _log(h), _log(1.0 - h), _log(model["embedded_transmat"])
    T, K = log_b.shape
    D = h.shape[1]
    delta = np.full((K, D), -np.inf)
    delta[:, 0] = _log(model["startprob"]) + log_b[0]
    back_regime = np.zeros((T, K), dtype=int)            # where a regime starting at t came from
    back_days = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        leaving = delta + log_h                          # (K, D) score of ending regime i after d days
        best_d = leaving.argmax(axis=1)
        best_leave = leaving.max(axis=1)
        cand = best_leave[:, None] + log_a               # from i to j
        src = cand.argmax(axis=0)
        new = np.full((K, D), -np.inf)
        new[:, 0] = cand.max(axis=0)
        new[:, 1:] = delta[:, :-1] + log_stay[:, :-1]
        delta = new + log_b[t][:, None]
        back_regime[t] = src
        back_days[t] = best_d[src]
    j, d = np.unravel_index(int(delta.argmax()), delta.shape)
    path = np.empty(T, dtype=int)
    for t in range(T - 1, -1, -1):
        path[t] = j
        if t == 0:
            break
        if d > 0:
            d -= 1                                       # same regime, one day younger
        else:
            j, d = back_regime[t, j], back_days[t, j]    # the regime started today
    return path
