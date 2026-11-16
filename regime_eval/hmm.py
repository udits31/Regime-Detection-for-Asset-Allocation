"""Plain Gaussian HMM: fitting, state ordering and state-count selection.

A fitted model is a plain dict of arrays so that the filter, the tests and
the semi-Markov model all share one representation:

    startprob (K,), transmat (K, K), means (K, d), variances (K, d),
    scaler_mean (d,), scaler_std (d,), n_states, train_loglik, seed

States are always ordered by ascending volatility: state 0 is the calmest.
"""
from __future__ import annotations

import warnings

import numpy as np

from regime_eval.filter import forward_filter

RETURN_COLUMN = 0  # position of log_return in the feature matrix


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the training rows only."""
    return X.mean(axis=0), X.std(axis=0)


def order_by_volatility(model: dict) -> dict:
    """Relabel states so that emission variance of the return is ascending."""
    order = np.argsort(model["variances"][:, RETURN_COLUMN], kind="stable")
    out = dict(model)
    out["startprob"] = model["startprob"][order]
    out["transmat"] = model["transmat"][np.ix_(order, order)]
    out["means"] = model["means"][order]
    out["variances"] = model["variances"][order]
    return out


def fit_hmm(X_train: np.ndarray, n_states: int, seeds: list[int]) -> dict:
    """Fit on the training rows with several random starts; keep the best training likelihood."""
    from hmmlearn.hmm import GaussianHMM

    mu, sd = fit_scaler(X_train)
    Z = (X_train - mu) / sd
    best = None
    for seed in seeds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, tol=1e-4, random_state=seed)
            try:
                m.fit(Z)
                ll = float(m.score(Z))
            except Exception:
                continue
        params_ok = all(np.isfinite(a).all() for a in (m.startprob_, m.transmat_, m.means_, m.covars_))
        if not np.isfinite(ll) or not params_ok:
            continue
        if best is None or ll > best["train_loglik"]:
            variances = np.array([np.diag(c) for c in m.covars_])   # hmmlearn returns full (d, d) matrices
            best = {
                "startprob": m.startprob_.copy(), "transmat": m.transmat_.copy(),
                "means": m.means_.copy(), "variances": variances,
                "scaler_mean": mu, "scaler_std": sd,
                "n_states": n_states, "train_loglik": ll, "seed": seed,
            }
    if best is None:
        raise RuntimeError(f"No HMM fit succeeded for n_states={n_states}")
    return order_by_volatility(best)


def filter_with(model: dict, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale X with the model's training-window scaler and run the forward filter."""
    Z = (X - model["scaler_mean"]) / model["scaler_std"]
    return forward_filter(Z, model["startprob"], model["transmat"], model["means"], model["variances"])


def heldout_loglik(model: dict, X_window: np.ndarray, n_fit: int) -> float:
    """log P(last rows | first n_fit rows) for a model fitted on the first n_fit rows.

    Computed as loglik(all rows) - loglik(first n_fit rows), both from the
    forward filter, so the held-out rows are scored given the history.
    """
    _, ll_all = filter_with(model, X_window)
    _, ll_fit = filter_with(model, X_window[:n_fit])
    return ll_all - ll_fit


def select_and_fit(X_window: np.ndarray, grid: list[int], seeds: list[int], validation_fraction: float, fit=fit_hmm, score=heldout_loglik) -> tuple[dict, dict]:
    """Choose the number of states inside one training window, then refit on all of it.

    The window is split in time order. Each candidate is fitted on the first
    part and scored on the last `validation_fraction` of the window. Nothing
    outside the window is touched. Returns (model, {n_states: held-out loglik}).
    """
    n_fit = int(round(len(X_window) * (1 - validation_fraction)))
    scores = {}
    for k in grid:
        try:
            scores[k] = score(fit(X_window[:n_fit], k, seeds), X_window, n_fit)
        except (RuntimeError, FloatingPointError):
            scores[k] = -np.inf
    best_k = max(grid, key=lambda k: (scores[k], -k))   # ties go to fewer states
    return fit(X_window, best_k, seeds), scores
