"""Explicit-duration hidden semi-Markov model with Gaussian emissions.

A plain HMM leaves a regime each day with a fixed probability, so regime
lengths are geometric. Here each regime j has its own duration
distribution: (length - 1) ~ NegativeBinomial(r_j, p_j), capped at
`max_duration` days. r_j = 1 is the geometric case, so the plain HMM is
nested inside this model.

Representation. The hidden state is the pair (regime j, days spent in it d).
Let h_j(d) = P(length = d | length >= d) be the hazard of leaving after d days.

    (j, d) -> (j, d+1)   with probability 1 - h_j(d)
    (j, d) -> (i, 1)     with probability h_j(d) * A[j, i],   i != j

A is the regime-to-regime matrix with a zero diagonal. At d = max_duration
the hazard is 1: the regime must end. Emissions depend on the regime only.

This is an ordinary HMM on K * max_duration states, so the forward filter
and expectation-maximisation are the standard ones, written to exploit the
sparse transition structure. Durations, means and variances are estimated
jointly. The last regime in the sample is right-censored, which the hazard
form handles without special treatment. The first regime is assumed to
start on the first day.

A fitted model is a dict with the same keys as the plain HMM's, except
`embedded_transmat` (zero diagonal) replaces `transmat`, plus `dur_r`,
`dur_p`, `max_duration` and diagnostics.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom

from regime_eval import hmm as hmm_mod
from regime_eval.config import load_settings
from regime_eval.filter import gaussian_log_emission

# Fitting constants, fixed before any out-of-sample run (configuration log entry 16).
MAX_EM_ITER = 30
EM_TOL = 1e-2          # stop when the training log-likelihood improves by less than this
VAR_FLOOR = 1e-3       # on scaled features, same as hmmlearn's default
R_BOUNDS = (0.05, 200.0)
P_BOUNDS = (1e-4, 1 - 1e-4)


# ── Durations ────────────────────────────────────────────────────────────────

def hazards(r: np.ndarray, p: np.ndarray, max_duration: int) -> np.ndarray:
    """h[j, d-1] = P(length = d | length >= d) for d = 1..max_duration. Shape (K, D)."""
    k = np.arange(max_duration)[None, :]                    # length - 1
    r, p = np.asarray(r, float)[:, None], np.asarray(p, float)[:, None]
    pmf = nbinom.pmf(k, r, p)
    surv = nbinom.sf(k - 1, r, p)                           # P(length - 1 >= k)
    h = np.where(surv > 1e-300, pmf / np.maximum(surv, 1e-300), 1.0)
    h = np.clip(h, 1e-12, 1.0)
    h[:, -1] = 1.0                                          # cap: the regime must end
    return h


def mean_duration(r, p) -> np.ndarray:
    """Mean regime length in days implied by the (uncapped) duration distribution."""
    r, p = np.asarray(r, float), np.asarray(p, float)
    return 1 + r * (1 - p) / p


def tail_beyond_cap(r, p, max_duration: int) -> np.ndarray:
    """P(length >= max_duration) under the uncapped distribution, per regime."""
    return nbinom.sf(max_duration - 2, np.asarray(r, float), np.asarray(p, float))


# ── Forward pass ─────────────────────────────────────────────────────────────

def _forward(log_b, startprob, A, h, store: bool = False):
    """Scaled forward recursion on (regime, days-in-regime).

    Returns per-regime filtered probabilities (T, K), the log-likelihood,
    and, if `store`, the full alpha array (T, K, D), the normalisers and
    the scaled emissions for the backward pass.
    """
    T, K = log_b.shape
    D = h.shape[1]
    stay = 1.0 - h
    shifts = log_b.max(axis=1)
    b = np.exp(log_b - shifts[:, None])
    probs = np.empty((T, K))
    c = np.empty(T)
    alphas = np.empty((T, K, D)) if store else None
    a = np.zeros((K, D))
    for t in range(T):
        new = np.zeros((K, D))
        if t == 0:
            new[:, 0] = startprob                           # the first regime starts on day 1
        else:
            new[:, 0] = (a * h).sum(axis=1) @ A             # regimes that ended, redistributed
            new[:, 1:] = a[:, :-1] * stay[:, :-1]           # regimes that continue, one day older
        new *= b[t][:, None]
        ct = new.sum()
        if not np.isfinite(ct) or ct <= 0:
            raise FloatingPointError(f"Semi-Markov filter failed on row {t}: normaliser {ct}")
        a = new / ct
        c[t] = ct
        probs[t] = a.sum(axis=1)
        if store:
            alphas[t] = a
    loglik = float(np.log(c).sum() + shifts.sum())
    return (probs, loglik, alphas, c, b) if store else (probs, loglik)


def hsmm_filter(model: dict, X: np.ndarray) -> tuple[np.ndarray, float]:
    """P(regime on day t | data up to day t), shape (T, K), and the log-likelihood.

    Same contract as `regime_eval.hmm.filter_with`. Uses no data after day t.
    """
    Z = (X - model["scaler_mean"]) / model["scaler_std"]
    log_b = gaussian_log_emission(Z, model["means"], model["variances"])
    h = hazards(model["dur_r"], model["dur_p"], model["max_duration"])
    return _forward(log_b, model["startprob"], model["embedded_transmat"], h)


# ── Expectation-maximisation ─────────────────────────────────────────────────

def _e_step(log_b, startprob, A, h):
    """Forward-backward. Returns the sufficient statistics and the log-likelihood."""
    T, K = log_b.shape
    D = h.shape[1]
    stay = 1.0 - h
    _, loglik, alphas, c, b = _forward(log_b, startprob, A, h, store=True)
    gamma = np.empty((T, K))
    exits = np.zeros((K, D))       # expected number of regimes of j that ended after exactly d days
    stays = np.zeros((K, D))       # expected number of times regime j continued past day d
    trans = np.zeros((K, K))       # expected number of j -> i regime changes
    beta = np.ones((K, D))
    gamma[T - 1] = alphas[T - 1].sum(axis=1)
    for t in range(T - 2, -1, -1):
        bb = b[t + 1]
        into_new = bb * beta[:, 0]                          # (K,) entering regime i at day 1
        enter = A @ into_new                                # (K,) leaving j, summed over destinations
        cont = np.zeros((K, D))
        cont[:, :-1] = bb[:, None] * beta[:, 1:]            # staying in j, one day older
        a = alphas[t]
        exit_post = a * h * enter[:, None] / c[t + 1]
        stay_post = a * stay * cont / c[t + 1]
        exits += exit_post
        stays += stay_post
        trans += (a * h).sum(axis=1)[:, None] * A * into_new[None, :] / c[t + 1]
        beta = (stay * cont + h * enter[:, None]) / c[t + 1]
        gamma[t] = (a * beta).sum(axis=1)
    return gamma, exits, stays, trans, loglik


def _duration_objective(theta, exits, stays, D):
    r = float(np.clip(np.exp(theta[0]), *R_BOUNDS))
    p = float(np.clip(1 / (1 + np.exp(-theta[1])), *P_BOUNDS))
    h = np.clip(hazards([r], [p], D)[0, :-1], 1e-12, 1 - 1e-12)
    return -float((exits[:-1] * np.log(h) + stays[:-1] * np.log1p(-h)).sum())


def _fit_duration(exits, stays, r0, p0, D):
    """Maximise the expected log-likelihood of one regime's durations over (r, p)."""
    theta0 = np.array([np.log(r0), np.log(p0 / (1 - p0))])
    f0 = _duration_objective(theta0, exits, stays, D)
    res = minimize(_duration_objective, theta0, args=(exits, stays, D), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 200})
    if not np.isfinite(res.fun) or res.fun > f0:
        return r0, p0                                       # never accept a worse fit
    r = float(np.clip(np.exp(res.x[0]), *R_BOUNDS))
    p = float(np.clip(1 / (1 + np.exp(-res.x[1])), *P_BOUNDS))
    return r, p


def _em(Z, init: dict, max_duration: int) -> dict:
    K = init["n_states"]
    startprob, A = init["startprob"].copy(), init["embedded_transmat"].copy()
    means, variances = init["means"].copy(), init["variances"].copy()
    r, p = init["dur_r"].copy(), init["dur_p"].copy()
    history = []
    best = None
    for _ in range(MAX_EM_ITER):
        h = hazards(r, p, max_duration)
        log_b = gaussian_log_emission(Z, means, variances)
        gamma, exits, stays, trans, loglik = _e_step(log_b, startprob, A, h)
        history.append(loglik)
        total_exits = exits.sum(axis=1)
        cap_share = np.where(total_exits > 1e-9, exits[:, -1] / np.maximum(total_exits, 1e-9), 0.0)
        best = {"startprob": startprob.copy(), "embedded_transmat": A.copy(), "means": means.copy(),
                "variances": variances.copy(), "dur_r": r.copy(), "dur_p": p.copy(),
                "train_loglik": loglik, "cap_hit_share": cap_share}
        if len(history) > 1 and history[-1] - history[-2] < EM_TOL:
            break
        # M-step
        w = gamma.sum(axis=0)
        means = (gamma.T @ Z) / w[:, None]
        variances = np.maximum((gamma.T @ Z**2) / w[:, None] - means**2, VAR_FLOOR)
        startprob = gamma[0] / gamma[0].sum()
        if K > 2:
            np.fill_diagonal(trans, 0.0)
            rows = trans.sum(axis=1, keepdims=True)
            A = np.where(rows > 1e-12, trans / np.maximum(rows, 1e-12), A)
        for j in range(K):
            r[j], p[j] = _fit_duration(exits[j], stays[j], r[j], p[j], max_duration)
    best["loglik_history"] = history
    return best


def _init_from_hmm(h: dict) -> dict:
    """Start from a plain HMM fit: the same model written in semi-Markov form (r = 1)."""
    K = h["n_states"]
    T = h["transmat"]
    stay = np.clip(np.diag(T), 1 - P_BOUNDS[1], 1 - P_BOUNDS[0])
    A = T.copy()
    np.fill_diagonal(A, 0.0)
    rows = A.sum(axis=1, keepdims=True)
    uniform = (np.ones((K, K)) - np.eye(K)) / (K - 1)
    A = np.where(rows > 1e-12, A / np.maximum(rows, 1e-12), uniform)
    return {"n_states": K, "startprob": h["startprob"], "embedded_transmat": A, "means": h["means"],
            "variances": np.maximum(h["variances"], VAR_FLOOR), "dur_r": np.ones(K), "dur_p": 1 - stay}


def _order_by_volatility(m: dict) -> dict:
    order = np.argsort(m["variances"][:, hmm_mod.RETURN_COLUMN], kind="stable")
    out = dict(m)
    for key in ("startprob", "means", "variances", "dur_r", "dur_p", "cap_hit_share"):
        out[key] = m[key][order]
    out["embedded_transmat"] = m["embedded_transmat"][np.ix_(order, order)]
    return out


def fit_hsmm(X_train: np.ndarray, n_states: int, seeds: list[int], max_duration: int | None = None) -> dict:
    """Fit by EM from several starts; keep the best training likelihood.

    Each start is the plain HMM fitted with that seed, rewritten in
    semi-Markov form. EM then frees the duration distributions.
    """
    if max_duration is None:
        max_duration = load_settings()["hsmm"]["max_duration_days"]
    mu, sd = hmm_mod.fit_scaler(X_train)
    Z = (X_train - mu) / sd
    best = None
    for seed in seeds:
        try:
            start = hmm_mod.fit_hmm(X_train, n_states, [seed])
            fitted = _em(Z, _init_from_hmm(start), max_duration)
        except (RuntimeError, FloatingPointError):
            continue
        if best is None or fitted["train_loglik"] > best["train_loglik"]:
            best = {**fitted, "seed": seed, "hmm_start_loglik": start["train_loglik"]}
    if best is None:
        raise RuntimeError(f"No semi-Markov fit succeeded for n_states={n_states}")
    best.update({"n_states": n_states, "scaler_mean": mu, "scaler_std": sd, "max_duration": max_duration})
    best = _order_by_volatility(best)
    best["mean_duration"] = mean_duration(best["dur_r"], best["dur_p"])
    best["tail_beyond_cap"] = tail_beyond_cap(best["dur_r"], best["dur_p"], max_duration)
    return best


def heldout_loglik(model: dict, X_window: np.ndarray, n_fit: int) -> float:
    """log P(last rows | first n_fit rows), as for the plain HMM."""
    _, ll_all = hsmm_filter(model, X_window)
    _, ll_fit = hsmm_filter(model, X_window[:n_fit])
    return ll_all - ll_fit


def fit_block(X_window: np.ndarray):
    """Walk-forward hook: choose the state count and fit inside one training window."""
    m = load_settings()["model"]
    fitted, scores = hmm_mod.select_and_fit(X_window, m["n_states_grid"], m["seeds"], m["validation_fraction"],
                                            fit=fit_hsmm, score=heldout_loglik)
    info = {"heldout": scores,
            "cap_hit_share_max": float(fitted["cap_hit_share"].max()),
            "tail_beyond_cap_max": float(fitted["tail_beyond_cap"].max())}
    return fitted, hsmm_filter, info


# ── Simulation (for tests and the synthetic benchmark only) ──────────────────

def simulate(T: int, means, sds, dur_r, dur_p, A, rng) -> tuple[np.ndarray, np.ndarray]:
    """Draw a one-feature semi-Markov series. Returns (observations (T, 1), true regimes (T,))."""
    K = len(means)
    states = np.empty(T, dtype=int)
    j = int(rng.integers(K))
    t = 0
    while t < T:
        length = 1 + int(rng.negative_binomial(dur_r[j], dur_p[j]))
        states[t:t + length] = j
        t += length
        j = int(rng.choice(K, p=A[j]))
    x = rng.normal(np.asarray(means)[states], np.asarray(sds)[states])
    return x[:, None], states
