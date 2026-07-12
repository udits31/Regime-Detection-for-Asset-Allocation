# ============================================================
# utils/statistical_tests.py
# Rigorous Statistical Testing for Forecasts and Portfolios
#
# Implements:
#   1. Diebold-Mariano (1995)  — equal predictive accuracy
#   2. Jobson-Korkie (1981)    — equal Sharpe ratios
#   3. Bootstrap Sharpe CI     — block bootstrap confidence intervals
#   4. Information Ratio test  — vs benchmark
#   5. Harvey et al. corrected DM for small samples
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist
from typing import Dict, List, Optional, Tuple
from loguru import logger


RISK_FREE_ANNUAL = 0.065


# ── Diebold-Mariano Test ─────────────────────────────────────────────────────

def diebold_mariano(
    e1:        np.ndarray,
    e2:        np.ndarray,
    h:         int   = 1,
    criterion: str   = "MSE",
    small_sample: bool = True,
) -> Dict:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    H0: E[L(e1_t) - L(e2_t)] = 0   (both models equally accurate)
    H1: one model is more accurate

    Parameters
    ──────────
    e1, e2   : Forecast errors from model 1 and model 2.
    h        : Forecast horizon (lag truncation for HAC variance).
    criterion: "MSE" (squared error) or "MAE" (absolute error).
    small_sample: Apply Harvey, Leybourne & Newbold (1997) correction.

    Returns
    ───────
    stat, p_value, better_model, reject_H0
    """
    e1, e2 = np.asarray(e1), np.asarray(e2)
    T = len(e1)

    # Loss differential
    if criterion == "MSE":
        d = e1 ** 2 - e2 ** 2
    elif criterion == "MAE":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("criterion must be 'MSE' or 'MAE'")

    d_bar = d.mean()

    # HAC (Newey-West) variance of d_bar
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = sum(
        2 * (1 - k / h) * _autocovariance(d, k)
        for k in range(1, h)
    ) if h > 1 else 0.0
    var_d_bar = (gamma_0 + gamma_sum) / T

    dm_stat = d_bar / np.sqrt(max(var_d_bar, 1e-12))

    # Harvey, Leybourne & Newbold (1997) small-sample correction
    if small_sample:
        correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
        dm_stat *= correction
        p_value = 2 * t_dist.sf(abs(dm_stat), df=T - 1)
    else:
        p_value = 2 * norm.sf(abs(dm_stat))

    better = "Model 1" if d_bar > 0 else "Model 2"   # positive d → model 2 is better

    return {
        "dm_stat":      round(float(dm_stat),   4),
        "p_value":      round(float(p_value),   4),
        "better_model": better,
        "reject_H0":    bool(p_value < 0.05),
        "criterion":    criterion,
        "n":            T,
        "interpretation": (
            f"p={p_value:.3f} — {'REJECT' if p_value < 0.05 else 'FAIL TO REJECT'} "
            f"H0 of equal accuracy at 5%. {better} is more accurate."
        ),
    }


def _autocovariance(x: np.ndarray, lag: int) -> float:
    n = len(x)
    x_dm = x - x.mean()
    return float(np.dot(x_dm[lag:], x_dm[:n - lag]) / n)


# ── Jobson-Korkie Test ────────────────────────────────────────────────────────

def jobson_korkie_test(
    r1:             pd.Series,
    r2:             pd.Series,
    risk_free_rate: float = RISK_FREE_ANNUAL,
) -> Dict:
    """
    Jobson & Korkie (1981) test for equality of two Sharpe ratios.

    H0: SR(r1) = SR(r2)
    Uses the asymptotic distribution of the Sharpe ratio difference.

    Parameters
    ──────────
    r1, r2 : Daily return series for two strategies.

    Returns
    ───────
    jk_stat, p_value, sr1, sr2, reject_H0
    """
    r1, r2 = np.asarray(r1), np.asarray(r2)
    rf_d   = risk_free_rate / 252
    n      = len(r1)

    e1 = r1 - rf_d; e2 = r2 - rf_d
    s1  = r1.std(ddof=1); s2 = r2.std(ddof=1)
    m1  = e1.mean();       m2 = e2.mean()
    s12 = np.cov(r1, r2)[0, 1]

    sr1 = (m1 / s1) * np.sqrt(252)
    sr2 = (m2 / s2) * np.sqrt(252)

    # JK asymptotic variance of (SR1 - SR2)
    theta = (
        (2 * s1**2 * s2**2)
        - (2 * s1 * s2 * s12)
        + (0.5 * m1**2 * s2**2)
        + (0.5 * m2**2 * s1**2)
        - ((m1 * m2 * s12**2) / (s1 * s2))
    ) / (n * s1**2 * s2**2)

    if theta <= 0:
        return {"jk_stat": np.nan, "p_value": np.nan, "sr1": sr1,
                "sr2": sr2, "reject_H0": False,
                "interpretation": "Numerical issue in JK variance (theta ≤ 0)"}

    jk_stat = (m1 / s1 - m2 / s2) / np.sqrt(theta)
    p_value = 2 * norm.sf(abs(jk_stat))

    return {
        "jk_stat":   round(float(jk_stat), 4),
        "p_value":   round(float(p_value), 4),
        "sr1_annual": round(sr1, 4),
        "sr2_annual": round(sr2, 4),
        "sr_diff":   round(sr1 - sr2, 4),
        "reject_H0": bool(p_value < 0.05),
        "interpretation": (
            f"p={p_value:.3f} — {'REJECT' if p_value < 0.05 else 'FAIL TO REJECT'} "
            f"H0 that SR1=SR2 at 5%. "
            f"SR1={sr1:.3f}, SR2={sr2:.3f} (annualised)."
        ),
    }


# ── Bootstrap Sharpe Confidence Intervals ────────────────────────────────────

def bootstrap_sharpe_ci(
    returns:     pd.Series,
    risk_free:   float = RISK_FREE_ANNUAL,
    n_boot:      int   = 2000,
    block_size:  int   = 20,
    ci_level:    float = 0.95,
) -> Dict:
    """
    Stationary block bootstrap confidence interval for the Sharpe ratio.

    Block bootstrap (block_size≈20d ≈ 1 month) preserves autocorrelation
    structure of daily returns, giving valid inference for time series.

    Returns
    ───────
    sharpe_estimate, ci_lower, ci_upper, std_error
    """
    ret   = np.asarray(returns)
    rf_d  = risk_free / 252
    n     = len(ret)

    def sharpe(r): 
        return (r.mean() - rf_d) / (r.std(ddof=1) + 1e-10) * np.sqrt(252)

    point_est = sharpe(ret)
    boot_dist = np.zeros(n_boot)

    for b in range(n_boot):
        sample = []
        while len(sample) < n:
            start = np.random.randint(0, n)
            block = ret[start: start + block_size]
            sample.extend(block.tolist())
        boot_dist[b] = sharpe(np.array(sample[:n]))

    alpha = 1 - ci_level
    ci_lo = float(np.percentile(boot_dist, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_dist, 100 * (1 - alpha / 2)))

    return {
        "sharpe":      round(point_est, 4),
        "ci_lower":    round(ci_lo,     4),
        "ci_upper":    round(ci_hi,     4),
        "std_error":   round(float(boot_dist.std()), 4),
        "ci_level":    ci_level,
        "n_bootstrap": n_boot,
    }


# ── Information Ratio ─────────────────────────────────────────────────────────

def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualise:         bool = True,
) -> Dict:
    """
    Information Ratio = (E[r_p - r_b]) / std(r_p - r_b)
    """
    active  = portfolio_returns - benchmark_returns.reindex(portfolio_returns.index).fillna(0)
    ir      = active.mean() / (active.std() + 1e-10)
    if annualise:
        ir *= np.sqrt(252)

    alpha_annual = float(active.mean() * 252)
    te_annual    = float(active.std() * np.sqrt(252))

    return {
        "information_ratio": round(float(ir), 4),
        "alpha_annual":      round(alpha_annual, 4),
        "tracking_error":    round(te_annual, 4),
        "hit_rate":          round(float((active > 0).mean()), 4),
    }


# ── Pairwise Sharpe significance table ───────────────────────────────────────

def pairwise_sharpe_tests(
    equity_curves:  Dict[str, pd.Series],
    risk_free_rate: float = RISK_FREE_ANNUAL,
) -> pd.DataFrame:
    """
    Run JK tests for all pairs of strategies. Returns a p-value matrix.
    """
    names = list(equity_curves.keys())
    n     = len(names)
    pvals = pd.DataFrame(np.nan, index=names, columns=names)

    for i in range(n):
        for j in range(i + 1, n):
            r1 = equity_curves[names[i]].pct_change().dropna()
            r2 = equity_curves[names[j]].pct_change().dropna()
            idx = r1.index.intersection(r2.index)
            res = jobson_korkie_test(r1.loc[idx], r2.loc[idx], risk_free_rate)
            pvals.loc[names[i], names[j]] = res["p_value"]
            pvals.loc[names[j], names[i]] = res["p_value"]

    for name in names:
        pvals.loc[name, name] = 1.0

    return pvals.round(3)
