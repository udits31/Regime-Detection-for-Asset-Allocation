"""Paired stationary block bootstrap for the difference between two Sharpe ratios.

Daily returns are serially dependent (volatility clusters), so resampling
single days would understate the uncertainty. The stationary bootstrap of
Politis and Romano (1994) resamples blocks of random length: from each
position the next one follows with probability 1 - 1/L, otherwise a new
block starts at a random day. L is the mean block length.

"Paired" means the same resampled days are used for the strategy and the
benchmark, so their common market exposure cancels in the difference.
"""
from __future__ import annotations

import numpy as np


def stationary_indices(n: int, mean_block: float, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """(n_resamples, n) array of day indices; blocks wrap around the end of the sample."""
    jump = rng.random((n_resamples, n)) < 1.0 / mean_block
    jump[:, 0] = True
    starts = rng.integers(0, n, size=(n_resamples, n))
    block_id = np.cumsum(jump, axis=1) - 1
    cols = np.arange(n)[None, :]
    # position of the most recent block start at or before each column
    last_jump = np.maximum.accumulate(np.where(jump, cols, 0), axis=1)
    block_start = np.take_along_axis(starts, last_jump, axis=1)
    del block_id
    return (block_start + (cols - last_jump)) % n


def _sharpe_rows(ex: np.ndarray, per_year: int) -> np.ndarray:
    return ex.mean(axis=1) / ex.std(axis=1, ddof=1) * np.sqrt(per_year)


def sharpe_difference_ci(strategy_excess, benchmark_excess, mean_block: float = 21, n_resamples: int = 10000,
                         seed: int = 0, level: float = 0.95, per_year: int = 252, chunk: int = 500) -> dict:
    """Point estimate and percentile interval for Sharpe(strategy) - Sharpe(benchmark).

    Inputs are daily returns in excess of the cash return, aligned day by day.
    """
    s, b = np.asarray(strategy_excess, float), np.asarray(benchmark_excess, float)
    if s.shape != b.shape:
        raise ValueError("Series must be aligned")
    n = len(s)
    point = float(_sharpe_rows(s[None, :], per_year)[0] - _sharpe_rows(b[None, :], per_year)[0])
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for start in range(0, n_resamples, chunk):
        m = min(chunk, n_resamples - start)
        idx = stationary_indices(n, mean_block, m, rng)
        diffs[start:start + m] = _sharpe_rows(s[idx], per_year) - _sharpe_rows(b[idx], per_year)
    lo, hi = np.quantile(diffs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"difference": point, "ci_low": float(lo), "ci_high": float(hi),
            "share_of_resamples_above_zero": float((diffs > 0).mean())}
