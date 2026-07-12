# ============================================================
# models/change_point_detector.py
# Change Point Detection — PELT + Bayesian Online CPD
#
# Reviewer requirement: "A significant body of literature employs
# change point detection techniques instead of HMMs. Examples
# include Bayesian CPD, PELT, Binary Segmentation."
#
# These are fundamentally different from HMMs:
#   • No hidden states — directly detects structural breaks
#   • Non-parametric — no distributional assumption on regimes
#   • Deterministic outputs — no probabilistic regime membership
#
# Implementation:
#   1. PELT (Pruned Exact Linear Time) via ruptures
#   2. Binary Segmentation via ruptures
#   3. Bayesian Online CPD (Adams & MacKay 2007)
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger

try:
    import ruptures as rpt
    _HAS_RUPTURES = True
except ImportError:
    _HAS_RUPTURES = False
    logger.warning("ruptures not installed. PELT/BinSeg unavailable.")


class ChangePointDetector:
    """
    Structural break detector for financial time series.

    Detects regime changes as change points — dates where the
    statistical properties of returns shift significantly.

    Three algorithms:
    ─────────────────
    PELT (Pruned Exact Linear Time):
      • Exact optimal segmentation (minimises piecewise cost)
      • O(n) average time complexity
      • Penalty parameter controls number of breakpoints

    Binary Segmentation:
      • Greedy top-down segmentation
      • O(n log n), faster for long series

    Bayesian Online CPD (Adams & MacKay 2007):
      • Real-time detection with uncertainty quantification
      • Outputs: run-length probabilities P(r_t | data_{1:t})
      • Detects new regimes as they emerge (online learning)
    """

    def __init__(
        self,
        model:       str   = "rbf",   # ruptures cost model
        penalty:     float = 10.0,    # PELT penalty (higher = fewer breakpoints)
        min_size:    int   = 20,      # Minimum regime length (trading days)
        hazard:      float = 1/252,   # BOCPD: prior probability of changepoint per day
    ):
        self.model      = model
        self.penalty    = penalty
        self.min_size   = min_size
        self.hazard     = hazard
        self.breakpoints_pelt:   Optional[List[int]] = None
        self.breakpoints_binseg: Optional[List[int]] = None
        self.bocpd_run_lengths:  Optional[np.ndarray] = None
        self.feature_cols: List[str] = []

    # ── PELT ─────────────────────────────────────────────────────────────────

    def fit_pelt(self, features_df: pd.DataFrame) -> "ChangePointDetector":
        """Pruned Exact Linear Time changepoint detection."""
        if not _HAS_RUPTURES:
            raise ImportError("pip install ruptures")
        self.feature_cols = list(features_df.columns)
        signal = features_df.values

        algo = rpt.Pelt(model=self.model, min_size=self.min_size).fit(signal)
        # Returns list of end indices of segments (last = len(signal))
        bps = algo.predict(pen=self.penalty)
        # Convert to 0-based dates (exclude final endpoint = len)
        self.breakpoints_pelt = bps[:-1]
        logger.info(f"PELT detected {len(self.breakpoints_pelt)} changepoints")
        return self

    # ── Binary Segmentation ───────────────────────────────────────────────────

    def fit_binseg(
        self, features_df: pd.DataFrame, n_bkps: Optional[int] = None
    ) -> "ChangePointDetector":
        """Binary Segmentation (greedy top-down)."""
        if not _HAS_RUPTURES:
            raise ImportError("pip install ruptures")
        signal = features_df.values
        n      = n_bkps or max(1, len(signal) // 252)   # ~1 break per year
        algo   = rpt.Binseg(model=self.model, min_size=self.min_size).fit(signal)
        bps    = algo.predict(n_bkps=n)
        self.breakpoints_binseg = bps[:-1]
        logger.info(f"BinSeg detected {len(self.breakpoints_binseg)} changepoints")
        return self

    # ── Bayesian Online CPD ───────────────────────────────────────────────────

    def fit_bocpd(self, returns_series: pd.Series) -> "ChangePointDetector":
        """
        Bayesian Online Changepoint Detection (Adams & MacKay 2007).

        Assumes Gaussian observations with unknown mean/variance.
        Uses Normal-Gamma conjugate prior.
        Run-length distribution P(r_t | x_{1:t}) is computed sequentially.
        """
        x    = returns_series.dropna().values
        T    = len(x)
        # Normal-Gamma prior hyperparameters
        mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, 0.01

        # R[t] = probability distribution over run lengths at time t
        R          = np.zeros((T + 1, T + 1))
        R[0, 0]    = 1.0
        run_length_map = np.zeros(T, dtype=int)

        # Sufficient statistics per run length
        mu_arr  = np.full(T + 1, mu0)
        kap_arr = np.full(T + 1, float(kappa0))
        alp_arr = np.full(T + 1, float(alpha0))
        bet_arr = np.full(T + 1, float(beta0))

        changepoints = []

        for t in range(T):
            xt = x[t]
            # Predictive probabilities for each run length
            pred_probs = self._t_predictive(xt, mu_arr, kap_arr, alp_arr, bet_arr, t+1)

            # Hazard function: constant hazard rate
            h = self.hazard
            cp_mass = (R[:t+1, t] * pred_probs[:t+1] * h).sum()
            R[0, t+1] = cp_mass

            for r in range(t + 1):
                R[r+1, t+1] = R[r, t] * pred_probs[r] * (1 - h)

            # Normalise
            norm = R[:t+2, t+1].sum()
            if norm > 0:
                R[:t+2, t+1] /= norm

            # Update sufficient statistics
            kap_new = kap_arr[:t+2] + 1
            mu_new  = (kap_arr[:t+2] * mu_arr[:t+2] + xt) / kap_new
            alp_new = alp_arr[:t+2] + 0.5
            bet_new = (
                bet_arr[:t+2]
                + (kap_arr[:t+2] * (xt - mu_arr[:t+2])**2) / (2 * kap_new)
            )
            mu_arr[:t+2]  = mu_new
            kap_arr[:t+2] = kap_new
            alp_arr[:t+2] = alp_new
            bet_arr[:t+2] = bet_new

            run_length_map[t] = int(R[:t+2, t+1].argmax())
            if t > 0 and run_length_map[t] < run_length_map[t-1]:
                changepoints.append(t)

        self.bocpd_run_lengths  = run_length_map
        self.bocpd_changepoints = changepoints
        self.bocpd_R            = R
        self.bocpd_index        = returns_series.dropna().index
        logger.info(f"BOCPD detected {len(changepoints)} changepoints")
        return self

    def _t_predictive(
        self, x: float,
        mu: np.ndarray, kap: np.ndarray,
        alp: np.ndarray, bet: np.ndarray,
        size: int,
    ) -> np.ndarray:
        """Student-t predictive probability under Normal-Gamma conjugate."""
        from scipy.stats import t as t_dist
        df  = 2 * alp[:size]
        loc = mu[:size]
        sc2 = bet[:size] * (kap[:size] + 1) / (alp[:size] * kap[:size])
        sc  = np.sqrt(np.maximum(sc2, 1e-12))
        return t_dist.pdf(x, df=df, loc=loc, scale=sc)

    # ── Regime label assignment from breakpoints ──────────────────────────────

    def label_segments(
        self,
        index:       pd.DatetimeIndex,
        breakpoints: List[int],
        returns:     pd.Series,
    ) -> pd.Series:
        """Assign regime labels to each segment between breakpoints."""
        n       = len(index)
        labels  = pd.Series(["Unknown"] * n, index=index)
        edges   = [0] + list(breakpoints) + [n]

        seg_stats = []
        for i in range(len(edges) - 1):
            s, e = edges[i], edges[i+1]
            seg_ret  = float(returns.iloc[s:e].mean() * 252)
            seg_vol  = float(returns.iloc[s:e].std()  * np.sqrt(252))
            seg_stats.append((i, seg_ret, seg_vol, s, e))

        # Rank and label
        seg_stats.sort(key=lambda x: x[1] - x[2])
        names = ["Crisis / Panic", "Bear Market", "Sideways / Range-bound",
                 "Recovery", "Bull Market"]
        n_segs = len(seg_stats)
        for rank, (idx, ret, vol, s, e) in enumerate(seg_stats):
            name_idx = min(int(rank * len(names) / n_segs), len(names) - 1)
            labels.iloc[s:e] = names[name_idx]

        return labels

    def get_pelt_regimes(
        self, index: pd.DatetimeIndex, returns: pd.Series
    ) -> Optional[pd.Series]:
        if self.breakpoints_pelt is None:
            return None
        return self.label_segments(index, self.breakpoints_pelt, returns)

    def get_bocpd_run_lengths(self) -> Optional[pd.Series]:
        if self.bocpd_run_lengths is None:
            return None
        return pd.Series(self.bocpd_run_lengths, index=self.bocpd_index,
                         name="run_length")

    def summary(self) -> Dict:
        return {
            "pelt_breakpoints":   self.breakpoints_pelt   or [],
            "binseg_breakpoints": self.breakpoints_binseg or [],
            "bocpd_changepoints": getattr(self, "bocpd_changepoints", []),
            "n_pelt":    len(self.breakpoints_pelt)   if self.breakpoints_pelt   else 0,
            "n_binseg":  len(self.breakpoints_binseg) if self.breakpoints_binseg else 0,
            "n_bocpd":   len(getattr(self, "bocpd_changepoints", [])),
        }
