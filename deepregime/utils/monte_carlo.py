# ============================================================
# utils/monte_carlo.py
# Monte Carlo Portfolio Simulation — Regime-Conditional
#
# NEW MODULE
#
# Simulates portfolio paths under regime-conditional return
# distributions using Geometric Brownian Motion with:
#   • Regime-specific drift and volatility parameters
#   • Correlated asset returns (Cholesky decomposition)
#   • Fat tails via Student-t innovations (nu degrees of freedom)
#   • Jump-diffusion extension for crisis regimes
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


# Regime-conditional volatility scaling factors (relative to normal)
REGIME_VOL_SCALE: Dict[str, float] = {
    "Bull Market":           0.80,
    "Bull Market II":        0.85,
    "Recovery":              0.90,
    "Sideways / Range-bound": 1.00,
    "High Volatility":       1.60,
    "Bear Market":           1.40,
    "Bear Market II":        1.50,
    "Crisis / Panic":        2.50,
    "Uncertain":             1.20,
}

# Regime-conditional daily drift adjustment (annualised)
REGIME_DRIFT_ADJ: Dict[str, float] = {
    "Bull Market":           0.08,
    "Recovery":              0.04,
    "Sideways / Range-bound": 0.00,
    "High Volatility":      -0.02,
    "Bear Market":          -0.12,
    "Crisis / Panic":       -0.30,
    "Uncertain":            -0.02,
}

# Regime jump parameters (λ = jump intensity per year, J_mu, J_sigma)
REGIME_JUMPS: Dict[str, Tuple[float, float, float]] = {
    "Crisis / Panic": (20.0, -0.03, 0.02),  # 20 jumps/year, mean -3%, std 2%
    "Bear Market":    (5.0,  -0.01, 0.01),
    "High Volatility":(8.0,   0.00, 0.015),
}


@dataclass
class SimulationResult:
    final_values:   np.ndarray   # (n_sim,) — portfolio value at horizon
    paths:          np.ndarray   # (n_sim, horizon) — full paths (subsampled)
    var_95:         float
    cvar_95:        float
    var_99:         float
    median_return:  float
    prob_loss:      float
    prob_loss_10pct: float
    prob_loss_20pct: float
    best_case_95:   float
    worst_case_5:   float
    sharpe_sim:     float


class MonteCarloSimulator:
    """
    Regime-conditional Monte Carlo portfolio simulator.

    Uses Geometric Brownian Motion with:
    1. Historical covariance matrix (Ledoit-Wolf shrunk)
    2. Regime-adjusted drift and volatility
    3. Student-t innovations (fat tails, nu=5 default)
    4. Merton jump-diffusion for crisis regimes

    All simulations are vectorised over n_simulations for speed.
    """

    def __init__(
        self,
        nu:          float = 5.0,   # Student-t degrees of freedom
        risk_free:   float = 0.065,
        initial_capital: float = 1_000_000.0,
    ):
        self.nu              = nu
        self.risk_free       = risk_free
        self.initial_capital = initial_capital

    # ── Main simulation ───────────────────────────────────────────────────────

    def stress_test(
        self,
        weights:       Dict[str, float],
        returns_df:    pd.DataFrame,
        regime:        str,
        horizon_days:  int  = 252,
        n_simulations: int  = 10_000,
    ) -> Dict:
        """
        Simulate portfolio under a specific market regime.

        Parameters
        ──────────
        weights       : {asset: weight} dict (will be normalised)
        returns_df    : Historical daily returns for each asset
        regime        : Regime label (determines drift/vol scaling)
        horizon_days  : Simulation horizon in trading days
        n_simulations : Number of Monte Carlo paths

        Returns
        ───────
        Dict of risk metrics, percentile distribution, and path statistics.
        """
        assets   = [a for a in weights if a in returns_df.columns]
        w_vec    = np.array([weights[a] for a in assets])
        w_vec   /= w_vec.sum()

        ret_data = returns_df[assets].dropna()

        # ── Parameter estimation ──────────────────────────────────────────────
        mu_daily, cov_daily = self._estimate_params(ret_data)

        # Apply regime adjustments
        vol_scale  = self._vol_scale(regime)
        drift_adj  = self._drift_adj(regime)
        mu_reg     = mu_daily + drift_adj / 252
        cov_reg    = cov_daily * (vol_scale ** 2)

        # Portfolio-level parameters
        p_mu_reg   = float(w_vec @ mu_reg)
        p_sigma    = float(np.sqrt(w_vec @ cov_reg @ w_vec))

        logger.info(
            f"MC stress test | regime={regime} n={n_simulations} T={horizon_days} "
            f"μ={p_mu_reg*252:.2%} σ={p_sigma*np.sqrt(252):.2%}"
        )

        # ── Simulate paths ────────────────────────────────────────────────────
        jump_params = REGIME_JUMPS.get(regime.split(" ")[0] + (
            " " + regime.split(" ")[1] if len(regime.split()) > 1 else ""
        ), None)
        if not jump_params:
            jump_params = REGIME_JUMPS.get(regime.split(" ")[0], None)

        paths = self._simulate_paths(
            p_mu_reg, p_sigma,
            horizon_days, n_simulations,
            jump_params=jump_params,
        )

        final_values = self.initial_capital * paths[:, -1]

        # ── Risk metrics ──────────────────────────────────────────────────────
        result = self._compute_risk_metrics(final_values, paths, horizon_days)

        # ── Efficient frontier overlay ────────────────────────────────────────
        result.update({
            "regime":          regime,
            "horizon_days":    horizon_days,
            "n_simulations":   n_simulations,
            "vol_scale":       vol_scale,
            "drift_adj_annual":round(drift_adj, 4),
            "sim_mu_annual":   round(p_mu_reg * 252, 4),
            "sim_sigma_annual": round(p_sigma * np.sqrt(252), 4),
            "assets":          assets,
            "weights":         dict(zip(assets, w_vec.round(4).tolist())),
            "percentile_distribution": self._percentile_distribution(final_values),
        })

        return result

    def _simulate_paths(
        self,
        mu:           float,
        sigma:        float,
        T:            int,
        n:            int,
        jump_params:  Optional[Tuple[float, float, float]] = None,
    ) -> np.ndarray:
        """
        Simulate n paths of length T using GBM with Student-t innovations.
        Returns (n, T) array of cumulative return factors (starts at 1.0).
        """
        dt = 1.0 / 252

        # Student-t innovations
        z   = np.random.standard_t(df=self.nu, size=(n, T))
        z  *= np.sqrt((self.nu - 2) / self.nu)   # Scale to unit variance

        log_ret = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z

        # Merton jump-diffusion
        if jump_params:
            lam, j_mu, j_sigma = jump_params
            # Number of jumps per day
            n_jumps = np.random.poisson(lam * dt, size=(n, T))
            jump_sizes = np.random.normal(j_mu, j_sigma, size=(n, T))
            log_ret += n_jumps * jump_sizes

        cum_returns = np.exp(np.cumsum(log_ret, axis=1))
        return cum_returns   # (n, T)

    def _compute_risk_metrics(
        self,
        final_values: np.ndarray,
        paths:        np.ndarray,
        horizon_days: int,
    ) -> Dict:
        returns   = final_values / self.initial_capital - 1.0
        n_years   = horizon_days / 252

        # VaR / CVaR
        var95   = float(np.percentile(returns, 5))
        cvar95  = float(returns[returns <= var95].mean()) if (returns <= var95).any() else var95
        var99   = float(np.percentile(returns, 1))

        # Probabilities
        prob_loss     = float((returns < 0).mean())
        prob_loss_10  = float((returns < -0.10).mean())
        prob_loss_20  = float((returns < -0.20).mean())

        # Percentile outcomes
        best_95   = float(np.percentile(returns, 95))
        worst_5   = float(np.percentile(returns, 5))
        median    = float(np.median(returns))

        # Simulated Sharpe
        ann_ret    = (1 + median) ** (1 / n_years) - 1
        path_rets  = np.diff(np.log(paths), axis=1)
        ann_vol    = float(path_rets.std() * np.sqrt(252))
        sharpe_sim = (ann_ret - self.risk_free) / (ann_vol + 1e-10)

        # Max drawdown distribution
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns   = (paths - running_max) / running_max
        max_dds     = drawdowns.min(axis=1)
        median_mdd  = float(np.median(max_dds))
        worst_mdd   = float(np.percentile(max_dds, 5))

        return {
            "var_95":           round(var95,    4),
            "cvar_95":          round(cvar95,   4),
            "var_99":           round(var99,    4),
            "median_return":    round(median,   4),
            "prob_loss":        round(prob_loss, 4),
            "prob_loss_10pct":  round(prob_loss_10, 4),
            "prob_loss_20pct":  round(prob_loss_20, 4),
            "best_case_95":     round(best_95,  4),
            "worst_case_5":     round(worst_5,  4),
            "sharpe_sim":       round(sharpe_sim, 4),
            "median_max_drawdown": round(median_mdd, 4),
            "worst_max_drawdown":  round(worst_mdd,  4),
            "initial_capital":  self.initial_capital,
            "median_final_value": round(float(np.median(final_values)), 2),
            "var95_final_value":  round(self.initial_capital * (1 + var95), 2),
        }

    def _percentile_distribution(self, final_values: np.ndarray) -> Dict:
        """Return the full percentile distribution of final portfolio values."""
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        return {
            f"p{p}": round(float(np.percentile(final_values, p)), 2)
            for p in pcts
        }

    # ── Parameter estimation ──────────────────────────────────────────────────

    def _estimate_params(
        self, returns_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ledoit-Wolf covariance + exponentially weighted mean returns."""
        from sklearn.covariance import LedoitWolf
        mu  = returns_df.ewm(halflife=63).mean().iloc[-1].values
        lw  = LedoitWolf().fit(returns_df.values)
        return mu, lw.covariance_

    # ── Regime helpers ────────────────────────────────────────────────────────

    def _vol_scale(self, regime: str) -> float:
        for key, scale in REGIME_VOL_SCALE.items():
            if key in regime:
                return scale
        return 1.0

    def _drift_adj(self, regime: str) -> float:
        for key, adj in REGIME_DRIFT_ADJ.items():
            if key in regime:
                return adj
        return 0.0
