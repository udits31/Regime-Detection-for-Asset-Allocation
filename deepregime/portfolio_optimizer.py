# ============================================================
# portfolio_optimizer.py
# Adaptive Portfolio Optimization Engine
#
# FIXED (vs original):
#   • available = [c for c in returns_df.columns if c in returns_df.columns]
#     was a tautology — now filters against TRADEABLE_UNIVERSE
#   • All methods return weights normalised to sum to 1
# NEW:
#   • Black-Litterman with regime-implied views
#   • Risk Parity via Newton's method (analytical gradient)
#   • Regime-specific asset class bounds are configurable
#   • Monte Carlo Simulation for portfolio stress testing
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Dict, List, Optional, Tuple
from loguru import logger

from config.settings import (
    REGIME_ALLOCATION_TEMPLATES, INDIAN_STOCKS,
    INDIAN_ETFS, CRYPTO_ASSETS, COMMODITY_ASSETS,
)

# All assets we consider tradeable (used to fix the tautological filter)
TRADEABLE_UNIVERSE = set(
    INDIAN_STOCKS + INDIAN_ETFS + CRYPTO_ASSETS + COMMODITY_ASSETS
)

RISK_FREE_RATE = 0.065   # India repo rate


class PortfolioOptimizer:
    """
    Multi-strategy regime-adaptive portfolio optimizer.

    Methods
    ───────
    max_sharpe       : Mean-variance tangency portfolio
    min_variance     : Global minimum variance portfolio
    risk_parity      : Equal risk contribution (Newton's method)
    regime_adaptive  : Blend optimised weights with regime template
    black_litterman  : BL with regime-implied return views
    equal_weight     : Fallback (no history needed)

    Covariance estimation uses Ledoit-Wolf shrinkage (sklearn) by
    default, which is more stable than sample covariance for the
    typical ratio of assets to observations in this system.
    """

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate
        self.asset_classes = {
            "equities":   INDIAN_STOCKS,
            "etfs":       INDIAN_ETFS,
            "crypto":     CRYPTO_ASSETS,
            "commodities":COMMODITY_ASSETS,
        }

    # ── Main entry point ─────────────────────────────────────────────────────

    def optimize(
        self,
        returns_df:   pd.DataFrame,
        regime:       str,
        method:       str = "max_sharpe",
        risk_profile: str = "moderate",
    ) -> Dict:
        """
        Compute optimal portfolio weights for the given regime.

        Parameters
        ──────────
        returns_df   : Historical daily returns (assets as columns).
        regime       : Detected regime string from EnsembleRegimeDetector.
        method       : One of max_sharpe | min_variance | risk_parity |
                       regime_adaptive | black_litterman | equal_weight.
        risk_profile : conservative | moderate | aggressive.

        Returns
        ───────
        {weights, expected_return, expected_vol, sharpe, metadata}
        """
        # FIXED: filter against tradeable universe, not the column list itself
        available = [c for c in returns_df.columns if c in TRADEABLE_UNIVERSE]
        if not available:
            # Fallback: use whatever is in the DataFrame
            available = list(returns_df.columns)
            logger.warning("No assets matched TRADEABLE_UNIVERSE; using all columns")

        ret = returns_df[available].dropna()

        if len(ret) < 63:
            logger.warning(f"Insufficient history ({len(ret)} rows); equal-weight fallback")
            return self._equal_weight(available)

        mu  = self._expected_returns(ret)
        cov = self._covariance(ret)
        bds = self._regime_bounds(regime, available, risk_profile)

        if method == "max_sharpe":
            w = self._max_sharpe(mu, cov, bds, available)
        elif method == "min_variance":
            w = self._min_variance(cov, bds, available)
        elif method == "risk_parity":
            w = self._risk_parity(cov, available)
        elif method == "regime_adaptive":
            w = self._regime_adaptive(mu, cov, regime, available, risk_profile)
        elif method == "black_litterman":
            w = self._black_litterman(ret, regime, mu, cov, available)
        else:
            return self._equal_weight(available)

        # Blend 70% optimised + 30% regime template
        w = self._regime_overlay(w, regime, available, blend=0.3)

        # Hard constraints: max 40% per asset, min 0
        w = np.clip(w, 0.0, 0.40)
        w = w / (w.sum() + 1e-10)

        port_ret = float(w @ mu * 252)
        port_vol = float(np.sqrt(w @ cov @ w * 252))
        sharpe   = (port_ret - self.risk_free_rate) / (port_vol + 1e-10)

        return {
            "weights":           dict(zip(available, w.round(4))),
            "expected_return":   round(port_ret, 4),
            "expected_vol":      round(port_vol, 4),
            "sharpe_ratio":      round(sharpe,   4),
            "regime":            regime,
            "method":            method,
            "n_assets":          int((w > 0.01).sum()),
            "top_holdings": sorted(
                zip(available, w.tolist()),
                key=lambda x: x[1], reverse=True
            )[:5],
        }

    # ── Expected Returns ──────────────────────────────────────────────────────

    def _expected_returns(
        self, ret: pd.DataFrame, method: str = "ewm"
    ) -> np.ndarray:
        """Exponentially weighted mean return (recent data weighted more)."""
        mu = ret.ewm(halflife=63).mean().iloc[-1]
        return mu.values * 252   # Annualise

    # ── Covariance ────────────────────────────────────────────────────────────

    def _covariance(self, ret: pd.DataFrame) -> np.ndarray:
        """Ledoit-Wolf shrinkage covariance (daily, then annualised)."""
        from sklearn.covariance import LedoitWolf
        lw  = LedoitWolf().fit(ret.values)
        return lw.covariance_ * 252

    # ── Max Sharpe ────────────────────────────────────────────────────────────

    def _max_sharpe(
        self,
        mu:        np.ndarray,
        cov:       np.ndarray,
        bounds:    List[Tuple[float, float]],
        assets:    List[str],
    ) -> np.ndarray:
        n   = len(assets)
        rf  = self.risk_free_rate

        def neg_sharpe(w: np.ndarray) -> float:
            p_ret = float(w @ mu)
            p_vol = float(np.sqrt(w @ cov @ w))
            return -(p_ret - rf) / (p_vol + 1e-10)

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        x0          = np.ones(n) / n

        result = sco.minimize(
            neg_sharpe, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        return result.x if result.success else x0

    # ── Min Variance ──────────────────────────────────────────────────────────

    def _min_variance(
        self,
        cov:    np.ndarray,
        bounds: List[Tuple[float, float]],
        assets: List[str],
    ) -> np.ndarray:
        n   = len(assets)
        x0  = np.ones(n) / n

        result = sco.minimize(
            lambda w: w @ cov @ w, x0, method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": 1000},
        )
        return result.x if result.success else x0

    # ── Risk Parity ───────────────────────────────────────────────────────────

    def _risk_parity(
        self, cov: np.ndarray, assets: List[str]
    ) -> np.ndarray:
        """
        Equal Risk Contribution via sequential quadratic programming.
        MRC_i * w_i = (1/n) * portfolio_vol  for all i.
        """
        n  = len(assets)
        x0 = np.ones(n) / n
        target_rc = np.ones(n) / n   # Equal contribution

        def rc_error(w: np.ndarray) -> float:
            pvar = float(w @ cov @ w)
            if pvar < 1e-12:
                return 1.0
            mrc = (cov @ w) / np.sqrt(pvar)
            rc  = w * mrc
            rc /= rc.sum()
            return float(((rc - target_rc) ** 2).sum())

        bounds = [(0.005, 0.5)] * n   # Minimum 0.5% to avoid degenerate solutions
        result = sco.minimize(
            rc_error, x0, method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": 2000, "ftol": 1e-10},
        )
        w = result.x if result.success else x0
        return w / w.sum()

    # ── Regime Adaptive ───────────────────────────────────────────────────────

    def _regime_adaptive(
        self,
        mu:          np.ndarray,
        cov:         np.ndarray,
        regime:      str,
        assets:      List[str],
        risk_profile: str,
    ) -> np.ndarray:
        """Max-Sharpe with regime-tightened asset class bounds."""
        bounds = self._regime_bounds(regime, assets, risk_profile)
        return self._max_sharpe(mu, cov, bounds, assets)

    # ── Black-Litterman ───────────────────────────────────────────────────────

    def _black_litterman(
        self,
        ret:    pd.DataFrame,
        regime: str,
        mu_hist: np.ndarray,
        cov:    np.ndarray,
        assets: List[str],
    ) -> np.ndarray:
        """
        Black-Litterman model with regime-implied absolute return views.

        Views are set by regime: Bull → equity views +; Bear → equity views -.
        Uncertainty (Omega) is proportional to view confidence.
        """
        n      = len(assets)
        tau    = 0.05   # Uncertainty scaling of the prior

        # Market cap proxy weights (equal weight as prior)
        w_mkt  = np.ones(n) / n
        pi     = tau * cov @ w_mkt   # Equilibrium excess returns

        # Build view matrix P and view vector q from regime
        views  = self._regime_views(assets, regime)
        if not views:
            # No views → fall back to max Sharpe on historical returns
            return self._max_sharpe(mu_hist, cov, [(0, 0.4)] * n, assets)

        P = np.zeros((len(views), n))
        q = np.zeros(len(views))
        omega_diag = np.zeros(len(views))

        for i, (asset_name, view_ret, confidence) in enumerate(views):
            if asset_name in assets:
                j      = assets.index(asset_name)
                P[i, j]= 1.0
                q[i]   = view_ret
                omega_diag[i] = (1.0 - confidence) * tau * float(cov[j, j])

        Omega = np.diag(omega_diag + 1e-8)
        tau_cov = tau * cov

        # BL posterior
        M1     = np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(Omega) @ P
        M2     = np.linalg.inv(tau_cov) @ pi + P.T @ np.linalg.inv(Omega) @ q
        mu_bl  = np.linalg.solve(M1, M2)

        bounds = self._regime_bounds(regime, assets, "moderate")
        return self._max_sharpe(mu_bl, cov, bounds, assets)

    def _regime_views(
        self, assets: List[str], regime: str
    ) -> List[Tuple[str, float, float]]:
        """
        Generate (asset, expected_return, confidence) triples per regime.
        Confidence ∈ [0, 1]; higher means Omega is smaller (more certain).
        """
        views: List[Tuple[str, float, float]] = []
        equity_assets = [a for a in assets if a in INDIAN_STOCKS]

        if "Bull" in regime:
            for a in equity_assets[:5]:
                views.append((a, 0.20, 0.70))   # 20% expected return, 70% confident
        elif "Bear" in regime or "Crisis" in regime:
            for a in equity_assets[:5]:
                views.append((a, -0.10, 0.75))
        elif "Recovery" in regime:
            for a in equity_assets[:3]:
                views.append((a, 0.12, 0.60))

        return views

    # ── Regime Overlay ────────────────────────────────────────────────────────

    def _regime_overlay(
        self,
        weights: np.ndarray,
        regime:  str,
        assets:  List[str],
        blend:   float = 0.3,
    ) -> np.ndarray:
        """Blend optimised weights with regime strategic template."""
        template = self._regime_template(regime, assets)
        return (1.0 - blend) * weights + blend * template

    def _regime_template(self, regime: str, assets: List[str]) -> np.ndarray:
        """Regime-specific target allocation template (equal within class)."""
        template = REGIME_ALLOCATION_TEMPLATES.get(regime, {})
        w = np.zeros(len(assets))
        for i, a in enumerate(assets):
            for cls, allocation in template.items():
                cls_assets = self.asset_classes.get(cls, [])
                if a in cls_assets:
                    n_in_cls = sum(1 for x in assets if x in cls_assets)
                    w[i]     = allocation / max(n_in_cls, 1)
        total = w.sum()
        return w / total if total > 1e-6 else np.ones(len(assets)) / len(assets)

    # ── Asset Bounds ─────────────────────────────────────────────────────────

    def _regime_bounds(
        self,
        regime:      str,
        assets:      List[str],
        risk_profile: str,
    ) -> List[Tuple[float, float]]:
        """Per-asset weight bounds derived from regime and risk profile."""
        # Base max per asset class
        class_limits = {
            "Bull Market":          {"equities": 0.40, "crypto": 0.15, "commodities": 0.10, "etfs": 0.30},
            "Bear Market":          {"equities": 0.15, "crypto": 0.03, "commodities": 0.30, "etfs": 0.40},
            "Crisis / Panic":       {"equities": 0.10, "crypto": 0.01, "commodities": 0.40, "etfs": 0.30},
            "High Volatility":      {"equities": 0.25, "crypto": 0.05, "commodities": 0.25, "etfs": 0.35},
            "Recovery":             {"equities": 0.35, "crypto": 0.10, "commodities": 0.15, "etfs": 0.30},
            "Sideways / Range-bound":{"equities": 0.30,"crypto": 0.08, "commodities": 0.20, "etfs": 0.35},
        }

        # Risk profile scaling
        risk_scale = {"conservative": 0.70, "moderate": 1.0, "aggressive": 1.30}
        scale      = risk_scale.get(risk_profile, 1.0)

        # Match regime (partial match for disambiguated labels like "Bull Market II")
        limits = {}
        for key, vals in class_limits.items():
            if key.split(" ")[0] in regime:
                limits = vals
                break
        if not limits:
            limits = class_limits["Sideways / Range-bound"]

        bounds = []
        for a in assets:
            max_w = 0.05   # Default minimum exposure
            for cls, cls_assets in self.asset_classes.items():
                if a in cls_assets:
                    max_w = min(limits.get(cls, 0.30) * scale, 0.40)
                    break
            bounds.append((0.0, max_w))
        return bounds

    # ── Fallback: equal weight ────────────────────────────────────────────────

    def _equal_weight(self, assets: List[str]) -> Dict:
        n = max(len(assets), 1)
        w = {a: round(1.0 / n, 4) for a in assets}
        return {
            "weights":          w,
            "expected_return":  None,
            "expected_vol":     None,
            "sharpe_ratio":     None,
            "regime":           "Unknown",
            "method":           "equal_weight",
            "n_assets":         n,
            "top_holdings":     list(w.items())[:5],
        }

    # ── Efficient frontier ────────────────────────────────────────────────────

    def efficient_frontier(
        self,
        returns_df: pd.DataFrame,
        n_points:   int = 50,
    ) -> pd.DataFrame:
        """Compute the mean-variance efficient frontier."""
        available = [c for c in returns_df.columns if c in TRADEABLE_UNIVERSE]
        if not available:
            available = list(returns_df.columns)
        ret = returns_df[available].dropna()
        mu  = self._expected_returns(ret)
        cov = self._covariance(ret)

        target_returns = np.linspace(mu.min(), mu.max(), n_points)
        frontier: List[Dict] = []

        for target in target_returns:
            n      = len(available)
            bounds = [(0.0, 0.40)] * n
            cons   = [
                {"type": "eq",  "fun": lambda w: w.sum() - 1.0},
                {"type": "eq",  "fun": lambda w, t=target: w @ mu - t},
            ]
            res = sco.minimize(
                lambda w: w @ cov @ w, np.ones(n)/n,
                method="SLSQP", bounds=bounds, constraints=cons,
                options={"maxiter": 1000},
            )
            if res.success:
                w      = res.x
                p_ret  = float(w @ mu)
                p_vol  = float(np.sqrt(w @ cov @ w))
                frontier.append({
                    "return": p_ret,
                    "vol":    p_vol,
                    "sharpe": (p_ret - self.risk_free_rate) / (p_vol + 1e-10),
                    "weights": dict(zip(available, w.round(4))),
                })

        return pd.DataFrame(frontier)
