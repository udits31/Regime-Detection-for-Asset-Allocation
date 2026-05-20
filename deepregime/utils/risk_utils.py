# ============================================================
# utils/risk_utils.py
# Portfolio Risk Analytics
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


RISK_FREE_RATE = 0.065   # India repo rate


def compute_risk_dashboard(
    returns_df: pd.DataFrame,
    regime:     Optional[str] = None,
) -> Dict:
    """Aggregate risk dashboard for a multi-asset return DataFrame."""
    metrics: Dict = {}

    port_ret = returns_df.mean(axis=1)
    equity   = (1 + port_ret).cumprod()
    n_years  = len(equity) / 252

    # Basic performance
    cagr    = (equity.iloc[-1] / equity.iloc[0]) ** (1 / max(n_years, 0.1)) - 1
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe  = (cagr - RISK_FREE_RATE) / (ann_vol + 1e-10)

    # Drawdown
    roll_max = equity.cummax()
    dd       = (equity - roll_max) / roll_max
    max_dd   = float(dd.min())

    # VaR / CVaR
    var95    = float(np.percentile(port_ret, 5))
    cvar95   = float(port_ret[port_ret <= var95].mean()) if (port_ret <= var95).any() else var95
    var99    = float(np.percentile(port_ret, 1))
    cvar99   = float(port_ret[port_ret <= var99].mean()) if (port_ret <= var99).any() else var99

    # Sortino
    downside  = port_ret[port_ret < RISK_FREE_RATE / 252]
    sortino   = (cagr - RISK_FREE_RATE) / (downside.std() * np.sqrt(252) + 1e-10)

    # Calmar
    calmar    = cagr / (abs(max_dd) + 1e-10)

    # Tail risk
    skewness  = float(port_ret.skew())
    kurtosis  = float(port_ret.kurt())

    # Asset-level correlations and individual VaRs
    corr_matrix = returns_df.corr()
    asset_vars  = {
        col: round(float(np.percentile(returns_df[col].dropna(), 5)), 4)
        for col in returns_df.columns
    }

    # Rolling 60-day Sharpe
    roll_sharpe = (
        port_ret.rolling(60).mean() * 252 /
        (port_ret.rolling(60).std() * np.sqrt(252) + 1e-10)
    ).dropna()

    metrics.update({
        "summary": {
            "cagr":          round(cagr,     4),
            "annual_vol":    round(ann_vol,  4),
            "sharpe":        round(sharpe,   4),
            "sortino":       round(sortino,  4),
            "calmar":        round(calmar,   4),
            "max_drawdown":  round(max_dd,   4),
            "var_95":        round(var95,    4),
            "cvar_95":       round(cvar95,   4),
            "var_99":        round(var99,    4),
            "cvar_99":       round(cvar99,   4),
            "skewness":      round(skewness, 4),
            "excess_kurtosis": round(kurtosis, 4),
        },
        "current_regime":     regime,
        "asset_var_95":       asset_vars,
        "correlation_matrix": corr_matrix.round(3).to_dict(),
        "rolling_sharpe_60d": roll_sharpe.tail(20).round(3).to_dict(),
    })

    return metrics


def compute_marginal_var(
    weights:    np.ndarray,
    returns_df: pd.DataFrame,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Component VaR: marginal contribution of each asset to portfolio VaR."""
    cov      = returns_df.cov().values * 252
    port_var = float(weights @ cov @ weights)
    port_vol = np.sqrt(max(port_var, 1e-10))

    # Marginal: dσ/dw_i = (Σw)_i / σ
    marginal = cov @ weights / port_vol

    # Component VaR = w_i * marginal_i * z_alpha
    from scipy.stats import norm
    z        = norm.ppf(1 - confidence)
    comp_var = weights * marginal * abs(z)

    return dict(zip(returns_df.columns, comp_var.tolist()))


def rolling_correlation_breakdown(
    returns_df: pd.DataFrame,
    window:     int = 60,
) -> pd.DataFrame:
    """
    Rolling average pairwise correlation — useful for detecting correlation
    spikes during crisis regimes (contagion signal).
    """
    avg_corr = []
    for i in range(window, len(returns_df) + 1):
        window_ret = returns_df.iloc[i - window: i]
        corr       = window_ret.corr()
        # Average of off-diagonal elements
        n         = len(corr)
        mask      = ~np.eye(n, dtype=bool)
        avg       = float(corr.values[mask].mean())
        avg_corr.append({"date": returns_df.index[i - 1], "avg_correlation": avg})

    return pd.DataFrame(avg_corr).set_index("date")
