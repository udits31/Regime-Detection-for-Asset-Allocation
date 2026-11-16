"""The four model features. Every one uses only the current and earlier days."""
from __future__ import annotations

import numpy as np
import pandas as pd

from regime_eval.config import load_settings


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Row t depends on panel rows up to and including t, never later ones.

    log_return       daily log return of the equity ETF
    realised_vol_21  annualised standard deviation of the last 21 log returns
    log_vix          log of India VIX
    drawdown_252     equity price relative to its highest close in the last 252 days
    """
    cfg = load_settings()
    f, per_year = cfg["features"], cfg["data"]["trading_days_per_year"]
    log_ret = np.log1p(panel["equity_ret"])
    out = pd.DataFrame({
        "log_return": log_ret,
        "realised_vol_21": log_ret.rolling(f["realised_vol_window"]).std() * np.sqrt(per_year),
        "log_vix": np.log(panel["vix"]),
        "drawdown_252": panel["equity_px"] / panel["equity_px"].rolling(f["drawdown_window"]).max() - 1,
    })
    return out[f["names"]]
