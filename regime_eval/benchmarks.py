"""Target weights for the benchmarks. Each row uses data up to that day only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from regime_eval.config import load_settings


def _frame(equity, gold, index) -> pd.DataFrame:
    return pd.DataFrame({"equity": equity, "gold": gold}, index=index)


def buy_and_hold(panel: pd.DataFrame) -> pd.DataFrame:
    b = load_settings()["benchmarks"]["buy_and_hold"]
    return _frame(b["equity"], 0.0, panel.index)


def static_mix(panel: pd.DataFrame) -> pd.DataFrame:
    b = load_settings()["benchmarks"]["static_mix"]
    return _frame(b["equity"], b["gold"], panel.index)


def moving_average(panel: pd.DataFrame) -> pd.DataFrame:
    """High equity weight when the close is above its trailing average, low when below."""
    b = load_settings()["benchmarks"]["moving_average"]
    px = panel["equity_px"]
    sma = px.rolling(b["window"]).mean()
    equity = pd.Series(np.where(px > sma, b["equity_above"], b["equity_below"]), index=panel.index)
    return _frame(equity.where(sma.notna()), b["gold"], panel.index)


def vol_target(panel: pd.DataFrame) -> pd.DataFrame:
    """Equity weight = target volatility / trailing realised volatility, bounded."""
    cfg = load_settings()
    b, per_year = cfg["benchmarks"]["vol_target"], cfg["data"]["trading_days_per_year"]
    rv = np.log1p(panel["equity_ret"]).rolling(b["realised_vol_window"]).std() * np.sqrt(per_year)
    equity = (b["target_annual_vol"] / rv).clip(b["equity_min"], b["equity_max"])
    return _frame(equity, b["gold"], panel.index)


def model_targets(signals: pd.DataFrame) -> pd.DataFrame:
    """Targets for the HMM and semi-Markov strategies from walk-forward signals."""
    gold = load_settings()["allocation"]["gold_weight"]
    return _frame(signals["equity_target"], gold, signals.index)


# name -> (target function, trading rule)
BENCHMARKS = {
    "Buy and hold NIFTYBEES": (buy_and_hold, "initial_only"),
    "Static mix 50/20/30": (static_mix, "monthly"),
    "200-day moving average": (moving_average, "band"),
    "Volatility targeting": (vol_target, "band"),
}
