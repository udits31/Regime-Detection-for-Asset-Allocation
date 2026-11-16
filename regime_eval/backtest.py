"""Turns target weights into after-cost daily returns, and computes the metrics.

Timing convention (the same for every strategy and benchmark):

    return of day u   = change from the close of day u-1 to the close of day u
    signal of day t   = target weights computed from data up to the close of day t
    execution         = at the close of day t+1
    first effect      = the return of day t+2

Trading cost is charged on the traded value of the two ETFs (cash moves are
free) and is taken out of the capital that earns the following day's return.
Between trades the weights drift with prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ASSETS = ["equity", "gold", "cash"]


def run_backtest(targets: pd.DataFrame, returns: pd.DataFrame, cost_bps: float,
                 rule: str = "band", band: float = 0.05) -> pd.DataFrame:
    """Simulate one strategy.

    targets : columns equity, gold (cash is the remainder), indexed by signal date.
    returns : columns equity_ret, gold_ret, cash_ret, one row per trading day.
    rule    : when a trade happens at an execution close
              "band"         when the target equity weight differs from the held
                             equity weight by more than `band`; all weights are
                             then set to target
              "monthly"      on the first execution close of each calendar month
              "initial_only" once, at the first execution close (buy and hold)

    Returns a frame indexed like `returns` with the net return, the weights
    held while earning it, the turnover traded at that day's close and the
    cost charged to that day. Before the first trade the portfolio is in cash.
    """
    if rule not in ("band", "monthly", "initial_only"):
        raise ValueError(rule)
    days = returns.index
    r = returns[["equity_ret", "gold_ret", "cash_ret"]].to_numpy()
    tgt = targets.reindex(days)[["equity", "gold"]].to_numpy()
    rate = cost_bps / 1e4
    w = np.array([0.0, 0.0, 1.0])            # held weights going into each day
    pending_cost = 0.0                       # cost of yesterday's trade, charged today
    traded_once = False
    last_trade_month = None
    out = np.zeros((len(days), 6))
    for u in range(len(days)):
        gross = float(w @ r[u])
        net = (1 - pending_cost) * (1 + gross) - 1
        out[u, :3] = w
        out[u, 3] = net
        out[u, 5] = pending_cost
        w = w * (1 + r[u]) / (1 + gross)     # drift to the close of day u
        pending_cost = 0.0
        # execution at the close of day u, using the signal of day u-1
        if u >= 1 and np.isfinite(tgt[u - 1]).all():
            want = np.array([tgt[u - 1, 0], tgt[u - 1, 1], 1 - tgt[u - 1].sum()])
            month = (days[u].year, days[u].month)
            if rule == "band":
                trade = (not traded_once) or abs(want[0] - w[0]) > band
            elif rule == "monthly":
                trade = (not traded_once) or month != last_trade_month
            else:
                trade = not traded_once
            if trade:
                turnover = abs(want[0] - w[0]) + abs(want[1] - w[1])
                out[u, 4] = turnover
                pending_cost = rate * turnover
                w = want
                traded_once = True
                last_trade_month = month
    return pd.DataFrame(out, index=days, columns=["w_equity", "w_gold", "w_cash", "net_return", "turnover", "cost"])


# ── Metrics ──────────────────────────────────────────────────────────────────

def annualised_return(r: np.ndarray, per_year: int = 252) -> float:
    r = np.asarray(r, float)
    return float(np.prod(1 + r) ** (per_year / len(r)) - 1)


def annualised_volatility(r: np.ndarray, per_year: int = 252) -> float:
    return float(np.std(np.asarray(r, float), ddof=1) * np.sqrt(per_year))


def sharpe_ratio(r: np.ndarray, rf: np.ndarray, per_year: int = 252) -> float:
    """Mean daily return in excess of the cash return, over its standard deviation, annualised."""
    ex = np.asarray(r, float) - np.asarray(rf, float)
    sd = np.std(ex, ddof=1)
    return float(np.mean(ex) / sd * np.sqrt(per_year)) if sd > 0 else float("nan")


def max_drawdown(r: np.ndarray) -> float:
    """Largest peak-to-trough fall of the cumulative value, as a negative number."""
    value = np.cumprod(1 + np.asarray(r, float))
    peak = np.maximum.accumulate(np.r_[1.0, value])[1:]
    return float((value / peak - 1).min())


def summarise(bt: pd.DataFrame, rf: pd.Series, per_year: int = 252) -> dict:
    r = bt["net_return"].to_numpy()
    years = len(r) / per_year
    return {
        "annual_return": annualised_return(r, per_year),
        "annual_volatility": annualised_volatility(r, per_year),
        "sharpe": sharpe_ratio(r, rf.reindex(bt.index).to_numpy(), per_year),
        "max_drawdown": max_drawdown(r),
        "annual_turnover": float(bt["turnover"].sum() / years),
        "annual_cost": float(bt["cost"].sum() / years),
        "mean_equity_weight": float(bt["w_equity"].mean()),
        "trades": int((bt["turnover"] > 0).sum()),
    }
