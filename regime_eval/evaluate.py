"""Out-of-sample evaluation: strategies, benchmarks, metrics and intervals.

`evaluate` is a pure function of the panel and the walk-forward signals, so
it can be tested on made-up data. `write_outputs` writes the tables with a
header stating the data source, window, benchmark and cost basis.

All strategies start in cash on the first out-of-sample day, receive their
first signal that day, trade at the next close, and are measured from the
third out-of-sample day, the first day any of them can earn a return.
"""
from __future__ import annotations

import pandas as pd

from regime_eval import backtest as bt
from regime_eval import benchmarks as bm
from regime_eval.bootstrap import sharpe_difference_ci
from regime_eval.config import load_settings

HMM_NAME = "Plain HMM strategy"
HSMM_NAME = "Semi-Markov strategy"

# Sub-periods fixed before the out-of-sample run (configuration log entry 21).
SUBPERIODS = {"2015-2019": ("2015-01-01", "2019-12-31"),
              "2020": ("2020-01-01", "2020-12-31"),
              "2021-2026": ("2021-01-01", "2026-10-30")}


def run_all(panel: pd.DataFrame, signals: dict[str, pd.DataFrame], cost_bps: float) -> dict[str, pd.DataFrame]:
    """Backtest every strategy and benchmark on the out-of-sample window at one cost level."""
    cfg = load_settings()
    w, band = cfg["windows"], cfg["trading"]["rebalance_band"]
    oos = panel.loc[w["oos_start"]:w["oos_end"]]
    out = {}
    for name, (target_fn, rule) in bm.BENCHMARKS.items():
        targets = target_fn(panel).loc[oos.index]           # computed with full history, then cut to the window
        out[name] = bt.run_backtest(targets, oos, cost_bps, rule, band)
    for name, sig in signals.items():
        out[name] = bt.run_backtest(bm.model_targets(sig).reindex(oos.index), oos, cost_bps, "band", band)
    first_day = oos.index[2]
    measured = {}
    for k, v in out.items():
        m = v.loc[first_day:].copy()
        # The opening trade happens the day before the first measured day and its cost is
        # charged on that first day; count its turnover there too so turnover and cost agree.
        m.iloc[0, m.columns.get_loc("turnover")] += v.loc[:first_day, "turnover"].iloc[:-1].sum()
        measured[k] = m
    return measured


def evaluate(panel: pd.DataFrame, signals: dict[str, pd.DataFrame], cost_bps: float, bootstrap: bool = True):
    """Return (metrics table, Sharpe-difference table, sub-period table, daily net returns)."""
    cfg = load_settings()
    b, per_year = cfg["bootstrap"], cfg["data"]["trading_days_per_year"]
    runs = run_all(panel, signals, cost_bps)
    rf = panel["cash_ret"]
    metrics = pd.DataFrame({k: bt.summarise(v, rf, per_year) for k, v in runs.items()}).T
    metrics.index.name = "strategy"
    daily = pd.DataFrame({k: v["net_return"] for k, v in runs.items()})
    excess = daily.sub(rf.reindex(daily.index), axis=0)

    diffs = []
    if bootstrap:
        pairs = [(s, bench) for s in signals for bench in bm.BENCHMARKS]
        if HMM_NAME in signals and HSMM_NAME in signals:
            pairs.append((HSMM_NAME, HMM_NAME))
        for strat, bench in pairs:
            ci = sharpe_difference_ci(excess[strat].to_numpy(), excess[bench].to_numpy(), b["mean_block_days"],
                                      b["n_resamples"], b["seed"], per_year=per_year)
            diffs.append({"strategy": strat, "benchmark": bench, "sharpe_strategy": metrics.loc[strat, "sharpe"],
                          "sharpe_benchmark": metrics.loc[bench, "sharpe"], **ci})

    sub = []
    for label, (a, z) in SUBPERIODS.items():
        d = daily.loc[a:z]
        if len(d) < 20:
            continue
        for k in d.columns:
            sub.append({"period": label, "strategy": k, "days": len(d),
                        "annual_return": bt.annualised_return(d[k].to_numpy(), per_year),
                        "sharpe": bt.sharpe_ratio(d[k].to_numpy(), rf.reindex(d.index).to_numpy(), per_year),
                        "max_drawdown": bt.max_drawdown(d[k].to_numpy())})
    return metrics, pd.DataFrame(diffs), pd.DataFrame(sub), daily


def header(daily: pd.DataFrame, cost_bps: float, what: str) -> str:
    cfg = load_settings()
    d, t = cfg["data"], cfg["trading"]
    return (f"# {what}\n"
            f"# Data source: real data. {d['tickers']['equity']} and {d['tickers']['gold']} ({d['source_prices']}, with the logged corrections); "
            f"cash and risk-free rate: {d['source_cash']}.\n"
            f"# Out-of-sample window: {daily.index[0].date()} to {daily.index[-1].date()} ({len(daily)} trading days). "
            f"Models refitted every {cfg['windows']['refit_every_days']} days on data before each block only.\n"
            f"# Costs: {'before costs' if cost_bps == 0 else f'after costs of {cost_bps:g} basis points per unit of ETF turnover'}. "
            f"Execution: {t['execution']}; rebalancing band {t['rebalance_band']:.0%}.\n"
            f"# Benchmarks: named in each row. Sharpe ratios use the cash return as the risk-free rate.\n")


def write_table(path, df: pd.DataFrame, head: str, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(head)
        df.to_csv(f, index=index, float_format="%.6f", lineterminator="\n")
