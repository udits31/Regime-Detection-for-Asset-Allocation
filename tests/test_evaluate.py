"""End-to-end check of the evaluation on a made-up panel. No real data, no model fitting."""
import numpy as np
import pandas as pd
import pytest

from regime_eval import benchmarks as bm
from regime_eval import evaluate as ev
from regime_eval.config import load_settings
from tests.test_leakage import first_leak


@pytest.fixture(scope="module")
def fake():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2013-06-03", "2016-12-30")
    eq = rng.normal(0.0004, 0.01, len(days))
    gold = rng.normal(0.0002, 0.008, len(days))
    panel = pd.DataFrame({"equity_ret": eq, "gold_ret": gold, "cash_ret": 0.0002,
                          "equity_px": 100 * np.cumprod(1 + eq), "gold_px": 50 * np.cumprod(1 + gold),
                          "rf_annual_pct": 5.0, "vix": 15.0}, index=days)
    oos = panel.loc[load_settings()["windows"]["oos_start"]:]
    sig = pd.DataFrame({"equity_target": rng.uniform(0.2, 0.8, len(oos))}, index=oos.index)
    return panel, {ev.HMM_NAME: sig, ev.HSMM_NAME: sig.copy()}


def test_evaluation_runs_and_reports_every_strategy(fake):
    panel, signals = fake
    metrics, diffs, sub, daily = ev.evaluate(panel, signals, cost_bps=10, bootstrap=False)
    assert set(metrics.index) == set(bm.BENCHMARKS) | set(signals)
    assert {"annual_return", "annual_volatility", "sharpe", "max_drawdown", "annual_turnover"} <= set(metrics.columns)
    assert daily.index[0] == panel.loc[load_settings()["windows"]["oos_start"]:].index[2]
    assert not daily.isna().any().any()


def test_identical_signals_give_identical_results_and_costs_reduce_returns(fake):
    panel, signals = fake
    with_cost, _, _, _ = ev.evaluate(panel, signals, cost_bps=10, bootstrap=False)
    no_cost, _, _, _ = ev.evaluate(panel, signals, cost_bps=0, bootstrap=False)
    assert with_cost.loc[ev.HMM_NAME].equals(with_cost.loc[ev.HSMM_NAME])
    traded = with_cost["annual_turnover"] > 0
    assert (with_cost.loc[traded, "annual_return"] < no_cost.loc[traded, "annual_return"]).all()
    assert with_cost.loc["Buy and hold NIFTYBEES", "trades"] == 1


def test_static_mix_holds_about_its_target(fake):
    panel, signals = fake
    metrics, _, _, _ = ev.evaluate(panel, signals, cost_bps=10, bootstrap=False)
    assert metrics.loc["Static mix 50/20/30", "mean_equity_weight"] == pytest.approx(0.50, abs=0.01)


def test_benchmark_signals_are_causal(fake):
    panel, _ = fake
    for fn in (bm.moving_average, bm.vol_target):
        assert first_leak(fn, panel, [300, 500, len(panel) - 1]) is None
