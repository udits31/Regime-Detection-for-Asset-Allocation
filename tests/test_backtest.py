"""Tests of the backtest mechanics, metrics and bootstrap on made-up data."""
import numpy as np
import pandas as pd
import pytest

from regime_eval import backtest as bt
from regime_eval.bootstrap import sharpe_difference_ci, stationary_indices

DAYS = pd.bdate_range("2021-01-01", periods=12)


def _returns(equity=0.0, gold=0.0, cash=0.0):
    return pd.DataFrame({"equity_ret": equity, "gold_ret": gold, "cash_ret": cash}, index=DAYS)


def _targets(equity, gold=0.0):
    return pd.DataFrame({"equity": equity, "gold": gold}, index=DAYS)


def test_signal_first_affects_the_return_two_days_later():
    """Signal on day 3 -> trade at the close of day 4 -> first earned return is day 5's."""
    equity = pd.Series(0.0, index=DAYS)
    equity.iloc[3:] = 1.0
    res = bt.run_backtest(_targets(equity), _returns(equity=0.01), cost_bps=0, rule="band")
    held = res["w_equity"].to_numpy()
    assert (held[:5] == 0).all()                    # days 0-4 still earn cash
    assert held[5] == pytest.approx(1.0)            # day 5 is the first equity return
    assert res["net_return"].iloc[4] == 0 and res["net_return"].iloc[5] == pytest.approx(0.01)


def test_cost_equals_turnover_times_rate_and_is_charged_next_day():
    res = bt.run_backtest(_targets(1.0), _returns(), cost_bps=10, rule="initial_only")
    assert res["turnover"].iloc[1] == pytest.approx(1.0)        # cash to 100% equity at the close of day 1
    assert res["turnover"].sum() == pytest.approx(1.0)
    assert res["net_return"].iloc[2] == pytest.approx(-0.001)   # 10 bps of 1.0, taken from day 2
    assert res["net_return"].drop(DAYS[2]).abs().max() == 0


def test_cash_leg_trades_are_free_and_both_etfs_are_charged():
    res = bt.run_backtest(_targets(0.5, 0.2), _returns(), cost_bps=10, rule="initial_only")
    assert res["turnover"].iloc[1] == pytest.approx(0.7)
    assert res["cost"].iloc[2] == pytest.approx(0.0007)


def test_buy_and_hold_matches_hand_calculation():
    equity = pd.Series([0.0, 0.0, 0.10, -0.05] + [0.0] * 8, index=DAYS)
    res = bt.run_backtest(_targets(1.0), _returns(equity=equity), cost_bps=0, rule="initial_only")
    assert np.prod(1 + res["net_return"]) == pytest.approx(1.10 * 0.95)


def test_weights_drift_between_trades():
    equity = pd.Series([0.0, 0.0, 0.10] + [0.0] * 9, index=DAYS)
    res = bt.run_backtest(_targets(0.5), _returns(equity=equity), cost_bps=0, rule="initial_only")
    assert res["w_equity"].iloc[2] == pytest.approx(0.5)
    assert res["w_equity"].iloc[3] == pytest.approx(0.55 / 1.05)   # drifted, not rebalanced


def test_band_blocks_small_changes_and_allows_large_ones():
    small = pd.Series(0.50, index=DAYS)
    small.iloc[5:] = 0.53
    assert (bt.run_backtest(_targets(small), _returns(), 10, "band", 0.05)["turnover"] > 0).sum() == 1
    large = pd.Series(0.50, index=DAYS)
    large.iloc[5:] = 0.56
    res = bt.run_backtest(_targets(large), _returns(), 10, "band", 0.05)
    assert (res["turnover"] > 0).sum() == 2
    assert res["turnover"].iloc[6] == pytest.approx(0.06)          # signal day 5, traded at the close of day 6


def test_monthly_rule_trades_once_per_month():
    days = pd.bdate_range("2021-01-01", "2021-04-30")
    rng = np.random.default_rng(0)
    rets = pd.DataFrame({"equity_ret": rng.normal(0, 0.01, len(days)), "gold_ret": 0.0, "cash_ret": 0.0}, index=days)
    tg = pd.DataFrame({"equity": 0.5, "gold": 0.2}, index=days)
    res = bt.run_backtest(tg, rets, 10, "monthly")
    assert (res["turnover"] > 0).groupby(res.index.to_period("M")).sum().tolist() == [1, 1, 1, 1]


def test_metrics_on_known_series():
    r = np.array([0.10, -0.50, 0.10])
    assert bt.max_drawdown(r) == pytest.approx(-0.50)
    assert bt.max_drawdown(np.array([-0.2, 0.1])) == pytest.approx(-0.2)   # a fall from the starting value counts
    assert bt.annualised_return(np.full(252, 0.001)) == pytest.approx(1.001**252 - 1)
    assert bt.annualised_volatility(np.array([0.01, -0.01] * 126)) == pytest.approx(np.std([0.01, -0.01] * 126, ddof=1) * np.sqrt(252))
    ex = np.array([0.002, 0.000] * 126)
    assert bt.sharpe_ratio(ex + 0.0003, np.full(252, 0.0003)) == pytest.approx(ex.mean() / ex.std(ddof=1) * np.sqrt(252))


def test_bootstrap_blocks_and_reproducibility():
    idx = stationary_indices(2000, 21, 200, np.random.default_rng(0))
    assert idx.min() >= 0 and idx.max() < 2000
    consecutive = (np.diff(idx, axis=1) % 2000 == 1).mean()
    assert consecutive == pytest.approx(1 - 1 / 21, abs=0.01)      # mean block length is about 21 days
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.0005, 0.01, 1500), rng.normal(0.0003, 0.01, 1500)
    first = sharpe_difference_ci(a, b, n_resamples=500, seed=0)
    assert first == sharpe_difference_ci(a, b, n_resamples=500, seed=0)
    assert first["ci_low"] < first["difference"] < first["ci_high"]


def test_bootstrap_of_identical_series_is_exactly_zero():
    a = np.random.default_rng(2).normal(0.0004, 0.01, 800)
    out = sharpe_difference_ci(a, a, n_resamples=300)
    assert out["difference"] == 0 and out["ci_low"] == 0 and out["ci_high"] == 0


def test_bootstrap_detects_a_clear_difference():
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 0.01, 3000)
    out = sharpe_difference_ci(base + 0.002, base, n_resamples=500)
    assert out["ci_low"] > 0
