"""Look-ahead test: deleting the future must not change the past.

For a signal function S and a cut day t:

    S(panel[:t])[u] == S(panel)[u]   for every u <= t

The toy cases show the test passes a causal signal and fails two leaky
ones. The last tests point it at the real features and the real
walk-forward pipeline on the cached panel.
"""
import numpy as np
import pandas as pd
import pytest

from regime_eval.config import PANEL_PATH, load_settings


def first_leak(signal_fn, data: pd.DataFrame, cuts: list[int], atol: float = 1e-8):
    """Return None if causal, else (cut, first differing date)."""
    full = signal_fn(data)
    for cut in cuts:
        part = signal_fn(data.iloc[:cut])
        ref = full.loc[part.index] if len(part) else full.iloc[:0]
        a, b = np.asarray(part, dtype=float), np.asarray(ref, dtype=float)
        bad = ~np.isclose(a, b, atol=atol, rtol=0, equal_nan=True)
        if bad.any():
            rows = bad.reshape(len(part), -1).any(axis=1)
            return cut, part.index[np.flatnonzero(rows)[0]]
    return None


# ── Toy cases with known answers ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=400)
    return pd.DataFrame({"r": rng.normal(0, 0.01, 400)}, index=idx)


CUTS = [150, 250, 399]


def test_trailing_mean_passes(toy):
    assert first_leak(lambda d: d["r"].rolling(21).mean(), toy, CUTS) is None


def test_centred_mean_fails(toy):
    leak = first_leak(lambda d: d["r"].rolling(21, center=True).mean(), toy, CUTS)
    assert leak is not None
    cut, date = leak
    assert toy.index.get_loc(date) >= cut - 21      # only the days just before the cut differ


def test_full_sample_standardisation_fails(toy):
    leak = first_leak(lambda d: (d["r"] - d["r"].mean()) / d["r"].std(), toy, CUTS)
    assert leak is not None
    assert leak[1] == toy.index[0]                  # every day differs, starting with the first


def test_shifted_back_one_day_fails(toy):
    assert first_leak(lambda d: d["r"].rolling(21).mean().shift(-1), toy, CUTS) is not None


# ── Real pipeline ────────────────────────────────────────────────────────────

needs_panel = pytest.mark.skipif(not PANEL_PATH.exists(), reason="data cache not built")


@pytest.fixture(scope="module")
def panel():
    from regime_eval.data import load_panel
    return load_panel()


def _oos_cut_positions(panel):
    """Row counts that end the day before a refit, on it, the day after, mid-block and late."""
    w = load_settings()["windows"]
    first = int(np.searchsorted(panel.index, pd.Timestamp(w["oos_start"])))
    step = w["refit_every_days"]
    # panel.iloc[:n] ends on out-of-sample day number n - first - 1 (0-based)
    return [first + step, first + step + 1, first + step + 2, first + step + 32, first + 5 * step + 10]


@needs_panel
def test_features_are_causal(panel):
    from regime_eval.features import build_features
    assert first_leak(build_features, panel, [300, 1500, 3000, len(panel) - 1]) is None


@needs_panel
def test_smoothed_probabilities_are_caught(panel):
    """Negative control: the library's smoothed probabilities must fail the test."""
    from hmmlearn.hmm import GaussianHMM
    from regime_eval.features import build_features

    def smoothed(p):
        f = build_features(p).dropna()
        m = GaussianHMM(n_components=2, covariance_type="diag", n_iter=30, random_state=0)
        z = (f - f.iloc[:1000].mean()) / f.iloc[:1000].std()
        m.fit(z.iloc[:1000].to_numpy())             # fixed training rows: parameters do not change
        return pd.DataFrame(m.predict_proba(z.to_numpy()), index=f.index).reindex(p.index)

    assert first_leak(smoothed, panel.iloc[:2000], [1500, 1800]) is not None


@needs_panel
@pytest.mark.slow
def test_walk_forward_signals_are_causal(panel):
    from regime_eval.walkforward import generate_signals

    cuts = _oos_cut_positions(panel)
    horizon = panel.iloc[: cuts[-1] + 40]            # enough history to contain every cut
    cols = ["equity_target", "n_states", "p_state_0", "p_state_1", "p_state_2", "p_state_3"]

    def signals(p):
        s = generate_signals(p)
        return s[cols] if len(s) else pd.DataFrame(columns=cols)

    leak = first_leak(signals, horizon, cuts)
    assert leak is None, f"look-ahead detected at cut {leak[0]}, first differing date {leak[1]}"


@needs_panel
@pytest.mark.slow
def test_semi_markov_walk_forward_signals_are_causal(panel):
    """Same property for the semi-Markov pipeline, over the first two blocks (fits are slower)."""
    from regime_eval.walkforward import generate_signals

    w = load_settings()["windows"]
    first = int(np.searchsorted(panel.index, pd.Timestamp(w["oos_start"])))
    step = w["refit_every_days"]
    cuts = [first + step, first + step + 1, first + step + 27]     # day before the refit, refit day, mid-block
    horizon = panel.iloc[: first + step + 37]
    cols = ["equity_target", "n_states", "p_state_0", "p_state_1", "p_state_2", "p_state_3"]

    def signals(p):
        s = generate_signals(p, model="hsmm")
        return s[cols] if len(s) else pd.DataFrame(columns=cols)

    leak = first_leak(signals, horizon, cuts)
    assert leak is None, f"look-ahead detected at cut {leak[0]}, first differing date {leak[1]}"
