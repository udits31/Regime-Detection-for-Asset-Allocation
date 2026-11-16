"""Known-answer tests for the forward filter."""
import numpy as np
import pytest
from scipy.stats import norm

from regime_eval.filter import forward_filter

A = np.array([[0.9, 0.1], [0.2, 0.8]])


def test_case_a_matches_hand_calculation():
    """Two states, one feature, two observations, worked out independently here."""
    x = np.array([[0.0], [3.0]])
    start = np.array([0.5, 0.5])
    means, var = np.array([[0.0], [3.0]]), np.ones((2, 1))
    b1 = np.array([norm.pdf(0, 0, 1), norm.pdf(0, 3, 1)])
    b2 = np.array([norm.pdf(3, 0, 1), norm.pdf(3, 3, 1)])
    a1 = start * b1
    f1 = a1 / a1.sum()
    a2 = (f1 @ A) * b2
    f2 = a2 / a2.sum()
    filtered, loglik = forward_filter(x, start, A, means, var)
    np.testing.assert_allclose(filtered, np.vstack([f1, f2]), atol=1e-12)
    np.testing.assert_allclose(loglik, np.log(a1.sum()) + np.log(a2.sum()), atol=1e-12)
    # the figures quoted in the specification
    np.testing.assert_allclose(filtered, [[0.98901, 0.01099], [0.08429, 0.91571]], atol=5e-6)
    assert loglik == pytest.approx(-4.6604, abs=5e-5)


def test_case_b_uninformative_data_stays_at_stationary_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 1))
    stationary = np.array([2 / 3, 1 / 3])
    filtered, _ = forward_filter(x, stationary, A, np.zeros((2, 1)), np.ones((2, 1)))
    np.testing.assert_allclose(filtered, np.tile(stationary, (50, 1)), atol=1e-12)


def test_case_c_agrees_with_hmmlearn_where_it_should_and_differs_where_it_should():
    from hmmlearn.hmm import GaussianHMM

    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal(0, 1, (150, 2)), rng.normal(3, 2, (150, 2)), rng.normal(0, 1, (100, 2))])
    m = GaussianHMM(n_components=2, covariance_type="diag", n_iter=50, random_state=0).fit(x)
    filtered, loglik = forward_filter(x, m.startprob_, m.transmat_, m.means_, np.array([np.diag(c) for c in m.covars_]))
    smoothed = m.predict_proba(x)
    assert loglik == pytest.approx(m.score(x), rel=1e-9)
    np.testing.assert_allclose(filtered[-1], smoothed[-1], atol=1e-8)   # on the last day there is no future
    assert np.abs(filtered - smoothed).max() > 1e-3                      # earlier days must differ


def test_case_d_extreme_observation_does_not_underflow():
    x = np.array([[0.0], [1000.0], [0.0]])
    filtered, loglik = forward_filter(x, [0.5, 0.5], A, np.array([[0.0], [3.0]]), np.ones((2, 1)))
    assert np.isfinite(filtered).all() and np.isfinite(loglik)
    np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-12)


def test_truncation_invariance():
    """Filtered probabilities up to day t do not change when later days are removed."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 3))
    args = ([0.5, 0.5], A, rng.normal(size=(2, 3)), np.ones((2, 3)))
    full, _ = forward_filter(x, *args)
    cut, _ = forward_filter(x[:120], *args)
    np.testing.assert_array_equal(full[:120], cut)
