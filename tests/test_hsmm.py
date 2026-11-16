"""Tests for the explicit-duration semi-Markov model."""
import numpy as np
import pytest

from regime_eval import hmm as hmm_mod
from regime_eval import hsmm
from regime_eval.filter import forward_filter


def _model(means, variances, r, p, A, start, D=252):
    k, d = np.asarray(means).shape
    return {"n_states": k, "startprob": np.asarray(start, float), "embedded_transmat": np.asarray(A, float),
            "means": np.asarray(means, float), "variances": np.asarray(variances, float),
            "dur_r": np.asarray(r, float), "dur_p": np.asarray(p, float), "max_duration": D,
            "scaler_mean": np.zeros(d), "scaler_std": np.ones(d)}


def test_geometric_durations_reproduce_the_plain_filter():
    """With r = 1 the model is a plain HMM, so the two filters must agree."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 2)) + np.repeat([[0, 0], [2, 1], [-1, 3]], 100, axis=0)
    transmat = np.array([[0.80, 0.15, 0.05], [0.10, 0.70, 0.20], [0.25, 0.15, 0.60]])
    means, variances = rng.normal(size=(3, 2)), np.array([[1.0, 0.5], [2.0, 1.0], [0.7, 1.5]])
    start = np.array([0.5, 0.3, 0.2])
    plain, ll_plain = forward_filter(x, start, transmat, means, variances)
    leave = 1 - np.diag(transmat)
    A = transmat.copy()
    np.fill_diagonal(A, 0)
    A /= A.sum(axis=1, keepdims=True)
    semi, ll_semi = hsmm.hsmm_filter(_model(means, variances, np.ones(3), leave, A, start), x)
    np.testing.assert_allclose(semi, plain, atol=1e-10)
    assert ll_semi == pytest.approx(ll_plain, abs=1e-8)


def test_hazard_of_geometric_is_constant_and_cap_forces_exit():
    h = hsmm.hazards([1.0], [0.1], 50)
    np.testing.assert_allclose(h[0, :-1], 0.1, atol=1e-10)
    assert h[0, -1] == 1.0


def test_filter_is_truncation_invariant():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(250, 1))
    m = _model([[0.0], [2.0]], [[1.0], [1.0]], [5.0, 3.0], [0.2, 0.1], [[0, 1], [1, 0]], [0.5, 0.5])
    full, _ = hsmm.hsmm_filter(m, x)
    cut, _ = hsmm.hsmm_filter(m, x[:140])
    np.testing.assert_array_equal(full[:140], cut)


@pytest.fixture(scope="module")
def simulated():
    """Two regimes with tightly concentrated lengths (mean 41 and 21 days): far from geometric."""
    rng = np.random.default_rng(7)
    r, p = np.array([20.0, 20.0]), np.array([1 / 3, 0.5])
    x, states = hsmm.simulate(3000, [0.0, 1.5], [1.0, 1.0], r, p, np.array([[0.0, 1.0], [1.0, 0.0]]), rng)
    return x, states, hsmm.mean_duration(r, p)


def test_fit_recovers_non_geometric_durations(simulated):
    x, states, true_mean = simulated
    fitted = hsmm.fit_hsmm(x, 2, [0, 1], max_duration=252)
    order = np.argsort(fitted["means"][:, 0])                     # match regimes by emission mean
    np.testing.assert_allclose(fitted["mean_duration"][order], true_mean, rtol=0.25)
    assert (fitted["dur_r"] > 3).all()                            # clearly not geometric (r = 1)
    assert fitted["train_loglik"] > fitted["hmm_start_loglik"]    # better than the plain HMM it started from


def test_em_never_decreases_the_likelihood(simulated):
    x, _, _ = simulated
    fitted = hsmm.fit_hsmm(x[:1200], 2, [0], max_duration=252)
    steps = np.diff(fitted["loglik_history"])
    assert (steps > -1e-6).all(), f"likelihood fell by {steps.min()}"


def test_semi_markov_filter_tracks_true_regimes_better_than_plain(simulated):
    """On data with non-geometric lengths, knowing the duration law should help classify days."""
    x, states, _ = simulated
    train, test = x[:2000], slice(2000, 3000)
    semi = hsmm.fit_hsmm(train, 2, [0, 1], max_duration=252)
    plain = hmm_mod.fit_hmm(train, 2, [0, 1])
    p_semi, _ = hsmm.hsmm_filter(semi, x)
    p_plain, _ = hmm_mod.filter_with(plain, x)

    def accuracy(probs, model):
        hi = int(np.argmax(model["means"][:, 0]))                 # the regime with the higher mean is true regime 1
        return float(((probs[test, hi] > 0.5).astype(int) == states[test]).mean())

    assert accuracy(p_semi, semi) >= accuracy(p_plain, plain)
