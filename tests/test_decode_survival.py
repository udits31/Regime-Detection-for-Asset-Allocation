"""Tests for path decoding and the Kaplan-Meier estimator."""
import numpy as np
import pytest

from regime_eval import hsmm
from regime_eval.decode import viterbi_hmm, viterbi_hsmm
from regime_eval.survival import kaplan_meier, km_median, run_lengths


def test_plain_viterbi_matches_hmmlearn():
    from hmmlearn.hmm import GaussianHMM

    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 1, (120, 2)), rng.normal(3, 1, (120, 2)), rng.normal(0, 1, (120, 2))])
    m = GaussianHMM(n_components=2, covariance_type="diag", n_iter=50, random_state=0).fit(x)
    model = {"startprob": m.startprob_, "transmat": m.transmat_, "means": m.means_,
             "variances": np.array([np.diag(c) for c in m.covars_]), "scaler_mean": np.zeros(2), "scaler_std": np.ones(2)}
    np.testing.assert_array_equal(viterbi_hmm(model, x), m.predict(x))


def test_semi_markov_viterbi_with_geometric_durations_matches_plain():
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal(0, 1, (100, 1)), rng.normal(3, 1, (60, 1)), rng.normal(0, 1, (90, 1))])
    transmat = np.array([[0.9, 0.1], [0.2, 0.8]])
    base = {"startprob": np.array([0.6, 0.4]), "means": np.array([[0.0], [3.0]]), "variances": np.ones((2, 1)),
            "scaler_mean": np.zeros(1), "scaler_std": np.ones(1)}
    plain = viterbi_hmm({**base, "transmat": transmat}, x)
    semi = viterbi_hsmm({**base, "embedded_transmat": np.array([[0.0, 1.0], [1.0, 0.0]]),
                         "dur_r": np.ones(2), "dur_p": np.array([0.1, 0.2]), "max_duration": 252}, x)
    np.testing.assert_array_equal(semi, plain)


def test_semi_markov_viterbi_recovers_simulated_regimes():
    rng = np.random.default_rng(3)
    r, p, A = np.array([20.0, 20.0]), np.array([1 / 3, 0.5]), np.array([[0.0, 1.0], [1.0, 0.0]])
    x, states = hsmm.simulate(1500, [0.0, 3.0], [1.0, 1.0], r, p, A, rng)
    model = {"startprob": np.array([0.5, 0.5]), "means": np.array([[0.0], [3.0]]), "variances": np.ones((2, 1)),
             "scaler_mean": np.zeros(1), "scaler_std": np.ones(1), "embedded_transmat": A,
             "dur_r": r, "dur_p": p, "max_duration": 252}
    assert (viterbi_hsmm(model, x) == states).mean() > 0.97


def test_run_lengths_and_censoring():
    runs = run_lengths(np.array([0, 0, 0, 1, 1, 0, 2, 2, 2, 2]))
    assert runs["state"].tolist() == [0, 1, 0, 2]
    assert runs["length"].tolist() == [3, 2, 1, 4]
    assert runs["observed"].tolist() == [False, True, True, False]     # first and last are cut by the sample edges


def test_kaplan_meier_without_censoring_is_the_empirical_survival():
    curve = kaplan_meier(np.array([1, 2, 2, 5]), np.array([True] * 4))
    assert curve["days"].tolist() == [1, 2, 5]
    np.testing.assert_allclose(curve["survival"], [0.75, 0.25, 0.0])
    assert km_median(curve) == 2


def test_kaplan_meier_textbook_case_with_censoring():
    """Lengths 2, 3+, 4, 5+ (plus = censored): S(2) = 3/4, S(4) = 3/4 * 1/2 = 3/8."""
    curve = kaplan_meier(np.array([2, 3, 4, 5]), np.array([True, False, True, False]))
    assert curve["days"].tolist() == [2, 4]
    np.testing.assert_allclose(curve["survival"], [0.75, 0.375])
    assert km_median(curve) == 4


def test_median_not_reached_is_nan():
    curve = kaplan_meier(np.array([2, 9, 9, 9]), np.array([True, False, False, False]))
    assert np.isnan(km_median(curve))
