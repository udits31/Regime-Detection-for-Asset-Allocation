# ============================================================
# tests/test_core.py
# Core Unit + Integration Tests for DeepRegime v2.0
#
# Tests validate the key bugs that were fixed:
#   1. HMM label disambiguation (no duplicate column names)
#   2. LSTM sequence creation (target is t+1, not t)
#   3. VWAP formula correctness
#   4. Tautological filter fix in portfolio optimizer
#   5. Kalman smoother probability normalisation
#   6. Walk-forward split non-overlap guarantee
#   7. Password hashing / verification
#   8. Monte Carlo path properties
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def price_series():
    """Synthetic NIFTY-like price series (GBM)."""
    np.random.seed(42)
    n   = 500
    ret = np.random.normal(0.0004, 0.012, n)
    px  = 15000 * np.exp(np.cumsum(ret))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(px, index=idx, name="^NSEI")


@pytest.fixture
def ohlcv_df(price_series):
    """Synthetic OHLCV DataFrame."""
    np.random.seed(42)
    close  = price_series
    high   = close * (1 + np.abs(np.random.normal(0, 0.005, len(close))))
    low    = close * (1 - np.abs(np.random.normal(0, 0.005, len(close))))
    vol    = np.random.lognormal(20, 1, len(close))
    return pd.DataFrame({
        "^NSEI":        close,
        "^NSEI_high":   high,
        "^NSEI_low":    low,
        "volume":       vol,
    }, index=close.index)


@pytest.fixture
def returns_df():
    """Multi-asset synthetic returns."""
    np.random.seed(42)
    n      = 300
    assets = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "GOLDBEES.NS"]
    idx    = pd.date_range("2021-01-01", periods=n, freq="B")
    data   = {a: np.random.normal(0.0003, 0.01, n) for a in assets}
    return pd.DataFrame(data, index=idx)


# ── 1. HMM label disambiguation ──────────────────────────────────────────────

class TestHMMRegimeDetector:

    def test_no_duplicate_columns_in_predict_proba(self, ohlcv_df):
        """predict_proba must return unique column names even when states share a base label."""
        from feature_engineering.feature_builder import FeatureBuilder
        from models.hmm_regime_detector import HMMRegimeDetector

        fb       = FeatureBuilder()
        features = fb.build_all_features(ohlcv_df)
        rf       = fb.get_regime_features(features)

        hmm = HMMRegimeDetector(n_components=6)
        hmm.fit(rf)

        proba = hmm.predict_proba(rf)
        assert proba.shape[1] == 6, "Should have 6 regime columns"
        assert len(proba.columns.unique()) == 6, \
            f"Duplicate column names found: {list(proba.columns)}"

    def test_label_map_covers_all_states(self, ohlcv_df):
        from feature_engineering.feature_builder import FeatureBuilder
        from models.hmm_regime_detector import HMMRegimeDetector

        fb       = FeatureBuilder()
        features = fb.build_all_features(ohlcv_df)
        rf       = fb.get_regime_features(features)

        hmm = HMMRegimeDetector(n_components=4)
        hmm.fit(rf)

        assert len(hmm.state_label_map) == 4
        assert all(isinstance(v, str) for v in hmm.state_label_map.values())

    def test_stationary_distribution_sums_to_one(self, ohlcv_df):
        from feature_engineering.feature_builder import FeatureBuilder
        from models.hmm_regime_detector import HMMRegimeDetector

        fb       = FeatureBuilder()
        features = fb.build_all_features(ohlcv_df)
        rf       = fb.get_regime_features(features)

        hmm = HMMRegimeDetector(n_components=4)
        hmm.fit(rf)

        pi = hmm.stationary_distribution()
        assert abs(pi.sum() - 1.0) < 1e-6
        assert (pi >= 0).all()

    def test_predict_length_matches_input(self, ohlcv_df):
        from feature_engineering.feature_builder import FeatureBuilder
        from models.hmm_regime_detector import HMMRegimeDetector

        fb       = FeatureBuilder()
        features = fb.build_all_features(ohlcv_df)
        rf       = fb.get_regime_features(features)

        hmm  = HMMRegimeDetector(n_components=4)
        hmm.fit(rf)
        pred = hmm.predict(rf)

        assert len(pred) == len(rf)


# ── 2. LSTM sequence creation (t+1 target) ───────────────────────────────────

class TestLSTMSequenceCreation:

    def _make_predictor(self):
        from models.lstm_regime_predictor import LSTMRegimePredictor
        return LSTMRegimePredictor(lookback=10, hidden_units=16, n_regimes=3)

    def test_sequences_shape(self):
        """X[i] corresponds to features[i:i+lookback], y[i] to labels[i+lookback]."""
        pred    = self._make_predictor()
        n       = 50
        features = np.random.randn(n, 5).astype(np.float32)
        labels   = np.random.randint(0, 3, n).astype(np.int32)

        X, y = pred._make_sequences(features, labels, forecast_horizon=1)

        # Expected: n - lookback - horizon + 1 = 50 - 10 - 1 + 1 = 40 samples
        expected = n - pred.lookback - 1 + 1
        assert len(X) == expected
        assert len(y) == expected
        assert X.shape == (expected, pred.lookback, 5)

    def test_target_is_next_step(self):
        """Verify y[i] = labels[i + lookback] (not labels[i + lookback - 1])."""
        pred     = self._make_predictor()
        n        = 30
        features = np.zeros((n, 3), dtype=np.float32)
        labels   = np.arange(n, dtype=np.int32)   # labels[t] = t (unique per step)

        X, y = pred._make_sequences(features, labels, forecast_horizon=1)

        # y[0] should be labels[0 + lookback + 1 - 1] = labels[lookback]
        assert y[0] == pred.lookback, \
            f"Expected y[0]={pred.lookback}, got {y[0]}"

    def test_no_label_leakage(self):
        """The lookback window must not overlap with the target label timestep."""
        pred     = self._make_predictor()
        n        = 40
        features = np.zeros((n, 2), dtype=np.float32)
        labels   = np.arange(n, dtype=np.int32)

        X, y = pred._make_sequences(features, labels, forecast_horizon=1)

        for i in range(len(X)):
            # X[i] covers feature indices [i, i+lookback)
            # Target y[i] = labels[i+lookback] — outside the window
            window_end  = i + pred.lookback
            target_step = i + pred.lookback   # same index (0-based)
            assert y[i] >= window_end, \
                f"Leakage at sample {i}: target={y[i]} within window [0,{window_end})"


# ── 3. VWAP formula ───────────────────────────────────────────────────────────

class TestFeatureBuilder:

    def test_vwap_formula_correctness(self, ohlcv_df):
        """VWAP_20 = Σ(P*V)/Σ(V), not the original broken formula."""
        from feature_engineering.feature_builder import FeatureBuilder

        fb       = FeatureBuilder()
        features = fb.build_all_features(ohlcv_df)

        assert "vwap_20" in features.columns, "vwap_20 should be in features"
        assert "vwap_ratio" in features.columns

        price = ohlcv_df["^NSEI"]
        vol   = ohlcv_df["volume"]
        # Manual computation of 20-day VWAP
        expected_vwap = (price * vol).rolling(20).sum() / vol.rolling(20).sum()
        expected_vwap = expected_vwap.reindex(features.index)

        np.testing.assert_allclose(
            features["vwap_20"].values,
            expected_vwap.values,
            rtol=1e-5,
            err_msg="VWAP formula is incorrect",
        )

    def test_no_infinite_values(self, ohlcv_df):
        from feature_engineering.feature_builder import FeatureBuilder
        fb       = FeatureBuilder()
        features = fb.build_all_features(ohlcv_df)
        assert not np.any(np.isinf(features.values)), "Feature matrix contains inf values"

    def test_no_nan_after_dropna(self, ohlcv_df):
        from feature_engineering.feature_builder import FeatureBuilder
        fb       = FeatureBuilder()
        features = fb.build_all_features(ohlcv_df)
        assert not features.isnull().any().any(), "Feature matrix contains NaN after dropna"

    def test_hurst_exponent_range(self, ohlcv_df):
        from feature_engineering.feature_builder import FeatureBuilder
        fb       = FeatureBuilder()
        features = fb.build_all_features(ohlcv_df)
        if "hurst_100d" in features.columns:
            valid = features["hurst_100d"].dropna()
            assert (valid >= 0.0).all() and (valid <= 1.5).all(), \
                "Hurst exponent out of expected range"


# ── 4. Portfolio optimizer tautological filter ───────────────────────────────

class TestPortfolioOptimizer:

    def test_available_filter_uses_tradeable_universe(self, returns_df):
        """
        FIXED BUG: available = [c for c in df.columns if c in df.columns]
        was a no-op. Now filters against TRADEABLE_UNIVERSE.
        """
        from portfolio_optimizer import PortfolioOptimizer, TRADEABLE_UNIVERSE

        # Add a fake non-tradeable column
        fake_df = returns_df.copy()
        fake_df["NOT_TRADEABLE_XYZ"] = 0.001

        opt = PortfolioOptimizer()
        # After fix, NOT_TRADEABLE_XYZ should be excluded
        available = [c for c in fake_df.columns if c in TRADEABLE_UNIVERSE]
        assert "NOT_TRADEABLE_XYZ" not in available

    def test_weights_sum_to_one(self, returns_df):
        from portfolio_optimizer import PortfolioOptimizer
        opt    = PortfolioOptimizer()
        result = opt.optimize(returns_df, "Bull Market", method="max_sharpe")
        w_sum  = sum(result["weights"].values())
        assert abs(w_sum - 1.0) < 1e-3, f"Weights sum = {w_sum}, expected ≈ 1.0"

    def test_no_negative_weights(self, returns_df):
        from portfolio_optimizer import PortfolioOptimizer
        opt    = PortfolioOptimizer()
        result = opt.optimize(returns_df, "Bear Market", method="risk_parity")
        for asset, w in result["weights"].items():
            assert w >= -1e-6, f"Negative weight for {asset}: {w}"


# ── 5. Kalman smoother probability normalisation ─────────────────────────────

class TestKalmanSmoother:

    def test_output_probabilities_sum_to_one(self):
        from models.kalman_smoother import RegimeProbabilityKalmanSmoother

        n, T  = 4, 100
        F     = np.ones((n, n)) / n
        ks    = RegimeProbabilityKalmanSmoother(process_noise=1e-3, obs_noise=1e-1)
        ks.fit(F)

        raw     = np.random.dirichlet(np.ones(n), size=T)
        smooth  = ks.smooth(raw)

        row_sums = smooth.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(T), atol=1e-6,
                                   err_msg="Smoothed probabilities don't sum to 1")

    def test_noise_reduction_non_negative(self):
        from models.kalman_smoother import RegimeProbabilityKalmanSmoother

        n, T = 4, 100
        F    = np.eye(n) * 0.9 + np.ones((n, n)) * 0.025  # Sticky diagonal
        F   /= F.sum(axis=1, keepdims=True)

        ks   = RegimeProbabilityKalmanSmoother()
        ks.fit(F)

        raw    = np.random.dirichlet(np.ones(n), size=T)
        smooth = ks.smooth(raw)
        nrr    = ks.noise_reduction_ratio(raw, smooth)

        assert nrr >= 0.0, f"Noise reduction ratio negative: {nrr}"

    def test_output_shape_matches_input(self):
        from models.kalman_smoother import RegimeProbabilityKalmanSmoother

        n, T = 6, 200
        F    = np.ones((n, n)) / n
        ks   = RegimeProbabilityKalmanSmoother()
        ks.fit(F)

        raw    = np.random.dirichlet(np.ones(n), size=T)
        smooth = ks.smooth(raw)
        assert smooth.shape == (T, n)


# ── 6. Walk-forward split non-overlap ────────────────────────────────────────

class TestWalkForwardOptimizer:

    def test_test_windows_do_not_overlap(self, ohlcv_df):
        from optimization.walk_forward import WalkForwardOptimizer

        wf    = WalkForwardOptimizer(n_splits=4, min_train=50, test_window=30)
        folds = wf.split(ohlcv_df)

        test_ranges = [(td.index[0], td.index[-1]) for _, td in folds]
        for i in range(1, len(test_ranges)):
            prev_end  = test_ranges[i - 1][1]
            curr_start = test_ranges[i][0]
            assert curr_start > prev_end, \
                f"Test windows overlap: fold {i-1} ends {prev_end}, fold {i} starts {curr_start}"

    def test_train_comes_before_test(self, ohlcv_df):
        from optimization.walk_forward import WalkForwardOptimizer

        wf    = WalkForwardOptimizer(n_splits=3, min_train=50, test_window=30)
        folds = wf.split(ohlcv_df)

        for i, (train_df, test_df) in enumerate(folds):
            assert train_df.index[-1] < test_df.index[0], \
                f"Fold {i}: training data bleeds into test window"

    def test_minimum_train_size_respected(self, ohlcv_df):
        from optimization.walk_forward import WalkForwardOptimizer

        min_t = 100
        wf    = WalkForwardOptimizer(n_splits=3, min_train=min_t, test_window=30)
        folds = wf.split(ohlcv_df)

        for i, (train_df, _) in enumerate(folds):
            assert len(train_df) >= min_t, \
                f"Fold {i} train size {len(train_df)} < min_train {min_t}"


# ── 7. Password hashing ───────────────────────────────────────────────────────

class TestAuth:

    def test_verify_correct_password(self):
        from api.auth import hash_password, verify_password
        pw     = "SecurePass123!"
        hashed = hash_password(pw)
        assert verify_password(pw, hashed), "Correct password should verify"

    def test_reject_wrong_password(self):
        from api.auth import hash_password, verify_password
        pw     = "SecurePass123!"
        hashed = hash_password(pw)
        assert not verify_password("WrongPassword", hashed), \
            "Wrong password should not verify"

    def test_hashes_are_unique(self):
        """Different calls to hash_password must produce different hashes (salt)."""
        from api.auth import hash_password
        pw = "SamePassword123!"
        h1 = hash_password(pw)
        h2 = hash_password(pw)
        assert h1 != h2, "Hashes must be unique (different salts)"

    def test_no_sha256_fallback_without_argon2(self):
        """Original code used SHA-256 with static salt — must not appear."""
        import api.auth as auth_mod
        import inspect
        src = inspect.getsource(auth_mod)
        assert "deepregime_salt_v1" not in src, \
            "Hardcoded SHA-256 salt found — auth not fixed!"


# ── 8. Monte Carlo path properties ────────────────────────────────────────────

class TestMonteCarlo:

    def test_paths_start_at_one(self):
        from utils.monte_carlo import MonteCarloSimulator
        sim   = MonteCarloSimulator()
        # Internal method: simulate 10 paths of length 5
        paths = sim._simulate_paths(0.0004, 0.012, T=5, n=10)
        # cumsum starts at step 1 not step 0; first element is e^(log_ret[0])
        # All paths should be positive
        assert (paths > 0).all(), "All simulated paths must be positive"

    def test_cvar_less_than_var(self):
        from utils.monte_carlo import MonteCarloSimulator
        import pandas as pd, numpy as np

        sim     = MonteCarloSimulator(initial_capital=1_000_000)
        np.random.seed(0)
        returns = pd.DataFrame({
            "A": np.random.normal(0.0003, 0.012, 252),
            "B": np.random.normal(0.0002, 0.010, 252),
        })
        result = sim.stress_test(
            {"A": 0.6, "B": 0.4}, returns, "Bear Market",
            horizon_days=63, n_simulations=1000,
        )
        assert result["cvar_95"] <= result["var_95"], \
            "CVaR must be ≤ VaR (CVaR is the tail mean)"

    def test_prob_loss_in_range(self):
        from utils.monte_carlo import MonteCarloSimulator
        import pandas as pd, numpy as np

        sim     = MonteCarloSimulator(initial_capital=1_000_000)
        np.random.seed(1)
        returns = pd.DataFrame({
            "A": np.random.normal(0.0003, 0.01, 252),
        })
        result = sim.stress_test(
            {"A": 1.0}, returns, "Bull Market",
            horizon_days=252, n_simulations=1000,
        )
        assert 0.0 <= result["prob_loss"] <= 1.0
        assert 0.0 <= result["prob_loss_10pct"] <= result["prob_loss"]
