# ============================================================
# regime_detector.py
# Ensemble Regime Detector: HMM (current) + LSTM (forecast) + Kalman
#
# FIXED (vs original):
#   • HMM and LSTM are genuinely complementary, not redundant:
#       HMM  → P(regime NOW  | features_1:t)   [structural]
#       LSTM → P(regime t+1  | features_t-L:t) [predictive]
#   • Kalman filter smooths HMM probability stream
#   • Ensemble weights auto-adjusted by model confidence
#   • Risk score uses calibrated multi-factor formula
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger
from datetime import datetime, timezone

from models.hmm_regime_detector import HMMRegimeDetector
from models.lstm_regime_predictor import LSTMRegimePredictor
from models.kalman_smoother import RegimeProbabilityKalmanSmoother
from feature_engineering.feature_builder import FeatureBuilder
from config.settings import REGIME_COLORS


class EnsembleRegimeDetector:
    """
    Three-stage regime detection pipeline.

    Stage 1 — Structural (HMM)
    ──────────────────────────
    Gaussian HMM learns the latent market structure from price features.
    Outputs: current-state posterior probabilities.

    Stage 2 — Temporal Smoothing (Kalman)
    ──────────────────────────────────────
    Kalman filter applied to the HMM probability stream damps spurious
    one-day regime flips while preserving genuine structural transitions.

    Stage 3 — Predictive (LSTM)
    ────────────────────────────
    BiLSTM+TCN+Attention predicts the NEXT DAY's regime given the past
    `lookback` days of features — a true one-step-ahead forecast.
    Temperature-calibrated probabilities are used.

    Ensemble
    ────────
    The two models answer different questions:
      • HMM:  what regime are we in right now?
      • LSTM: what regime will we be in tomorrow?

    They are combined via a confidence-weighted average with a fallback
    to HMM-only when LSTM cannot run (insufficient history).
    """

    def __init__(
        self,
        hmm_weight:     float = 0.45,
        lstm_weight:    float = 0.55,
        kalman_q:       float = 1e-3,
        kalman_r:       float = 1e-1,
        forecast_horizon: int = 1,
    ):
        self.hmm      = HMMRegimeDetector()
        self.lstm     = LSTMRegimePredictor()
        self.kalman   = RegimeProbabilityKalmanSmoother(kalman_q, kalman_r)
        self.fb       = FeatureBuilder()
        self.base_hmm_weight    = hmm_weight
        self.base_lstm_weight   = lstm_weight
        self.forecast_horizon   = forecast_horizon
        self.feature_df: Optional[pd.DataFrame] = None
        self.regime_history: Optional[pd.DataFrame] = None

    # ── Full Training Pipeline ────────────────────────────────────────────────

    def fit(
        self,
        prices_df: pd.DataFrame,
        macro_df:  Optional[pd.DataFrame] = None,
    ) -> "EnsembleRegimeDetector":
        """
        End-to-end training on historical price data.

        Steps
        ─────
        1.  Feature engineering (full historical set — used for training only,
            NOT for the backtest which recomputes features per fold).
        2.  HMM training (unsupervised).
        3.  Kalman smoother initialised from HMM transition matrix.
        4.  LSTM training with target = regime[t+1] (true forecast target).
        5.  Store smoothed historical regime timeline.
        """
        logger.info("Starting full Ensemble Regime Detector training …")

        # ── Features ──────────────────────────────────────────────────────────
        self.feature_df   = self.fb.build_all_features(prices_df, macro_df)
        regime_features   = self.fb.get_regime_features(self.feature_df)

        # ── HMM ───────────────────────────────────────────────────────────────
        logger.info("Phase 1: Training HMM …")
        self.hmm.fit(regime_features)

        # ── Kalman ────────────────────────────────────────────────────────────
        logger.info("Phase 2: Initialising Kalman smoother …")
        self.kalman.fit(self.hmm.model.transmat_)

        # Smooth historical HMM probabilities
        raw_probs     = self.hmm.predict_proba(regime_features)
        smoothed_probs = self.kalman.smooth_dataframe(raw_probs, self.hmm.state_label_map)
        hmm_labels    = pd.Series(
            smoothed_probs.idxmax(axis=1).values,
            index=regime_features.index,
            name="regime",
        )

        nr = self.kalman.noise_reduction_ratio(raw_probs.values, smoothed_probs.values)
        logger.info(f"Kalman noise reduction ratio: {nr:.3f}")
        logger.info(f"Smoothed regime distribution:\n{hmm_labels.value_counts()}")

        # ── LSTM ──────────────────────────────────────────────────────────────
        # Uses smoothed labels (not raw Viterbi) as targets → less label noise.
        # Target is label[t+1] (one-step forecast), not label[t].
        logger.info("Phase 3: Training LSTM (next-day forecaster) …")
        self.lstm.fit(regime_features, hmm_labels,
                      forecast_horizon=self.forecast_horizon)

        # ── Historical regime timeline ─────────────────────────────────────────
        self.regime_history = self._build_regime_history(regime_features)

        logger.info("Ensemble Regime Detector fully trained.")
        return self

    # ── Real-Time Detection ────────────────────────────────────────────────────

    def detect_current_regime(self, recent_prices_df: pd.DataFrame) -> Dict:
        """
        Full regime assessment for the most recent market state.

        Returns
        ───────
        Comprehensive dict: current regime, next-day forecast, confidence,
        risk score, volatility classification, key market signals.
        """
        features        = self.fb.build_all_features(recent_prices_df)
        regime_features = self.fb.get_regime_features(features)

        # ── HMM current state ──────────────────────────────────────────────────
        raw_hmm    = self.hmm.predict_latest(regime_features)

        # Apply Kalman smoothing to the last window of HMM probabilities
        raw_prob_series = self.hmm.predict_proba(regime_features)
        smoothed        = self.kalman.smooth(raw_prob_series.values)
        smoothed_now    = smoothed[-1]
        smoothed_state  = int(smoothed_now.argmax())
        smoothed_regime = self.hmm.state_label_map.get(smoothed_state, "Unknown")
        smoothed_conf   = float(smoothed_now[smoothed_state])

        # ── LSTM next-day forecast ─────────────────────────────────────────────
        lstm_available = len(regime_features) >= self.lstm.lookback + 5
        if lstm_available:
            try:
                lstm_result = self.lstm.predict_next_regime(regime_features)
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
                lstm_available = False

        if not lstm_available:
            return self._hmm_only_response(
                smoothed_regime, smoothed_conf, raw_hmm, features
            )

        # ── Confidence-weighted ensemble ───────────────────────────────────────
        ensemble = self._ensemble(
            smoothed_now,
            lstm_result["all_probabilities"],
            smoothed_conf,
            lstm_result["confidence"],
        )

        return self._build_response(ensemble, raw_hmm, lstm_result, features,
                                    smoothed_regime, smoothed_conf)

    def _ensemble(
        self,
        hmm_probs:    np.ndarray,
        lstm_probs:   Dict[str, float],
        hmm_conf:     float,
        lstm_conf:    float,
    ) -> Dict:
        """
        Confidence-adjusted weighted average of HMM and LSTM distributions.
        When one model is highly confident, its weight is boosted.
        """
        # Adaptive weights: more confidence → more weight
        total_conf = hmm_conf + lstm_conf + 1e-9
        w_hmm  = self.base_hmm_weight  * (hmm_conf  / total_conf) * 2
        w_lstm = self.base_lstm_weight * (lstm_conf / total_conf) * 2
        norm   = w_hmm + w_lstm
        w_hmm /= norm; w_lstm /= norm

        all_regimes = set(self.hmm.state_label_map.values()) | set(lstm_probs.keys())
        ensemble_p: Dict[str, float] = {}

        for regime in all_regimes:
            # HMM probability for this regime (find by label match)
            hmm_p = 0.0
            for state_id, lbl in self.hmm.state_label_map.items():
                if lbl == regime:
                    hmm_p += float(hmm_probs[state_id])
            lstm_p = lstm_probs.get(regime, 0.0)
            ensemble_p[regime] = w_hmm * hmm_p + w_lstm * lstm_p

        # Normalise
        total = sum(ensemble_p.values())
        if total > 1e-9:
            ensemble_p = {k: v / total for k, v in ensemble_p.items()}

        final = max(ensemble_p, key=ensemble_p.get)
        return {
            "regime":        final,
            "confidence":    ensemble_p[final],
            "probs":         ensemble_p,
            "w_hmm":         round(w_hmm,  3),
            "w_lstm":        round(w_lstm, 3),
            "models_agree":  (
                self.hmm.state_label_map.get(
                    int(np.array(list(self.hmm.state_label_map.values()))
                        .tolist().index(
                            max(self.hmm.state_label_map.values(),
                                key=lambda r: float(hmm_probs[
                                    [k for k,v in self.hmm.state_label_map.items() if v==r][0]
                                ])
                            )
                        )) if len(self.hmm.state_label_map) else 0,
                    "Unknown"
                ) == max(lstm_probs, key=lstm_probs.get)
            ),
        }

    def _build_response(
        self, ensemble: Dict, hmm: Dict, lstm: Dict,
        features: pd.DataFrame, smoothed_regime: str, smoothed_conf: float,
    ) -> Dict:
        latest = features.iloc[-1]
        return {
            # ── Core
            "regime":             ensemble["regime"],
            "confidence":         ensemble["confidence"],
            "color":              REGIME_COLORS.get(ensemble["regime"], "#888888"),
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            # ── HMM (current state)
            "hmm_regime":         smoothed_regime,
            "hmm_confidence":     smoothed_conf,
            "hmm_raw_regime":     hmm["regime"],
            # ── LSTM (next-day forecast)
            "lstm_next_regime":   lstm["predicted_regime"],
            "lstm_confidence":    lstm["confidence"],
            "lstm_calibrated":    lstm.get("is_calibrated", False),
            # ── Ensemble metadata
            "models_agree":       ensemble["models_agree"],
            "ensemble_weights":   {"hmm": ensemble["w_hmm"], "lstm": ensemble["w_lstm"]},
            "regime_probabilities": ensemble["probs"],
            "transition_probs":   hmm.get("transition_probs", {}),
            # ── Risk
            "risk_score":         self._risk_score(latest),
            "volatility_level":   self._vol_class(latest),
            # ── Market signals snapshot
            "key_signals": {
                "rsi_14":         float(latest.get("rsi_14",    np.nan)),
                "macd_hist":      float(latest.get("macd_hist", np.nan)),
                "vol_20d":        float(latest.get("vol_20d",   np.nan)),
                "drawdown":       float(latest.get("drawdown",  np.nan)),
                "golden_cross":   bool(latest.get("golden_cross", False)),
                "bb_width":       float(latest.get("bb_width",  np.nan)),
                "crisis_score":   float(latest.get("crisis_score", np.nan)),
            },
        }

    def _hmm_only_response(
        self, regime: str, conf: float, raw_hmm: Dict, features: pd.DataFrame
    ) -> Dict:
        latest = features.iloc[-1]
        return {
            "regime":             regime,
            "confidence":         conf,
            "color":              REGIME_COLORS.get(regime, "#888888"),
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "hmm_regime":         regime,
            "hmm_confidence":     conf,
            "hmm_raw_regime":     raw_hmm["regime"],
            "lstm_next_regime":   None,
            "lstm_confidence":    None,
            "lstm_calibrated":    False,
            "models_agree":       None,
            "ensemble_weights":   {"hmm": 1.0, "lstm": 0.0},
            "regime_probabilities": raw_hmm.get("state_probabilities", {}),
            "transition_probs":   raw_hmm.get("transition_probs", {}),
            "risk_score":         self._risk_score(latest),
            "volatility_level":   self._vol_class(latest),
            "key_signals":        {},
            "source":             "HMM only (LSTM insufficient history)",
        }

    def _build_regime_history(self, features: pd.DataFrame) -> pd.DataFrame:
        raw     = self.hmm.predict_proba(features)
        smooth  = self.kalman.smooth_dataframe(raw, self.hmm.state_label_map)
        labels, confs = self.kalman.extract_regimes(
            smooth.values, self.hmm.state_label_map
        )
        return pd.DataFrame({
            "regime":        labels,
            "confidence":    confs,
            "regime_change": pd.Series(labels).ne(pd.Series(labels).shift(1)).values,
        }, index=features.index)

    # ── Risk metrics ─────────────────────────────────────────────────────────

    def _risk_score(self, f: pd.Series) -> float:
        """Multi-factor risk score ∈ [0, 100]. Higher = riskier."""
        s = 50.0
        vol = f.get("vol_20d", 0.15)
        dd  = abs(f.get("drawdown", 0.0))
        rsi = f.get("rsi_14", 50.0)
        cs  = f.get("crisis_score", 0.0)

        s += np.clip((vol - 0.15) * 150, -15, 25)
        s += np.clip(dd * 80,            0,   20)
        s += np.clip(cs * 30,            0,   10)
        if rsi < 30: s += 8
        if rsi > 75: s -= 5
        return float(np.clip(s, 0, 100))

    def _vol_class(self, f: pd.Series) -> str:
        vol = f.get("vol_20d", 0.15)
        if vol < 0.10: return "Very Low"
        if vol < 0.15: return "Low"
        if vol < 0.22: return "Normal"
        if vol < 0.30: return "Elevated"
        if vol < 0.45: return "High"
        return "Extreme"

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        self.hmm.save()
        self.lstm.save()
        self.kalman.save()
        logger.info("All ensemble models saved.")

    def load(self) -> "EnsembleRegimeDetector":
        self.hmm    = HMMRegimeDetector.load()
        self.lstm   = LSTMRegimePredictor.load()
        self.kalman = RegimeProbabilityKalmanSmoother.load()
        logger.info("All ensemble models loaded.")
        return self
