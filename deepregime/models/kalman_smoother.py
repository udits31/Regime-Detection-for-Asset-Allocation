# ============================================================
# models/kalman_smoother.py
# Kalman Filter Regime Probability Smoother
#
# NEW MODULE — post-processes the HMM probability vector
# to reduce spurious one-day regime transitions (noise).
#
# Rationale: HMM Viterbi can flip between adjacent regimes
# on individual noisy days. A linear Kalman filter treating
# the probability vector as the "state" damps these flips
# while preserving genuine structural transitions.
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from loguru import logger
import joblib
from pathlib import Path


class RegimeProbabilityKalmanSmoother:
    """
    Linear Kalman filter applied to a T × n_regimes probability matrix.

    State:  x_t  = regime probability vector  ∈ R^n
    Obs:    z_t  = raw HMM posterior probs    ∈ R^n

    Dynamics:   x_t = F · x_{t-1} + w_t       w ~ N(0, Q)
    Observation: z_t = H · x_t   + v_t         v ~ N(0, R)

    F is the HMM transition matrix (already known from the HMM fit).
    H = I (we observe the full state directly).

    Parameters
    ──────────
    process_noise  : Q diagonal scalar — controls how quickly the filter
                     allows the regime distribution to change.
                     High → responsive; Low → smoother.
    obs_noise      : R diagonal scalar — trust in raw HMM probabilities.
                     High → less trust in observations.

    The filter is re-normalised at each step so the output remains a
    valid probability vector.
    """

    SAVE_PATH = Path("models/saved/kalman_smoother.pkl")

    def __init__(
        self,
        process_noise: float = 1e-3,
        obs_noise:     float = 1e-1,
        min_prob:      float = 1e-6,
    ):
        self.process_noise = process_noise
        self.obs_noise     = obs_noise
        self.min_prob      = min_prob
        self.F: Optional[np.ndarray] = None   # transition matrix from HMM
        self.n: int = 0
        self.is_fitted = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def fit(self, transition_matrix: np.ndarray) -> "RegimeProbabilityKalmanSmoother":
        """
        Initialise the filter from the HMM's transition matrix.

        Parameters
        ──────────
        transition_matrix : (n, n) ndarray, rows sum to 1.
        """
        self.F = transition_matrix.copy()
        self.n = transition_matrix.shape[0]
        self.is_fitted = True
        logger.info(
            f"Kalman smoother fitted | n_states={self.n} "
            f"Q={self.process_noise} R={self.obs_noise}"
        )
        return self

    # ── Smoothing ─────────────────────────────────────────────────────────────

    def smooth(self, prob_matrix: np.ndarray) -> np.ndarray:
        """
        Forward Kalman filter pass over the probability sequence.

        Parameters
        ──────────
        prob_matrix : (T, n) raw HMM posterior probabilities

        Returns
        ───────
        (T, n) smoothed probability matrix (rows sum to 1)
        """
        self._check_fitted()
        T, n = prob_matrix.shape
        assert n == self.n, f"Expected {self.n} regimes, got {n}"

        Q = self.process_noise * np.eye(n)
        R = self.obs_noise     * np.eye(n)
        H = np.eye(n)

        # Initial state = first observation
        x = prob_matrix[0].copy()
        P = np.eye(n) * 0.1

        smoothed = np.zeros((T, n))
        smoothed[0] = self._normalise(x)

        for t in range(1, T):
            # ── Predict ───────────────────────────────────────────────────────
            x_pred = self.F.T @ x          # Chapman-Kolmogorov propagation
            P_pred = self.F.T @ P @ self.F + Q

            # ── Update ────────────────────────────────────────────────────────
            z   = prob_matrix[t]
            S   = H @ P_pred @ H.T + R
            K   = P_pred @ H.T @ np.linalg.inv(S)   # Kalman gain
            x   = x_pred + K @ (z - H @ x_pred)
            P   = (np.eye(n) - K @ H) @ P_pred

            # Re-normalise to valid probability simplex
            x = np.clip(x, self.min_prob, None)
            x = self._normalise(x)
            smoothed[t] = x

        return smoothed

    def smooth_dataframe(
        self,
        prob_df: pd.DataFrame,
        label_map: Dict[int, str],
    ) -> pd.DataFrame:
        """
        Convenience wrapper: DataFrame in, DataFrame out.

        Parameters
        ──────────
        prob_df   : (T, n) DataFrame of raw HMM posteriors
        label_map : {state_id → label_string} from HMMRegimeDetector

        Returns
        ───────
        Smoothed probability DataFrame with same columns and index.
        """
        smoothed_arr = self.smooth(prob_df.values)
        return pd.DataFrame(
            smoothed_arr,
            index   = prob_df.index,
            columns = prob_df.columns,
        )

    # ── Regime extraction from smoothed probs ─────────────────────────────────

    def extract_regimes(
        self,
        smoothed: np.ndarray,
        label_map: Dict[int, str],
        min_confidence: float = 0.35,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert smoothed probability matrix to regime labels and confidences.
        Observations below min_confidence are labelled "Uncertain".

        Returns
        ───────
        labels      : (T,) array of regime label strings
        confidences : (T,) float array
        """
        max_probs  = smoothed.max(axis=1)
        max_states = smoothed.argmax(axis=1)
        labels = np.array([
            label_map.get(s, f"State_{s}") if c >= min_confidence else "Uncertain"
            for s, c in zip(max_states, max_probs)
        ])
        return labels, max_probs

    # ── Transition analysis ───────────────────────────────────────────────────

    def count_transitions(
        self, labels: np.ndarray
    ) -> Dict[str, int]:
        """Count regime transition events in the smoothed sequence."""
        changes = [
            f"{labels[i-1]} → {labels[i]}"
            for i in range(1, len(labels))
            if labels[i] != labels[i - 1]
        ]
        from collections import Counter
        return dict(Counter(changes).most_common())

    def mean_holding_period(self, labels: np.ndarray) -> Dict[str, float]:
        """Average consecutive days in each regime."""
        from itertools import groupby
        holding: Dict[str, list] = {}
        for regime, group in groupby(labels):
            dur = sum(1 for _ in group)
            holding.setdefault(regime, []).append(dur)
        return {r: float(np.mean(durations)) for r, durations in holding.items()}

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def noise_reduction_ratio(
        self, raw: np.ndarray, smoothed: np.ndarray
    ) -> float:
        """
        Fraction of raw day-to-day probability variance removed by smoothing.
        Higher → smoother output; 0 → no change; 1 → fully flat.
        """
        raw_var      = np.var(np.diff(raw, axis=0))
        smoothed_var = np.var(np.diff(smoothed, axis=0))
        if raw_var < 1e-12:
            return 0.0
        return float(1.0 - smoothed_var / raw_var)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        self.SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, self.SAVE_PATH)
        logger.info(f"Kalman smoother saved → {self.SAVE_PATH}")

    @classmethod
    def load(cls) -> "RegimeProbabilityKalmanSmoother":
        obj = joblib.load(cls.SAVE_PATH)
        logger.info("Kalman smoother loaded")
        return obj

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        s = x.sum()
        return x / s if s > 1e-12 else np.ones(self.n) / self.n

    def _check_fitted(self) -> None:
        if not self.is_fitted or self.F is None:
            raise RuntimeError(
                "Kalman smoother not fitted. Call .fit(transition_matrix) first."
            )
