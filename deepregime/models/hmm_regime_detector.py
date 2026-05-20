# ============================================================
# models/hmm_regime_detector.py
# Hidden Markov Model — Market Regime Detection
# FIXED: label collision disambiguation, dead PCA removed,
#        robust stationary distribution, proper save/load
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger

from config.settings import settings, REGIME_COLORS


# ── Regime taxonomy ───────────────────────────────────────────────────────────
#  Ordered by severity so disambiguation picks the "stronger" duplicate last.
_REGIME_PRIORITY = [
    "Crisis / Panic",
    "Bear Market",
    "High Volatility",
    "Sideways / Range-bound",
    "Recovery",
    "Bull Market",
]
_SUFFIX = ["", " II", " III", " IV"]   # up to 4 states can share a base label


class HMMRegimeDetector:
    """
    Gaussian Hidden Markov Model for unsupervised market regime detection.

    Architecture
    ────────────
    Input  → StandardScaler → GaussianHMM (n_components states, full covariance)
                                        ↓
                              Viterbi state sequence
                                        ↓
                         Post-hoc label assignment (return / vol / drawdown)
                         with collision disambiguation (Bull II, Bear II …)

    Key design decisions
    ────────────────────
    • Full covariance captures cross-feature correlations inside each regime.
    • State labelling is done by ranking states on (annualised return, realised
      vol, max drawdown) — NOT via hard-coded thresholds — so it generalises
      across market periods.
    • Duplicate labels get roman-numeral suffixes so predict_proba() always
      produces a DataFrame with unique column names.
    • PCA is NOT used here: dimensionality reduction before HMM is debatable
      and was dead code in the original; removed for clarity.
    """

    MODEL_SAVE_PATH  = Path("models/saved/hmm_model.pkl")
    SCALER_SAVE_PATH = Path("models/saved/hmm_scaler.pkl")
    LABEL_MAP_PATH   = Path("models/saved/hmm_label_map.pkl")

    def __init__(self, n_components: int | None = None):
        self.n_components    = n_components or settings.HMM_N_COMPONENTS
        self.model: hmm.GaussianHMM | None = None
        self.scaler          = StandardScaler()
        self.state_label_map: Dict[int, str] = {}
        self.is_fitted       = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, features_df: pd.DataFrame) -> "HMMRegimeDetector":
        """
        Train the HMM on historical feature data.

        Steps
        ─────
        1. Scale features (zero mean, unit variance).
        2. Fit Gaussian HMM via Baum-Welch (EM).
        3. Decode Viterbi state sequence.
        4. Label states by ranked regime characteristics (no hard thresholds).
        """
        logger.info(
            f"Training HMM | n_states={self.n_components} | "
            f"samples={len(features_df)} | features={features_df.shape[1]}"
        )

        X_scaled = self.scaler.fit_transform(features_df.values)

        self.model = hmm.GaussianHMM(
            n_components    = self.n_components,
            covariance_type = settings.HMM_COVARIANCE_TYPE,   # "full"
            n_iter          = settings.HMM_N_ITER,             # 300
            random_state    = 42,
            verbose         = False,
        )
        self.model.fit(X_scaled)

        states = self.model.predict(X_scaled)
        self.state_label_map = self._label_states(features_df, states)

        self.is_fitted = True
        ll = self.model.score(X_scaled)
        logger.info(f"HMM trained  | log-likelihood={ll:.2f}")
        logger.info(f"State map    | {self.state_label_map}")
        return self

    def _label_states(
        self, features_df: pd.DataFrame, states: np.ndarray
    ) -> Dict[int, str]:
        """
        Rank-based, threshold-free state labelling.

        For each hidden state compute:
          • annualised mean return
          • mean 20-day realised vol
          • mean drawdown depth

        States are sorted by (return DESC, vol ASC) and mapped to the six
        canonical regime names in _REGIME_PRIORITY order (worst → best).
        Duplicate base-labels get roman-numeral suffixes.
        """
        df = features_df.copy()
        df["_state"] = states

        ret_col = "returns"   if "returns"  in df.columns else df.columns[0]
        vol_col = "vol_20d"   if "vol_20d"  in df.columns else None
        dd_col  = "drawdown"  if "drawdown" in df.columns else None

        agg: Dict[str, object] = {ret_col: "mean"}
        if vol_col: agg[vol_col] = "mean"
        if dd_col:  agg[dd_col]  = "mean"

        stats = df.groupby("_state").agg(agg)
        stats["ann_return"] = stats[ret_col] * 252
        if vol_col:
            stats["mean_vol"] = stats[vol_col]
        else:
            stats["mean_vol"] = 0.0
        if dd_col:
            stats["mean_dd"] = stats[dd_col]
        else:
            stats["mean_dd"] = 0.0

        # Score: higher return → better; higher vol/deeper dd → worse
        stats["score"] = (
            stats["ann_return"].rank()
            - stats["mean_vol"].rank()
            - (-stats["mean_dd"]).rank()   # dd is negative, flip sign
        )

        # Sort states worst → best by score
        sorted_states = stats["score"].sort_values().index.tolist()
        n = len(sorted_states)

        # Map sorted position → canonical label (spread evenly across 6 labels)
        canonical = _REGIME_PRIORITY
        label_counts: Dict[str, int] = {}
        label_map: Dict[int, str] = {}

        for rank, state_id in enumerate(sorted_states):
            # Scale rank into [0, len(canonical)-1]
            idx        = min(int(rank * len(canonical) / n), len(canonical) - 1)
            base       = canonical[idx]
            count      = label_counts.get(base, 0)
            suffix     = _SUFFIX[count] if count < len(_SUFFIX) else f" {count+1}"
            label_map[state_id] = base + suffix
            label_counts[base]  = count + 1

        return label_map

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        """Return regime label series (Viterbi path)."""
        self._check_fitted()
        X       = self.scaler.transform(features_df.values)
        states  = self.model.predict(X)
        labels  = [self.state_label_map.get(s, f"State_{s}") for s in states]
        return pd.Series(labels, index=features_df.index, name="regime")

    def predict_proba(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return posterior state probabilities as a DataFrame.
        Columns are unique regime label strings (disambiguation guaranteed).
        """
        self._check_fitted()
        X     = self.scaler.transform(features_df.values)
        probs = self.model.predict_proba(X)
        cols  = [self.state_label_map.get(i, f"State_{i}")
                 for i in range(self.n_components)]
        return pd.DataFrame(probs, index=features_df.index, columns=cols)

    def predict_latest(self, features_df: pd.DataFrame) -> Dict:
        """
        Full regime assessment for the most recent observation.

        Returns
        ───────
        regime, confidence, state_probabilities (dict),
        transition_probs (next-state distribution), hmm_state_id
        """
        self._check_fitted()
        X          = self.scaler.transform(features_df.tail(1).values)
        state      = int(self.model.predict(X)[0])
        probs      = self.model.predict_proba(X)[0]
        regime     = self.state_label_map.get(state, f"State_{state}")
        confidence = float(probs[state])

        trans_probs = {
            self.state_label_map.get(i, f"State_{i}"): float(p)
            for i, p in enumerate(self.model.transmat_[state])
        }

        return {
            "regime":              regime,
            "confidence":          confidence,
            "state_probabilities": {
                self.state_label_map.get(i, f"State_{i}"): float(p)
                for i, p in enumerate(probs)
            },
            "transition_probs":    trans_probs,
            "hmm_state_id":        state,
        }

    def get_regime_sequence(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Full historical regime timeline with confidence and change flags."""
        labels = self.predict(features_df)
        probs  = self.predict_proba(features_df)
        return pd.DataFrame({
            "regime":        labels,
            "confidence":    probs.max(axis=1),
            "regime_change": labels != labels.shift(1),
        })

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self, features_df: pd.DataFrame, returns_series: pd.Series
    ) -> Dict:
        """Regime-stratified return statistics + Markov chain diagnostics."""
        self._check_fitted()
        seq = self.get_regime_sequence(features_df)
        seq["returns"] = returns_series.reindex(seq.index)

        regime_stats = (
            seq.groupby("regime")["returns"]
            .agg(
                count     = "count",
                mean_ret  = "mean",
                std_ret   = "std",
                sharpe    = lambda x: (
                    x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0
                ),
            )
        )

        X_scaled = self.scaler.transform(features_df.values)
        return {
            "log_likelihood":     float(self.model.score(X_scaled)),
            "n_states":           self.n_components,
            "aic":                self._aic(X_scaled),
            "bic":                self._bic(X_scaled),
            "regime_statistics":  regime_stats.to_dict(),
            "transition_matrix":  self.model.transmat_.tolist(),
            "stationary_dist":    self.stationary_distribution().tolist(),
            "mean_regime_duration": self._mean_durations(),
        }

    def _aic(self, X_scaled: np.ndarray) -> float:
        k = self._n_free_params()
        return -2 * self.model.score(X_scaled) * len(X_scaled) + 2 * k

    def _bic(self, X_scaled: np.ndarray) -> float:
        k = self._n_free_params()
        n = len(X_scaled)
        return -2 * self.model.score(X_scaled) * n + k * np.log(n)

    def _n_free_params(self) -> int:
        n, d = self.n_components, self.model.means_.shape[1]
        cov_params = n * d * (d + 1) // 2   # full covariance per state
        return n * d + cov_params + n * (n - 1)   # means + cov + transitions

    def stationary_distribution(self) -> np.ndarray:
        """Closed-form stationary distribution via dominant eigenvector."""
        A  = self.model.transmat_
        ev = np.linalg.eig(A.T)
        pi = np.real(ev[1][:, np.argmax(np.real(ev[0]))])
        pi = np.abs(pi)
        return pi / pi.sum()

    def _mean_durations(self) -> Dict[str, float]:
        """Expected duration in each regime = 1 / (1 - p_ii)."""
        return {
            self.state_label_map.get(i, f"State_{i}"): 1.0 / max(1 - p, 1e-6)
            for i, p in enumerate(np.diag(self.model.transmat_))
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        self.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model,           self.MODEL_SAVE_PATH)
        joblib.dump(self.scaler,          self.SCALER_SAVE_PATH)
        joblib.dump(self.state_label_map, self.LABEL_MAP_PATH)
        logger.info(f"HMM saved → {self.MODEL_SAVE_PATH.parent}")

    @classmethod
    def load(cls) -> "HMMRegimeDetector":
        inst               = cls()
        inst.model         = joblib.load(cls.MODEL_SAVE_PATH)
        inst.scaler        = joblib.load(cls.SCALER_SAVE_PATH)
        inst.state_label_map = joblib.load(cls.LABEL_MAP_PATH)
        inst.n_components  = inst.model.n_components
        inst.is_fitted     = True
        logger.info("HMM loaded from disk")
        return inst

    def _check_fitted(self) -> None:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() or .load() first.")
