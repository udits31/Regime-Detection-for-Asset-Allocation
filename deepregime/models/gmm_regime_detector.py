# ============================================================
# models/gmm_regime_detector.py
# Gaussian Mixture Model — Regime Detector
#
# Reviewer requirement: "Compare HMM, GMM, LSTM Autoencoder,
# Transformer Encoder" (Phase I)
#
# GMM treats each observation independently (no temporal
# dependencies unlike HMM). It's a useful baseline because:
#   • Simpler (fewer parameters)
#   • No Markov assumption — regimes don't "stick"
#   • BIC/AIC for model selection is straightforward
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path
from loguru import logger

_REGIME_NAMES = ["Bull Market", "Bear Market", "Recovery", "Crisis / Panic",
                 "Sideways / Range-bound", "High Volatility"]


class GMMRegimeDetector:
    """
    Gaussian Mixture Model for market regime detection.

    Unlike HMM, GMM treats observations as i.i.d. samples from a
    mixture distribution — no temporal transitions. This makes it
    a simpler baseline, but it ignores the sequential structure
    of financial time series.

    Used in Phase I of the reviewer's recommended framework as
    one of four competing regime detection approaches.
    """

    SAVE_PATH = Path("models/saved/gmm_model.pkl")

    def __init__(self, n_components: int = 4, covariance_type: str = "full"):
        self.n_components     = n_components
        self.covariance_type  = covariance_type
        self.model: Optional[GaussianMixture] = None
        self.scaler           = StandardScaler()
        self.state_label_map: Dict[int, str] = {}
        self.is_fitted        = False

    def fit(self, features_df: pd.DataFrame) -> "GMMRegimeDetector":
        logger.info(f"Training GMM | n_components={self.n_components}")
        X = self.scaler.fit_transform(features_df.values)
        self.model = GaussianMixture(
            n_components    = self.n_components,
            covariance_type = self.covariance_type,
            n_init          = 10,
            random_state    = 42,
            max_iter        = 300,
        )
        self.model.fit(X)
        states = self.model.predict(X)
        self.state_label_map = self._label_states(features_df, states)
        self.is_fitted = True
        ll = self.model.score(X)
        logger.info(f"GMM trained | log-likelihood={ll:.4f} BIC={self.bic(X):.2f}")
        return self

    def _label_states(self, features_df: pd.DataFrame, states: np.ndarray) -> Dict[int, str]:
        df = features_df.copy()
        df["_state"] = states
        ret_col = "returns" if "returns" in df.columns else df.columns[0]
        vol_col = "vol_20d" if "vol_20d" in df.columns else None
        dd_col  = "drawdown" if "drawdown" in df.columns else None

        agg: Dict = {ret_col: "mean"}
        if vol_col: agg[vol_col] = "mean"
        if dd_col:  agg[dd_col] = "mean"
        stats = df.groupby("_state").agg(agg)
        stats["ann_return"] = stats[ret_col] * 252
        stats["vol"]   = stats.get(vol_col, pd.Series(0, index=stats.index))
        stats["dd"]    = stats.get(dd_col, pd.Series(0, index=stats.index))
        stats["score"] = (
            stats["ann_return"].rank() - stats["vol"].rank() - (-stats["dd"]).rank()
        )
        sorted_states = stats["score"].sort_values().index.tolist()
        label_counts: Dict[str, int] = {}
        label_map: Dict[int, str] = {}
        names = _REGIME_NAMES
        for rank, sid in enumerate(sorted_states):
            idx   = min(int(rank * len(names) / len(sorted_states)), len(names) - 1)
            base  = names[idx]
            cnt   = label_counts.get(base, 0)
            sfx   = ["", " II", " III", " IV"][cnt] if cnt < 4 else f" {cnt+1}"
            label_map[sid] = base + sfx
            label_counts[base] = cnt + 1
        return label_map

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        self._check_fitted()
        X      = self.scaler.transform(features_df.values)
        states = self.model.predict(X)
        return pd.Series(
            [self.state_label_map.get(s, f"State_{s}") for s in states],
            index=features_df.index, name="regime"
        )

    def predict_proba(self, features_df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        X     = self.scaler.transform(features_df.values)
        probs = self.model.predict_proba(X)
        cols  = [self.state_label_map.get(i, f"State_{i}") for i in range(self.n_components)]
        return pd.DataFrame(probs, index=features_df.index, columns=cols)

    def bic(self, X_scaled: np.ndarray) -> float:
        return float(self.model.bic(X_scaled))

    def aic(self, X_scaled: np.ndarray) -> float:
        return float(self.model.aic(X_scaled))

    def evaluate(self, features_df: pd.DataFrame, returns_series: pd.Series) -> Dict:
        self._check_fitted()
        X = self.scaler.transform(features_df.values)
        labels = self.predict(features_df)
        merged = pd.DataFrame({"regime": labels, "returns": returns_series.reindex(labels.index)})
        regime_stats = merged.groupby("regime")["returns"].agg(
            count="count",
            mean_ret="mean",
            std_ret="std",
            sharpe=lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0,
        )
        return {
            "model":          "GMM",
            "n_components":   self.n_components,
            "log_likelihood": float(self.model.score(X)),
            "bic":            self.bic(X),
            "aic":            self.aic(X),
            "regime_stats":   regime_stats.to_dict(),
        }

    def save(self) -> None:
        self.SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, self.SAVE_PATH)

    @classmethod
    def load(cls) -> "GMMRegimeDetector":
        return joblib.load(cls.SAVE_PATH)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Not fitted. Call .fit() first.")
