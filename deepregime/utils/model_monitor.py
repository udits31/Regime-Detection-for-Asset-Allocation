# ============================================================
# utils/model_monitor.py
# Production Model Monitoring & Registry
#
# Implements:
#   1. ModelDriftDetector  — rolling log-likelihood monitoring
#   2. ModelRegistry       — versioned model storage with metadata
#   3. Regime drift alerts — P(current_state) vs historical baseline
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json
from loguru import logger


# ── Model Drift Detector ─────────────────────────────────────────────────────

class ModelDriftDetector:
    """
    Monitors a deployed HMM for statistical drift using rolling log-likelihood.

    Drift = new data has significantly lower likelihood under the trained model
    than the training data distribution → model has become stale.

    Algorithm
    ─────────
    1. At training time, record mean/std of rolling-window log-likelihoods.
    2. At deployment, compute rolling log-lik on new data.
    3. Alert if rolling log-lik < mean − k*std  (k=2 by default).
    """

    def __init__(self, window: int = 60, k: float = 2.0):
        self.window        = window
        self.k             = k
        self.train_mean_ll: Optional[float] = None
        self.train_std_ll:  Optional[float] = None
        self.history:       List[Dict]      = []

    def fit_baseline(
        self,
        model,          # fitted HMMRegimeDetector
        features_df: pd.DataFrame,
    ) -> "ModelDriftDetector":
        """Establish baseline log-likelihood distribution from training data."""
        scaler = model.scaler
        X      = scaler.transform(features_df.values)
        lls    = []
        for i in range(self.window, len(X)):
            chunk_ll = model.model.score(X[i - self.window: i])
            lls.append(chunk_ll)
        if lls:
            self.train_mean_ll = float(np.mean(lls))
            self.train_std_ll  = float(np.std(lls))
            logger.info(
                f"Drift baseline: mean_ll={self.train_mean_ll:.3f}  "
                f"std_ll={self.train_std_ll:.3f}  "
                f"alert_threshold={self.train_mean_ll - self.k * self.train_std_ll:.3f}"
            )
        return self

    def check(
        self,
        model,
        new_features_df: pd.DataFrame,
        timestamp:       Optional[str] = None,
    ) -> Dict:
        """Check whether new data shows significant drift from the trained model."""
        if self.train_mean_ll is None:
            raise RuntimeError("Call fit_baseline() first.")

        scaler  = model.scaler
        X_in    = new_features_df.values if hasattr(new_features_df, 'values') else new_features_df
        X       = scaler.transform(X_in)
        current_ll = float(model.model.score(X))

        threshold  = self.train_mean_ll - self.k * self.train_std_ll
        z_score    = (current_ll - self.train_mean_ll) / (self.train_std_ll + 1e-10)
        is_drifted = current_ll < threshold

        result = {
            "timestamp":     timestamp or datetime.now(timezone.utc).isoformat(),
            "current_ll":    round(current_ll,        4),
            "train_mean_ll": round(self.train_mean_ll, 4),
            "threshold":     round(threshold,           4),
            "z_score":       round(z_score,            4),
            "is_drifted":    is_drifted,
            "alert": (
                f"⚠ DRIFT DETECTED: LL={current_ll:.3f} < threshold={threshold:.3f} "
                f"(z={z_score:.2f}). Consider retraining."
                if is_drifted else
                f"✓ No drift. LL={current_ll:.3f} (z={z_score:.2f})."
            ),
        }
        self.history.append(result)
        if is_drifted:
            logger.warning(result["alert"])
        return result

    def drift_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


# ── Model Registry ────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Versioned model storage — each training run is saved with metadata.
    Supports listing, loading by version, and comparing training runs.

    Storage: JSON manifest + model files per version.
    """

    REGISTRY_PATH = Path("models/saved/registry.json")

    def __init__(self):
        self.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> List[Dict]:
        if self.REGISTRY_PATH.exists():
            with open(self.REGISTRY_PATH) as f:
                return json.load(f)
        return []

    def _save_manifest(self) -> None:
        with open(self.REGISTRY_PATH, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def register(
        self,
        model_name:    str,
        version:       str,
        metrics:       Dict[str, Any],
        train_start:   str,
        train_end:     str,
        n_features:    int,
        n_states:      int,
        notes:         str = "",
    ) -> str:
        """Register a new model version. Returns the version ID."""
        entry = {
            "model_name":  model_name,
            "version":     version,
            "registered":  datetime.now(timezone.utc).isoformat(),
            "train_start": train_start,
            "train_end":   train_end,
            "n_features":  n_features,
            "n_states":    n_states,
            "metrics":     metrics,
            "notes":       notes,
        }
        self._manifest.append(entry)
        self._save_manifest()
        logger.info(f"Model registered: {model_name} v{version}")
        return version

    def list_models(self) -> pd.DataFrame:
        if not self._manifest:
            return pd.DataFrame()
        rows = []
        for e in self._manifest:
            row = {
                "model":     e["model_name"],
                "version":   e["version"],
                "registered":e["registered"][:10],
                "train_end": e["train_end"],
                "n_states":  e["n_states"],
            }
            row.update({f"metric_{k}": v for k, v in e.get("metrics", {}).items()})
            rows.append(row)
        return pd.DataFrame(rows).set_index("version")

    def get_best_version(self, model_name: str, metric: str = "bic") -> Optional[Dict]:
        """Return the registered version with the best (lowest) metric value."""
        candidates = [
            e for e in self._manifest
            if e["model_name"] == model_name and metric in e.get("metrics", {})
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda e: e["metrics"][metric])

    def compare(self) -> pd.DataFrame:
        """Compare all registered models on their stored metrics."""
        return self.list_models()


# ── Regime Transition Probability Signal ─────────────────────────────────────

class RegimeTransitionSignal:
    """
    Pre-emptive position sizing based on HMM forward transition probabilities.

    Instead of only using the *current* regime label, this uses:
        P(next_state ∈ {crisis_states} | current_state) = A[current_state, crisis_states].sum()

    If this forward crisis probability exceeds a threshold, equity weight
    is reduced proportionally BEFORE the regime officially changes.
    This is the key predictive use of the transition matrix that most
    implementations miss.
    """

    def __init__(
        self,
        crisis_threshold: float = 0.20,
        min_equity:       float = 0.10,
        max_equity:       float = 0.80,
    ):
        self.crisis_threshold = crisis_threshold
        self.min_equity       = min_equity
        self.max_equity       = max_equity

    def compute_signal(
        self,
        hmm_model,                  # fitted HMMRegimeDetector
        current_state_id:  int,
    ) -> Dict:
        """
        Compute the forward crisis probability and recommended equity weight.

        Returns
        ───────
        crisis_prob, recommended_equity_weight, signal_strength
        """
        A = hmm_model.model.transmat_
        label_map = hmm_model.state_label_map

        # Identify crisis and bear states
        crisis_states = [
            sid for sid, lbl in label_map.items()
            if any(kw in lbl for kw in ["Crisis", "Bear", "Panic"])
        ]
        if not crisis_states:
            crisis_states = []

        crisis_prob = float(A[current_state_id, crisis_states].sum()) if crisis_states else 0.0

        # Linear scaling: equity_weight = max_equity - (max-min) * (crisis_prob / threshold)
        if crisis_prob >= self.crisis_threshold:
            signal = min(1.0, crisis_prob / (self.crisis_threshold + 1e-10))
            eq_weight = max(
                self.min_equity,
                self.max_equity - (self.max_equity - self.min_equity) * signal
            )
        else:
            signal    = 0.0
            eq_weight = self.max_equity

        next_state_dist = {
            label_map.get(i, f"S{i}"): round(float(A[current_state_id, i]), 4)
            for i in range(A.shape[0])
        }

        return {
            "current_state":         label_map.get(current_state_id, f"S{current_state_id}"),
            "crisis_prob":           round(crisis_prob, 4),
            "signal_strength":       round(signal, 4),
            "recommended_eq_weight": round(eq_weight, 4),
            "next_state_dist":       next_state_dist,
            "pre_emptive_action": (
                f"Reduce equity to {eq_weight:.0%} — forward crisis probability={crisis_prob:.1%}"
                if signal > 0 else "Hold current equity allocation"
            ),
        }
