# ============================================================
# models/hsmm_detector.py
# Hidden Semi-Markov Model — Regime Detector
#
# Reviewer requirement: "HSMM explicitly models state duration
# distributions and is frequently used in financial regime
# identification."
#
# Key difference vs HMM:
#   • Standard HMM: duration ~ Geometric(1 - p_ii)
#     (often too short — regimes flip day-to-day)
#   • HSMM: explicit Negative Binomial or Poisson duration
#     distribution per state — much more realistic
#
# Implementation:
#   1. Fit base GaussianHMM
#   2. Extract Viterbi state sequence to empirically fit
#      duration distributions (Negative Binomial per state)
#   3. Re-decode using duration-adjusted Viterbi (semi-Markov)
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, geom
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path
from loguru import logger


class HSMMRegimeDetector:
    """
    Hidden Semi-Markov Model with Negative Binomial duration distributions.

    Standard HMMs assign implicit Geometric duration to each state.
    Geometric distributions are memoryless and tend to produce
    unrealistically brief regimes (1–3 days). HSMMs break this
    by explicitly modelling how long the system stays in each state.

    Duration model: D_i ~ NegBin(r_i, p_i) per state i
    Parameters estimated empirically from the base HMM's Viterbi path.

    The semi-Markov Viterbi then incorporates a duration penalty:
      score(state, duration) = emission_loglik + duration_logprob
    """

    SAVE_PATH = Path("models/saved/hsmm_model.pkl")

    def __init__(
        self,
        n_components:  int   = 4,
        max_duration:  int   = 120,   # Maximum regime duration (trading days)
        covariance_type: str = "full",
    ):
        self.n_components   = n_components
        self.max_duration   = max_duration
        self.covariance_type = covariance_type
        self.base_hmm: Optional[hmm.GaussianHMM] = None
        self.scaler         = StandardScaler()
        self.state_label_map: Dict[int, str] = {}
        # Duration distribution parameters: {state: (r, p)} for NegBin
        self.duration_params: Dict[int, Tuple[float, float]] = {}
        self.is_fitted       = False

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, features_df: pd.DataFrame) -> "HSMMRegimeDetector":
        logger.info(f"Training HSMM | n_states={self.n_components}")
        X = self.scaler.fit_transform(features_df.values)

        # Step 1: Fit base HMM
        self.base_hmm = hmm.GaussianHMM(
            n_components    = self.n_components,
            covariance_type = self.covariance_type,
            n_iter          = 200, random_state=42,
        )
        self.base_hmm.fit(X)

        # Step 2: Viterbi decode to get empirical durations
        states  = self.base_hmm.predict(X)
        durations = self._extract_durations(states)

        # Step 3: Fit NegBin duration model per state
        for state_id in range(self.n_components):
            d = durations.get(state_id, [1])
            self.duration_params[state_id] = self._fit_negbin(np.array(d))

        # Step 4: Label states (reuse rank-based approach from HMM)
        self.state_label_map = self._label_states(features_df, states)
        self.is_fitted = True

        # Log mean durations comparison: Geometric (HMM) vs NegBin (HSMM)
        hmm_mean_dur = {
            i: 1.0 / max(1 - self.base_hmm.transmat_[i, i], 1e-6)
            for i in range(self.n_components)
        }
        hsmm_mean_dur = {
            i: nbinom.mean(*self.duration_params[i])
            for i in range(self.n_components)
        }
        for i in range(self.n_components):
            lbl = self.state_label_map.get(i, f"S{i}")
            logger.info(
                f"  {lbl:25s} | HMM dur={hmm_mean_dur[i]:.1f}d "
                f"| HSMM dur={hsmm_mean_dur[i]:.1f}d"
            )
        return self

    # ── Duration utilities ────────────────────────────────────────────────────

    def _extract_durations(self, states: np.ndarray) -> Dict[int, List[int]]:
        """Extract consecutive run lengths per state from Viterbi path."""
        durations: Dict[int, List[int]] = {i: [] for i in range(self.n_components)}
        if len(states) == 0:
            return durations
        curr_state = states[0]
        run_len    = 1
        for s in states[1:]:
            if s == curr_state:
                run_len += 1
            else:
                durations[curr_state].append(run_len)
                curr_state = s
                run_len    = 1
        durations[curr_state].append(run_len)
        return durations

    def _fit_negbin(self, durations: np.ndarray) -> Tuple[float, float]:
        """
        Fit Negative Binomial distribution to observed durations.
        NegBin(r, p): mean = r(1-p)/p, var = r(1-p)/p^2
        Method of moments estimator.
        """
        d   = durations[durations >= 1]
        if len(d) < 2:
            return (1.0, 0.5)
        mu  = float(d.mean())
        var = float(d.var()) if d.var() > 0 else mu
        # MoM: p = mu/var, r = mu^2/(var - mu)
        if var <= mu:
            # Under-dispersed → Poisson approximation (p→1 limit)
            return (max(mu, 1.0), 0.5)
        p = mu / var
        r = mu ** 2 / (var - mu)
        return (max(r, 0.1), min(max(p, 0.01), 0.99))

    def _duration_log_prob(self, state: int, duration: int) -> float:
        """Log P(duration | state) under NegBin model."""
        r, p = self.duration_params.get(state, (1.0, 0.5))
        return float(nbinom.logpmf(duration - 1, r, p))   # duration ≥ 1

    # ── Semi-Markov Decoding ──────────────────────────────────────────────────

    def _semimarkov_viterbi(self, X: np.ndarray) -> np.ndarray:
        """
        Duration-aware Viterbi decoding.

        DP: delta[t][i] = max log-prob of sequence ending in state i at time t
        Incorporates duration log-probability for runs of length d.
        """
        T     = len(X)
        K     = self.n_components
        D_max = min(self.max_duration, T)

        # Emission log-likelihoods: T × K
        log_emit = np.zeros((T, K))
        for k in range(K):
            mu     = self.base_hmm.means_[k]
            cov    = self.base_hmm.covars_[k]
            diff   = X - mu
            sign, logdet = np.linalg.slogdet(cov)
            inv_cov = np.linalg.inv(cov)
            for t in range(T):
                log_emit[t, k] = (
                    -0.5 * (diff[t] @ inv_cov @ diff[t]) - 0.5 * logdet
                )

        # Cumulative emission for runs: cum_emit[t][k][d] = Σ log_emit[t-d:t][k]
        def cum_emit_range(k: int, t_end: int, d: int) -> float:
            t_start = max(0, t_end - d)
            return float(log_emit[t_start:t_end, k].sum())

        log_trans   = np.log(self.base_hmm.transmat_ + 1e-12)
        log_startprob = np.log(self.base_hmm.startprob_ + 1e-12)

        # delta[t][k] = best log-prob ending at t in state k
        delta   = np.full((T, K), -np.inf)
        psi_dur = np.zeros((T, K), dtype=int)   # best duration at (t, k)
        psi_st  = np.zeros((T, K), dtype=int)   # best previous state at (t, k)

        # Initialise
        for k in range(K):
            for d in range(1, min(D_max + 1, T + 1)):
                v = (
                    log_startprob[k]
                    + cum_emit_range(k, d, d)
                    + self._duration_log_prob(k, d)
                )
                if v > delta[d - 1, k]:
                    delta[d - 1, k] = v
                    psi_dur[d - 1, k] = d
                    psi_st[d - 1, k]  = k

        # Recursion
        for t in range(1, T):
            for k in range(K):
                for d in range(1, min(D_max + 1, t + 1)):
                    t_prev = t - d   # time just before this segment
                    if t_prev < 0:
                        continue
                    dur_ll = self._duration_log_prob(k, d)
                    emit_ll = cum_emit_range(k, t + 1, d)
                    for j in range(K):
                        if j == k:
                            continue
                        v = delta[t_prev, j] + log_trans[j, k] + emit_ll + dur_ll
                        if v > delta[t, k]:
                            delta[t, k]    = v
                            psi_dur[t, k]  = d
                            psi_st[t, k]   = j

        # Backtrack
        states    = np.zeros(T, dtype=int)
        t         = T - 1
        curr_state = int(np.argmax(delta[t]))

        while t >= 0:
            d          = int(psi_dur[t, curr_state])
            d          = max(d, 1)
            t_start    = max(0, t - d + 1)
            states[t_start: t + 1] = curr_state
            if t_start == 0:
                break
            curr_state = int(psi_st[t_start - 1, curr_state])
            t          = t_start - 1

        return states

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        """Predict regimes using semi-Markov Viterbi."""
        self._check_fitted()
        X      = self.scaler.transform(features_df.values)
        # Use fast Viterbi for long sequences; semi-Markov for short
        if len(X) > 2000:
            logger.info("Long sequence: using base HMM Viterbi (semi-Markov too slow)")
            states = self.base_hmm.predict(X)
        else:
            states = self._semimarkov_viterbi(X)
        return pd.Series(
            [self.state_label_map.get(s, f"State_{s}") for s in states],
            index=features_df.index, name="regime"
        )

    def predict_fast(self, features_df: pd.DataFrame) -> pd.Series:
        """Fast Viterbi (base HMM) — use for long sequences."""
        self._check_fitted()
        X      = self.scaler.transform(features_df.values)
        states = self.base_hmm.predict(X)
        return pd.Series(
            [self.state_label_map.get(s, f"State_{s}") for s in states],
            index=features_df.index, name="regime"
        )

    def get_duration_stats(self) -> pd.DataFrame:
        """Compare HMM geometric vs HSMM NegBin mean durations."""
        rows = []
        for i in range(self.n_components):
            r, p = self.duration_params.get(i, (1.0, 0.5))
            rows.append({
                "state":           self.state_label_map.get(i, f"State_{i}"),
                "hmm_mean_dur":    round(1.0 / max(1 - self.base_hmm.transmat_[i, i], 1e-6), 1),
                "hsmm_mean_dur":   round(float(nbinom.mean(r, p)) + 1, 1),
                "hsmm_std_dur":    round(float(nbinom.std(r, p)), 1),
                "negbin_r":        round(r, 3),
                "negbin_p":        round(p, 3),
            })
        return pd.DataFrame(rows).set_index("state")

    def _label_states(self, features_df: pd.DataFrame, states: np.ndarray) -> Dict[int, str]:
        df = features_df.copy()
        df["_state"] = states
        ret_col = "returns" if "returns" in df.columns else df.columns[0]
        vol_col = "vol_20d" if "vol_20d" in df.columns else None
        agg: Dict = {ret_col: "mean"}
        if vol_col: agg[vol_col] = "mean"
        stats = df.groupby("_state").agg(agg)
        stats["ann"] = stats[ret_col] * 252
        stats["vol"] = stats.get(vol_col, pd.Series(0, index=stats.index))
        stats["score"] = stats["ann"].rank() - stats["vol"].rank()
        sorted_states = stats["score"].sort_values().index.tolist()
        names = ["Crisis / Panic", "Bear Market", "Sideways / Range-bound",
                 "Recovery", "Bull Market", "High Volatility"]
        label_counts: Dict[str, int] = {}
        label_map: Dict[int, str] = {}
        for rank, sid in enumerate(sorted_states):
            idx  = min(int(rank * len(names) / len(sorted_states)), len(names)-1)
            base = names[idx]
            cnt  = label_counts.get(base, 0)
            sfx  = ["", " II", " III"][cnt] if cnt < 3 else f" {cnt+1}"
            label_map[sid] = base + sfx
            label_counts[base] = cnt + 1
        return label_map

    def save(self) -> None:
        self.SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, self.SAVE_PATH)

    @classmethod
    def load(cls) -> "HSMMRegimeDetector":
        return joblib.load(cls.SAVE_PATH)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")
