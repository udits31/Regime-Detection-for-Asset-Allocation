# ============================================================
# utils/stability_analysis.py
# Regime Model Stability & Selection Tools
#
# Implements:
#   1. BIC/AIC sweep → automatic n_states selection
#   2. Hungarian algorithm label alignment across WF folds
#   3. Regime stability test (20 seeds, Adjusted Rand Index)
#   4. Kaplan-Meier survival curves per regime
#   5. Log-rank test for regime duration equality
#   6. Conditional drawdown analysis per regime
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm as hmmlearn_hmm
from typing import Dict, List, Optional, Tuple
from loguru import logger


# ── 1. BIC / AIC Sweep ───────────────────────────────────────────────────────

def bic_sweep(
    features_df: pd.DataFrame,
    n_range:     range = range(2, 9),
    n_init:      int   = 5,
    covariance_type: str = "full",
) -> pd.DataFrame:
    """
    Fit Gaussian HMMs for each n in n_range and return BIC/AIC/log-lik.
    Optimal n_states = argmin BIC (lower is better).

    Parameters
    ──────────
    features_df : Feature matrix (samples × features).
    n_range     : Range of state counts to sweep.
    n_init      : Number of random initialisations per n (takes best).

    Returns
    ───────
    DataFrame indexed by n with columns: log_lik, bic, aic, n_params.
    """
    logger.info(f"BIC sweep: n_states ∈ {list(n_range)}")
    scaler = StandardScaler()
    X      = scaler.fit_transform(features_df.values)
    T, d   = X.shape

    rows = []
    for n in n_range:
        best_ll, best_bic = -np.inf, np.inf
        for _ in range(n_init):
            try:
                model = hmmlearn_hmm.GaussianHMM(
                    n_components    = n,
                    covariance_type = covariance_type,
                    n_iter          = 150,
                    random_state    = np.random.randint(0, 9999),
                )
                model.fit(X)
                ll = model.score(X)
                if ll > best_ll:
                    best_ll = ll
                    # Free params: means(n*d) + full_cov(n*d*(d+1)/2) + transitions(n*(n-1))
                    k      = n * d + n * d * (d + 1) // 2 + n * (n - 1)
                    bic    = -2 * ll * T + k * np.log(T)
                    aic    = -2 * ll * T + 2 * k
                    best_bic = bic
            except Exception:
                continue

        rows.append({
            "n_states": n,
            "log_lik":  round(best_ll, 2),
            "bic":      round(best_bic, 1),
            "aic":      round(aic, 1),
            "n_params": k,
        })
        logger.info(f"  n={n}: log-lik={best_ll:.2f}  BIC={best_bic:.1f}  AIC={aic:.1f}")

    df = pd.DataFrame(rows).set_index("n_states")

    optimal_n = int(df["bic"].idxmin())
    logger.info(f"BIC-optimal n_states = {optimal_n}")
    df["optimal"] = df.index == optimal_n
    return df


def find_elbow(bic_series: pd.Series) -> int:
    """
    Detect elbow point in BIC curve using maximum curvature (Kneedle).
    Returns the n_states at the elbow.
    """
    n_vals = bic_series.index.values.astype(float)
    bics   = bic_series.values.astype(float)
    # Normalise to [0,1]
    n_norm = (n_vals - n_vals.min()) / (n_vals.max() - n_vals.min() + 1e-10)
    b_norm = (bics - bics.min())    / (bics.max()   - bics.min() + 1e-10)
    # Kneedle: distance from the diagonal
    diff   = b_norm - n_norm
    elbow_idx = int(diff.argmax())
    return int(n_vals[elbow_idx])


# ── 2. Hungarian Label Alignment ──────────────────────────────────────────────

def align_labels_hungarian(
    ref_means:  np.ndarray,
    fold_means: np.ndarray,
) -> np.ndarray:
    """
    Align fold HMM state indices to reference state indices using the
    Hungarian algorithm (linear_sum_assignment on Euclidean distances).

    Ensures that "state 0" in Fold 1 represents the same market regime
    as "state 0" in Fold 2, making fold results directly comparable.

    Parameters
    ──────────
    ref_means  : (n_states, n_features) reference HMM emission means.
    fold_means : (n_states, n_features) fold HMM emission means.

    Returns
    ───────
    perm : length-n array. perm[fold_state] = ref_state.
    """
    n = ref_means.shape[0]
    # Cost matrix: L2 distance between each pair of means
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = np.linalg.norm(ref_means[i] - fold_means[j])

    # Hungarian: minimise total cost
    row_ind, col_ind = linear_sum_assignment(cost)
    # col_ind[i] = fold state mapped to reference state i
    # We want perm[fold_state] = ref_state
    perm = np.zeros(n, dtype=int)
    for ref_s, fold_s in zip(row_ind, col_ind):
        perm[fold_s] = ref_s
    return perm


def relabel_with_perm(
    labels: np.ndarray,
    perm:   np.ndarray,
    label_map: Dict[int, str],
) -> np.ndarray:
    """Apply state permutation to a label array and return string labels."""
    return np.array([
        label_map.get(int(perm[s]), f"State_{perm[s]}") for s in labels
    ])


# ── 3. Regime Detection Stability Test ───────────────────────────────────────

def regime_stability_test(
    features_df:  pd.DataFrame,
    n_components: int = 4,
    n_seeds:      int = 20,
    covariance_type: str = "full",
) -> Dict:
    """
    Measures how stable HMM regime assignments are across random seeds.

    Runs HMM with n_seeds different initialisations, computes pairwise
    Adjusted Rand Index (ARI) between all pairs of Viterbi paths.

    ARI ∈ [-1, 1]:
      1.0  = identical (perfect stability)
      0.0  = random (completely unstable)
      >0.7 = acceptable for regime detection

    Returns
    ───────
    mean_ari, min_ari, std_ari, all_labels (n_seeds × T), stable (bool)
    """
    logger.info(f"Regime stability test: {n_seeds} seeds, n_states={n_components}")
    scaler = StandardScaler()
    X      = scaler.fit_transform(features_df.values)

    all_labels = []
    for seed in range(n_seeds):
        try:
            model = hmmlearn_hmm.GaussianHMM(
                n_components    = n_components,
                covariance_type = covariance_type,
                n_iter          = 150,
                random_state    = seed,
            )
            model.fit(X)
            labels = model.predict(X)
            all_labels.append(labels)
        except Exception:
            pass

    if len(all_labels) < 2:
        return {"mean_ari": np.nan, "stable": False, "error": "Too few converged runs"}

    # Pairwise ARI
    ari_values = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            ari_values.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    mean_ari = float(np.mean(ari_values))
    min_ari  = float(np.min(ari_values))
    std_ari  = float(np.std(ari_values))
    stable   = mean_ari >= 0.70

    logger.info(
        f"  ARI: mean={mean_ari:.3f}  min={min_ari:.3f}  std={std_ari:.3f}  "
        f"stable={stable}"
    )

    return {
        "mean_ari":   round(mean_ari, 4),
        "min_ari":    round(min_ari,  4),
        "std_ari":    round(std_ari,  4),
        "n_converged":len(all_labels),
        "stable":     stable,
        "verdict": (
            f"STABLE (mean ARI={mean_ari:.3f} ≥ 0.70)"
            if stable else
            f"UNSTABLE (mean ARI={mean_ari:.3f} < 0.70) — "
            f"consider fewer states or different features"
        ),
    }


# ── 4. Kaplan-Meier Survival Analysis ────────────────────────────────────────

def kaplan_meier(durations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kaplan-Meier estimator for a single group of event times.

    Returns (times, survival_probability).
    All events are assumed observed (no censoring — regimes always end).
    """
    durations = np.sort(durations[durations > 0])
    if len(durations) == 0:
        return np.array([0]), np.array([1.0])

    times   = np.unique(durations)
    S       = 1.0
    t_list  = [0.0]
    S_list  = [1.0]

    for t in times:
        n_at_risk = int((durations >= t).sum())
        n_events  = int((durations == t).sum())
        if n_at_risk > 0:
            S *= 1.0 - n_events / n_at_risk
        t_list.append(float(t))
        S_list.append(float(S))

    return np.array(t_list), np.array(S_list)


def log_rank_test(
    durations1: np.ndarray,
    durations2: np.ndarray,
) -> Dict:
    """
    Log-rank test for equality of two survival functions.

    H0: S1(t) = S2(t) for all t.
    Used to test whether two regimes have significantly different durations.

    Returns chi2 statistic, p-value, and interpretation.
    """
    d1  = durations1[durations1 > 0]
    d2  = durations2[durations2 > 0]
    all_times = np.sort(np.unique(np.concatenate([d1, d2])))

    O1 = O2 = E1 = E2 = 0.0
    for t in all_times:
        n1 = float((d1 >= t).sum()); n2 = float((d2 >= t).sum())
        n  = n1 + n2
        if n == 0:
            continue
        ev1 = float((d1 == t).sum()); ev2 = float((d2 == t).sum())
        ev  = ev1 + ev2
        E1 += ev * n1 / n; E2 += ev * n2 / n
        O1 += ev1;         O2 += ev2

    if E1 < 1e-6 or E2 < 1e-6:
        return {"chi2": np.nan, "p_value": np.nan, "reject_H0": False}

    chi2_stat = (O1 - E1) ** 2 / E1 + (O2 - E2) ** 2 / E2
    p_value   = float(1 - chi2_dist.cdf(chi2_stat, df=1))

    return {
        "chi2":      round(float(chi2_stat), 4),
        "p_value":   round(p_value, 4),
        "reject_H0": bool(p_value < 0.05),
        "interpretation": (
            f"p={p_value:.3f} — {'DIFFERENT' if p_value < 0.05 else 'SIMILAR'} "
            f"duration distributions at 5% significance."
        ),
    }


def regime_survival_analysis(
    labels:  pd.Series,
    returns: pd.Series,
) -> Dict:
    """
    Run survival analysis across all detected regimes.

    Extracts run lengths per regime, fits KM curves, runs pairwise
    log-rank tests, and reports median survival times.

    Returns
    ───────
    Dict of per-regime KM results and pairwise log-rank p-values.
    """
    # Extract run lengths per regime
    regime_durations: Dict[str, List[int]] = {}
    current   = labels.iloc[0]
    run_count = 1
    for i in range(1, len(labels)):
        if labels.iloc[i] == current:
            run_count += 1
        else:
            regime_durations.setdefault(current, []).append(run_count)
            current   = labels.iloc[i]
            run_count = 1
    regime_durations.setdefault(current, []).append(run_count)

    results: Dict = {}
    km_curves:    Dict[str, Tuple] = {}
    median_durs:  Dict[str, float] = {}

    for regime, durs in regime_durations.items():
        d_arr        = np.array(durs)
        t_km, S_km   = kaplan_meier(d_arr)
        # Median = first t where S(t) ≤ 0.5
        below        = t_km[S_km <= 0.5]
        median_dur   = float(below[0]) if len(below) > 0 else float(d_arr.max())
        km_curves[regime]   = (t_km, S_km)
        median_durs[regime] = median_dur
        results[regime]     = {
            "n_episodes":   len(durs),
            "mean_dur":     round(float(np.mean(d_arr)), 1),
            "median_dur":   round(median_dur,  1),
            "max_dur":      int(d_arr.max()),
            "durations":    d_arr.tolist(),
        }

    # Pairwise log-rank tests
    regimes   = list(regime_durations.keys())
    pairwise  = {}
    for i in range(len(regimes)):
        for j in range(i + 1, len(regimes)):
            r1, r2 = regimes[i], regimes[j]
            lr     = log_rank_test(
                np.array(regime_durations[r1]),
                np.array(regime_durations[r2]),
            )
            pairwise[f"{r1} vs {r2}"] = lr

    return {
        "per_regime":       results,
        "median_durations": median_durs,
        "km_curves":        km_curves,
        "pairwise_logrank": pairwise,
    }


# ── 5. Conditional Drawdown by Regime ────────────────────────────────────────

def conditional_drawdown(
    equity_curve: pd.Series,
    labels:       pd.Series,
) -> pd.DataFrame:
    """
    Maximum drawdown of the portfolio separately for each detected regime.
    Shows whether the model correctly reduces drawdowns in Bear/Crisis periods.
    """
    roll_max = equity_curve.cummax()
    dd       = (equity_curve - roll_max) / roll_max
    merged   = pd.DataFrame({
        "dd":     dd,
        "regime": labels.reindex(dd.index, method="ffill"),
        "ret":    equity_curve.pct_change(),
    }).dropna()

    rows = []
    for regime, grp in merged.groupby("regime"):
        rows.append({
            "Regime":        regime,
            "Max DD (%)":    round(float(grp["dd"].min()) * 100, 2),
            "Mean DD (%)":   round(float(grp["dd"].mean()) * 100, 2),
            "Obs":           len(grp),
            "Worst Day (%)": round(float(grp["ret"].min()) * 100, 2),
            "Mean Ret (%)":  round(float(grp["ret"].mean()) * 252 * 100, 2),
        })
    return (
        pd.DataFrame(rows)
        .set_index("Regime")
        .sort_values("Max DD (%)")
    )


# ── 6. IS/OOS Gap Report ─────────────────────────────────────────────────────

def is_oos_gap(
    is_returns:   pd.Series,
    oos_returns:  pd.Series,
    risk_free:    float = 0.065,
) -> Dict:
    """
    Report in-sample vs out-of-sample Sharpe gap.
    Gap > 0.5 suggests potential overfitting.
    """
    rf = risk_free / 252

    def sharpe(r):
        return float((r.mean() - rf) / (r.std() + 1e-10) * np.sqrt(252))

    sh_is  = sharpe(is_returns)
    sh_oos = sharpe(oos_returns)
    gap    = sh_is - sh_oos

    return {
        "is_sharpe":  round(sh_is,  4),
        "oos_sharpe": round(sh_oos, 4),
        "gap":        round(gap,    4),
        "overfit_flag": gap > 0.5,
        "verdict": (
            f"IS={sh_is:.3f}  OOS={sh_oos:.3f}  Gap={gap:.3f} — "
            + ("⚠ POTENTIAL OVERFITTING (gap > 0.5)" if gap > 0.5 else "✓ Gap acceptable")
        ),
    }
