"""Kaplan-Meier survival curves of regime lengths, for both models.

IN-SAMPLE AND DESCRIPTIVE. Both models are fitted once on the whole real
sample, including the out-of-sample evaluation window, and decoded with the
most likely path. Nothing here feeds the trading evaluation.

    python -m regime_eval.survival

State names are the same for both models: "State 1" is the calmest regime
(lowest return variance) and "State K" the most volatile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from regime_eval import hmm as hmm_mod
from regime_eval import hsmm as hsmm_mod
from regime_eval.config import ROOT, load_settings
from regime_eval.decode import viterbi_hmm, viterbi_hsmm

OUT_DIR = ROOT / "results" / "real"


def run_lengths(path: np.ndarray) -> pd.DataFrame:
    """One row per regime episode: state, length, and whether its full length was observed.

    The first and last episodes are cut by the sample edges, so their true
    lengths are at least what is observed: they are treated as censored.
    """
    change = np.flatnonzero(np.diff(path)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(path)]
    out = pd.DataFrame({"state": path[starts], "length": ends - starts, "observed": True})
    out.loc[[0, len(out) - 1], "observed"] = False
    return out


def kaplan_meier(lengths: np.ndarray, observed: np.ndarray) -> pd.DataFrame:
    """S(d) = P(regime lasts longer than d days), with right-censoring.

    At each observed ending time d: S *= 1 - (endings at d) / (episodes still running at d).
    """
    lengths, observed = np.asarray(lengths), np.asarray(observed, dtype=bool)
    rows, s = [], 1.0
    for d in np.unique(lengths[observed]):
        at_risk = int((lengths >= d).sum())
        ended = int(((lengths == d) & observed).sum())
        s *= 1 - ended / at_risk
        rows.append({"days": int(d), "at_risk": at_risk, "ended": ended, "survival": s})
    return pd.DataFrame(rows, columns=["days", "at_risk", "ended", "survival"])


def km_median(curve: pd.DataFrame) -> float:
    """First day on which the survival curve is at or below one half; NaN if never reached."""
    below = curve.loc[curve["survival"] <= 0.5, "days"]
    return float(below.iloc[0]) if len(below) else float("nan")


def implied_survival_hmm(model: dict, state: int, days: np.ndarray) -> np.ndarray:
    """P(length > d) under the plain HMM: geometric, set by the stay probability."""
    return model["transmat"][state, state] ** np.asarray(days, float)


def implied_survival_hsmm(model: dict, state: int, days: np.ndarray) -> np.ndarray:
    """P(length > d) under the fitted negative binomial (length - 1 ~ NB)."""
    return nbinom.sf(np.asarray(days) - 1, model["dur_r"][state], model["dur_p"][state])


def analyse(X: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit both models with the same number of states and build the curves."""
    m = load_settings()["model"]
    plain, scores = hmm_mod.select_and_fit(X, m["n_states_grid"], m["seeds"], m["validation_fraction"])
    k = plain["n_states"]
    semi = hsmm_mod.fit_hsmm(X, k, m["seeds"])
    paths = {"plain HMM": viterbi_hmm(plain, X), "semi-Markov": viterbi_hsmm(semi, X)}
    implied = {"plain HMM": lambda s, d: implied_survival_hmm(plain, s, d),
               "semi-Markov": lambda s, d: implied_survival_hsmm(semi, s, d)}
    summary, curves = [], []
    for name, path in paths.items():
        runs = run_lengths(path)
        for s in range(k):
            r = runs[runs["state"] == s]
            label = f"State {s + 1} of {k}" + (" (calmest)" if s == 0 else " (most volatile)" if s == k - 1 else "")
            curve = kaplan_meier(r["length"].to_numpy(), r["observed"].to_numpy())
            curve["model_implied_survival"] = implied[name](s, curve["days"].to_numpy()) if len(curve) else []
            curve.insert(0, "state", label)
            curve.insert(0, "model", name)
            curves.append(curve)
            implied_median = np.nan
            grid = np.arange(1, 2001)
            below = grid[implied[name](s, grid) <= 0.5]
            if len(below):
                implied_median = float(below[0])
            summary.append({
                "model": name, "state": label, "episodes": len(r), "censored": int((~r["observed"]).sum()),
                "days_in_state": int(r["length"].sum()),
                "km_median_days": km_median(curve), "longest_days": int(r["length"].max()) if len(r) else 0,
                "model_implied_median_days": implied_median,
            })
    info = {"n_states": k, "heldout_scores": scores,
            "agreement": float((paths["plain HMM"] == paths["semi-Markov"]).mean()),
            "semi_cap_hit_share_max": float(semi["cap_hit_share"].max()),
            "semi_tail_beyond_cap_max": float(semi["tail_beyond_cap"].max()),
            "loglik_plain": plain["train_loglik"], "loglik_semi": semi["train_loglik"]}
    return pd.DataFrame(summary), pd.concat(curves, ignore_index=True), info


def main() -> None:
    from regime_eval.data import load_panel
    from regime_eval.features import build_features

    w = load_settings()["windows"]
    feats = build_features(load_panel()).loc[w["train_start"]:w["oos_end"]]
    summary, curves, info = analyse(feats.to_numpy())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header = (f"# IN-SAMPLE, DESCRIPTIVE. Data: NIFTYBEES.NS features, {feats.index[0].date()} to {feats.index[-1].date()} "
              f"(real data; includes the out-of-sample evaluation window). Both models fitted once on the full sample "
              f"with {info['n_states']} states; regime lengths from the most likely path. Not a trading result.\n")
    for name, df in (("in_sample_survival_summary.csv", summary), ("in_sample_survival_curves.csv", curves)):
        with open(OUT_DIR / name, "w", encoding="utf-8", newline="") as f:
            f.write(header)
            df.to_csv(f, index=False, float_format="%.4f")
    print(header.strip())
    print(summary.to_string(index=False))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()})


if __name__ == "__main__":
    main()
