"""Walk-forward signal generation.

For each block of `refit_every_days` out-of-sample days:

  1. Take the training window: train_start up to the day before the block.
  2. Inside that window only: choose the number of states, fit the scaler
     and the model, and order the states by volatility.
  3. Run the forward filter from train_start to the end of the block with
     that model. The rows that fall in the block are the signals.

The signal for day t is the equity target weight computed from data up to
the close of day t. Lagging it for execution is the backtest's job.

Block boundaries are counted forward from the first out-of-sample day, so
removing later data never moves them.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from regime_eval import hmm as hmm_mod
from regime_eval.config import load_settings
from regime_eval.features import build_features


def state_weights(n_states: int, calmest: float, most_volatile: float) -> np.ndarray:
    """Equity weight per state, evenly spaced, state 0 (calmest) highest."""
    return np.linspace(calmest, most_volatile, n_states)


def generate_signals(panel: pd.DataFrame, model: str = "hmm", fit_block: Callable | None = None) -> pd.DataFrame:
    """Return one row per out-of-sample day present in `panel`.

    Columns: equity_target, n_states, block, train_end, p_state_0..p_state_3
    (unused states are NaN), plus diagnostics of the fit.

    `fit_block(X_window) -> (model_dict, filter_fn, info)` lets another model
    (the semi-Markov one) reuse this loop unchanged.
    """
    cfg = load_settings()
    w, m, a = cfg["windows"], cfg["model"], cfg["allocation"]
    feats = build_features(panel).loc[w["train_start"]:]
    if feats.isna().any().any():
        raise ValueError("Features contain missing values inside the evaluation range")
    X = feats.to_numpy()
    dates = feats.index
    oos_pos = np.flatnonzero((dates >= pd.Timestamp(w["oos_start"])) & (dates <= pd.Timestamp(w["oos_end"])))
    if len(oos_pos) == 0:
        return pd.DataFrame()
    first, last = oos_pos[0], oos_pos[-1]
    max_k = max(m["n_states_grid"])

    if fit_block is None:
        if model != "hmm":
            raise ValueError("Pass fit_block for models other than the plain HMM")

        def fit_block(X_window):
            fitted, scores = hmm_mod.select_and_fit(X_window, m["n_states_grid"], m["seeds"], m["validation_fraction"])
            return fitted, hmm_mod.filter_with, {"heldout": scores}

    rows = []
    for block, start in enumerate(range(first, last + 1, w["refit_every_days"])):
        end = min(start + w["refit_every_days"], last + 1)          # exclusive
        fitted, filter_fn, info = fit_block(X[:start])              # rows before the block only
        probs, _ = filter_fn(fitted, X[:end])                       # history plus this block
        k = fitted["n_states"]
        weights = state_weights(k, a["equity_weight_calmest"], a["equity_weight_most_volatile"])
        for t in range(start, end):
            p = np.full(max_k, np.nan)
            p[:k] = probs[t]
            rows.append({
                "date": dates[t], "equity_target": float(probs[t] @ weights), "n_states": k,
                "block": block, "train_end": dates[start - 1], "seed": fitted.get("seed"),
                **{f"p_state_{j}": p[j] for j in range(max_k)},
                **{f"heldout_{kk}": v for kk, v in info.get("heldout", {}).items()},
                **{kk: v for kk, v in info.items() if kk != "heldout"},
            })
    return pd.DataFrame(rows).set_index("date")
