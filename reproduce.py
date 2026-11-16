"""Regenerates every reported number from the cached data with fixed seeds.

    python reproduce.py                 # use cached walk-forward signals if present
    python reproduce.py --refit         # refit both models from scratch (about an hour)
    python reproduce.py --skip-survival # skip the in-sample survival curves

Requires the data cache: run `python -m regime_eval.data download` once,
then this script builds the panel itself.

Real-data outputs go to results/real/. Nothing here writes synthetic output;
simulated series are used only inside the tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd

from regime_eval import evaluate as ev
from regime_eval import survival
from regime_eval.config import ROOT, SETTINGS_PATH, load_settings
from regime_eval.data import build
from regime_eval.runlog import append_log
from regime_eval.walkforward import generate_signals

OUT = ROOT / "results" / "real"
MODELS = {"hmm": ev.HMM_NAME, "hsmm": ev.HSMM_NAME}


def settings_hash() -> str:
    return hashlib.sha256(SETTINGS_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12]


def load_or_fit_signals(panel: pd.DataFrame, model: str, refit: bool) -> pd.DataFrame:
    path = OUT / f"oos_signals_{model}.csv"
    meta_path = OUT / f"oos_signals_{model}.meta.json"
    if path.exists() and meta_path.exists() and not refit:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("settings_hash") == settings_hash():
            print(f"  {model}: using cached signals")
            return pd.read_csv(path, index_col="date", parse_dates=True)
    print(f"  {model}: fitting walk-forward signals")
    sig = generate_signals(panel, model=model)
    OUT.mkdir(parents=True, exist_ok=True)
    sig.to_csv(path, float_format="%.10g", lineterminator="\n")
    meta_path.write_text(json.dumps({"settings_hash": settings_hash(), "rows": len(sig)}, indent=2), encoding="utf-8")
    counts = sig.groupby("block")["n_states"].first().value_counts().sort_index().to_dict()
    append_log("configuration run", f"Walk-forward signals: {MODELS[model]}",
               f"Frozen settings (hash {settings_hash()}). {sig['block'].nunique()} refits. States chosen per refit: {counts}.",
               "reproduce.py", f"{len(sig)} out-of-sample signal days")
    return sig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--skip-survival", action="store_true")
    args = ap.parse_args()
    cfg = load_settings()

    print("1. Building the panel from the raw cache")
    panel = build()

    if not args.skip_survival:
        print("2. In-sample survival curves (descriptive)")
        survival.main()

    print("3. Walk-forward signals")
    signals = {name: load_or_fit_signals(panel, model, args.refit) for model, name in MODELS.items()}

    print("4. Out-of-sample evaluation")
    base = cfg["trading"]["cost_bps_per_unit_turnover"]
    for cost in (base, 0, cfg["trading"]["cost_bps_sensitivity"]):
        tag = "main" if cost == base else ("before_costs" if cost == 0 else f"cost_{cost:g}bps")
        metrics, diffs, sub, daily = ev.evaluate(panel, signals, cost)
        ev.write_table(OUT / f"oos_results_{tag}.csv", metrics, ev.header(daily, cost, "Out-of-sample performance of each strategy and benchmark."))
        ev.write_table(OUT / f"oos_sharpe_differences_{tag}.csv", diffs,
                       ev.header(daily, cost, "Sharpe ratio difference (strategy minus benchmark) with a 95% paired stationary block-bootstrap interval."), index=False)
        ev.write_table(OUT / f"oos_subperiods_{tag}.csv", sub, ev.header(daily, cost, "Out-of-sample performance by pre-declared sub-period."), index=False)
        if cost == base:
            daily.to_csv(OUT / "oos_daily_returns_main.csv", float_format="%.8f", lineterminator="\n")
            print(metrics.round(4).to_string())
            print(diffs.round(4).to_string(index=False))
        append_log("configuration run", f"Out-of-sample evaluation at {cost:g} bps",
                   f"Frozen settings (hash {settings_hash()}). Strategies: {', '.join(metrics.index)}. "
                   f"Window {daily.index[0].date()} to {daily.index[-1].date()}, {len(daily)} days.",
                   "reproduce.py", f"results/real/oos_results_{tag}.csv")

    # Refit diagnostics
    rows = []
    for model, name in MODELS.items():
        g = signals[name].groupby("block").first()
        g.insert(0, "model", name)
        rows.append(g.drop(columns=[c for c in g.columns if c.startswith("p_state_") or c == "equity_target"]))
    pd.concat(rows).to_csv(OUT / "oos_refit_log.csv", lineterminator="\n")
    print("Done. Tables are in results/real/.")


if __name__ == "__main__":
    main()
