"""Data cache for the out-of-sample evaluation.

Two steps, run separately:

    python -m regime_eval.data download   # fetch raw series once, save as CSV
    python -m regime_eval.data build      # apply logged corrections, build the panel

Everything downstream reads data/processed/panel.csv and nothing else.
There is no synthetic fallback: if a download or a check fails, this raises.

The corrections applied in `build` are the ones recorded in
results/config_log.csv (entries 6 and 7) and fixed in settings.toml.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys

import numpy as np
import pandas as pd

from regime_eval.config import PANEL_PATH, PROCESSED_DIR, RAW_DIR, load_settings

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRSTCI01INM156N"


# ── Step 1: download ─────────────────────────────────────────────────────────

def download(force: bool = False) -> None:
    """Fetch the raw series and save them unchanged.

    Refuses to overwrite existing raw files unless `force` is set, because
    Yahoo revises adjusted prices and the committed files are the record.
    """
    import requests
    import yfinance as yf

    s = load_settings()["data"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in RAW_DIR.glob("*.csv")]
    if existing and not force:
        raise RuntimeError(f"Raw data already cached ({existing}). Pass --force to replace it.")

    end_exclusive = (pd.Timestamp(s["end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for name, ticker in s["tickers"].items():
        df = yf.download(ticker, start=s["start"], end=end_exclusive, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close", "Volume"]].dropna(subset=["Close"])
        if df.empty:
            raise RuntimeError(f"No data returned for {ticker}")
        df.index.name = "date"
        df.columns = ["close", "volume"]
        df.to_csv(RAW_DIR / f"{name}.csv")
        print(f"  {name:<7} {ticker:<13} {len(df)} rows  {df.index[0].date()} to {df.index[-1].date()}")

    r = requests.get(FRED_CSV, timeout=60)
    r.raise_for_status()
    rate = pd.read_csv(io.StringIO(r.text))
    if rate.shape[1] != 2:
        raise RuntimeError("Unexpected FRED response")
    rate.columns = ["month", "rate_pct"]
    rate.to_csv(RAW_DIR / "cash_rate.csv", index=False)
    print(f"  cash    FRED          {len(rate)} months  {rate['month'].iloc[0]} to {rate['month'].iloc[-1]}")


# ── Step 2: build ────────────────────────────────────────────────────────────

def _read_close(name: str) -> pd.Series:
    return pd.read_csv(RAW_DIR / f"{name}.csv", index_col="date", parse_dates=True)["close"]


def _fix_split(close: pd.Series, ratio: float, ex_date: str, dates: list[str], name: str) -> pd.Series:
    """Multiply the closes on the listed ex-split days by the published ratio.

    Guard: each of those closes must be about 1/ratio of the last close before
    the ex-date. If it is not, the raw data differ from what was inspected and
    the correction must not be applied blindly.
    """
    close = close.copy()
    ref = close.loc[close.index < pd.Timestamp(ex_date)].iloc[-1]
    for d in pd.to_datetime(dates):
        if abs(close.loc[d] * ratio / ref - 1) > 0.20:
            raise RuntimeError(f"{name} {d.date()}: close {close.loc[d]} is not on the mis-scaled basis; inspect before correcting")
        close.loc[d] = close.loc[d] * ratio
    return close


def build() -> pd.DataFrame:
    cfg = load_settings()
    s, c = cfg["data"], cfg["corrections"]
    per_year = s["trading_days_per_year"]

    equity = _fix_split(_read_close("equity"), c["split_ratio_equity"], c["split_ex_date"], c["split_dates"], "equity")
    gold = _fix_split(_read_close("gold"), c["split_ratio_gold"], c["split_ex_date"], c["split_dates"], "gold")
    index = _read_close("index")
    vix = _read_close("vix")

    # Calendar: days on which both traded assets have a close.
    cal = equity.index.intersection(gold.index)
    cal = cal[(cal >= pd.Timestamp(s["start"])) & (cal <= pd.Timestamp(s["end"]))]

    equity_ret = equity.reindex(cal).pct_change()
    gold_ret = gold.reindex(cal).pct_change()

    # 2010 bad prices: replace the ETF return with the index return over the same day.
    replaced = {}
    for d in pd.to_datetime(c["replace_equity_return_with_index"]):
        prev = cal[cal.get_loc(d) - 1]
        index_ret = index.loc[d] / index.loc[prev] - 1
        if not np.isfinite(index_ret):
            raise RuntimeError(f"Index return unavailable on {d.date()}")
        replaced[str(d.date())] = (float(equity_ret.loc[d]), float(index_ret))
        equity_ret.loc[d] = index_ret

    # Price levels rebuilt from the corrected returns, so that features built
    # on levels (drawdown, moving average) see the corrected series.
    equity_px = equity.reindex(cal).iloc[0] * (1 + equity_ret.fillna(0)).cumprod()
    gold_px = gold.reindex(cal).iloc[0] * (1 + gold_ret.fillna(0)).cumprod()

    # India VIX: carry the last value forward, at most N calendar-of-trading days.
    limit = s["vix_ffill_max_days"]
    vix_on_cal = vix.reindex(cal)
    vix_filled = vix.reindex(cal.union(vix.index)).ffill(limit=limit).reindex(cal)
    n_vix_filled = int((vix_on_cal.isna() & vix_filled.notna()).sum())
    if vix_filled.isna().any():
        raise RuntimeError(f"India VIX missing on {int(vix_filled.isna().sum())} days after carrying forward {limit} days")

    # Cash: previous month's rate, converted to a daily return.
    rate = pd.read_csv(RAW_DIR / "cash_rate.csv", parse_dates=["month"])
    rate = pd.Series(pd.to_numeric(rate["rate_pct"], errors="coerce").values, index=rate["month"].dt.to_period("M")).dropna()
    prev_month = cal.to_period("M") - 1
    rf_pct = pd.Series(rate.reindex(prev_month).values, index=cal)
    if rf_pct.isna().any():
        raise RuntimeError(f"Cash rate missing for {int(rf_pct.isna().sum())} days")
    cash_ret = (1 + rf_pct / 100) ** (1 / per_year) - 1

    panel = pd.DataFrame({
        "equity_px": equity_px, "gold_px": gold_px,
        "equity_ret": equity_ret, "gold_ret": gold_ret, "cash_ret": cash_ret,
        "rf_annual_pct": rf_pct, "vix": vix_filled,
    })
    panel.index.name = "date"

    # Verification: rerun the two split checks on the corrected series (counts only).
    index_ret_all = index.reindex(cal).pct_change()
    flag_equity = int(((panel["equity_ret"] - index_ret_all).abs() > 0.05).sum())
    flag_gold = int((panel["gold_ret"].abs() > 0.15).sum())
    if flag_equity or flag_gold:
        raise RuntimeError(f"Split checks still flag days after correction: equity {flag_equity}, gold {flag_gold}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_PATH, float_format="%.10g")
    meta = {
        "rows": len(panel), "first_date": str(cal[0].date()), "last_date": str(cal[-1].date()),
        "vix_days_carried_forward": n_vix_filled,
        "equity_returns_replaced_with_index": replaced,
        "split_closes_rescaled": c["split_dates"],
        "split_checks_after_correction": {"equity_vs_index_gt_5pts": flag_equity, "gold_move_gt_15pct": flag_gold},
        "raw_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW_DIR.glob("*.csv"))},
        "panel_sha256": hashlib.sha256(PANEL_PATH.read_bytes()).hexdigest(),
    }
    (PROCESSED_DIR / "panel_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return panel


def load_panel() -> pd.DataFrame:
    """The only entry point the rest of the package uses to read data."""
    if not PANEL_PATH.exists():
        raise FileNotFoundError("data/processed/panel.csv not found. Run: python -m regime_eval.data build")
    return pd.read_csv(PANEL_PATH, index_col="date", parse_dates=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "download":
        download(force="--force" in sys.argv)
    elif cmd == "build":
        p = build()
        print(f"panel: {len(p)} rows, {p.index[0].date()} to {p.index[-1].date()}")
        print((PROCESSED_DIR / "panel_meta.json").read_text(encoding="utf-8"))
    else:
        sys.exit("usage: python -m regime_eval.data [download [--force] | build]")
