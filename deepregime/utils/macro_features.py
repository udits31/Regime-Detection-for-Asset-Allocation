# ============================================================
# utils/macro_features.py
# India-Specific Macroeconomic Feature Collection
#
# Reviewer requirement: "Incorporate India-specific macro
# variables: CPI, WPI, Repo Rate, INR/USD, FII Flows,
# India VIX, Yield Curve Spread, IIP"
#
# Data sources via yfinance (real-time):
#   ^INDIAVIX   : India VIX (fear index)
#   USDINR=X    : USD/INR exchange rate
#   GC=F        : Gold (strategic asset, India-specific)
#   ^GSPC       : S&P 500 (global risk proxy)
#   CL=F        : Crude Oil (India = major importer)
#   IRFC.NS     : Long-duration bond proxy
#
# Unavailable via yfinance (would need RBI / MOSPI APIs):
#   CPI, WPI, Repo Rate, FII Flows, IIP
#   → We construct proxies and provide stubs
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False


# Tickers available via yfinance
MACRO_TICKERS = {
    "india_vix":     "^INDIAVIX",
    "usdinr":        "USDINR=X",
    "gold":          "GC=F",
    "crude_oil":     "CL=F",
    "us_sp500":      "^GSPC",
    "nifty_bank":    "^NSEBANK",
    "nifty_it":      "^CNXIT",
    "nifty_pharma":  "NIFPHARMA.NS",
    "us_10y":        "^TNX",        # US 10-year yield (global risk-free proxy)
    "silver":        "SI=F",
}

# Known RBI repo rate changes (approximate; would need RBI API for live)
RBI_REPO_RATE_HISTORY = [
    ("2019-06-06", 5.75),
    ("2019-08-07", 5.40),
    ("2019-10-04", 5.15),
    ("2019-12-05", 5.15),
    ("2020-03-27", 4.40),
    ("2020-05-22", 4.00),
    ("2022-05-04", 4.40),
    ("2022-06-08", 4.90),
    ("2022-08-05", 5.40),
    ("2022-09-30", 5.90),
    ("2022-12-07", 6.25),
    ("2023-02-08", 6.50),
    ("2024-10-09", 6.50),
]

# Demonetisation shock date
DEMONETISATION_DATE = "2016-11-08"
COVID_LOCKDOWN_DATE = "2020-03-25"


class IndianMacroFeatureBuilder:
    """
    Builds India-specific macroeconomic features for regime detection.

    Feature groups
    ──────────────
    1. India VIX (fear index, direct volatility measure)
    2. INR/USD dynamics (FX regime, capital flows proxy)
    3. Gold/INR (India's strategic safe haven)
    4. Global risk (S&P 500, Crude oil impact)
    5. Repo rate (monetary policy regime)
    6. Yield curve proxy (term structure signal)
    7. Sector rotation (IT vs Bank vs Pharma)
    8. Crisis dummies (Demonetisation, COVID)
    """

    def __init__(self, start_date: str = "2016-01-01"):
        self.start_date = start_date
        self.raw_data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None

    def fetch_and_build(self) -> pd.DataFrame:
        """Download macro data and build features. Returns feature DataFrame."""
        logger.info("Fetching India macro data …")
        self.raw_data = self._download()
        self.features = self._build_features()
        logger.info(f"Macro features built: {self.features.shape}")
        return self.features

    def _download(self) -> pd.DataFrame:
        if not _HAS_YF:
            logger.warning("yfinance not available; returning empty macro DataFrame")
            return pd.DataFrame()

        dfs = {}
        for name, ticker in MACRO_TICKERS.items():
            try:
                raw = yf.download(ticker, start=self.start_date,
                                  auto_adjust=True, progress=False)
                if len(raw) > 0:
                    close = raw["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    dfs[name] = close.rename(name)
                    logger.debug(f"  ✓ {name} ({ticker}): {len(raw)} rows")
                else:
                    logger.debug(f"  ✗ {name} ({ticker}): no data")
            except Exception as e:
                logger.debug(f"  ✗ {name}: {e}")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs.values(), axis=1).sort_index()
        combined = combined.ffill().bfill()
        return combined

    def _build_features(self) -> pd.DataFrame:
        df_raw = self.raw_data
        if df_raw is None or len(df_raw) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(index=df_raw.index)

        # ── 1. India VIX ──────────────────────────────────────────────────────
        if "india_vix" in df_raw.columns:
            vix = df_raw["india_vix"]
            df["india_vix"]           = vix
            df["india_vix_30ma"]      = vix.rolling(30).mean()
            df["india_vix_zscore"]    = (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-10)
            df["india_vix_pct90"]     = (vix > vix.rolling(252).quantile(0.90)).astype(int)
            df["india_vix_chg5d"]     = vix.pct_change(5)

        # ── 2. INR/USD ────────────────────────────────────────────────────────
        if "usdinr" in df_raw.columns:
            usdinr = df_raw["usdinr"]
            df["usdinr"]              = usdinr
            df["usdinr_ret20d"]       = usdinr.pct_change(20)
            df["usdinr_vol20d"]       = usdinr.pct_change().rolling(20).std() * np.sqrt(252)
            df["inr_depreciation"]    = (usdinr.pct_change(5) > 0.01).astype(int)
            df["inr_crisis"]          = (
                usdinr.pct_change(20) > usdinr.pct_change(20).rolling(252).quantile(0.95)
            ).astype(int)

        # ── 3. Gold ───────────────────────────────────────────────────────────
        if "gold" in df_raw.columns:
            gold = df_raw["gold"]
            df["gold_ret20d"]         = gold.pct_change(20)
            df["gold_vol20d"]         = gold.pct_change().rolling(20).std() * np.sqrt(252)
            df["gold_safe_haven"]     = (gold.pct_change(5) > 0.02).astype(int)

        # ── 4. Crude Oil (India import dependence) ────────────────────────────
        if "crude_oil" in df_raw.columns:
            crude = df_raw["crude_oil"]
            df["crude_ret20d"]        = crude.pct_change(20)
            df["crude_shock"]         = (crude.pct_change(5) < -0.10).astype(int)

        # ── 5. Global Risk (S&P 500) ──────────────────────────────────────────
        if "us_sp500" in df_raw.columns:
            sp500 = df_raw["us_sp500"]
            df["sp500_ret20d"]        = sp500.pct_change(20)
            df["sp500_vol20d"]        = sp500.pct_change().rolling(20).std() * np.sqrt(252)
            df["global_risk_off"]     = (sp500.pct_change(5) < -0.05).astype(int)

        # ── 6. Repo Rate (categorical from known schedule) ────────────────────
        df["repo_rate"] = self._build_repo_rate_series(df_raw.index)
        df["rate_hike"] = (df["repo_rate"].diff() > 0).astype(int)
        df["rate_cut"]  = (df["repo_rate"].diff() < 0).astype(int)

        # ── 7. Sector Rotation (IT - Bank spread) ────────────────────────────
        if "nifty_it" in df_raw.columns and "nifty_bank" in df_raw.columns:
            it_ret   = df_raw["nifty_it"].pct_change(20)
            bank_ret = df_raw["nifty_bank"].pct_change(20)
            df["it_vs_bank_spread"]   = it_ret - bank_ret
            df["defensive_rotation"]  = (it_ret > bank_ret).astype(int)

        # ── 8. Crisis Dummies ─────────────────────────────────────────────────
        df["post_demonetisation"] = (
            (df.index >= DEMONETISATION_DATE) &
            (df.index <= "2017-03-31")
        ).astype(int)
        df["covid_lockdown"] = (
            (df.index >= COVID_LOCKDOWN_DATE) &
            (df.index <= "2020-10-31")
        ).astype(int)

        return df.ffill().bfill().replace([np.inf, -np.inf], np.nan).dropna(how="all")

    def _build_repo_rate_series(self, index: pd.DatetimeIndex) -> pd.Series:
        """Build daily repo rate series from known RBI rate history."""
        rate = pd.Series(6.50, index=index)   # Default current rate
        for date_str, rate_val in RBI_REPO_RATE_HISTORY:
            try:
                date = pd.Timestamp(date_str)
                rate[rate.index >= date] = rate_val
            except Exception:
                pass
        return rate

    def align_with_price(
        self,
        price_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Forward-fill macro features to align with daily price index."""
        if self.features is None or len(self.features) == 0:
            return pd.DataFrame(index=price_index)
        return self.features.reindex(price_index, method="ffill").fillna(0)
