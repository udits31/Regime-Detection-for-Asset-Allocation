# ============================================================
# realtime_engine/data_ingestion.py  — ROBUST VERSION
# ============================================================

import time
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from loguru import logger

warnings.filterwarnings("ignore")

# ── Verified working tickers for Indian market ─────────────
VERIFIED_INDIAN_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "LT.NS", "ITC.NS",
    "BHARTIARTL.NS", "SUNPHARMA.NS",
]

VERIFIED_ETFS = [
    "NIFTYBEES.NS", "GOLDBEES.NS", "BANKBEES.NS",
]

VERIFIED_CRYPTO = [
    "BTC-USD", "ETH-USD",
]

VERIFIED_COMMODITIES = [
    "GC=F",   # Gold Futures
]

VERIFIED_INDICES = [
    "^NSEI",    # NIFTY 50  — primary signal
    "^BSESN",   # SENSEX
    "USDINR=X", # USD/INR
]

ALL_VERIFIED = (
    VERIFIED_INDIAN_STOCKS +
    VERIFIED_ETFS +
    VERIFIED_CRYPTO +
    VERIFIED_COMMODITIES
)


class MarketDataIngestion:

    def __init__(self):
        self.yf_cache: Dict[str, pd.DataFrame] = {}

    # ── Single ticker download with retry ─────────────────

    def _download_one(
        self,
        ticker: str,
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
        retries: int = 3,
        delay: float = 2.0,
    ) -> Optional[pd.DataFrame]:
        """
        Download one ticker with retry logic and validation.
        Returns None if data is unusable.
        """
        end = end or datetime.now().strftime("%Y-%m-%d")

        for attempt in range(retries):
            try:
                # Download — use group_by="ticker" to get clean columns
                raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True,   # Adjusts for splits/dividends
                    progress=False,
                    threads=False,      # Single-threaded avoids rate limits
                )

                if raw is None or raw.empty:
                    logger.warning(f"  ✗ {ticker}: empty response (attempt {attempt+1})")
                    time.sleep(delay * (attempt + 1))
                    continue

                # Flatten multi-level columns if present
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

                # Rename 'adj_close' → 'close' if needed
                if "adj_close" in raw.columns and "close" not in raw.columns:
                    raw["close"] = raw["adj_close"]

                if "close" not in raw.columns:
                    logger.warning(f"  ✗ {ticker}: no close price column")
                    return None

                # ── Validation checks ──────────────────────
                df = raw[["close"]].copy()

                # Must have at least 60 rows for any meaningful analysis
                if len(df) < 60:
                    logger.warning(f"  ✗ {ticker}: only {len(df)} rows — too few")
                    return None

                # Drop NaN close prices
                df = df.dropna(subset=["close"])

                # Drop zero or negative prices (data error)
                df = df[df["close"] > 0]

                # Check for suspicious price jumps (>80% in one day = likely error)
                daily_returns = df["close"].pct_change().abs()
                bad_days = (daily_returns > 0.80).sum()
                if bad_days > 3:
                    logger.warning(
                        f"  ⚠ {ticker}: {bad_days} suspicious price jumps detected — "
                        f"clipping outliers"
                    )
                    # Cap extreme single-day moves at 50% — keeps data but limits damage
                    df["close"] = self._clip_price_jumps(df["close"], max_jump=0.50)

                # Add OHLC if available
                for col in ["open", "high", "low", "volume"]:
                    if col in raw.columns:
                        df[col] = raw[col]

                # Add computed columns
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df["returns"]     = df["close"].pct_change()
                df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
                df["ticker"]      = ticker

                # Final NaN drop
                df = df.replace([np.inf, -np.inf], np.nan)

                logger.debug(f"  ✓ {ticker}: {len(df)} rows, "
                           f"{df.index[0].date()} → {df.index[-1].date()}")
                return df

            except Exception as e:
                logger.warning(f"  ✗ {ticker}: error on attempt {attempt+1}: {e}")
                time.sleep(delay * (attempt + 1))

        logger.error(f"  ✗ {ticker}: all {retries} attempts failed — skipping")
        return None

    def _clip_price_jumps(
        self, prices: pd.Series, max_jump: float = 0.50
    ) -> pd.Series:
        """
        Replace single-day jumps larger than max_jump with interpolated values.
        Prevents data errors from corrupting feature calculations.
        """
        prices = prices.copy()
        returns = prices.pct_change().abs()
        bad_idx = returns[returns > max_jump].index
        prices.loc[bad_idx] = np.nan
        prices = prices.interpolate(method="time")
        return prices

    # ── Batch download with rate limiting ─────────────────

    def fetch_historical(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d",
        batch_size: int = 5,       # Download in small batches to avoid rate limits
        sleep_between: float = 1.5 # Seconds between batches
    ) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for multiple tickers safely.
        Uses batching and delays to avoid yfinance rate limiting.
        """
        result = {}
        failed = []

        logger.info(
            f"📥 Fetching {len(tickers)} tickers "
            f"[{start_date} → {end_date or 'today'}] "
            f"in batches of {batch_size}"
        )

        # Process in small batches
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            logger.info(f"  Batch {i//batch_size + 1}: {batch}")

            for ticker in batch:
                df = self._download_one(ticker, start_date, end_date, interval)
                if df is not None:
                    result[ticker] = df
                else:
                    failed.append(ticker)

            # Pause between batches — critical for avoiding rate limits
            if i + batch_size < len(tickers):
                time.sleep(sleep_between)

        logger.info(
            f"✅ Download complete: "
            f"{len(result)}/{len(tickers)} successful, "
            f"{len(failed)} failed"
        )
        if failed:
            logger.warning(f"  Failed tickers: {failed}")

        return result

    # ── Aligned price + returns matrix ────────────────────

    def fetch_multi_asset_returns(
        self,
        start_date: str = "2018-01-01",
        end_date: Optional[str] = None,
        min_coverage: float = 0.80,  # Drop assets with >20% missing data
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns aligned (prices_df, returns_df) for the verified asset universe.

        Key quality steps:
        1. Download with retry + validation
        2. Align all assets to a common calendar (NSE trading days)
        3. Forward-fill short gaps (≤5 days) — handles holidays
        4. Drop assets with too many missing values
        5. Winsorize extreme returns (caps outliers)
        """
        all_tickers = ALL_VERIFIED + VERIFIED_INDICES

        raw_data = self.fetch_historical(
            all_tickers, start_date, end_date, batch_size=4, sleep_between=2.0
        )

        if not raw_data:
            raise RuntimeError("No data downloaded. Check your internet connection.")

        # Build aligned close price matrix
        price_dict = {}
        for ticker, df in raw_data.items():
            if "close" in df.columns:
                price_dict[ticker] = df["close"]

        prices_df = pd.DataFrame(price_dict)

        # Use NIFTY 50 trading calendar as the reference
        if "^NSEI" in prices_df.columns:
            reference_index = prices_df["^NSEI"].dropna().index
            prices_df = prices_df.reindex(reference_index)

        # Forward-fill gaps up to 5 days (handles bank holidays etc.)
        prices_df = prices_df.ffill(limit=5)

        # Drop assets with too little data coverage
        coverage = prices_df.notna().mean()
        good_cols = coverage[coverage >= min_coverage].index.tolist()
        dropped = [c for c in prices_df.columns if c not in good_cols]
        if dropped:
            logger.warning(f"  Dropping low-coverage assets: {dropped}")
        prices_df = prices_df[good_cols]

        # Drop any remaining NaN rows
        prices_df = prices_df.dropna(how="all")

        # Compute log returns
        returns_df = np.log(prices_df / prices_df.shift(1))

        # Winsorize returns — cap at ±15% per day
        # This removes data errors without losing the row
        returns_df = returns_df.clip(lower=-0.15, upper=0.15)

        returns_df = returns_df.dropna(how="all")

        logger.info(
            f"✅ Final matrices: "
            f"prices={prices_df.shape}, returns={returns_df.shape}\n"
            f"  Assets: {list(prices_df.columns)}\n"
            f"  Date range: {prices_df.index[0].date()} → {prices_df.index[-1].date()}"
        )
        return prices_df, returns_df

    # ── Live snapshot ──────────────────────────────────────

    def fetch_live_snapshot(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch current prices for a list of tickers."""
        records = []
        for ticker in tickers:
            try:
                t    = yf.Ticker(ticker)
                info = t.fast_info
                price = getattr(info, "last_price", None)
                prev  = getattr(info, "previous_close", None)
                if price and prev and price > 0:
                    records.append({
                        "ticker":     ticker,
                        "price":      round(price, 2),
                        "prev_close": round(prev, 2),
                        "change_pct": round((price - prev) / prev * 100, 2),
                    })
            except Exception as e:
                logger.debug(f"Live snapshot failed for {ticker}: {e}")
        return pd.DataFrame(records)

    # ── Macro indicators ───────────────────────────────────

    def fetch_macro_indicators(
        self, start_date: str = "2018-01-01"
    ) -> pd.DataFrame:
        """Fetch NIFTY 50, SENSEX, USD/INR as macro features."""
        macro_map = {
            "^NSEI":    "nifty_50",
            "^BSESN":   "sensex",
            "USDINR=X": "usd_inr",
        }
        frames = {}
        for ticker, col in macro_map.items():
            df = self._download_one(ticker, start_date)
            if df is not None:
                frames[col] = df["close"]

        if not frames:
            logger.warning("No macro data available")
            return pd.DataFrame()

        macro_df = pd.DataFrame(frames).ffill(limit=5).dropna(how="all")

        if "nifty_50" in macro_df.columns:
            macro_df["nifty_returns"] = macro_df["nifty_50"].pct_change()
            macro_df["nifty_vol_20"] = (
                macro_df["nifty_returns"].rolling(20).std() * np.sqrt(252)
            )

        return macro_df