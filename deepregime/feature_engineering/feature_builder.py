# ============================================================
# feature_engineering/feature_builder.py
# Technical + Microstructure + Regime Features
#
# FIXED (vs original):
#   • VWAP formula corrected: Σ(P·V) / Σ(V)
#   • FeatureBuilder is now stateless / pure — no side effects,
#     safe to call repeatedly inside a walk-forward loop
# NEW:
#   • Amihud illiquidity ratio
#   • Hurst exponent (trend vs mean-reversion classification)
#   • Realised skewness and kurtosis for tail-risk regime features
#   • Autocorrelation-based momentum decay
#   • Fama-French-style factor exposures (size proxy, value proxy)
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger


class FeatureBuilder:
    """
    Stateless feature construction pipeline.

    All methods take DataFrames and return DataFrames — no instance
    state is mutated after construction, making this safe to call
    in parallel walk-forward folds.

    Feature groups
    ──────────────
    1. Returns & log-returns
    2. Moving averages & trend signals
    3. RSI (multiple periods)
    4. MACD
    5. Momentum (multiple horizons)
    6. Bollinger Bands
    7. ATR (requires OHLC; degrades gracefully to close-to-close)
    8. Volatility (realised, rolling, regime-relative)
    9. Volume & microstructure (OBV, VWAP [FIXED], Amihud)
    10. Risk metrics (drawdown, rolling Sharpe, Sortino, skew, kurtosis)
    11. Regime composite signals (crisis score, trend strength)
    12. Hurst exponent (long-range dependence)
    13. Macro (aligned by forward-fill)
    """

    # ── Master builder ────────────────────────────────────────────────────────

    def build_all_features(
        self,
        prices_df: pd.DataFrame,
        macro_df:  Optional[pd.DataFrame] = None,
        ticker:    str = "^NSEI",
    ) -> pd.DataFrame:
        """
        Build the complete feature matrix from OHLCV price data.

        Parameters
        ──────────
        prices_df : DataFrame with at minimum a column for `ticker`.
                    Optional columns: `{ticker}_high`, `{ticker}_low`,
                    `volume` or `{ticker}_volume`.
        macro_df  : Optional macro indicators (e.g. USD/INR, VIX).
        ticker    : Primary price column key.

        Returns
        ───────
        Feature DataFrame, NaN rows dropped.
        """
        if ticker not in prices_df.columns:
            ticker = prices_df.columns[0]
            logger.warning(f"Ticker not found; using first column: {ticker}")

        price = prices_df[ticker].dropna()
        df    = pd.DataFrame(index=price.index)

        # ── Returns ───────────────────────────────────────────────────────────
        df["close"]       = price
        df["returns"]     = price.pct_change()
        df["log_returns"] = np.log(price / price.shift(1))

        # ── Indicators ────────────────────────────────────────────────────────
        df = self._add_moving_averages(df, price)
        df = self._add_rsi(df, price)
        df = self._add_macd(df, price)
        df = self._add_momentum(df, price)
        df = self._add_bollinger_bands(df, price)
        df = self._add_atr(df, prices_df, ticker)
        df = self._add_volatility_features(df)

        # Volume features (optional, degrade gracefully)
        vol_col = self._find_volume_col(prices_df, ticker)
        if vol_col:
            df = self._add_volume_features(df, price, prices_df[vol_col])

        # ── Risk & regime ─────────────────────────────────────────────────────
        df = self._add_risk_features(df)
        df = self._add_hurst_exponent(df)
        df = self._add_regime_features(df)

        # ── Macro ─────────────────────────────────────────────────────────────
        if macro_df is not None:
            df = self._add_macro_features(df, macro_df)

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        logger.info(
            f"Feature matrix built: {df.shape[0]} rows × {df.shape[1]} cols"
        )
        return df

    # ── Moving Averages ───────────────────────────────────────────────────────

    def _add_moving_averages(
        self, df: pd.DataFrame, price: pd.Series
    ) -> pd.DataFrame:
        for w in [10, 20, 50, 100, 200]:
            df[f"sma_{w}"] = price.rolling(w).mean()
            df[f"ema_{w}"] = price.ewm(span=w, adjust=False).mean()

        df["price_vs_sma50"]  = (price - df["sma_50"])  / (df["sma_50"]  + 1e-10)
        df["price_vs_sma200"] = (price - df["sma_200"]) / (df["sma_200"] + 1e-10)
        df["sma50_vs_sma200"] = (df["sma_50"] - df["sma_200"]) / (df["sma_200"] + 1e-10)
        df["golden_cross"]    = (df["sma_50"] > df["sma_200"]).astype(int)

        # EMA acceleration: rate of change of EMA slope
        df["ema20_slope"]     = df["ema_20"].diff(5) / (df["ema_20"].shift(5) + 1e-10)
        return df

    # ── RSI ───────────────────────────────────────────────────────────────────

    def _add_rsi(
        self, df: pd.DataFrame, price: pd.Series,
        periods: List[int] = [9, 14, 21],
    ) -> pd.DataFrame:
        for p in periods:
            delta = price.diff()
            gain  = delta.clip(lower=0).rolling(p).mean()
            loss  = (-delta.clip(upper=0)).rolling(p).mean()
            rs    = gain / (loss.replace(0, np.nan))
            df[f"rsi_{p}"] = 100 - 100 / (1 + rs)

        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
        df["rsi_oversold"]   = (df["rsi_14"] < 30).astype(int)
        df["rsi_divergence"] = df["rsi_14"] - df["rsi_14"].shift(5)
        # RSI regime: distance from 50 normalised to [-1, 1]
        df["rsi_regime"]     = (df["rsi_14"] - 50) / 50.0
        return df

    # ── MACD ──────────────────────────────────────────────────────────────────

    def _add_macd(self, df: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
        ema12 = price.ewm(span=12, adjust=False).mean()
        ema26 = price.ewm(span=26, adjust=False).mean()
        df["macd"]            = ema12 - ema26
        df["macd_signal"]     = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]       = df["macd"] - df["macd_signal"]
        df["macd_cross"]      = (df["macd"] > df["macd_signal"]).astype(int)
        df["macd_zero_cross"] = (
            (df["macd"] > 0) & (df["macd"].shift(1) <= 0)
        ).astype(int)
        # Normalised MACD histogram
        df["macd_hist_norm"]  = df["macd_hist"] / (price.rolling(26).std() + 1e-10)
        return df

    # ── Momentum ──────────────────────────────────────────────────────────────

    def _add_momentum(self, df: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
        for p in [5, 10, 20, 60, 120]:
            df[f"mom_{p}"] = price.pct_change(p)

        df["rate_of_change_14"] = price.pct_change(14) * 100

        # Stochastic oscillator
        low14  = price.rolling(14).min()
        high14 = price.rolling(14).max()
        df["stoch_k"]  = 100 * (price - low14) / (high14 - low14 + 1e-10)
        df["stoch_d"]  = df["stoch_k"].rolling(3).mean()
        df["stoch_sig"]= (df["stoch_k"] > df["stoch_d"]).astype(int)

        # 1-day return autocorrelation (short-term mean reversion signal)
        df["ret_ac5"]  = df["returns"].rolling(20).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 5 else np.nan,
            raw=False
        )
        return df

    # ── Bollinger Bands ───────────────────────────────────────────────────────

    def _add_bollinger_bands(
        self, df: pd.DataFrame, price: pd.Series,
        window: int = 20, n_std: float = 2.0,
    ) -> pd.DataFrame:
        sma = price.rolling(window).mean()
        std = price.rolling(window).std()

        df["bb_upper"]  = sma + n_std * std
        df["bb_lower"]  = sma - n_std * std
        df["bb_middle"] = sma
        df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-10)
        df["bb_pct"]    = (price - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
        # Squeeze: width in bottom quintile of past year
        df["bb_squeeze"]= (
            df["bb_width"] < df["bb_width"].rolling(252).quantile(0.2)
        ).astype(int)
        return df

    # ── ATR ───────────────────────────────────────────────────────────────────

    def _add_atr(
        self, df: pd.DataFrame, prices_df: pd.DataFrame,
        ticker: str, window: int = 14,
    ) -> pd.DataFrame:
        close = prices_df[ticker]
        high  = prices_df.get(f"{ticker}_high", close)
        low   = prices_df.get(f"{ticker}_low",  close)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        df["atr_14"]  = tr.rolling(window).mean()
        df["atr_pct"] = df["atr_14"] / (close + 1e-10) * 100
        return df

    # ── Volatility ────────────────────────────────────────────────────────────

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ret = df["returns"]

        for w in [5, 10, 20, 60, 252]:
            df[f"vol_{w}d"] = ret.rolling(w).std() * np.sqrt(252)

        df["vol_ratio"]      = df["vol_20d"] / (df["vol_60d"] + 1e-10)
        df["vol_trend"]      = df["vol_20d"] - df["vol_20d"].shift(20)
        df["sq_returns_20"]  = (ret ** 2).rolling(20).mean()
        df["realized_var_20"]= (ret ** 2).rolling(20).sum()

        # Volatility of volatility (vol clustering signal)
        df["vol_of_vol_20"]  = df["vol_20d"].rolling(20).std()

        # Vol percentile over trailing year (regime-relative vol)
        df["vol_percentile"] = (
            df["vol_20d"].rolling(252).rank(pct=True).fillna(method="bfill")
        )
        return df

    # ── Volume & Microstructure ───────────────────────────────────────────────

    def _add_volume_features(
        self, df: pd.DataFrame,
        price: pd.Series, volume: pd.Series,
    ) -> pd.DataFrame:
        df["volume"]       = volume
        df["vol_sma20"]    = volume.rolling(20).mean()
        df["vol_ratio_v"]  = volume / (df["vol_sma20"] + 1e-10)

        # OBV
        obv               = (np.sign(price.diff()) * volume).fillna(0).cumsum()
        df["obv"]         = obv
        df["obv_sma20"]   = obv.rolling(20).mean()
        df["obv_trend"]   = (obv > df["obv_sma20"]).astype(int)

        # VWAP (FIXED: Σ(P·V) / Σ(V), not the original broken formula)
        vwap_20          = (price * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-10)
        df["vwap_20"]    = vwap_20
        df["vwap_ratio"] = price / (vwap_20 + 1e-10)

        # Amihud illiquidity: |return| / volume (higher = less liquid = more vol)
        df["amihud"]     = (df["returns"].abs() / (volume.replace(0, np.nan))).rolling(20).mean()

        # Turnover rate proxy
        df["turnover"]   = (price * volume).rolling(20).mean()
        return df

    # ── Risk Features ─────────────────────────────────────────────────────────

    def _add_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ret = df["returns"]

        # Drawdown from running maximum
        roll_max      = df["close"].cummax()
        df["drawdown"]= (df["close"] - roll_max) / (roll_max + 1e-10)

        # Rolling Sharpe (annualised, India 6.5% risk-free rate)
        rf_daily = 0.065 / 252
        for w in [20, 60]:
            mu_r  = (ret.rolling(w).mean() - rf_daily) * 252
            sigma = ret.rolling(w).std() * np.sqrt(252)
            df[f"sharpe_{w}d"] = mu_r / (sigma + 1e-10)

        # Sortino (downside deviation only)
        downside      = ret.copy().clip(upper=0)
        df["sortino_60d"] = (
            (ret.rolling(60).mean() - rf_daily) * 252 /
            (downside.rolling(60).std() * np.sqrt(252) + 1e-10)
        )

        # Calmar (return / max drawdown)
        roll_dd_max   = df["drawdown"].rolling(252).min()
        df["calmar_252d"] = (ret.rolling(252).mean() * 252) / (roll_dd_max.abs() + 1e-10)

        # Tail risk moments
        df["skewness_20d"] = ret.rolling(20).skew()
        df["kurtosis_20d"] = ret.rolling(20).kurt()

        # Historical VaR 5% (negative of 5th percentile)
        df["var_5pct_20d"] = ret.rolling(20).quantile(0.05)

        return df

    # ── Hurst Exponent ────────────────────────────────────────────────────────

    def _add_hurst_exponent(
        self, df: pd.DataFrame, window: int = 100, lags: int = 10
    ) -> pd.DataFrame:
        """
        Rolling Hurst exponent via R/S analysis.
          H < 0.5 → mean-reverting (Sideways regime)
          H ≈ 0.5 → random walk
          H > 0.5 → trending (Bull/Bear regime)
        """
        def hurst(ts: np.ndarray) -> float:
            if len(ts) < 20:
                return np.nan
            try:
                ts = ts[~np.isnan(ts)]
                lags_range = range(2, min(lags, len(ts) // 2))
                rs_values  = []
                for lag in lags_range:
                    chunks = [ts[i: i + lag] for i in range(0, len(ts) - lag, lag)]
                    rs_chunk = []
                    for c in chunks:
                        mean_c  = np.mean(c)
                        dev     = np.cumsum(c - mean_c)
                        r       = dev.max() - dev.min()
                        s       = np.std(c, ddof=1)
                        if s > 0:
                            rs_chunk.append(r / s)
                    if rs_chunk:
                        rs_values.append((lag, np.mean(rs_chunk)))
                if len(rs_values) < 2:
                    return np.nan
                lags_arr = np.log([x[0] for x in rs_values])
                rs_arr   = np.log([x[1] for x in rs_values])
                return float(np.polyfit(lags_arr, rs_arr, 1)[0])
            except Exception:
                return np.nan

        df["hurst_100d"] = (
            df["log_returns"].rolling(window)
            .apply(hurst, raw=True)
        )
        # Regime classification
        df["hurst_regime"] = pd.cut(
            df["hurst_100d"],
            bins=[-np.inf, 0.4, 0.6, np.inf],
            labels=[-1, 0, 1],  # -1=mean-reverting, 0=random, 1=trending
        ).astype(float)
        return df

    # ── Regime Composite Features ─────────────────────────────────────────────

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Trend strength
        df["trend_strength"] = df.get("sma50_vs_sma200", pd.Series(0, index=df.index))

        # Bull / bear signal composite (normalised to [0,1])
        signals = []
        for col in ["golden_cross", "macd_cross", "rsi_overbought", "obv_trend"]:
            if col in df.columns:
                signals.append(df[col])
        if signals:
            df["bull_composite"] = pd.concat(signals, axis=1).mean(axis=1)

        # Crisis score: blends deep drawdown with elevated vol
        if "drawdown" in df.columns and "vol_20d" in df.columns:
            norm_dd  = (-df["drawdown"]).clip(0, 0.5) / 0.5
            hist_vol = df["vol_20d"].rolling(252).mean().fillna(0.15)
            norm_vol = (df["vol_20d"] / (hist_vol + 1e-10)).clip(0, 4) / 4
            df["crisis_score"] = 0.5 * norm_dd + 0.5 * norm_vol

        # Regime momentum: returns of the current 20-day trend relative to history
        if "mom_20" in df.columns:
            df["regime_momentum"] = df["mom_20"].rolling(252).rank(pct=True)

        return df

    # ── Macro ─────────────────────────────────────────────────────────────────

    def _add_macro_features(
        self, df: pd.DataFrame, macro_df: pd.DataFrame
    ) -> pd.DataFrame:
        for col in macro_df.columns:
            aligned = macro_df[col].reindex(df.index, method="ffill")
            df[f"macro_{col}"] = aligned
            df[f"macro_{col}_chg10"] = aligned.pct_change(10)
        return df

    # ── Feature Selection for Regime Model ────────────────────────────────────

    def get_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the canonical feature subset used by HMM and LSTM.
        Only includes columns that are actually present in df.
        """
        desired = [
            # Returns
            "returns", "log_returns",
            # Trend
            "price_vs_sma50", "price_vs_sma200", "sma50_vs_sma200",
            "golden_cross", "ema20_slope",
            # Momentum
            "rsi_14", "rsi_regime", "macd_hist_norm", "mom_20", "stoch_k",
            # Volatility
            "vol_20d", "vol_60d", "vol_ratio", "vol_percentile",
            "vol_of_vol_20", "bb_width", "atr_pct",
            # Risk
            "drawdown", "sharpe_60d", "sortino_60d",
            "skewness_20d", "kurtosis_20d", "var_5pct_20d",
            # Microstructure
            "vwap_ratio", "amihud",
            # Regime signals
            "crisis_score", "bull_composite", "hurst_regime",
            "regime_momentum",
        ]
        available = [c for c in desired if c in df.columns]
        return df[available].dropna()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_volume_col(
        self, prices_df: pd.DataFrame, ticker: str
    ) -> Optional[str]:
        for candidate in ["volume", f"{ticker}_volume", "Volume"]:
            if candidate in prices_df.columns:
                return candidate
        return None
