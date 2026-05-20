# ============================================================
# config/settings.py
# Centralised configuration — all env-overridable via .env
# ============================================================

from __future__ import annotations
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────
    APP_NAME:    str = "DeepRegime"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = "development"   # development | production

    # ── Security ─────────────────────────────────────────────
    SECRET_KEY:      str = "change-me-in-production-32-chars!!"
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8501"

    # ── Database ─────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./deepregime.db"
    REDIS_URL:    str = "redis://localhost:6379/0"

    # ── HMM ──────────────────────────────────────────────────
    HMM_N_COMPONENTS:    int = 6
    HMM_COVARIANCE_TYPE: str = "full"
    HMM_N_ITER:          int = 300

    # ── LSTM ─────────────────────────────────────────────────
    LSTM_LOOKBACK_DAYS: int   = 60
    LSTM_HIDDEN_UNITS:  int   = 128
    LSTM_DROPOUT:       float = 0.2
    LSTM_EPOCHS:        int   = 100
    LSTM_BATCH_SIZE:    int   = 64

    # ── Kalman ───────────────────────────────────────────────
    KALMAN_PROCESS_NOISE: float = 1e-3
    KALMAN_OBS_NOISE:     float = 1e-1

    # ── Backtesting ──────────────────────────────────────────
    BACKTEST_START_DATE:      str   = "2015-01-01"
    INITIAL_CAPITAL:          float = 1_000_000.0
    TRANSACTION_COST_BPS:     float = 10.0
    SLIPPAGE_BPS:             float = 5.0
    WALK_FORWARD_FOLDS:       int   = 5

    # ── Live data ─────────────────────────────────────────────
    LIVE_DATA_INTERVAL_SECONDS: int = 60
    DEBUG: bool = False

    # ── External APIs ─────────────────────────────────────────
    ALPHA_VANTAGE_KEY: str = ""
    FINNHUB_KEY:       str = ""
    NEWS_API_KEY:      str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()


# ── Asset universe ────────────────────────────────────────────────────────────

INDIAN_STOCKS: List[str] = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "AXISBANK.NS", "LT.NS", "ITC.NS", "BHARTIARTL.NS",
    "ASIANPAINT.NS", "SUNPHARMA.NS", "KOTAKBANK.NS", "HINDUNILVR.NS",
    "BAJFINANCE.NS",
]

INDIAN_ETFS: List[str] = [
    "NIFTYBEES.NS", "BANKBEES.NS", "ITBEES.NS", "GOLDBEES.NS", "CPSEETF.NS",
]

CRYPTO_ASSETS: List[str] = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]

COMMODITY_ASSETS: List[str] = ["GC=F", "SI=F", "CL=F"]   # Gold, Silver, Crude

INDEX_ASSETS: List[str] = ["^NSEI", "^BSESN", "^NSEBANK", "^INDIAVIX", "USDINR=X"]

ALL_ASSETS: List[str] = (
    INDIAN_STOCKS + INDIAN_ETFS + CRYPTO_ASSETS + COMMODITY_ASSETS + INDEX_ASSETS
)


# ── Regime definitions ────────────────────────────────────────────────────────

REGIME_LABELS: List[str] = [
    "Bull Market",
    "Bear Market",
    "Sideways / Range-bound",
    "High Volatility",
    "Recovery",
    "Crisis / Panic",
]

REGIME_COLORS: dict = {
    "Bull Market":            "#00C851",
    "Bull Market II":         "#00A844",
    "Bear Market":            "#FF4444",
    "Bear Market II":         "#CC3333",
    "Sideways / Range-bound": "#FFB300",
    "High Volatility":        "#FF6D00",
    "Recovery":               "#00BCD4",
    "Crisis / Panic":         "#880E4F",
    "Uncertain":              "#9E9E9E",
}


# ── Regime portfolio templates ────────────────────────────────────────────────
# Target allocation fractions per asset class per regime.
# Rows sum to ~1.0; actual weights depend on available assets.

REGIME_ALLOCATION_TEMPLATES: dict = {
    "Bull Market": {
        "equities":    0.65,
        "etfs":        0.20,
        "crypto":      0.10,
        "commodities": 0.05,
    },
    "Bear Market": {
        "equities":    0.20,
        "etfs":        0.40,
        "crypto":      0.02,
        "commodities": 0.38,
    },
    "Crisis / Panic": {
        "equities":    0.10,
        "etfs":        0.30,
        "crypto":      0.01,
        "commodities": 0.59,
    },
    "High Volatility": {
        "equities":    0.30,
        "etfs":        0.35,
        "crypto":      0.05,
        "commodities": 0.30,
    },
    "Recovery": {
        "equities":    0.50,
        "etfs":        0.30,
        "crypto":      0.08,
        "commodities": 0.12,
    },
    "Sideways / Range-bound": {
        "equities":    0.40,
        "etfs":        0.35,
        "crypto":      0.07,
        "commodities": 0.18,
    },
}
