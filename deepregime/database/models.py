# ============================================================
# database/models.py
# SQLAlchemy ORM Models — Full Database Schema
# ============================================================

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, Text, JSON, Enum, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()


# ── Enums ──────────────────────────────────────────────────

class MarketRegime(str, enum.Enum):
    BULL         = "Bull Market"
    BEAR         = "Bear Market"
    SIDEWAYS     = "Sideways / Range-bound"
    HIGH_VOL     = "High Volatility"
    RECOVERY     = "Recovery Phase"
    CRISIS       = "Crisis / Panic Phase"

class AssetClass(str, enum.Enum):
    EQUITY       = "equity"
    ETF          = "etf"
    CRYPTO       = "crypto"
    COMMODITY    = "commodity"
    FIXED_INCOME = "fixed_income"
    REIT         = "reit"

class ModelType(str, enum.Enum):
    HMM          = "hmm"
    LSTM         = "lstm"
    ENSEMBLE     = "ensemble"


# ── Users ──────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id             = Column(Integer, primary_key=True, index=True)
    username       = Column(String(50), unique=True, nullable=False)
    email          = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active      = Column(Boolean, default=True)
    is_admin       = Column(Boolean, default=False)
    risk_profile   = Column(String(20), default="moderate")  # conservative/moderate/aggressive
    initial_capital = Column(Float, default=1_000_000.0)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())
    updated_at     = Column(DateTime(timezone=True), onupdate=func.now())

    portfolios     = relationship("Portfolio", back_populates="user")
    watchlists     = relationship("Watchlist", back_populates="user")


# ── Assets ─────────────────────────────────────────────────

class Asset(Base):
    __tablename__ = "assets"

    id             = Column(Integer, primary_key=True, index=True)
    ticker         = Column(String(20), unique=True, nullable=False, index=True)
    name           = Column(String(100))
    asset_class    = Column(Enum(AssetClass), nullable=False)
    sector         = Column(String(50))
    exchange       = Column(String(20))
    currency       = Column(String(10), default="INR")
    country        = Column(String(50), default="India")
    is_active      = Column(Boolean, default=True)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())

    market_data    = relationship("MarketData", back_populates="asset")
    tech_indicators = relationship("TechnicalIndicator", back_populates="asset")

    __table_args__ = (
        Index("ix_assets_class_sector", "asset_class", "sector"),
    )


# ── Market Data ────────────────────────────────────────────

class MarketData(Base):
    __tablename__ = "market_data"

    id             = Column(Integer, primary_key=True, index=True)
    asset_id       = Column(Integer, ForeignKey("assets.id"), nullable=False)
    timestamp      = Column(DateTime(timezone=True), nullable=False)
    open           = Column(Float)
    high           = Column(Float)
    low            = Column(Float)
    close          = Column(Float, nullable=False)
    volume         = Column(Float)
    adj_close      = Column(Float)
    returns        = Column(Float)
    log_returns    = Column(Float)
    data_source    = Column(String(30), default="yfinance")
    is_live        = Column(Boolean, default=False)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())

    asset          = relationship("Asset", back_populates="market_data")

    __table_args__ = (
        Index("ix_market_data_asset_ts", "asset_id", "timestamp"),
    )


# ── Technical Indicators ───────────────────────────────────

class TechnicalIndicator(Base):
    __tablename__ = "technical_indicators"

    id             = Column(Integer, primary_key=True, index=True)
    asset_id       = Column(Integer, ForeignKey("assets.id"), nullable=False)
    timestamp      = Column(DateTime(timezone=True), nullable=False)

    # Trend
    sma_20         = Column(Float)
    sma_50         = Column(Float)
    sma_200        = Column(Float)
    ema_12         = Column(Float)
    ema_26         = Column(Float)

    # Momentum
    rsi_14         = Column(Float)
    macd           = Column(Float)
    macd_signal    = Column(Float)
    macd_hist      = Column(Float)
    momentum_10    = Column(Float)

    # Volatility
    bb_upper       = Column(Float)
    bb_lower       = Column(Float)
    bb_middle      = Column(Float)
    bb_bandwidth   = Column(Float)
    atr_14         = Column(Float)

    # Volume
    obv            = Column(Float)
    vwap           = Column(Float)

    # Risk
    rolling_vol_20 = Column(Float)
    rolling_vol_60 = Column(Float)
    drawdown       = Column(Float)

    asset          = relationship("Asset", back_populates="tech_indicators")

    __table_args__ = (
        Index("ix_tech_ind_asset_ts", "asset_id", "timestamp"),
    )


# ── Regime Predictions ─────────────────────────────────────

class RegimePrediction(Base):
    __tablename__ = "regime_predictions"

    id               = Column(Integer, primary_key=True, index=True)
    timestamp        = Column(DateTime(timezone=True), nullable=False)
    detected_regime  = Column(Enum(MarketRegime), nullable=False)
    predicted_regime = Column(Enum(MarketRegime))
    confidence_score = Column(Float)
    model_type       = Column(Enum(ModelType))
    volatility_level = Column(String(20))   # low/medium/high/extreme
    risk_score       = Column(Float)        # 0-100
    regime_duration_days = Column(Integer)  # How long current regime has lasted
    transition_probability = Column(JSON)   # {regime: probability}
    hmm_state_probs  = Column(JSON)
    feature_snapshot = Column(JSON)         # Key features at detection time
    created_at       = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_regime_pred_ts", "timestamp"),
    )


# ── Portfolio ──────────────────────────────────────────────

class Portfolio(Base):
    __tablename__ = "portfolios"

    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    name           = Column(String(100), nullable=False)
    strategy       = Column(String(50), default="adaptive_ai")
    initial_capital = Column(Float, nullable=False)
    current_value  = Column(Float)
    cash_balance   = Column(Float)
    is_active      = Column(Boolean, default=True)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())
    updated_at     = Column(DateTime(timezone=True), onupdate=func.now())

    user           = relationship("User", back_populates="portfolios")
    allocations    = relationship("PortfolioAllocation", back_populates="portfolio")
    trades         = relationship("Trade", back_populates="portfolio")


# ── Portfolio Allocations ──────────────────────────────────

class PortfolioAllocation(Base):
    __tablename__ = "portfolio_allocations"

    id             = Column(Integer, primary_key=True, index=True)
    portfolio_id   = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    asset_id       = Column(Integer, ForeignKey("assets.id"), nullable=False)
    timestamp      = Column(DateTime(timezone=True), nullable=False)
    target_weight  = Column(Float, nullable=False)   # 0-1
    actual_weight  = Column(Float)
    shares         = Column(Float)
    market_value   = Column(Float)
    regime_at_allocation = Column(Enum(MarketRegime))
    rebalance_trigger = Column(String(50))           # regime_change/threshold/scheduled
    created_at     = Column(DateTime(timezone=True), server_default=func.now())

    portfolio      = relationship("Portfolio", back_populates="allocations")

    __table_args__ = (
        Index("ix_alloc_portfolio_ts", "portfolio_id", "timestamp"),
    )


# ── Trades ─────────────────────────────────────────────────

class Trade(Base):
    __tablename__ = "trades"

    id             = Column(Integer, primary_key=True, index=True)
    portfolio_id   = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    asset_id       = Column(Integer, ForeignKey("assets.id"), nullable=False)
    timestamp      = Column(DateTime(timezone=True), nullable=False)
    action         = Column(String(10), nullable=False)  # BUY/SELL/REBALANCE
    shares         = Column(Float, nullable=False)
    price          = Column(Float, nullable=False)
    value          = Column(Float, nullable=False)
    commission     = Column(Float, default=0.0)
    regime_at_trade = Column(Enum(MarketRegime))
    rationale      = Column(Text)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())

    portfolio      = relationship("Portfolio", back_populates="trades")


# ── Risk Metrics ───────────────────────────────────────────

class RiskMetric(Base):
    __tablename__ = "risk_metrics"

    id             = Column(Integer, primary_key=True, index=True)
    portfolio_id   = Column(Integer, ForeignKey("portfolios.id"))
    timestamp      = Column(DateTime(timezone=True), nullable=False)
    sharpe_ratio   = Column(Float)
    sortino_ratio  = Column(Float)
    calmar_ratio   = Column(Float)
    max_drawdown   = Column(Float)
    current_drawdown = Column(Float)
    var_95         = Column(Float)   # Value at Risk 95%
    cvar_95        = Column(Float)   # Conditional VaR
    beta           = Column(Float)
    alpha          = Column(Float)
    portfolio_volatility = Column(Float)
    annual_return  = Column(Float)
    cagr           = Column(Float)
    portfolio_turnover = Column(Float)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())


# ── Backtest Results ───────────────────────────────────────

class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id             = Column(Integer, primary_key=True, index=True)
    strategy_name  = Column(String(100), nullable=False)
    start_date     = Column(DateTime(timezone=True))
    end_date       = Column(DateTime(timezone=True))
    initial_capital = Column(Float)
    final_value    = Column(Float)
    total_return   = Column(Float)
    cagr           = Column(Float)
    sharpe_ratio   = Column(Float)
    sortino_ratio  = Column(Float)
    max_drawdown   = Column(Float)
    volatility     = Column(Float)
    calmar_ratio   = Column(Float)
    win_rate       = Column(Float)
    total_trades   = Column(Integer)
    portfolio_turnover = Column(Float)
    equity_curve   = Column(JSON)    # [{date, value}, ...]
    regime_returns = Column(JSON)    # {regime: return}
    monthly_returns = Column(JSON)
    benchmark_comparison = Column(JSON)
    parameters     = Column(JSON)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())


# ── Macro Data ─────────────────────────────────────────────

class MacroData(Base):
    __tablename__ = "macro_data"

    id             = Column(Integer, primary_key=True, index=True)
    timestamp      = Column(DateTime(timezone=True), nullable=False)
    india_vix      = Column(Float)
    nifty_50       = Column(Float)
    sensex         = Column(Float)
    usd_inr        = Column(Float)
    repo_rate      = Column(Float)
    cpi_yoy        = Column(Float)
    gdp_growth     = Column(Float)
    fii_flow       = Column(Float)    # Foreign Institutional Investor flows
    dii_flow       = Column(Float)    # Domestic Institutional Investor flows
    created_at     = Column(DateTime(timezone=True), server_default=func.now())


# ── Watchlist ──────────────────────────────────────────────

class Watchlist(Base):
    __tablename__ = "watchlists"

    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    asset_id       = Column(Integer, ForeignKey("assets.id"), nullable=False)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())

    user           = relationship("User", back_populates="watchlists")
