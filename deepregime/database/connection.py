# ============================================================
# database/connection.py
# Async Database Engine, Session, and Initialization
# ============================================================

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text
from loguru import logger
from pathlib import Path

from database.models import Base
from config.settings import settings

# Ensure DB directory exists
Path("database").mkdir(exist_ok=True)

# ── Async Engine ───────────────────────────────────────────
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ── Dependency ─────────────────────────────────────────────
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Init DB ────────────────────────────────────────────────
async def init_db():
    """Create all tables and seed initial asset data."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables created successfully")
    await seed_assets()


async def seed_assets():
    """Seed the assets table with our universe of instruments."""
    from database.models import Asset, AssetClass
    from config.settings import (
        INDIAN_STOCKS, INDIAN_ETFS, CRYPTO_ASSETS, COMMODITY_ASSETS
    )
    from sqlalchemy import select

    ASSET_METADATA = {
        # Indian Stocks
        "RELIANCE.NS":  ("Reliance Industries",  AssetClass.EQUITY,    "Energy",        "NSE"),
        "TCS.NS":       ("Tata Consultancy Svc",  AssetClass.EQUITY,    "IT",            "NSE"),
        "INFY.NS":      ("Infosys",               AssetClass.EQUITY,    "IT",            "NSE"),
        "HDFCBANK.NS":  ("HDFC Bank",             AssetClass.EQUITY,    "Banking",       "NSE"),
        "ICICIBANK.NS": ("ICICI Bank",            AssetClass.EQUITY,    "Banking",       "NSE"),
        "SBIN.NS":      ("State Bank of India",   AssetClass.EQUITY,    "Banking",       "NSE"),
        "AXISBANK.NS":  ("Axis Bank",             AssetClass.EQUITY,    "Banking",       "NSE"),
        "LT.NS":        ("Larsen & Toubro",       AssetClass.EQUITY,    "Infrastructure","NSE"),
        "ITC.NS":       ("ITC Limited",           AssetClass.EQUITY,    "FMCG",          "NSE"),
        "BHARTIARTL.NS":("Bharti Airtel",         AssetClass.EQUITY,    "Telecom",       "NSE"),
        "ASIANPAINT.NS":("Asian Paints",          AssetClass.EQUITY,    "FMCG",          "NSE"),
        "SUNPHARMA.NS": ("Sun Pharma",            AssetClass.EQUITY,    "Pharma",        "NSE"),
        # ETFs
        "NIFTYBEES.NS": ("Nifty BeES ETF",        AssetClass.ETF,       "Index",         "NSE"),
        "BANKBEES.NS":  ("Bank BeES ETF",         AssetClass.ETF,       "Banking",       "NSE"),
        "ITBEES.NS":    ("IT BeES ETF",           AssetClass.ETF,       "IT",            "NSE"),
        "GOLDBEES.NS":  ("Gold BeES ETF",         AssetClass.ETF,       "Commodity",     "NSE"),
        "CPSEETF.NS":   ("CPSE ETF",              AssetClass.ETF,       "PSU",           "NSE"),
        # Crypto
        "BTC-USD":      ("Bitcoin",               AssetClass.CRYPTO,    "Crypto",        "CRYPTO"),
        "ETH-USD":      ("Ethereum",              AssetClass.CRYPTO,    "Crypto",        "CRYPTO"),
        "SOL-USD":      ("Solana",                AssetClass.CRYPTO,    "Crypto",        "CRYPTO"),
        "BNB-USD":      ("BNB",                   AssetClass.CRYPTO,    "Crypto",        "CRYPTO"),
        # Commodities
        "GC=F":         ("Gold Futures",          AssetClass.COMMODITY, "Commodity",     "CME"),
        "SI=F":         ("Silver Futures",        AssetClass.COMMODITY, "Commodity",     "CME"),
        "CL=F":         ("Crude Oil WTI",         AssetClass.COMMODITY, "Commodity",     "CME"),
    }

    async with AsyncSessionLocal() as session:
        for ticker, (name, asset_class, sector, exchange) in ASSET_METADATA.items():
            result = await session.execute(
                select(Asset).where(Asset.ticker == ticker)
            )
            if not result.scalar_one_or_none():
                currency = "INR" if ".NS" in ticker else "USD"
                country  = "India" if ".NS" in ticker else "Global"
                asset = Asset(
                    ticker=ticker, name=name, asset_class=asset_class,
                    sector=sector, exchange=exchange, currency=currency, country=country
                )
                session.add(asset)
        await session.commit()
    logger.info("✅ Asset universe seeded")
