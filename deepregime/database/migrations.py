# ============================================================
# database/migrations.py
# Database Migration Helpers (Alembic-compatible)
# ============================================================

import asyncio
from loguru import logger
from database.connection import engine, init_db
from database.models import Base


async def create_tables():
    """Create all database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ All tables created")


async def drop_tables():
    """Drop all database tables (CAUTION: destroys data)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("⚠️  All tables dropped")


async def reset_database():
    """Full reset: drop + recreate all tables."""
    await drop_tables()
    await create_tables()
    logger.info("✅ Database reset complete")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "create"
    if cmd == "create":
        asyncio.run(create_tables())
    elif cmd == "drop":
        asyncio.run(drop_tables())
    elif cmd == "reset":
        asyncio.run(reset_database())
    elif cmd == "init":
        asyncio.run(init_db())
