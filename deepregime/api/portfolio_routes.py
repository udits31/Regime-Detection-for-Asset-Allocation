# ============================================================
# api/portfolio_routes.py
# Portfolio Management API Routes
# ============================================================

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from database.connection import get_db
from database.models import Portfolio, PortfolioAllocation, Trade, RiskMetric, MarketRegime
from api.auth import get_current_user

router = APIRouter(prefix="/api/portfolios", tags=["Portfolio Management"])


# ── Schemas ────────────────────────────────────────────────

class PortfolioCreate(BaseModel):
    name: str
    strategy: str = "adaptive_ai"
    initial_capital: float = 1_000_000.0

class TradeCreate(BaseModel):
    asset_ticker: str
    action: str          # BUY / SELL
    shares: float
    price: float
    rationale: Optional[str] = None


# ── Routes ─────────────────────────────────────────────────

@router.get("/", response_model=List[dict])
async def list_portfolios(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """List all portfolios for current user."""
    result = await db.execute(
        select(Portfolio)
        .where(Portfolio.user_id == current_user.id)
        .order_by(desc(Portfolio.created_at))
    )
    portfolios = result.scalars().all()
    return [
        {
            "id":              p.id,
            "name":            p.name,
            "strategy":        p.strategy,
            "initial_capital": p.initial_capital,
            "current_value":   p.current_value or p.initial_capital,
            "return_pct":      (
                (p.current_value - p.initial_capital) / p.initial_capital * 100
                if p.current_value else 0.0
            ),
            "is_active":       p.is_active,
            "created_at":      p.created_at.isoformat() if p.created_at else None,
        }
        for p in portfolios
    ]


@router.post("/", status_code=201)
async def create_portfolio(
    payload: PortfolioCreate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Create a new portfolio."""
    portfolio = Portfolio(
        user_id=current_user.id,
        name=payload.name,
        strategy=payload.strategy,
        initial_capital=payload.initial_capital,
        current_value=payload.initial_capital,
        cash_balance=payload.initial_capital,
        is_active=True,
    )
    db.add(portfolio)
    await db.commit()
    await db.refresh(portfolio)
    return {
        "id":      portfolio.id,
        "name":    portfolio.name,
        "message": "Portfolio created successfully"
    }


@router.get("/{portfolio_id}")
async def get_portfolio(
    portfolio_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Get detailed portfolio information."""
    result = await db.execute(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == current_user.id
        )
    )
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(404, "Portfolio not found")

    # Get latest allocations
    alloc_result = await db.execute(
        select(PortfolioAllocation)
        .where(PortfolioAllocation.portfolio_id == portfolio_id)
        .order_by(desc(PortfolioAllocation.timestamp))
        .limit(50)
    )
    allocations = alloc_result.scalars().all()

    # Get recent trades
    trade_result = await db.execute(
        select(Trade)
        .where(Trade.portfolio_id == portfolio_id)
        .order_by(desc(Trade.timestamp))
        .limit(20)
    )
    trades = trade_result.scalars().all()

    return {
        "portfolio": {
            "id":              portfolio.id,
            "name":            portfolio.name,
            "strategy":        portfolio.strategy,
            "initial_capital": portfolio.initial_capital,
            "current_value":   portfolio.current_value,
            "cash_balance":    portfolio.cash_balance,
            "return_pct":      (
                (portfolio.current_value - portfolio.initial_capital) / portfolio.initial_capital * 100
                if portfolio.current_value else 0.0
            ),
        },
        "allocations": [
            {
                "asset_id":     a.asset_id,
                "target_weight": a.target_weight,
                "actual_weight": a.actual_weight,
                "market_value": a.market_value,
                "regime":       a.regime_at_allocation,
                "timestamp":    a.timestamp.isoformat() if a.timestamp else None,
            }
            for a in allocations
        ],
        "recent_trades": [
            {
                "asset_id":  t.asset_id,
                "action":    t.action,
                "shares":    t.shares,
                "price":     t.price,
                "value":     t.value,
                "timestamp": t.timestamp.isoformat() if t.timestamp else None,
            }
            for t in trades
        ]
    }


@router.post("/{portfolio_id}/trades")
async def record_trade(
    portfolio_id: int,
    payload: TradeCreate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Record a trade in the portfolio."""
    result = await db.execute(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == current_user.id
        )
    )
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(404, "Portfolio not found")

    trade_value = payload.shares * payload.price
    commission  = trade_value * 0.001  # 10bps

    trade = Trade(
        portfolio_id=portfolio_id,
        asset_id=1,                      # Simplified; real impl looks up asset by ticker
        timestamp=datetime.utcnow(),
        action=payload.action.upper(),
        shares=payload.shares,
        price=payload.price,
        value=trade_value,
        commission=commission,
        rationale=payload.rationale,
    )
    db.add(trade)

    # Update portfolio value
    if payload.action.upper() == "BUY":
        portfolio.cash_balance = (portfolio.cash_balance or 0) - trade_value - commission
    else:
        portfolio.cash_balance = (portfolio.cash_balance or 0) + trade_value - commission

    await db.commit()
    return {
        "message":   "Trade recorded",
        "trade_id":  trade.id,
        "value":     round(trade_value, 2),
        "commission": round(commission, 2),
    }


@router.get("/{portfolio_id}/performance")
async def get_portfolio_performance(
    portfolio_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Get portfolio performance metrics."""
    result = await db.execute(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == current_user.id
        )
    )
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(404, "Portfolio not found")

    # Get latest risk metrics
    risk_result = await db.execute(
        select(RiskMetric)
        .where(RiskMetric.portfolio_id == portfolio_id)
        .order_by(desc(RiskMetric.timestamp))
        .limit(1)
    )
    risk = risk_result.scalar_one_or_none()

    return {
        "portfolio_id":    portfolio_id,
        "current_value":   portfolio.current_value,
        "initial_capital": portfolio.initial_capital,
        "total_return":    (
            (portfolio.current_value - portfolio.initial_capital) / portfolio.initial_capital
            if portfolio.current_value else 0
        ),
        "risk_metrics": {
            "sharpe_ratio":   risk.sharpe_ratio if risk else None,
            "sortino_ratio":  risk.sortino_ratio if risk else None,
            "max_drawdown":   risk.max_drawdown if risk else None,
            "portfolio_vol":  risk.portfolio_volatility if risk else None,
            "var_95":         risk.var_95 if risk else None,
        } if risk else {}
    }


@router.delete("/{portfolio_id}")
async def delete_portfolio(
    portfolio_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Soft-delete a portfolio."""
    result = await db.execute(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == current_user.id
        )
    )
    portfolio = result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(404, "Portfolio not found")

    portfolio.is_active = False
    await db.commit()
    return {"message": "Portfolio deactivated"}
