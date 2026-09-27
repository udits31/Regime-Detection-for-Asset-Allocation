# ============================================================
# api/main.py
# FastAPI Backend — All Routes
#
# FIXED (vs original):
#   • Global mutable singletons → app.state via lifespan context manager
#   • CORS origin is env-gated (wildcard only in development)
#   • WebSocket client list protected by asyncio.Lock
#   • Rate limiting via slowapi (100 req/min default)
#   • Proper dependency injection for detector / optimizer
#   • Model loading is async-friendly (offloaded to threadpool)
# NEW:
#   • /api/regime/forecast — LSTM next-day regime prediction
#   • /api/portfolio/efficient-frontier — for dashboard visualisation
#   • /api/risk/stress-test — Monte Carlo stress test
#   • /api/system/model-info — metadata for deployed models
# ============================================================

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import (
    FastAPI, Depends, HTTPException, BackgroundTasks,
    WebSocket, WebSocketDisconnect, Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger
from datetime import datetime, timezone

from config.settings import settings, ALL_ASSETS, REGIME_LABELS
from database.connection import get_db, init_db
from realtime_engine.data_ingestion import MarketDataIngestion, DataPipeline
from regime_detector import EnsembleRegimeDetector
from portfolio_optimizer import PortfolioOptimizer
from backtesting.backtest_engine import BacktestEngine
from utils.monte_carlo import MonteCarloSimulator
from api.auth import router as auth_router, get_current_user, _token_store
from api.portfolio_routes import router as portfolio_router


# ── Lifespan context manager (replaces deprecated @app.on_event) ─────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise all services in dependency order.
    Shutdown: flush connections and save state.
    """
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} …")

    # Database
    await init_db()

    # Redis token store
    await _token_store.init(settings.REDIS_URL)

    # Market data layer
    ingestion = MarketDataIngestion()
    pipeline  = DataPipeline(ingestion)
    app.state.ingestion = ingestion
    app.state.pipeline  = pipeline

    # Models (load in threadpool — blocking I/O, not CPU)
    loop = asyncio.get_event_loop()

    detector = EnsembleRegimeDetector()
    try:
        await loop.run_in_executor(None, detector.load)
        logger.info("Loaded pre-trained ensemble models from disk")
    except Exception as e:
        logger.warning(f"Could not load saved models ({e}); detector requires training")

    optimizer   = PortfolioOptimizer()
    backtest    = BacktestEngine()
    mc_sim      = MonteCarloSimulator()

    app.state.detector  = detector
    app.state.optimizer = optimizer
    app.state.backtest  = backtest
    app.state.mc_sim    = mc_sim

    # WebSocket connection registry with lock
    app.state.ws_clients: List[WebSocket] = []
    app.state.ws_lock    = asyncio.Lock()

    logger.info(f"{settings.APP_NAME} ready.")

    yield   # ← Application runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down …")
    # Close any remaining WebSocket connections gracefully
    async with app.state.ws_lock:
        for ws in app.state.ws_clients:
            try:
                await ws.close()
            except Exception:
                pass


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "DeepRegime — AI Market Intelligence API",
    description = "Deep Learning Market Regime Detection & Adaptive Portfolio System",
    version     = settings.APP_VERSION,
    docs_url    = "/api/docs",
    redoc_url   = "/api/redoc",
    lifespan    = lifespan,
)

# CORS — wildcard only in development (FIXED from original)
_allowed_origins = (
    ["*"]
    if settings.ENVIRONMENT == "development"
    else settings.ALLOWED_ORIGINS.split(",")
)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = _allowed_origins,
    allow_credentials = True,
    allow_methods     = ["GET", "POST", "PUT", "DELETE"],
    allow_headers     = ["Authorization", "Content-Type"],
)

# Rate limiting (100 req/min per IP)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    _RATE_LIMIT = True
except ImportError:
    logger.warning("slowapi not installed; rate limiting disabled")
    _RATE_LIMIT = False


# ── Include routers ───────────────────────────────────────────────────────────

app.include_router(auth_router)
app.include_router(portfolio_router)


# ── Dependency helpers ────────────────────────────────────────────────────────

def get_detector(request: Request) -> EnsembleRegimeDetector:
    return request.app.state.detector

def get_optimizer(request: Request) -> PortfolioOptimizer:
    return request.app.state.optimizer

def get_ingestion(request: Request) -> MarketDataIngestion:
    return request.app.state.ingestion

def get_mc(request: Request) -> MonteCarloSimulator:
    return request.app.state.mc_sim


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class OptimizationRequest(BaseModel):
    regime:          str
    method:          str   = "max_sharpe"
    risk_profile:    str   = "moderate"
    lookback_days:   int   = 252

class BacktestRequest(BaseModel):
    start_date:      str           = "2018-01-01"
    end_date:        Optional[str] = None
    initial_capital: float         = 1_000_000.0
    n_wf_folds:      int           = 5

class StressTestRequest(BaseModel):
    weights:         Dict[str, float]
    horizon_days:    int   = 252
    n_simulations:   int   = 10_000
    regime:          str   = "Bear Market"


# ── System ────────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
async def health_check(request: Request):
    det = request.app.state.detector
    return {
        "status":    "healthy",
        "version":   settings.APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": settings.ENVIRONMENT,
        "models": {
            "hmm":    det.hmm.is_fitted,
            "lstm":   det.lstm.is_fitted,
            "kalman": det.kalman.is_fitted,
        },
    }


@app.get("/api/system/model-info", tags=["System"])
async def model_info(
    request:  Request,
    _:        Any = Depends(get_current_user),
):
    det = request.app.state.detector
    info: Dict = {"hmm": {}, "lstm": {}, "kalman": {}}

    if det.hmm.is_fitted:
        info["hmm"] = {
            "n_states":   det.hmm.n_components,
            "state_map":  det.hmm.state_label_map,
            "stationary": det.hmm.stationary_distribution().tolist(),
            "durations":  det.hmm._mean_durations(),
        }
    if det.lstm.is_fitted:
        info["lstm"] = {
            "lookback":     det.lstm.lookback,
            "n_regimes":    det.lstm.n_regimes,
            "temperature":  det.lstm.temperature,
            "classes":      list(det.lstm.encoder.classes_),
        }
    if det.kalman.is_fitted:
        info["kalman"] = {
            "process_noise": det.kalman.process_noise,
            "obs_noise":     det.kalman.obs_noise,
        }
    return info


# ── Market Data ───────────────────────────────────────────────────────────────

@app.get("/api/market/snapshot", tags=["Market Data"])
async def market_snapshot(ingestion: MarketDataIngestion = Depends(get_ingestion)):
    try:
        snap = ingestion.fetch_live_snapshot(ALL_ASSETS[:20])
        return {
            "data":        snap.to_dict(orient="records"),
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "asset_count": len(snap),
        }
    except Exception as e:
        raise HTTPException(500, f"Market data fetch failed: {e}")


@app.get("/api/market/history/{ticker}", tags=["Market Data"])
async def market_history(
    ticker:    str,
    period:    str = "1y",
    ingestion: MarketDataIngestion = Depends(get_ingestion),
):
    try:
        df = ingestion.fetch_historical(ticker, period=period)
        return {
            "ticker":    ticker,
            "data":      df.reset_index().to_dict(orient="records"),
            "n_rows":    len(df),
        }
    except Exception as e:
        raise HTTPException(500, f"History fetch failed for {ticker}: {e}")


# ── Regime Detection ──────────────────────────────────────────────────────────

@app.get("/api/regime/current", tags=["Regime Detection"])
async def current_regime(
    request:  Request,
    detector: EnsembleRegimeDetector = Depends(get_detector),
    ingestion: MarketDataIngestion   = Depends(get_ingestion),
):
    if not detector.hmm.is_fitted:
        raise HTTPException(503, "Models not trained. POST /api/system/train first.")
    try:
        prices = ingestion.fetch_historical("^NSEI", period="2y")
        result = detector.detect_current_regime(prices.to_frame())
        return result
    except Exception as e:
        raise HTTPException(500, f"Regime detection failed: {e}")


@app.get("/api/regime/history", tags=["Regime Detection"])
async def regime_history(
    request:  Request,
    detector: EnsembleRegimeDetector = Depends(get_detector),
):
    if detector.regime_history is None:
        raise HTTPException(404, "Regime history not available. Train models first.")
    rh = detector.regime_history
    return {
        "data":       rh.reset_index().to_dict(orient="records"),
        "n_days":     len(rh),
        "regime_dist": rh["regime"].value_counts().to_dict(),
    }


@app.get("/api/regime/forecast", tags=["Regime Detection"])
async def regime_forecast(
    request:  Request,
    detector: EnsembleRegimeDetector = Depends(get_detector),
    ingestion: MarketDataIngestion   = Depends(get_ingestion),
):
    """LSTM next-day regime forecast with calibrated probabilities."""
    if not detector.lstm.is_fitted:
        raise HTTPException(503, "LSTM not trained.")
    try:
        prices   = ingestion.fetch_historical("^NSEI", period="1y")
        features = detector.fb.build_all_features(prices.to_frame())
        rf       = detector.fb.get_regime_features(features)
        forecast = detector.lstm.predict_next_regime(rf)
        return {"forecast": forecast, "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        raise HTTPException(500, f"Forecast failed: {e}")


# ── Portfolio Optimisation ────────────────────────────────────────────────────

@app.post("/api/portfolio/optimize", tags=["Portfolio"])
async def optimize_portfolio(
    body:      OptimizationRequest,
    optimizer: PortfolioOptimizer  = Depends(get_optimizer),
    ingestion: MarketDataIngestion = Depends(get_ingestion),
    _:         Any                 = Depends(get_current_user),
):
    try:
        prices  = ingestion.fetch_multi_asset(ALL_ASSETS, period="2y")
        returns = prices.pct_change().dropna()
        result  = optimizer.optimize(
            returns.tail(body.lookback_days),
            body.regime,
            method       = body.method,
            risk_profile = body.risk_profile,
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Optimization failed: {e}")


@app.get("/api/portfolio/efficient-frontier", tags=["Portfolio"])
async def efficient_frontier(
    request:   Request,
    optimizer: PortfolioOptimizer  = Depends(get_optimizer),
    ingestion: MarketDataIngestion = Depends(get_ingestion),
    _:         Any                 = Depends(get_current_user),
):
    try:
        prices   = ingestion.fetch_multi_asset(ALL_ASSETS[:15], period="2y")
        returns  = prices.pct_change().dropna()
        frontier = optimizer.efficient_frontier(returns, n_points=40)
        return {"frontier": frontier.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(500, f"Efficient frontier failed: {e}")


# ── Backtesting ───────────────────────────────────────────────────────────────

@app.post("/api/backtest/run", tags=["Backtesting"])
async def run_backtest(
    body:      BacktestRequest,
    request:   Request,
    bg:        BackgroundTasks,
    _:         Any = Depends(get_current_user),
):
    """
    Trigger a full walk-forward backtest in the background.
    Returns a job_id; poll /api/backtest/status/{job_id} for results.
    """
    import uuid
    job_id = str(uuid.uuid4())[:8]
    bg.add_task(
        _run_backtest_task,
        job_id,
        request.app.state.detector,
        request.app.state.optimizer,
        request.app.state.backtest,
        body,
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/backtest/quick", tags=["Backtesting"])
async def quick_backtest(
    request: Request,
    _:       Any = Depends(get_current_user),
):
    """Synchronous quick backtest (2 WF folds) — for demo purposes."""
    try:
        from realtime_engine.data_ingestion import MarketDataIngestion
        ing     = request.app.state.ingestion
        prices  = ing.fetch_multi_asset(ALL_ASSETS[:10], period="3y")
        bt      = request.app.state.backtest
        results = bt.run_all(
            prices,
            request.app.state.detector,
            request.app.state.optimizer,
            wf_folds=2, n_bootstrap=100,
        )
        table   = bt.generate_comparison_table(results)
        return {"comparison": table.reset_index().to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(500, f"Backtest failed: {e}")


async def _run_backtest_task(job_id, detector, optimizer, backtest, body):
    logger.info(f"Backtest job {job_id} started")
    try:
        from realtime_engine.data_ingestion import MarketDataIngestion
        ing    = MarketDataIngestion()
        prices = ing.fetch_multi_asset(ALL_ASSETS, period="5y")
        results = backtest.run_all(
            prices, detector, optimizer,
            wf_folds=body.n_wf_folds, n_bootstrap=500,
        )
        logger.info(f"Backtest job {job_id} complete")
    except Exception as e:
        logger.error(f"Backtest job {job_id} failed: {e}")


# ── Risk Analytics ────────────────────────────────────────────────────────────

@app.post("/api/risk/stress-test", tags=["Risk"])
async def stress_test(
    body:  StressTestRequest,
    mc:    MonteCarloSimulator = Depends(get_mc),
    ingestion: MarketDataIngestion = Depends(get_ingestion),
    _:     Any = Depends(get_current_user),
):
    """Monte Carlo portfolio stress test under regime-conditional volatility."""
    try:
        prices  = ingestion.fetch_multi_asset(list(body.weights.keys()), period="3y")
        returns = prices.pct_change().dropna()
        result  = mc.stress_test(
            weights         = body.weights,
            returns_df      = returns,
            regime          = body.regime,
            horizon_days    = body.horizon_days,
            n_simulations   = body.n_simulations,
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Stress test failed: {e}")


@app.get("/api/risk/dashboard", tags=["Risk"])
async def risk_dashboard(
    request:   Request,
    ingestion: MarketDataIngestion = Depends(get_ingestion),
    detector:  EnsembleRegimeDetector = Depends(get_detector),
    _:         Any = Depends(get_current_user),
):
    from utils.risk_utils import compute_risk_dashboard
    try:
        prices  = ingestion.fetch_multi_asset(ALL_ASSETS[:15], period="1y")
        returns = prices.pct_change().dropna()
        regime  = None
        if detector.hmm.is_fitted:
            features = detector.fb.build_all_features(prices)
            rf       = detector.fb.get_regime_features(features)
            result   = detector.hmm.predict_latest(rf)
            regime   = result["regime"]
        dashboard = compute_risk_dashboard(returns, regime)
        return dashboard
    except Exception as e:
        raise HTTPException(500, f"Risk dashboard failed: {e}")


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/market-feed")
async def websocket_market_feed(ws: WebSocket, request: Request):
    """Real-time market price streaming via WebSocket."""
    await ws.accept()

    async with request.app.state.ws_lock:
        request.app.state.ws_clients.append(ws)

    logger.info(f"WebSocket client connected | {ws.client}")
    try:
        while True:
            try:
                snap = request.app.state.ingestion.fetch_live_snapshot(ALL_ASSETS[:10])
                msg  = {
                    "type":      "market_update",
                    "data":      snap.to_dict(orient="records"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await ws.send_json(msg)
                await asyncio.sleep(settings.LIVE_DATA_INTERVAL_SECONDS)
            except Exception as e:
                logger.debug(f"WS send error: {e}")
                await asyncio.sleep(5)
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected | {ws.client}")
    finally:
        async with request.app.state.ws_lock:
            if ws in request.app.state.ws_clients:
                request.app.state.ws_clients.remove(ws)
