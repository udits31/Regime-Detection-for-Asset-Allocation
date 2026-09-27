# ============================================================
# api/auth.py
# JWT Authentication & User Management
#
# FIXED (vs original):
#   • SHA-256 + static salt → argon2 (time_cost=3, memory=64MB)
#   • In-memory token dict → Redis-backed with TTL + automatic cleanup
#   • Tokens are cryptographically random (32-byte URL-safe)
#   • Added brute-force protection: 5 failed attempts → 15-min lockout
#   • Refresh token mechanism
# ============================================================

from __future__ import annotations

import secrets
import json
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from loguru import logger

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

try:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError, VerificationError
    _ARGON2_AVAILABLE = True
    _ph = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2, hash_len=32)
except ImportError:
    _ARGON2_AVAILABLE = False
    logger.warning(
        "argon2-cffi not installed; falling back to PBKDF2-SHA256. "
        "Install argon2-cffi for production security."
    )

from database.connection import get_db
from database.models import User
from config.settings import settings

router       = APIRouter(prefix="/api/auth", tags=["Authentication"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# ── Sentinel constant ─────────────────────────────────────────────────────────
ACCESS_TOKEN_TTL  = timedelta(hours=24)
REFRESH_TOKEN_TTL = timedelta(days=30)
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION    = timedelta(minutes=15)


# ── Token store (Redis-backed or in-memory fallback) ─────────────────────────

class TokenStore:
    """
    Redis-backed token store with TTL and atomic operations.
    Falls back to an in-memory dict if Redis is unavailable
    (for development; NOT for production multi-instance deployments).
    """

    def __init__(self):
        self._redis = None
        self._mem:  Dict[str, str] = {}   # fallback
        self._fail: Dict[str, Dict] = {}  # brute-force tracking

    async def init(self, redis_url: str) -> None:
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(redis_url, decode_responses=True)
            await self._redis.ping()
            logger.info("Token store: Redis connected")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}); using in-memory token store")
            self._redis = None

    async def set(self, key: str, value: Dict, ttl: timedelta) -> None:
        serialised = json.dumps(value)
        if self._redis:
            await self._redis.setex(key, int(ttl.total_seconds()), serialised)
        else:
            self._mem[key] = serialised   # no TTL in fallback; acceptable for dev

    async def get(self, key: str) -> Optional[Dict]:
        if self._redis:
            raw = await self._redis.get(key)
            return json.loads(raw) if raw else None
        raw = self._mem.get(key)
        return json.loads(raw) if raw else None

    async def delete(self, key: str) -> None:
        if self._redis:
            await self._redis.delete(key)
        else:
            self._mem.pop(key, None)

    async def record_failed_attempt(self, username: str) -> int:
        """Track brute-force attempts. Returns current attempt count."""
        key = f"fail:{username}"
        if self._redis:
            count = await self._redis.incr(key)
            if count == 1:
                await self._redis.expire(key, int(LOCKOUT_DURATION.total_seconds()))
            return int(count)
        else:
            entry = self._fail.setdefault(username, {"count": 0, "since": datetime.now(timezone.utc)})
            if datetime.now(timezone.utc) - entry["since"] > LOCKOUT_DURATION:
                entry["count"] = 0
                entry["since"] = datetime.now(timezone.utc)
            entry["count"] += 1
            return entry["count"]

    async def clear_failed_attempts(self, username: str) -> None:
        if self._redis:
            await self._redis.delete(f"fail:{username}")
        else:
            self._fail.pop(username, None)

    async def is_locked(self, username: str) -> bool:
        if self._redis:
            return await self._redis.exists(f"fail:{username}") and \
                   int(await self._redis.get(f"fail:{username}") or 0) >= MAX_FAILED_ATTEMPTS
        entry = self._fail.get(username)
        if not entry:
            return False
        if datetime.now(timezone.utc) - entry["since"] > LOCKOUT_DURATION:
            return False
        return entry["count"] >= MAX_FAILED_ATTEMPTS


_token_store = TokenStore()


# ── Password hashing ─────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Hash password with argon2 (preferred) or PBKDF2-SHA256 (fallback)."""
    if _ARGON2_AVAILABLE:
        return _ph.hash(password)
    # PBKDF2 fallback (still secure, not as memory-hard as argon2)
    salt = secrets.token_hex(16)
    dk   = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 310_000)
    return f"pbkdf2:{salt}:{dk.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against stored hash (supports both argon2 and pbkdf2)."""
    if hashed.startswith("$argon2"):
        if not _ARGON2_AVAILABLE:
            raise RuntimeError("argon2-cffi required to verify this hash")
        try:
            return _ph.verify(hashed, password)
        except (VerifyMismatchError, VerificationError):
            return False
    elif hashed.startswith("pbkdf2:"):
        _, salt, stored_dk = hashed.split(":", 2)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 310_000)
        return secrets.compare_digest(dk.hex(), stored_dk)
    return False


# ── Token creation ───────────────────────────────────────────────────────────

def _new_token() -> str:
    return secrets.token_urlsafe(32)


async def create_access_token(user_id: int, username: str) -> str:
    token = _new_token()
    await _token_store.set(
        f"access:{token}",
        {"user_id": user_id, "username": username, "type": "access"},
        ACCESS_TOKEN_TTL,
    )
    return token


async def create_refresh_token(user_id: int, username: str) -> str:
    token = _new_token()
    await _token_store.set(
        f"refresh:{token}",
        {"user_id": user_id, "username": username, "type": "refresh"},
        REFRESH_TOKEN_TTL,
    )
    return token


async def verify_token(token: str, token_type: str = "access") -> Optional[Dict]:
    return await _token_store.get(f"{token_type}:{token}")


async def revoke_token(token: str, token_type: str = "access") -> None:
    await _token_store.delete(f"{token_type}:{token}")


# ── Dependency: current user ─────────────────────────────────────────────────

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db:    AsyncSession = Depends(get_db),
) -> User:
    credentials_exc = HTTPException(
        status_code = status.HTTP_401_UNAUTHORIZED,
        detail      = "Invalid or expired token",
        headers     = {"WWW-Authenticate": "Bearer"},
    )
    data = await verify_token(token)
    if not data:
        raise credentials_exc

    result = await db.execute(
        select(User).where(User.id == data["user_id"], User.is_active == True)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise credentials_exc
    return user


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username:        str
    email:           EmailStr
    password:        str
    risk_profile:    str  = "moderate"
    initial_capital: float = 1_000_000.0

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 12:
            raise ValueError("Password must be at least 12 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain an uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain a digit")
        return v

    @field_validator("risk_profile")
    @classmethod
    def valid_risk_profile(cls, v: str) -> str:
        if v not in {"conservative", "moderate", "aggressive"}:
            raise ValueError("risk_profile must be conservative | moderate | aggressive")
        return v


class UserResponse(BaseModel):
    id:              int
    username:        str
    email:           str
    risk_profile:    str
    initial_capital: float
    is_active:       bool
    created_at:      datetime


class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    expires_in:    int = int(ACCESS_TOKEN_TTL.total_seconds())


class LoginRequest(BaseModel):
    username: str
    password: str


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    body: UserCreate,
    db:   AsyncSession = Depends(get_db),
):
    """Register a new user."""
    # Check uniqueness
    existing = await db.execute(
        select(User).where(
            (User.username == body.username) | (User.email == body.email)
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(400, "Username or email already registered")

    user = User(
        username        = body.username,
        email           = body.email,
        password_hash   = hash_password(body.password),
        risk_profile    = body.risk_profile,
        initial_capital = body.initial_capital,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    logger.info(f"New user registered: {user.username}")
    return user


@router.post("/token", response_model=TokenResponse)
async def login(
    request: Request,
    body:    LoginRequest,
    db:      AsyncSession = Depends(get_db),
):
    """Authenticate and issue access + refresh tokens."""
    # Brute-force check
    if await _token_store.is_locked(body.username):
        raise HTTPException(
            status_code = status.HTTP_429_TOO_MANY_REQUESTS,
            detail      = f"Account locked for {LOCKOUT_DURATION.seconds // 60} minutes "
                          f"due to failed login attempts",
        )

    result = await db.execute(
        select(User).where(User.username == body.username, User.is_active == True)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.password_hash):
        count = await _token_store.record_failed_attempt(body.username)
        remaining = MAX_FAILED_ATTEMPTS - count
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = f"Invalid credentials. {max(0, remaining)} attempts remaining",
        )

    await _token_store.clear_failed_attempts(body.username)
    access  = await create_access_token(user.id, user.username)
    refresh = await create_refresh_token(user.id, user.username)
    logger.info(f"User {user.username} logged in from {request.client.host}")

    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(refresh_token: str, db: AsyncSession = Depends(get_db)):
    """Exchange a refresh token for a new access token."""
    data = await verify_token(refresh_token, token_type="refresh")
    if not data:
        raise HTTPException(401, "Invalid or expired refresh token")

    await revoke_token(refresh_token, "refresh")
    access  = await create_access_token(data["user_id"], data["username"])
    new_ref = await create_refresh_token(data["user_id"], data["username"])
    return TokenResponse(access_token=access, refresh_token=new_ref)


@router.post("/logout")
async def logout(
    token:        str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_user),
):
    """Revoke the current access token."""
    await revoke_token(token)
    return {"message": f"User {current_user.username} logged out"}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user
