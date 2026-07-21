"""
AuthBrain AI Face Analysis Engine
Security — JWT Authentication & RBAC

Implements JWT token generation/validation, password hashing,
and role-based access control for all API endpoints and WebSocket connections.

Roles (in ascending privilege):
  employee → manager → administrator → researcher
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

# ── JWT Configuration ─────────────────────────────────────────────────────────
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Bearer token extractor
_bearer_scheme = HTTPBearer(auto_error=True)


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class UserRole(str, Enum):
    """RBAC roles in ascending privilege order."""
    EMPLOYEE     = "employee"
    MANAGER      = "manager"
    RESEARCHER   = "researcher"
    ADMINISTRATOR = "administrator"


# ══════════════════════════════════════════════════════════════════════════════
# Token Models
# ══════════════════════════════════════════════════════════════════════════════

class TokenPayload(BaseModel):
    """JWT payload fields."""
    sub: str              # User ID (UUID string)
    email: str
    role: UserRole
    org_id: str           # Organization/tenant ID
    exp: datetime
    iat: datetime
    jti: str              # JWT ID for revocation support


class TokenPair(BaseModel):
    """Access + refresh token pair returned on login."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


class LoginRequest(BaseModel):
    email: str
    password: str
    org_id: str | None = None


class RegisterRequest(BaseModel):
    full_name: str
    email: str
    password: str
    organization_name: str | None = "AuthBrain AI"


class FaceEnrollmentRequest(BaseModel):
    user_id: str | None = None
    frontal_embedding: list[float]
    left_embedding: list[float]
    right_embedding: list[float]
    upward_embedding: list[float]


class LoginResponse(BaseModel):
    tokens: TokenPair
    user_id: str
    email: str
    role: UserRole
    org_id: str
    full_name: str


import bcrypt

# Password Utilities
# ══════════════════════════════════════════════════════════════════════════════

def hash_password(plain: str) -> str:
    """Hash a plain-text password using native bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain-text password against a native bcrypt hash."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# JWT Token Operations
# ══════════════════════════════════════════════════════════════════════════════

def create_access_token(
    user_id: str,
    email: str,
    role: UserRole,
    org_id: str,
    expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
) -> str:
    """Create a signed JWT access token."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub":    user_id,
        "email":  email,
        "role":   role.value,
        "org_id": org_id,
        "iat":    now,
        "exp":    now + timedelta(minutes=expires_minutes),
        "jti":    str(uuid.uuid4()),
        "type":   "access",
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str, org_id: str) -> str:
    """Create a long-lived refresh token."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub":    user_id,
        "org_id": org_id,
        "iat":    now,
        "exp":    now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        "jti":    str(uuid.uuid4()),
        "type":   "refresh",
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT token.

    Raises:
        HTTPException 401 if token is invalid or expired
    """
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as exc:
        logger.warning("jwt_decode_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Dependencies
# ══════════════════════════════════════════════════════════════════════════════

class CurrentUser(BaseModel):
    """Authenticated user context injected via Depends()."""
    user_id: str
    email: str
    role: UserRole
    org_id: str


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(_bearer_scheme),
) -> CurrentUser:
    """
    FastAPI dependency: validate Bearer token and return current user.

    Usage:
        @router.get("/protected")
        async def endpoint(user: CurrentUser = Depends(get_current_user)):
            ...
    """
    payload = decode_token(credentials.credentials)

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh tokens cannot be used for API access",
        )

    return CurrentUser(
        user_id=payload["sub"],
        email=payload["email"],
        role=UserRole(payload["role"]),
        org_id=payload["org_id"],
    )


def require_role(*roles: UserRole):
    """
    RBAC dependency factory — requires at least one of the given roles.

    Usage:
        @router.get("/admin")
        async def admin_endpoint(
            user: CurrentUser = Depends(require_role(UserRole.ADMINISTRATOR))
        ):
            ...
    """
    async def role_checker(
        current_user: CurrentUser = Depends(get_current_user),
    ) -> CurrentUser:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This endpoint requires one of: {[r.value for r in roles]}",
            )
        return current_user
    return role_checker


async def get_ws_user(token: str) -> CurrentUser:
    """
    Validate JWT token from WebSocket query parameter.

    WebSocket Authorization flow:
        ws://host/ws/analyze?token=<JWT>

    Returns CurrentUser or raises HTTPException.
    """
    payload = decode_token(token)
    return CurrentUser(
        user_id=payload["sub"],
        email=payload["email"],
        role=UserRole(payload["role"]),
        org_id=payload["org_id"],
    )


def generate_consent_token() -> str:
    """Generate a cryptographically secure consent/analysis token."""
    return secrets.token_urlsafe(32)
