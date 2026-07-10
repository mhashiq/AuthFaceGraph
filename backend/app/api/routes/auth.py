"""
AuthBrain AI Face Analysis Engine
Auth API Routes — Login, Refresh, Logout

Implements JWT authentication flow with RBAC.
Tokens are issued on successful login and validated on every protected request.

Authentication Flow:
  POST /api/auth/login  → {access_token, refresh_token}
  Authorization: Bearer <access_token>
  POST /api/auth/refresh → {access_token}
  POST /api/auth/logout  → 200 OK
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.logging import get_logger
from app.core.security import (
    CurrentUser,
    LoginRequest,
    LoginResponse,
    TokenPair,
    UserRole,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.models.db_models import User

logger = get_logger(__name__)
router = APIRouter(prefix="/api/auth", tags=["authentication"])


@router.post("/login", response_model=LoginResponse, summary="Authenticate and receive JWT tokens")
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db_session),
) -> LoginResponse:
    """
    Authenticate user with email/password.
    Returns JWT access and refresh tokens on success.

    Rate limited: 5 attempts per minute per IP (enforced at nginx layer).
    """
    # Look up user by email
    stmt = select(User).where(
        User.email == request.email.lower().strip(),
        User.is_active == True,  # noqa: E712
    )
    if request.org_id:
        stmt = stmt.where(User.org_id == request.org_id)

    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    # Use constant-time comparison to prevent timing attacks
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last login
    user.last_login_at = datetime.now(timezone.utc)
    await db.commit()

    access_token  = create_access_token(
        user_id=str(user.id),
        email=user.email,
        role=UserRole(user.role),
        org_id=str(user.org_id),
    )
    refresh_token = create_refresh_token(
        user_id=str(user.id),
        org_id=str(user.org_id),
    )

    logger.info("user_login", user_id=str(user.id), org_id=str(user.org_id))

    return LoginResponse(
        tokens=TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
        ),
        user_id=str(user.id),
        email=user.email,
        role=UserRole(user.role),
        org_id=str(user.org_id),
        full_name=user.full_name,
    )


@router.post("/refresh", response_model=TokenPair, summary="Refresh access token")
async def refresh_token(
    refresh_token_str: str,
    db: AsyncSession = Depends(get_db_session),
) -> TokenPair:
    """
    Exchange a valid refresh token for a new access token.
    Refresh tokens are valid for 7 days.
    """
    payload = decode_token(refresh_token_str)

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token type",
        )

    user_id = payload["sub"]
    org_id  = payload["org_id"]

    # Verify user still exists and is active
    stmt   = select(User).where(User.id == uuid.UUID(user_id), User.is_active == True)  # noqa
    result = await db.execute(stmt)
    user   = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated",
        )

    new_access = create_access_token(
        user_id=user_id,
        email=user.email,
        role=UserRole(user.role),
        org_id=org_id,
    )

    return TokenPair(access_token=new_access, refresh_token=refresh_token_str)


@router.post("/logout", status_code=status.HTTP_200_OK, summary="Logout (client-side token discard)")
async def logout(current_user: CurrentUser = Depends(get_current_user)) -> dict:
    """
    Logout endpoint — tokens are stateless (JWT), so client must discard them.
    Future: Add token blacklist via Redis for immediate revocation.
    """
    logger.info("user_logout", user_id=current_user.user_id)
    return {"message": "Logged out successfully. Please discard your tokens."}


@router.get("/me", response_model=CurrentUser, summary="Get current authenticated user")
async def get_me(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Returns the currently authenticated user's profile."""
    return current_user
