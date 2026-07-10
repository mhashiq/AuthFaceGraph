"""
AuthBrain AI Face Analysis Engine
Consent API Routes — GDPR-Compliant Consent Management

Records explicit user consent before any webcam analysis begins.
Consent is time-stamped, versioned, and stored in PostgreSQL.

Authentication required: Employee role minimum.
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.logging import get_logger
from app.core.security import CurrentUser, UserRole, get_current_user
from app.models.db_models import ConsentRecord
from app.models.schemas import ConsentRequest, ConsentResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/api/consent", tags=["consent"])


@router.post(
    "/",
    response_model=ConsentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record user consent for AI analysis",
)
async def record_consent(
    request: ConsentRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> ConsentResponse:
    """
    Record user consent before enabling webcam and AI analysis.

    GDPR compliance:
    - Consent is explicit (must be True)
    - Consent version is recorded for audit trails
    - Timestamp is stored
    - Consent can be revoked at any time

    Returns an analysis_token used to authenticate WebSocket connections.
    """
    if not request.consent_granted:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Consent must be explicitly granted (consent_granted=true)",
        )

    # Generate secure analysis token
    analysis_token = secrets.token_urlsafe(32)
    expires_at     = datetime.now(timezone.utc) + timedelta(hours=1)
    session_id     = request.session_id or str(uuid.uuid4())

    # Store consent record
    consent_record = ConsentRecord(
        session_id=session_id,
        user_id=uuid.UUID(current_user.user_id),
        org_id=uuid.UUID(current_user.org_id),
        consent_granted=True,
        consent_text_version=request.consent_text_version,
        user_agent=request.user_agent,
        analysis_token=analysis_token,
        expires_at=expires_at,
    )
    db.add(consent_record)
    await db.commit()

    logger.info(
        "consent_recorded",
        session_id=session_id,
        user_id=current_user.user_id,
        version=request.consent_text_version,
    )

    return ConsentResponse(
        session_id=session_id,
        consent_granted=True,
        analysis_token=analysis_token,
        expires_at=expires_at,
        message="Consent recorded. You may now start the AI analysis session.",
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Revoke consent and stop analysis",
)
async def revoke_consent(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> dict:
    """
    Revoke consent for a session — marks consent as withdrawn.
    Analysis will stop immediately when consent is revoked.
    """
    stmt = select(ConsentRecord).where(
        ConsentRecord.session_id == session_id,
        ConsentRecord.user_id == uuid.UUID(current_user.user_id),
    )
    result = await db.execute(stmt)
    record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Consent record not found for this session",
        )

    record.consent_granted = False
    record.revoked_at = datetime.now(timezone.utc)
    await db.commit()

    logger.info("consent_revoked", session_id=session_id, user_id=current_user.user_id)

    # Close active WebSocket for this session
    from app.api.websocket.manager import ws_manager
    await ws_manager.disconnect(session_id)

    return {"message": "Consent revoked. Analysis session terminated.", "session_id": session_id}


@router.get(
    "/history",
    summary="Get consent history for current user",
)
async def get_consent_history(
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
    limit: int = 20,
) -> list[dict]:
    """Returns the authenticated user's consent history (GDPR right to access)."""
    stmt = (
        select(ConsentRecord)
        .where(ConsentRecord.user_id == uuid.UUID(current_user.user_id))
        .order_by(ConsentRecord.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    records = result.scalars().all()

    return [
        {
            "session_id": r.session_id,
            "consent_granted": r.consent_granted,
            "created_at": r.created_at.isoformat(),
            "expires_at": r.expires_at.isoformat(),
            "consent_version": r.consent_text_version,
        }
        for r in records
    ]
