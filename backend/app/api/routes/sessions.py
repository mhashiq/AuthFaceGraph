"""
AuthBrain AI Face Analysis Engine
Sessions API Routes

Provides CRUD for analysis sessions, including summaries, metrics,
and historical session data stored in PostgreSQL.

Authentication: Employee (own sessions), Manager/Admin (org-wide).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.logging import get_logger
from app.core.security import CurrentUser, UserRole, get_current_user, require_role
from app.models.db_models import AnalysisSession
from app.models.schemas import SessionStatus, SessionSummary

logger = get_logger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get(
    "/",
    response_model=list[SessionSummary],
    summary="List analysis sessions",
)
async def list_sessions(
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status_filter: SessionStatus | None = None,
) -> list[SessionSummary]:
    """
    List analysis sessions.
    - Employees see only their own sessions.
    - Managers/Admins see all org sessions.
    """
    stmt = select(AnalysisSession).order_by(AnalysisSession.started_at.desc())

    # Role-based filtering
    if current_user.role == UserRole.EMPLOYEE:
        stmt = stmt.where(AnalysisSession.user_id == uuid.UUID(current_user.user_id))
    else:
        stmt = stmt.where(AnalysisSession.org_id == uuid.UUID(current_user.org_id))

    if status_filter:
        stmt = stmt.where(AnalysisSession.status == status_filter.value)

    stmt = stmt.limit(limit).offset(offset)
    result = await db.execute(stmt)
    sessions = result.scalars().all()

    return [_session_to_summary(s) for s in sessions]


@router.get(
    "/{session_id}",
    response_model=SessionSummary,
    summary="Get session details",
)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> SessionSummary:
    """Retrieve a specific analysis session by ID."""
    stmt = select(AnalysisSession).where(AnalysisSession.session_id == session_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    # Employees can only access their own sessions
    if (current_user.role == UserRole.EMPLOYEE and
            str(session.user_id) != current_user.user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    return _session_to_summary(session)


@router.get(
    "/stats/summary",
    summary="Get aggregate statistics for the organization",
)
async def org_stats(
    db: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(
        require_role(UserRole.MANAGER, UserRole.ADMINISTRATOR)
    ),
) -> dict:
    """
    Returns aggregate session statistics for the entire organization.
    Requires Manager or Administrator role.
    """
    org_uuid = uuid.UUID(current_user.org_id)

    # Total sessions
    total_stmt = select(func.count(AnalysisSession.id)).where(
        AnalysisSession.org_id == org_uuid
    )
    total_result = await db.execute(total_stmt)
    total_sessions = total_result.scalar() or 0

    # Average metrics
    avg_stmt = select(
        func.avg(AnalysisSession.avg_ear).label("avg_ear"),
        func.avg(AnalysisSession.face_quality_score).label("avg_quality"),
        func.avg(AnalysisSession.total_blinks).label("avg_blinks"),
    ).where(AnalysisSession.org_id == org_uuid)

    avg_result = await db.execute(avg_stmt)
    avgs = avg_result.one()

    return {
        "org_id": current_user.org_id,
        "total_sessions": total_sessions,
        "avg_ear": round(avgs.avg_ear or 0, 3),
        "avg_quality_score": round(avgs.avg_quality or 0, 3),
        "avg_blinks_per_session": round(avgs.avg_blinks or 0, 1),
    }


def _session_to_summary(s: AnalysisSession) -> SessionSummary:
    """Convert ORM model to Pydantic schema."""
    return SessionSummary(
        session_id=s.session_id,
        started_at=s.started_at,
        ended_at=s.ended_at,
        status=SessionStatus(s.status),
        total_frames=s.total_frames,
        total_blinks=s.total_blinks,
        avg_ear=s.avg_ear,
        avg_head_yaw=s.avg_head_yaw,
        avg_head_pitch=s.avg_head_pitch,
        dominant_attention_state=s.dominant_attention_state,
        face_quality_score=s.face_quality_score,
    )
