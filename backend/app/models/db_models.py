"""
AuthBrain AI Face Analysis Engine
SQLAlchemy ORM Database Models (Updated with User, Org, Multi-tenant)

Defines all PostgreSQL tables with multi-tenant organization support,
JWT-compatible user model, RBAC roles, and GDPR consent tracking.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


# ══════════════════════════════════════════════════════════════════════════════
# Organization (Multi-tenant)
# ══════════════════════════════════════════════════════════════════════════════

class Organization(Base):
    """Represents a company/tenant in the multi-tenant system."""
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    plan: Mapped[str] = mapped_column(String(50), default="standard")   # standard/pro/enterprise
    research_mode_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    users: Mapped[list["User"]] = relationship("User", back_populates="organization", lazy="select")

    def __repr__(self) -> str:
        return f"<Organization {self.slug}>"


# ══════════════════════════════════════════════════════════════════════════════
# User
# ══════════════════════════════════════════════════════════════════════════════

class User(Base):
    """System user with RBAC role assignment."""
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="employee")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_enrolled: Mapped[bool] = mapped_column(Boolean, default=False)
    enrolled_face_embedding: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON 512-d vector
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    organization: Mapped["Organization"] = relationship("Organization", back_populates="users")

    def __repr__(self) -> str:
        return f"<User {self.email} role={self.role}>"


# ══════════════════════════════════════════════════════════════════════════════
# Consent Record
# ══════════════════════════════════════════════════════════════════════════════

class ConsentRecord(Base):
    """GDPR-compliant consent records with revocation support."""
    __tablename__ = "consent_records"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    consent_granted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    consent_text_version: Mapped[str] = mapped_column(String(10), default="1.0")
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    analysis_token: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<ConsentRecord session={self.session_id} granted={self.consent_granted}>"


# ══════════════════════════════════════════════════════════════════════════════
# Analysis Session
# ══════════════════════════════════════════════════════════════════════════════

class AnalysisSession(Base):
    """
    Tracks an analysis session lifecycle with aggregated metrics.
    Stores ONLY session summaries — no raw video or per-frame data by default.
    """
    __tablename__ = "analysis_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="active")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Aggregate session metrics (stored by default)
    total_frames: Mapped[int] = mapped_column(Integer, default=0)
    total_blinks: Mapped[int] = mapped_column(Integer, default=0)
    avg_ear: Mapped[float] = mapped_column(Float, default=0.0)
    max_fatigue_score: Mapped[float] = mapped_column(Float, default=0.0)
    avg_fatigue_score: Mapped[float] = mapped_column(Float, default=0.0)
    avg_focus_score: Mapped[float] = mapped_column(Float, default=0.0)
    avg_stress_risk: Mapped[float] = mapped_column(Float, default=0.0)
    max_stress_risk: Mapped[float] = mapped_column(Float, default=0.0)
    avg_head_yaw: Mapped[float] = mapped_column(Float, default=0.0)
    avg_head_pitch: Mapped[float] = mapped_column(Float, default=0.0)
    avg_inference_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    face_quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    dominant_attention_state: Mapped[str] = mapped_column(String(20), default="unknown")
    ai_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    expert_system_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON summary

    def __repr__(self) -> str:
        return f"<AnalysisSession {self.session_id} status={self.status}>"


# ══════════════════════════════════════════════════════════════════════════════
# Research Frame Data (Optional — only when research_mode_enabled=True)
# ══════════════════════════════════════════════════════════════════════════════

class ResearchFrameData(Base):
    """
    Optional per-frame data for research mode.
    Only stored when:
    1. Organization has research_mode_enabled=True
    2. User has explicitly consented to research data collection
    3. Administrator has explicitly enabled for the session
    """
    __tablename__ = "research_frame_data"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    frame_id: Mapped[str] = mapped_column(String(64), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Extracted features (no raw images stored)
    face_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    ear: Mapped[float] = mapped_column(Float, default=0.0)
    mar: Mapped[float] = mapped_column(Float, default=0.0)
    head_pitch: Mapped[float] = mapped_column(Float, default=0.0)
    head_yaw: Mapped[float] = mapped_column(Float, default=0.0)
    head_roll: Mapped[float] = mapped_column(Float, default=0.0)
    smile_intensity: Mapped[float] = mapped_column(Float, default=0.0)
    blink_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    face_quality: Mapped[float] = mapped_column(Float, default=0.0)
    fatigue_score: Mapped[float] = mapped_column(Float, default=0.0)
    attention_state: Mapped[str] = mapped_column(String(20), default="unknown")
    inference_time_ms: Mapped[float] = mapped_column(Float, default=0.0)

    def __repr__(self) -> str:
        return f"<ResearchFrameData session={self.session_id} frame={self.frame_id}>"


# ══════════════════════════════════════════════════════════════════════════════
# Audit Log
# ══════════════════════════════════════════════════════════════════════════════

class AuditLog(Base):
    """Immutable audit trail for security-relevant actions."""
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    def __repr__(self) -> str:
        return f"<AuditLog {self.action} by user={self.user_id}>"
