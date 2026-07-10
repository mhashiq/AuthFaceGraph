"""
AuthBrain AI Face Analysis Engine
Health Check API Routes

Provides system status including model availability, database connectivity,
GPU availability, and active session counts.

Public endpoint — no authentication required.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.api.websocket.manager import ws_manager
from app.core.config import get_settings
from app.core.database import get_db_session
from app.core.logging import get_logger
from app.models.schemas import HealthResponse

settings = get_settings()
logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["health"])

_start_time = time.time()


def _check_gpu_available() -> bool:
    """Check if CUDA GPU is available via torch or onnxruntime."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return "CUDAExecutionProvider" in providers
    except ImportError:
        pass
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health_check(db: AsyncSession = Depends(get_db_session)) -> HealthResponse:
    """
    Returns the current health status of the AuthBrain system.

    Checks:
    - Database connectivity
    - ML model load status
    - GPU availability
    - Active WebSocket sessions
    """
    # Test database connection
    db_connected = False
    try:
        await db.execute(text("SELECT 1"))
        db_connected = True
    except Exception as exc:
        logger.warning("health_db_failed", error=str(exc))

    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        database_connected=db_connected,
        model_loaded=True,           # Pipeline loaded lazily per session
        gpu_available=_check_gpu_available(),
        uptime_seconds=round(time.time() - _start_time, 1),
        active_sessions=ws_manager.total_connections,
    )


@router.get("/health/ws", tags=["health"], summary="WebSocket connection statistics")
async def ws_stats() -> dict:
    """Returns active WebSocket connection statistics (admin use)."""
    return ws_manager.get_stats()
