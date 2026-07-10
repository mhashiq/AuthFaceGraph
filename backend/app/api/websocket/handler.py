"""
AuthBrain AI Face Analysis Engine
WebSocket Frame Handler

Processes incoming JPEG frames over WebSocket.
Sends annotated frames + JSON analysis results back to the client.

Authentication: JWT token required via query param ?token=<JWT>
Protocol:
  CLIENT → SERVER: raw JPEG bytes (binary)
  SERVER → CLIENT: JSON (analysis result) + binary (annotated frame)
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import WebSocket, WebSocketDisconnect

from app.analysis.pipeline import FaceAnalysisPipeline
from app.api.websocket.manager import WebSocketClient, ws_manager
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.security import get_ws_user
from app.models.schemas import WSMessage, WSMessageType

settings = get_settings()
logger = get_logger(__name__)

# Thread pool for running CPU-bound pipeline inference
# Isolates OpenCV/MediaPipe from the async event loop
_thread_pool = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="analysis_worker",
)

# Per-session pipeline cache: session_id → FaceAnalysisPipeline
_pipeline_registry: dict[str, FaceAnalysisPipeline] = {}


def _get_or_create_pipeline(session_id: str) -> FaceAnalysisPipeline:
    """Get existing pipeline or create and load a new one for a session."""
    if session_id not in _pipeline_registry:
        pipeline = FaceAnalysisPipeline(session_id=session_id)
        pipeline.load()
        _pipeline_registry[session_id] = pipeline
        logger.info("pipeline_created", session_id=session_id)
    return _pipeline_registry[session_id]


def _cleanup_pipeline(session_id: str) -> None:
    """Release pipeline resources for a disconnected session."""
    pipeline = _pipeline_registry.pop(session_id, None)
    if pipeline:
        pipeline.close()
        logger.info("pipeline_released", session_id=session_id)


async def save_session_stats(session_id: str) -> None:
    """Save the final aggregated session metrics to the database on disconnect."""
    pipeline = _pipeline_registry.get(session_id)
    if not pipeline:
        return

    # Only save if we actually processed some frames to avoid empty sessions
    stats = pipeline.get_session_stats()
    if stats.get("total_frames", 0) == 0:
        return

    client = ws_manager.get_client(session_id)
    if not client:
        return

    try:
        from app.core.database import AsyncSessionLocal
        from app.models.db_models import AnalysisSession
        from datetime import datetime, timezone
        from sqlalchemy import select
        import uuid

        async with AsyncSessionLocal() as db:
            # Check if this session already exists
            stmt = select(AnalysisSession).where(AnalysisSession.session_id == session_id)
            res = await db.execute(stmt)
            session_rec = res.scalar_one_or_none()

            if not session_rec:
                session_rec = AnalysisSession(
                    session_id=session_id,
                    user_id=uuid.UUID(client.user_id),
                    org_id=uuid.UUID(client.org_id),
                )
                db.add(session_rec)

            # Update fields
            session_rec.status = "completed"
            session_rec.ended_at = datetime.now(timezone.utc)
            session_rec.total_frames = stats["total_frames"]
            session_rec.total_blinks = stats["total_blinks"]
            session_rec.avg_ear = stats["avg_ear"]
            session_rec.max_fatigue_score = stats["max_fatigue"]
            session_rec.avg_fatigue_score = stats["avg_fatigue"]
            session_rec.avg_focus_score = stats["avg_focus"]
            session_rec.face_quality_score = stats["avg_quality"]
            session_rec.avg_head_yaw = stats["avg_yaw"]
            session_rec.avg_head_pitch = stats["avg_pitch"]
            session_rec.avg_inference_time_ms = stats["avg_inference_ms"]
            session_rec.dominant_attention_state = stats["dominant_attention"]

            await db.commit()
            logger.info("session_stats_saved", session_id=session_id, frames=session_rec.total_frames)
    except Exception as exc:
        logger.error("failed_to_save_session_stats", session_id=session_id, error=str(exc))


async def handle_websocket(websocket: WebSocket, token: str, session_id: str) -> None:
    """
    Main WebSocket handler coroutine.

    Flow per frame:
    1. Receive binary JPEG bytes
    2. Offload to thread pool: pipeline.process_frame()
    3. Send JSON result (analysis metrics)
    4. Send binary annotated JPEG
    """
    # ── Auth ──────────────────────────────────────────────────────────────────
    try:
        user = await get_ws_user(token)
    except Exception as exc:
        await websocket.accept()
        await websocket.send_json({"type": "error", "detail": "Unauthorized"})
        await websocket.close(code=4001)
        logger.warning("ws_auth_failed", error=str(exc))
        return

    # ── Connect ───────────────────────────────────────────────────────────────
    client = await ws_manager.connect(
        websocket=websocket,
        session_id=session_id,
        user_id=user.user_id,
        org_id=user.org_id,
    )
    if client is None:
        return  # Connection rejected (capacity exceeded)

    # Send connected status
    await ws_manager.send_json(session_id, {
        "type": WSMessageType.STATUS,
        "payload": {
            "status": "connected",
            "session_id": session_id,
            "user_id": user.user_id,
            "fps_target": settings.TARGET_FPS,
        },
    })

    loop = asyncio.get_event_loop()

    try:
        while True:
            # ── Receive Frame ──────────────────────────────────────────────────
            try:
                raw_data = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=5.0,  # Disconnect if no frames for 5s
                )
            except asyncio.TimeoutError:
                # Send ping to check if client is alive
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
                continue
            except WebSocketDisconnect:
                break

            if not raw_data:
                continue

            ws_manager.increment_frame_count(session_id)

            # ── Process Frame in Thread Pool ──────────────────────────────────
            active_face = client.active_face_index

            try:
                result, annotated_bytes = await loop.run_in_executor(
                    _thread_pool,
                    lambda: _get_or_create_pipeline(session_id).process_frame(
                        raw_data, active_face, draw_overlay=True
                    ),
                )
            except Exception as exc:
                logger.error("frame_processing_error", session_id=session_id, error=str(exc))
                await ws_manager.send_json(session_id, {
                    "type": WSMessageType.ERROR,
                    "payload": {"code": "PROCESSING_ERROR", "message": str(exc)},
                })
                continue

            # ── Send JSON Analysis Result ─────────────────────────────────────
            result_dict = result.model_dump(mode="json")
            await ws_manager.send_json(session_id, {
                "type": WSMessageType.ANALYSIS_RESULT,
                "payload": result_dict,
            })

            # ── Send Annotated Frame ──────────────────────────────────────────
            await ws_manager.send_bytes(session_id, annotated_bytes)

    except WebSocketDisconnect:
        logger.info("ws_graceful_disconnect", session_id=session_id)
    except Exception as exc:
        logger.error("ws_unexpected_error", session_id=session_id, error=str(exc))
    finally:
        # Save session stats to database before removing the client metadata
        await save_session_stats(session_id)
        await ws_manager.disconnect(session_id)
        # Clean up pipeline in thread pool to avoid blocking the event loop
        loop.run_in_executor(_thread_pool, lambda: _cleanup_pipeline(session_id))
