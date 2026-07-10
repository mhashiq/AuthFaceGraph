"""
AuthBrain AI Face Analysis Engine
FastAPI Application Entrypoint

Configures the FastAPI app with:
- JWT authentication middleware
- CORS for frontend (React)
- API route registration
- WebSocket endpoint
- Startup/shutdown lifecycle hooks
- Structured logging
- OpenAPI docs
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import auth, consent, health, sessions, models
from app.api.websocket.handler import handle_websocket
from app.core.config import get_settings
from app.core.database import close_db, init_db
from app.core.logging import configure_logging, get_logger

settings = get_settings()
configure_logging()
logger = get_logger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    logger.info(
        "authbrain_startup",
        app=settings.APP_NAME,
        version=settings.APP_VERSION,
        env=settings.ENVIRONMENT,
    )

    # Initialize database tables
    await init_db()

    yield  # Application is running

    # Cleanup
    await close_db()
    logger.info("authbrain_shutdown")


# ── Application Factory ────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "AuthBrain AI Face Analysis Engine — Real-time facial behavior analysis "
            "using MediaPipe, OpenCV, and an Explainable AI expert system. "
            "All analysis requires explicit user consent. No facial recognition."
        ),
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── REST API Routes ────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(consent.router)
    app.include_router(sessions.router)
    app.include_router(models.router)

    # ── WebSocket Endpoint ─────────────────────────────────────────────────────
    @app.websocket("/ws/analyze")
    async def websocket_analyze(
        websocket: WebSocket,
        token: str = Query(..., description="JWT access token"),
        session_id: str = Query(..., description="Analysis session ID"),
    ) -> None:
        """
        Real-time face analysis WebSocket endpoint.

        Protocol:
          1. Connect with ?token=<JWT>&session_id=<UUID>
          2. Send JPEG frames as binary messages
          3. Receive JSON analysis results + binary annotated frames

        Authentication: JWT Bearer token (required)
        Authorization: Employee role minimum
        """
        await handle_websocket(
            websocket=websocket,
            token=token,
            session_id=session_id,
        )

    # ── Root redirect ──────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse({
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/api/docs",
            "health": "/api/health",
        })

    return app


app = create_app()
