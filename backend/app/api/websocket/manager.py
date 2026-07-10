"""
AuthBrain AI Face Analysis Engine
WebSocket Connection Manager

Manages active WebSocket connections with per-session state.
Handles registration, broadcasting, cleanup, and message routing.

Thread-safe: uses asyncio.Lock for connection map mutations.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WebSocketClient:
    """Represents one active WebSocket client connection."""
    websocket: WebSocket
    session_id: str
    user_id: str
    org_id: str
    connected_at: float = field(default_factory=time.time)
    frame_count: int = 0
    last_ping: float = field(default_factory=time.time)
    active_face_index: int = 0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.connected_at


class ConnectionManager:
    """
    Manages all active WebSocket connections.

    Design:
    - session_id → WebSocketClient mapping
    - Max connections per org enforced
    - Automatic cleanup on disconnect
    """

    MAX_CONNECTIONS_PER_ORG = 50  # Safety limit
    MAX_TOTAL_CONNECTIONS   = 500

    def __init__(self) -> None:
        # session_id → WebSocketClient
        self._connections: dict[str, WebSocketClient] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        session_id: str,
        user_id: str,
        org_id: str,
    ) -> WebSocketClient | None:
        """
        Accept and register a new WebSocket connection.

        Returns:
            WebSocketClient on success, None if limits exceeded
        """
        await websocket.accept()

        async with self._lock:
            # Check global limit
            if len(self._connections) >= self.MAX_TOTAL_CONNECTIONS:
                await websocket.close(code=1008, reason="Server at capacity")
                logger.warning("ws_capacity_exceeded", total=len(self._connections))
                return None

            # Check per-org limit
            org_count = sum(1 for c in self._connections.values() if c.org_id == org_id)
            if org_count >= self.MAX_CONNECTIONS_PER_ORG:
                await websocket.close(code=1008, reason="Organization connection limit reached")
                logger.warning("org_capacity_exceeded", org_id=org_id, count=org_count)
                return None

            # If session already exists, close old connection
            if session_id in self._connections:
                old = self._connections[session_id]
                try:
                    await old.websocket.close(code=1000, reason="Session reconnected")
                except Exception:
                    pass

            client = WebSocketClient(
                websocket=websocket,
                session_id=session_id,
                user_id=user_id,
                org_id=org_id,
            )
            self._connections[session_id] = client

        logger.info(
            "ws_connected",
            session_id=session_id,
            user_id=user_id,
            total_connections=len(self._connections),
        )
        return client

    async def disconnect(self, session_id: str) -> None:
        """Remove a client connection and clean up resources."""
        async with self._lock:
            client = self._connections.pop(session_id, None)

        if client:
            logger.info(
                "ws_disconnected",
                session_id=session_id,
                frames_processed=client.frame_count,
                uptime_s=round(client.uptime_seconds, 1),
            )

    async def send_json(self, session_id: str, data: dict[str, Any]) -> bool:
        """
        Send a JSON message to a specific session.

        Returns:
            True on success, False if client not found or send fails
        """
        client = self._connections.get(session_id)
        if not client:
            return False
        try:
            await client.websocket.send_json(data)
            return True
        except Exception as exc:
            logger.warning("ws_send_failed", session_id=session_id, error=str(exc))
            await self.disconnect(session_id)
            return False

    async def send_bytes(self, session_id: str, data: bytes) -> bool:
        """Send binary data (JPEG frame) to a specific session."""
        client = self._connections.get(session_id)
        if not client:
            return False
        try:
            await client.websocket.send_bytes(data)
            return True
        except Exception as exc:
            logger.warning("ws_bytes_send_failed", session_id=session_id, error=str(exc))
            await self.disconnect(session_id)
            return False

    def get_client(self, session_id: str) -> WebSocketClient | None:
        """Get client by session ID."""
        return self._connections.get(session_id)

    def increment_frame_count(self, session_id: str) -> None:
        """Increment processed frame counter for a session."""
        if session_id in self._connections:
            self._connections[session_id].frame_count += 1

    @property
    def total_connections(self) -> int:
        return len(self._connections)

    @property
    def session_ids(self) -> list[str]:
        return list(self._connections.keys())

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate connection statistics."""
        return {
            "total_connections": len(self._connections),
            "sessions": [
                {
                    "session_id": s,
                    "user_id": c.user_id,
                    "org_id": c.org_id,
                    "uptime_s": round(c.uptime_seconds, 1),
                    "frames": c.frame_count,
                }
                for s, c in self._connections.items()
            ],
        }


# Global singleton — shared across all WebSocket routes
ws_manager = ConnectionManager()
