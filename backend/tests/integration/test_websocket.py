"""
AuthBrain AI Face Analysis Engine
Integration Tests — WebSocket API

Tests real-time analysis WebSocket pipeline:
1. JWT authentication gate
2. Handshake message
3. Core JSON result + annotated JPEG frame loop
"""

from __future__ import annotations

import io
from fastapi.testclient import TestClient
import pytest
from PIL import Image


def _create_mock_jpeg() -> bytes:
    """Generate a valid dummy 10x10 JPEG image in memory."""
    img = Image.new("RGB", (10, 10), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestWebSocketInference:

    def test_ws_rejects_unauthorized_token(self, client: TestClient):
        """Websocket should reject connections with invalid token by sending an error and closing."""
        with client.websocket_connect("/ws/analyze?token=invalid_token&session_id=123") as websocket:
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "Unauthorized" in data["detail"]

    def test_ws_accepts_valid_token_and_handshakes(self, client: TestClient, seed_data: dict):
        """Websocket should authorize valid JWT and send status handshake."""
        token = seed_data["employee_token"]
        session_id = "test-ws-session-handshake"

        with client.websocket_connect(f"/ws/analyze?token={token}&session_id={session_id}") as websocket:
            # First message should be connected status
            data = websocket.receive_json()
            assert data["type"] == "status"
            assert data["payload"]["status"] == "connected"
            assert data["payload"]["session_id"] == session_id

    def test_ws_processes_frame_successfully(self, client: TestClient, seed_data: dict):
        """Websocket should receive a JPEG frame, process it, and return JSON + Binary annotated image."""
        token = seed_data["employee_token"]
        session_id = "test-ws-inference-session"
        frame_bytes = _create_mock_jpeg()

        with client.websocket_connect(f"/ws/analyze?token={token}&session_id={session_id}") as websocket:
            # Consume handshake
            websocket.receive_json()

            # Send raw JPEG bytes
            websocket.send_bytes(frame_bytes)

            # Receive JSON analysis result
            json_msg = websocket.receive_json()
            assert json_msg["type"] == "analysis_result"
            assert "payload" in json_msg
            payload = json_msg["payload"]
            assert payload["session_id"] == session_id
            assert payload["face_detected"] is False  # A solid blue image has no face

            # Receive Binary annotated frame
            annotated_bytes = websocket.receive_bytes()
            assert len(annotated_bytes) > 0
            assert annotated_bytes[:2] == b"\xff\xd8"  # JPEG Magic Number check
