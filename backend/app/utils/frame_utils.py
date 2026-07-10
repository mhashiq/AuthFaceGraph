"""
AuthBrain AI Face Analysis Engine
Frame Utilities — OpenCV JPEG Encode/Decode

Handles efficient conversion between binary JPEG bytes and numpy arrays.
All operations are optimized for the WebSocket real-time pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

# JPEG encoding parameters — pre-built for performance
_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY]


def jpeg_bytes_to_frame(data: bytes) -> NDArray[np.uint8] | None:
    """
    Decode JPEG bytes (from WebSocket) into an OpenCV BGR numpy array.

    Args:
        data: Raw JPEG bytes received over WebSocket

    Returns:
        BGR numpy array (H, W, 3) or None if decode fails
    """
    try:
        np_arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame  # type: ignore[return-value]
    except Exception as exc:
        logger.warning("frame_decode_failed", error=str(exc))
        return None


def frame_to_jpeg_bytes(
    frame: NDArray[np.uint8],
    quality: int | None = None,
) -> bytes:
    """
    Encode a BGR numpy array to JPEG bytes for WebSocket transmission.

    Args:
        frame: BGR numpy array (H, W, 3)
        quality: JPEG quality (1–100). Uses settings.JPEG_QUALITY if None.

    Returns:
        JPEG bytes
    """
    params = _ENCODE_PARAMS
    if quality is not None:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    success, encoded = cv2.imencode(".jpg", frame, params)
    if not success:
        raise RuntimeError("Failed to encode frame to JPEG")
    return bytes(encoded)  # type: ignore[arg-type]


def bgr_to_rgb(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert OpenCV BGR frame to RGB (required by MediaPipe)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore[return-value]


def rgb_to_bgr(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert RGB back to BGR for OpenCV display/encoding."""
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # type: ignore[return-value]


def compute_sharpness(frame: NDArray[np.uint8]) -> float:
    """
    Compute image sharpness using the Laplacian variance method.

    A higher value means the image is sharper (more edges).
    Typical thresholds:
        < 100  = blurry
        100-500 = acceptable
        > 500  = sharp

    Args:
        frame: BGR numpy array

    Returns:
        Laplacian variance (normalized to [0, 1] by clamping at 1000)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(np.clip(laplacian_var / 1000.0, 0.0, 1.0))


def compute_illumination(frame: NDArray[np.uint8]) -> float:
    """
    Estimate illumination quality from the frame's brightness distribution.

    Returns:
        Float in [0, 1] — 1.0 = ideal lighting, 0.0 = too dark or too bright
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    # Penalize too dark (<50) and too bright (>200)
    # Ideal range: 80–170
    if mean_brightness < 50:
        return mean_brightness / 50.0
    elif mean_brightness > 200:
        return (255 - mean_brightness) / 55.0
    else:
        return 1.0


def draw_face_mesh_overlay(
    frame: NDArray[np.uint8],
    landmarks: list[tuple[float, float]],
    connections: list[tuple[int, int]] | None = None,
    landmark_color: tuple[int, int, int] = (0, 255, 100),
    connection_color: tuple[int, int, int] = (0, 200, 60),
    landmark_radius: int = 1,
    connection_thickness: int = 1,
) -> NDArray[np.uint8]:
    """
    Draw MediaPipe face mesh landmarks and connections on a frame.

    Args:
        frame: BGR numpy array to draw on (in-place)
        landmarks: List of (x, y) pixel coordinates
        connections: List of (idx_a, idx_b) pairs to draw lines between
        landmark_color: BGR color for dots
        connection_color: BGR color for lines
        landmark_radius: Dot radius in pixels
        connection_thickness: Line thickness in pixels

    Returns:
        Annotated BGR frame
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw connections first (underneath landmarks)
    if connections:
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                pt1 = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
                pt2 = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
                cv2.line(annotated, pt1, pt2, connection_color, connection_thickness, cv2.LINE_AA)

    # Draw landmark dots
    for lm_x, lm_y in landmarks:
        cx, cy = int(lm_x * w), int(lm_y * h)
        cv2.circle(annotated, (cx, cy), landmark_radius, landmark_color, -1, cv2.LINE_AA)

    return annotated


def draw_bounding_box(
    frame: NDArray[np.uint8],
    x: float, y: float, w_ratio: float, h_ratio: float,
    label: str = "",
    color: tuple[int, int, int] = (0, 255, 120),
    thickness: int = 2,
) -> NDArray[np.uint8]:
    """
    Draw a labeled bounding box on the frame.

    Args:
        frame: BGR frame (in-place modification)
        x, y: Top-left corner (normalized 0–1)
        w_ratio, h_ratio: Width/height (normalized 0–1)
        label: Optional text label above the box
        color: BGR color tuple
        thickness: Line thickness in pixels

    Returns:
        Annotated frame
    """
    h, w = frame.shape[:2]
    x1 = int(x * w)
    y1 = int(y * h)
    x2 = int((x + w_ratio) * w)
    y2 = int((y + h_ratio) * h)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    if label:
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA
        )
    return frame


def resize_for_inference(
    frame: NDArray[np.uint8],
    max_width: int = 1280,
    max_height: int = 720,
) -> tuple[NDArray[np.uint8], float]:
    """
    Resize frame to fit within max dimensions while preserving aspect ratio.

    Returns:
        (resized_frame, scale_factor)
    """
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    return frame, 1.0
