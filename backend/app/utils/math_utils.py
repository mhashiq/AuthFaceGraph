"""
AuthBrain AI Face Analysis Engine
Math Utilities

Provides optimized geometric calculation functions used across the analysis pipeline.
Uses numpy for vectorized operations to maintain performance targets.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """
    Calculate 2D Euclidean distance between two points.

    Args:
        p1: First point (x, y)
        p2: Second point (x, y)

    Returns:
        Scalar distance
    """
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def euclidean_distance_3d(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
    """Calculate 3D Euclidean distance."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def compute_ear(eye_landmarks: list[tuple[float, float]]) -> float:
    """
    Compute Eye Aspect Ratio (EAR) from 6 landmark points.

    Formula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Where p1..p6 are the 6 eye landmarks in order:
        p1 = left corner
        p2 = upper-left
        p3 = upper-right
        p4 = right corner
        p5 = lower-right
        p6 = lower-left

    Args:
        eye_landmarks: List of 6 (x, y) tuples

    Returns:
        EAR value (typically 0.0 – 0.4; <0.25 indicates closed eye)
    """
    if len(eye_landmarks) != 6:
        raise ValueError(f"EAR requires exactly 6 landmarks, got {len(eye_landmarks)}")

    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in eye_landmarks]

    vertical_1 = float(np.linalg.norm(p2 - p6))
    vertical_2 = float(np.linalg.norm(p3 - p5))
    horizontal = float(np.linalg.norm(p1 - p4))

    if horizontal < 1e-6:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(np.clip(ear, 0.0, 1.0))


def compute_mar(mouth_landmarks: list[tuple[float, float]]) -> float:
    """
    Compute Mouth Aspect Ratio (MAR) from 6 landmark points.

    Similar structure to EAR — measures mouth openness.
    MAR > 0.6 typically indicates yawning.

    Args:
        mouth_landmarks: 6 (x, y) tuples [left, upper-left, upper-center,
                         right, lower-right, lower-left]

    Returns:
        MAR value (0.0 = fully closed, ~0.6+ = wide open/yawn)
    """
    if len(mouth_landmarks) != 6:
        raise ValueError(f"MAR requires exactly 6 landmarks, got {len(mouth_landmarks)}")

    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in mouth_landmarks]

    vertical_1 = float(np.linalg.norm(p2 - p6))
    vertical_2 = float(np.linalg.norm(p3 - p5))
    horizontal = float(np.linalg.norm(p1 - p4))

    if horizontal < 1e-6:
        return 0.0

    mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(mar)


def compute_smile_intensity(
    left_corner: tuple[float, float],
    right_corner: tuple[float, float],
    mouth_top: tuple[float, float],
    mouth_bottom: tuple[float, float],
    left_cheek: tuple[float, float] | None = None,
    right_cheek: tuple[float, float] | None = None,
) -> float:
    """
    Estimate smile intensity from mouth shape.
    Uses face-width normalized lip-stretch to detect smiling.

    Normalization uses jaw landmarks (234, 454) as the face-width reference.
    - Jaw-normalized mouth width at rest: ~0.42–0.46
    - Jaw-normalized mouth width smiling: ~0.52–0.60
    - So threshold is 0.44 (neutral) to 0.60 (full smile) → range 0.16

    Returns:
        Float in [0, 1] — 0 = neutral/sad, 1 = maximum smile
    """
    mouth_width = euclidean_distance(left_corner, right_corner)

    # Use face width normalization for robust, non-flickering smile detection
    if left_cheek is not None and right_cheek is not None:
        face_width = euclidean_distance(left_cheek, right_cheek)
        if face_width < 1e-6:
            return 0.0
        # Jaw-normalized horizontal stretch:
        # At rest, mouth corners (61, 291) span ~50% of jaw width (234, 454)
        # Neutral: ~0.48-0.52, broad smile: ~0.62+
        stretch = mouth_width / face_width
        smile_score = float(np.clip((stretch - 0.48) / 0.14, 0.0, 1.0))
        return smile_score

    # Fallback: use mouth width vs height aspect ratio
    mouth_height = euclidean_distance(mouth_top, mouth_bottom)
    if mouth_width < 1e-6:
        return 0.0
    # Width/height > 3.0 = likely smiling (corners drawn back)
    ratio = mouth_width / max(mouth_height, 1e-6)
    smile_score = np.clip((ratio - 2.0) / 3.0, 0.0, 1.0)
    return float(smile_score)


def compute_facial_symmetry(
    left_points: NDArray[np.float64],
    right_points: NDArray[np.float64],
    face_width: float,
) -> float:
    """
    Calculate facial symmetry score by comparing mirrored landmark distances.

    Args:
        left_points: Nx2 array of left-side landmarks
        right_points: Nx2 array of right-side landmarks (mirrored)
        face_width: Reference face width for normalization

    Returns:
        Float in [0, 1] — 1.0 = perfect symmetry
    """
    if len(left_points) == 0 or face_width < 1e-6:
        return 1.0

    # Normalize points by face width
    l_norm = left_points / face_width
    r_norm = right_points / face_width

    # Average point-wise distance
    avg_distance = float(np.mean(np.linalg.norm(l_norm - r_norm, axis=1)))

    # Convert to symmetry score (lower distance = higher symmetry)
    symmetry = float(np.clip(1.0 - avg_distance * 5.0, 0.0, 1.0))
    return symmetry


def rotation_matrix_to_euler_angles(rvec: NDArray[np.float64]) -> tuple[float, float, float]:
    """
    Convert OpenCV rotation vector (Rodrigues) to Euler angles.

    Args:
        rvec: 3x1 rotation vector from solvePnP

    Returns:
        (pitch, yaw, roll) in degrees
    """
    import cv2

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    # Extract Euler angles from rotation matrix
    sy = float(np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
    singular = sy < 1e-6

    if not singular:
        pitch = float(np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])))
        yaw   = float(np.degrees(np.arctan2(-rotation_matrix[2, 0], sy)))
        roll  = float(np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])))
    else:
        pitch = float(np.degrees(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])))
        yaw   = float(np.degrees(np.arctan2(-rotation_matrix[2, 0], sy)))
        roll  = 0.0

    return pitch, yaw, roll


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-180, 180] range."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def compute_landmark_velocity(
    current: NDArray[np.float64],
    previous: NDArray[np.float64],
    dt_ms: float,
) -> float:
    """
    Compute mean landmark displacement velocity in normalized units per second.

    Args:
        current: Nx2 current landmark positions
        previous: Nx2 previous landmark positions
        dt_ms: Time delta in milliseconds

    Returns:
        Average velocity in landmarks-per-second
    """
    if dt_ms < 1.0 or len(current) == 0:
        return 0.0
    displacement = np.mean(np.linalg.norm(current - previous, axis=1))
    velocity = displacement / (dt_ms / 1000.0)
    return float(velocity)
