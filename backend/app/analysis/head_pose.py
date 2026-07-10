"""
AuthBrain AI Face Analysis Engine
Head Pose Estimator

Uses OpenCV's solvePnP algorithm with the 6-point 3D face model
to estimate pitch, yaw, and roll from 2D landmark positions.

Algorithm:
1. Map 6 known 3D face model points to detected 2D landmark pixels
2. Use solvePnP (Levenberg-Marquardt) to compute rotation vector
3. Convert rotation vector to Euler angles via Rodrigues decomposition
"""

from __future__ import annotations

import numpy as np
import cv2
from numpy.typing import NDArray

from app.analysis.face_detector import RawFaceData
from app.analysis.landmark_indices import LANDMARKS
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import HeadPoseResult
from app.utils.math_utils import rotation_matrix_to_euler_angles, normalize_angle

settings = get_settings()
logger = get_logger(__name__)

# 3D canonical face model points (in mm) — matches mediapipe landmark order
# Indexed to align with LANDMARKS.HEAD_POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)
_3D_MODEL_POINTS = np.array([
    [0.0,      0.0,    0.0  ],   # Index 1  — Nose tip
    [0.0,   -330.0, -65.0  ],   # Index 152 — Chin
    [-225.0,  170.0, -135.0],   # Index 33  — Left eye left corner
    [225.0,   170.0, -135.0],   # Index 263 — Right eye right corner
    [-150.0, -150.0, -125.0],   # Index 61  — Left mouth corner
    [150.0,  -150.0, -125.0],   # Index 291 — Right mouth corner
], dtype=np.float64)

# Threshold angles for "facing forward"
_YAW_FORWARD_THRESHOLD   = 20.0  # degrees
_PITCH_FORWARD_THRESHOLD = 20.0  # degrees


class HeadPoseEstimator:
    """
    Estimates head orientation (pitch, yaw, roll) from 2D facial landmarks.

    Uses a simplified pinhole camera model with focal length estimated
    from frame dimensions — sufficient for real-time behavioral analysis.
    """

    def __init__(self) -> None:
        self._prev_pitch: float = 0.0
        self._prev_yaw: float = 0.0
        self._prev_roll: float = 0.0

        # Self-calibration attributes (requires first 30 frames of face detection)
        self._calibration_frames: list[tuple[float, float, float]] = []
        self._pitch_offset: float = 0.0
        self._yaw_offset: float = 0.0
        self._roll_offset: float = 0.0

    def estimate(
        self,
        face_data: RawFaceData,
    ) -> HeadPoseResult:
        """
        Estimate head pose from facial landmarks.

        Args:
            face_data: RawFaceData from FaceDetector containing head pose points

        Returns:
            HeadPoseResult with pitch, yaw, roll in degrees
        """
        if len(face_data.head_pose_pts) < 6:
            # Fallback: return previous values if landmarks incomplete
            return HeadPoseResult(
                pitch=self._prev_pitch,
                yaw=self._prev_yaw,
                roll=self._prev_roll,
                is_facing_forward=True,
            )

        # Build 2D image points from normalized landmarks → pixel space
        image_pts = np.array([
            [pt[0] * face_data.frame_width, pt[1] * face_data.frame_height]
            for pt in face_data.head_pose_pts[:6]
        ], dtype=np.float64)

        # Build camera matrix using simplified pinhole model
        # focal_length ≈ frame_width (common approximation for webcam)
        focal_length = float(face_data.frame_width)
        center = (face_data.frame_width / 2.0, face_data.frame_height / 2.0)

        camera_matrix = np.array([
            [focal_length, 0,            center[0]],
            [0,            focal_length, center[1]],
            [0,            0,            1        ],
        ], dtype=np.float64)

        # No lens distortion assumed for webcam
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve Perspective-n-Point
        success, rotation_vec, translation_vec = cv2.solvePnP(
            _3D_MODEL_POINTS,
            image_pts,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            logger.debug("solvePnP_failed", face_index=face_data.face_index)
            return HeadPoseResult(
                pitch=self._prev_pitch,
                yaw=self._prev_yaw,
                roll=self._prev_roll,
                is_facing_forward=True,
            )

        # Convert rotation vector to Euler angles
        # Cast to float64 array to satisfy type checker (solvePnP output is a generic Mat type)
        rotation_vec_f64: np.ndarray = np.asarray(rotation_vec, dtype=np.float64)
        pitch, yaw, roll = rotation_matrix_to_euler_angles(rotation_vec_f64)

        # Calibration phase: collect first 30 frames of face detection to establish baseline offsets
        if len(self._calibration_frames) < 30:
            self._calibration_frames.append((pitch, yaw, roll))
            if len(self._calibration_frames) == 30:
                self._pitch_offset = float(np.mean([f[0] for f in self._calibration_frames]))
                self._yaw_offset   = float(np.mean([f[1] for f in self._calibration_frames]))
                self._roll_offset  = float(np.mean([f[2] for f in self._calibration_frames]))
                logger.info(
                    "head_pose_calibrated",
                    pitch_offset=self._pitch_offset,
                    yaw_offset=self._yaw_offset,
                    roll_offset=self._roll_offset
                )

        # Subtract baseline offsets to get angles relative to front-facing calibration position
        pitch = pitch - self._pitch_offset
        yaw   = yaw - self._yaw_offset
        roll  = roll - self._roll_offset

        # Normalize and apply temporal smoothing (EMA with α=0.4)
        alpha = 0.4
        pitch = normalize_angle(alpha * pitch + (1 - alpha) * self._prev_pitch)
        yaw   = normalize_angle(alpha * yaw   + (1 - alpha) * self._prev_yaw)
        roll  = normalize_angle(alpha * roll  + (1 - alpha) * self._prev_roll)

        # Update history
        self._prev_pitch = pitch
        self._prev_yaw   = yaw
        self._prev_roll  = roll

        is_facing_forward = (
            abs(yaw)   < _YAW_FORWARD_THRESHOLD and
            abs(pitch) < _PITCH_FORWARD_THRESHOLD
        )

        return HeadPoseResult(
            pitch=round(pitch, 2),
            yaw=round(yaw, 2),
            roll=round(roll, 2),
            is_facing_forward=is_facing_forward,
        )

    def reset(self) -> None:
        """Reset accumulated state (call on new session)."""
        self._prev_pitch = 0.0
        self._prev_yaw   = 0.0
        self._prev_roll  = 0.0
        self._calibration_frames.clear()
        self._pitch_offset = 0.0
        self._yaw_offset = 0.0
        self._roll_offset = 0.0
