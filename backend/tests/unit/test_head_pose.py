"""
AuthBrain AI Face Analysis Engine
Unit Tests — Head Pose Estimator

Tests head pose estimation, facing forward threshold rules, and validation fallbacks.
"""

from __future__ import annotations

import pytest
import numpy as np

from app.analysis.head_pose import HeadPoseEstimator
from app.analysis.face_detector import RawFaceData
from app.models.schemas import Landmark, FaceBoundingBox


def make_raw_face_data(head_pose_pts: list[tuple[float, float]]) -> RawFaceData:
    """Helper: create minimal RawFaceData for head pose testing."""
    landmarks = [Landmark(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    return RawFaceData(
        landmarks=landmarks,
        landmarks_px=[(320, 240)] * 478,
        bounding_box=FaceBoundingBox(x=0.2, y=0.1, width=0.6, height=0.8),
        detection_confidence=0.99,
        face_index=0,
        frame_width=640,
        frame_height=480,
        timestamp_ms=33,
        left_eye_ear_pts=[],
        right_eye_ear_pts=[],
        mouth_mar_pts=[],
        left_iris_pts=[],
        right_iris_pts=[],
        head_pose_pts=head_pose_pts,
    )


class TestHeadPoseEstimator:

    def test_incomplete_landmarks_fallback(self):
        """Should fall back to previous values (initially 0.0) if points count < 6."""
        estimator = HeadPoseEstimator()
        face_data = make_raw_face_data([(0.5, 0.5)] * 5)  # only 5 points

        result = estimator.estimate(face_data)
        assert result.pitch == 0.0
        assert result.yaw == 0.0
        assert result.roll == 0.0
        assert result.is_facing_forward is True

    def test_frontal_face_zero_rotation(self):
        """A symmetrical projection of the 6 key points should produce minimal rotations."""
        estimator = HeadPoseEstimator()
        # Mathematically projected points for [Nose, Chin, L eye, R eye, L mouth, R mouth]
        pts = [
            (0.5, 0.5),
            (0.5, 0.0294),
            (0.2399, 0.762),
            (0.7601, 0.762),
            (0.3286, 0.2714),
            (0.6714, 0.2714),
        ]
        face_data = make_raw_face_data(pts)
        result = estimator.estimate(face_data)

        # Check values are within reasonable forward-facing bounds
        assert abs(result.yaw) < 5.0
        assert abs(result.pitch) < 5.0
        assert abs(result.roll) < 5.0
        assert result.is_facing_forward is True

    def test_turned_face_yaw_threshold(self):
        """Turning the head to the right (around Y axis) should trigger large yaw and is_facing_forward=False."""
        estimator = HeadPoseEstimator()

        # Mathematically projected points for head turned right (yaw ~ 57 deg)
        pts_turned_right = [
            (0.5, 0.5),
            (0.4433, 0.0440),
            (0.2894, 0.7030),
            (0.5108, 0.8072),
            (0.3241, 0.3111),
            (0.4701, 0.2519),
        ]
        face_data = make_raw_face_data(pts_turned_right)
        result = estimator.estimate(face_data)

        assert abs(result.yaw) > 20.0
        assert result.is_facing_forward is False
