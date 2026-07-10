"""
AuthBrain AI Face Analysis Engine
Unit Tests — Eye Analyzer

Tests EAR computation, blink detection, and gaze direction classification.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
import numpy as np

from app.analysis.eye_analyzer import EyeAnalyzer
from app.analysis.face_detector import RawFaceData
from app.models.schemas import Landmark, FaceBoundingBox


def make_face_data(
    left_ear_pts: list[tuple[float, float]],
    right_ear_pts: list[tuple[float, float]],
    iris_pts: list[tuple[float, float]] | None = None,
) -> RawFaceData:
    """Helper: create a minimal RawFaceData for testing."""
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
        left_eye_ear_pts=left_ear_pts,
        right_eye_ear_pts=right_ear_pts,
        mouth_mar_pts=[],
        left_iris_pts=iris_pts or [],
        right_iris_pts=iris_pts or [],
        head_pose_pts=[],
    )


# ── EAR Calculation ────────────────────────────────────────────────────────────

class TestEARComputation:

    def test_open_eye_ear_above_threshold(self):
        """Open eye should produce EAR > 0.25."""
        # Simulate open eye: vertical separation ~0.15, horizontal ~0.3
        eye_pts = [
            (0.0, 0.5),   # p1 inner corner
            (0.1, 0.35),  # p2 upper-left
            (0.2, 0.35),  # p3 upper-right
            (0.3, 0.5),   # p4 outer corner
            (0.2, 0.65),  # p5 lower-right
            (0.1, 0.65),  # p6 lower-left
        ]
        face = make_face_data(eye_pts, eye_pts)
        analyzer = EyeAnalyzer()
        result = analyzer.analyze(face)
        assert result.average_ear > 0.25, f"Expected open EAR > 0.25, got {result.average_ear}"

    def test_closed_eye_ear_below_threshold(self):
        """Closed eye should produce EAR < 0.25."""
        # Near-zero vertical separation
        eye_pts = [
            (0.0, 0.5),   # inner corner
            (0.1, 0.49),  # upper-left (nearly same as lower)
            (0.2, 0.49),  # upper-right
            (0.3, 0.5),   # outer corner
            (0.2, 0.51),  # lower-right
            (0.1, 0.51),  # lower-left
        ]
        face = make_face_data(eye_pts, eye_pts)
        analyzer = EyeAnalyzer()
        result = analyzer.analyze(face)
        assert result.average_ear < 0.25, f"Expected closed EAR < 0.25, got {result.average_ear}"

    def test_ear_clipped_to_valid_range(self):
        """EAR should always be in [0, 1]."""
        eye_pts = [(float(i) * 0.05, 0.5) for i in range(6)]
        face = make_face_data(eye_pts, eye_pts)
        analyzer = EyeAnalyzer()
        result = analyzer.analyze(face)
        assert 0.0 <= result.left.ear <= 1.0
        assert 0.0 <= result.right.ear <= 1.0
        assert 0.0 <= result.average_ear <= 1.0


# ── Blink Detection ────────────────────────────────────────────────────────────

class TestBlinkDetection:

    def _closed_eye_pts(self) -> list[tuple[float, float]]:
        return [
            (0.0, 0.5), (0.1, 0.49), (0.2, 0.49),
            (0.3, 0.5), (0.2, 0.51), (0.1, 0.51),
        ]

    def _open_eye_pts(self) -> list[tuple[float, float]]:
        return [
            (0.0, 0.5), (0.1, 0.35), (0.2, 0.35),
            (0.3, 0.5), (0.2, 0.65), (0.1, 0.65),
        ]

    def test_blink_detected_after_3_closed_frames(self):
        """Blink should be detected after 3 consecutive frames below threshold."""
        analyzer = EyeAnalyzer()
        closed_pts = self._closed_eye_pts()
        open_pts   = self._open_eye_pts()
        closed_face = make_face_data(closed_pts, closed_pts)
        open_face   = make_face_data(open_pts,   open_pts)

        # 3 closed frames (trigger)
        for _ in range(3):
            result = analyzer.analyze(closed_face)
        assert not result.blink_detected, "Blink should not yet be detected during closure"

        # 1 open frame (blink completes)
        result = analyzer.analyze(open_face)
        assert result.blink_detected, "Blink should be detected on eye open"
        assert result.blink_count == 1

    def test_no_blink_on_single_closed_frame(self):
        """Single closed frame should not count as blink."""
        analyzer   = EyeAnalyzer()
        closed_pts = self._closed_eye_pts()
        open_pts   = self._open_eye_pts()

        closed_face = make_face_data(closed_pts, closed_pts)
        open_face   = make_face_data(open_pts,   open_pts)

        analyzer.analyze(closed_face)  # 1 closed frame
        result = analyzer.analyze(open_face)  # open — insufficient consec frames
        assert not result.blink_detected
        assert result.blink_count == 0

    def test_blink_count_accumulates(self):
        """Multiple blinks should be counted correctly."""
        analyzer   = EyeAnalyzer()
        closed_pts = self._closed_eye_pts()
        open_pts   = self._open_eye_pts()
        closed_face = make_face_data(closed_pts, closed_pts)
        open_face   = make_face_data(open_pts,   open_pts)

        total_blinks = 0
        for _ in range(3):  # 3 blinks
            for _ in range(3):
                analyzer.analyze(closed_face)
            result = analyzer.analyze(open_face)
            if result.blink_detected:
                total_blinks += 1

        assert total_blinks == 3, f"Expected 3 blinks, got {total_blinks}"

    def test_reset_clears_blink_count(self):
        """Reset should clear all accumulated state."""
        analyzer   = EyeAnalyzer()
        closed_pts = self._closed_eye_pts()
        open_pts   = self._open_eye_pts()
        closed_face = make_face_data(closed_pts, closed_pts)
        open_face   = make_face_data(open_pts,   open_pts)

        for _ in range(3):
            analyzer.analyze(closed_face)
        analyzer.analyze(open_face)

        analyzer.reset()
        result = analyzer.analyze(open_face)
        assert result.blink_count == 0


# ── Math Utilities ────────────────────────────────────────────────────────────

class TestMathUtils:

    def test_compute_ear_standard_values(self):
        from app.utils.math_utils import compute_ear
        # Open eye
        open_pts = [
            (0.0, 0.5), (0.1, 0.45), (0.2, 0.45),
            (0.3, 0.5), (0.2, 0.55), (0.1, 0.55),
        ]
        ear = compute_ear(open_pts)
        assert 0.25 < ear < 0.6

    def test_compute_ear_raises_on_wrong_count(self):
        from app.utils.math_utils import compute_ear
        with pytest.raises(ValueError):
            compute_ear([(0, 0)] * 5)  # needs exactly 6

    def test_euclidean_distance_correct(self):
        from app.utils.math_utils import euclidean_distance
        d = euclidean_distance((0, 0), (3, 4))
        assert abs(d - 5.0) < 1e-6

    def test_smile_intensity_range(self):
        from app.utils.math_utils import compute_smile_intensity
        intensity = compute_smile_intensity(
            left_corner=(0.3, 0.7),
            right_corner=(0.7, 0.7),
            mouth_top=(0.5, 0.65),
            mouth_bottom=(0.5, 0.75),
        )
        assert 0.0 <= intensity <= 1.0
