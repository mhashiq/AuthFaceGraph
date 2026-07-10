"""
AuthBrain AI Face Analysis Engine
Eye Analyzer

Calculates Eye Aspect Ratio (EAR), detects blinks, measures eye closure
duration, and estimates gaze direction from iris landmarks.

EAR Formula (Soukupová & Čech, 2016):
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

Blink detection requires EAR < threshold for N consecutive frames.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from app.analysis.face_detector import RawFaceData
from app.analysis.landmark_indices import LANDMARKS
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import EyeAnalysisResult, EyeResult, GazeDirection
from app.utils.math_utils import compute_ear, euclidean_distance

settings = get_settings()
logger = get_logger(__name__)

# Rolling window for blink frequency calculation (1 minute of timestamps)
_BLINK_HISTORY_SECONDS = 60.0


class EyeAnalyzer:
    """
    Analyzes eye state including EAR, blink detection, closure duration,
    and gaze direction from MediaPipe 478-landmark face data.
    """

    def __init__(self) -> None:
        # Blink state machine
        self._consec_below_threshold: int = 0
        self._blink_count: int = 0
        self._eye_closed_since: float | None = None
        self._closure_duration_ms: float = 0.0

        # Rolling blink timestamps for blinks-per-minute calculation
        self._blink_timestamps: deque[float] = deque()

        # Previous iris positions for gaze velocity
        self._prev_left_iris:  tuple[float, float] | None = None
        self._prev_right_iris: tuple[float, float] | None = None

    def analyze(self, face_data: RawFaceData) -> EyeAnalysisResult:
        """
        Perform full eye analysis for a single face.

        Args:
            face_data: RawFaceData with pre-extracted eye and iris landmarks

        Returns:
            EyeAnalysisResult with all eye metrics
        """
        now = time.time()

        # ── EAR Calculation ────────────────────────────────────────────────────
        left_ear = self._compute_eye_ear(face_data.left_eye_ear_pts, "left")
        right_ear = self._compute_eye_ear(face_data.right_eye_ear_pts, "right")
        avg_ear = (left_ear + right_ear) / 2.0

        # ── Blink Detection (state machine) ───────────────────────────────────
        blink_detected = False
        is_closed = avg_ear < settings.EAR_BLINK_THRESHOLD

        if is_closed:
            self._consec_below_threshold += 1
            if self._eye_closed_since is None:
                self._eye_closed_since = now
        else:
            if self._consec_below_threshold >= settings.EAR_BLINK_CONSEC_FRAMES:
                # Valid blink completed
                blink_detected = True
                self._blink_count += 1
                self._blink_timestamps.append(now)
            self._consec_below_threshold = 0

            if self._eye_closed_since is not None:
                self._closure_duration_ms = (now - self._eye_closed_since) * 1000.0
                self._eye_closed_since = None

        # ── Current closure duration ───────────────────────────────────────────
        if self._eye_closed_since is not None:
            current_closure_ms = (now - self._eye_closed_since) * 1000.0
        else:
            current_closure_ms = self._closure_duration_ms

        # ── Blinks per minute (rolling 60s window) ────────────────────────────
        cutoff = now - _BLINK_HISTORY_SECONDS
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()
        blinks_per_minute = len(self._blink_timestamps)  # count in last 60s = bpm

        # ── Gaze Direction (from iris position relative to eye corners) ────────
        gaze_dir = self._compute_gaze_direction(
            face_data.left_iris_pts,
            face_data.right_iris_pts,
            face_data.landmarks,
            is_closed,
        )

        # ── Gaze offset (normalized) ───────────────────────────────────────────
        left_gaze_x, left_gaze_y = self._compute_gaze_offset(
            face_data.left_iris_pts,
            [face_data.landmarks[i] for i in LANDMARKS.LEFT_EYE_EAR_POINTS if i < len(face_data.landmarks)]
        )
        right_gaze_x, right_gaze_y = self._compute_gaze_offset(
            face_data.right_iris_pts,
            [face_data.landmarks[i] for i in LANDMARKS.RIGHT_EYE_EAR_POINTS if i < len(face_data.landmarks)]
        )

        return EyeAnalysisResult(
            left=EyeResult(
                ear=round(left_ear, 4),
                is_open=left_ear >= settings.EAR_BLINK_THRESHOLD,
                gaze_x=round(left_gaze_x, 4),
                gaze_y=round(left_gaze_y, 4),
            ),
            right=EyeResult(
                ear=round(right_ear, 4),
                is_open=right_ear >= settings.EAR_BLINK_THRESHOLD,
                gaze_x=round(right_gaze_x, 4),
                gaze_y=round(right_gaze_y, 4),
            ),
            average_ear=round(avg_ear, 4),
            blink_detected=blink_detected,
            blink_count=self._blink_count,
            eye_closure_duration_ms=round(current_closure_ms, 2),
            gaze_direction=gaze_dir,
            blinks_per_minute=round(blinks_per_minute, 1),
        )

    def _compute_eye_ear(
        self,
        eye_pts: list[tuple[float, float]],
        side: str,
    ) -> float:
        """Calculate EAR for one eye. Returns 0.3 (open estimate) if data missing."""
        if len(eye_pts) < 6:
            logger.debug("insufficient_ear_landmarks", side=side, count=len(eye_pts))
            return 0.30
        try:
            return compute_ear(eye_pts[:6])
        except Exception as exc:
            logger.warning("ear_computation_failed", side=side, error=str(exc))
            return 0.30

    def _compute_gaze_direction(
        self,
        left_iris: list[tuple[float, float]],
        right_iris: list[tuple[float, float]],
        landmarks: list,
        is_closed: bool,
    ) -> GazeDirection:
        """
        Estimate gaze direction from iris center position relative to eye corners.

        The iris center (normalized) is compared against the eye corner midpoint.
        Displacement determines left/right/up/down gaze.
        """
        if is_closed:
            return GazeDirection.CLOSED

        if not left_iris or not right_iris:
            return GazeDirection.CENTER

        # Average iris X position (0=left of frame, 1=right of frame)
        left_iris_x = np.mean([p[0] for p in left_iris])
        right_iris_x = np.mean([p[0] for p in right_iris])

        # Eye corner reference for left eye
        left_inner = landmarks[LANDMARKS.LEFT_EYE_EAR_POINTS[0]] if len(landmarks) > LANDMARKS.LEFT_EYE_EAR_POINTS[0] else None
        left_outer = landmarks[LANDMARKS.LEFT_EYE_EAR_POINTS[3]] if len(landmarks) > LANDMARKS.LEFT_EYE_EAR_POINTS[3] else None

        if left_inner and left_outer:
            eye_center_x = (left_inner.x + left_outer.x) / 2.0
            offset_x = left_iris_x - eye_center_x

            if offset_x < -0.02:
                return GazeDirection.LEFT
            elif offset_x > 0.02:
                return GazeDirection.RIGHT

        # Check vertical gaze from iris Y vs eye center Y
        left_iris_y = np.mean([p[1] for p in left_iris])
        upper_y = landmarks[LANDMARKS.LEFT_EYE_EAR_POINTS[1]].y if len(landmarks) > LANDMARKS.LEFT_EYE_EAR_POINTS[1] else 0
        lower_y = landmarks[LANDMARKS.LEFT_EYE_EAR_POINTS[4]].y if len(landmarks) > LANDMARKS.LEFT_EYE_EAR_POINTS[4] else 0
        eye_center_y = (upper_y + lower_y) / 2.0
        offset_y = left_iris_y - eye_center_y

        if offset_y < -0.015:
            return GazeDirection.UP
        elif offset_y > 0.015:
            return GazeDirection.DOWN

        return GazeDirection.CENTER

    def _compute_gaze_offset(
        self,
        iris_pts: list[tuple[float, float]],
        eye_landmarks: list,
    ) -> tuple[float, float]:
        """Return normalized gaze offset (dx, dy) from iris to eye center."""
        if not iris_pts or not eye_landmarks:
            return 0.0, 0.0

        iris_x = float(np.mean([p[0] for p in iris_pts]))
        iris_y = float(np.mean([p[1] for p in iris_pts]))

        eye_xs = [lm.x for lm in eye_landmarks]
        eye_ys = [lm.y for lm in eye_landmarks]

        if not eye_xs:
            return 0.0, 0.0

        center_x = float(np.mean(eye_xs))
        center_y = float(np.mean(eye_ys))

        return iris_x - center_x, iris_y - center_y

    def reset(self) -> None:
        """Reset all accumulated state for a new session."""
        self._consec_below_threshold = 0
        self._blink_count = 0
        self._eye_closed_since = None
        self._closure_duration_ms = 0.0
        self._blink_timestamps.clear()
        self._prev_left_iris = None
        self._prev_right_iris = None
