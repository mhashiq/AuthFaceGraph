"""
AuthBrain AI Face Analysis Engine
Mouth Analyzer

Detects mouth state including Mouth Aspect Ratio (MAR), yawning,
and smile intensity from MediaPipe facial landmarks.

MAR > 0.6 for 15+ consecutive frames = yawn detected.
Smile intensity is derived from lip corner position relative to face width.
"""

from __future__ import annotations

import numpy as np

from app.analysis.face_detector import RawFaceData
from app.analysis.landmark_indices import LANDMARKS
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import MouthAnalysisResult
from app.utils.math_utils import compute_mar, compute_smile_intensity, euclidean_distance

settings = get_settings()
logger = get_logger(__name__)


class MouthAnalyzer:
    """
    Analyzes mouth and facial expression metrics from face landmark data.

    Tracks:
    - MAR (Mouth Aspect Ratio)
    - Yawn detection with temporal confirmation
    - Smile intensity from lip corner geometry
    - Mouth openness percentage
    """

    def __init__(self) -> None:
        self._yawn_consec_frames: int = 0
        self._yawn_count: int = 0
        self._prev_mar: float = 0.0

    def analyze(self, face_data: RawFaceData) -> MouthAnalysisResult:
        """
        Analyze mouth state for a single face.

        Args:
            face_data: RawFaceData with pre-extracted mouth landmarks

        Returns:
            MouthAnalysisResult with all mouth metrics
        """
        landmarks = face_data.landmarks

        if len(landmarks) < 400:
            return self._default_result()

        # ── MAR Calculation ────────────────────────────────────────────────────
        mar = self._compute_mar(face_data)
        is_open = mar > 0.3

        # ── Temporal smoothing (EMA) ───────────────────────────────────────────
        alpha = 0.35
        mar = alpha * mar + (1 - alpha) * self._prev_mar
        self._prev_mar = mar

        # ── Yawn Detection ─────────────────────────────────────────────────────
        yawn_detected = False
        yawn_confidence = 0.0

        if mar >= settings.MAR_YAWN_THRESHOLD:
            self._yawn_consec_frames += 1
        else:
            self._yawn_consec_frames = 0

        if self._yawn_consec_frames >= settings.MAR_YAWN_CONSEC_FRAMES:
            yawn_detected = True
            # Scale confidence based on how far above threshold
            yawn_confidence = float(np.clip(
                (mar - settings.MAR_YAWN_THRESHOLD) / 0.3 + 0.7,
                0.7, 1.0
            ))

        # ── Smile Intensity ────────────────────────────────────────────────────
        smile_intensity = self._compute_smile_intensity(face_data, mar)

        # ── Mouth Openness Percentage ─────────────────────────────────────────
        mouth_openness_pct = float(np.clip(mar / 0.8 * 100.0, 0.0, 100.0))

        return MouthAnalysisResult(
            mar=round(mar, 4),
            is_open=is_open,
            yawn_detected=yawn_detected,
            yawn_confidence=round(yawn_confidence, 3),
            smile_intensity=round(smile_intensity, 3),
            mouth_openness_percent=round(mouth_openness_pct, 1),
        )

    def _compute_mar(self, face_data: RawFaceData) -> float:
        """Calculate MAR from pre-extracted mouth points."""
        if len(face_data.mouth_mar_pts) < 6:
            logger.debug("insufficient_mar_landmarks", count=len(face_data.mouth_mar_pts))
            return 0.0
        try:
            return compute_mar(face_data.mouth_mar_pts[:6])
        except Exception as exc:
            logger.warning("mar_computation_failed", error=str(exc))
            return 0.0

    def _compute_smile_intensity(self, face_data: RawFaceData, mar: float) -> float:
        """
        Compute smile intensity from mouth width vs height ratio.
        Also suppresses smile during yawning (wide mouth ≠ smile).
        """
        landmarks = face_data.landmarks

        if len(landmarks) <= max(
            LANDMARKS.SMILE_LEFT,
            LANDMARKS.SMILE_RIGHT,
            LANDMARKS.MOUTH_TOP,
            LANDMARKS.MOUTH_BOTTOM,
            LANDMARKS.JAW_LEFT,
            LANDMARKS.JAW_RIGHT,
        ):
            return 0.0

        left_corner = (landmarks[LANDMARKS.SMILE_LEFT].x, landmarks[LANDMARKS.SMILE_LEFT].y)
        right_corner = (landmarks[LANDMARKS.SMILE_RIGHT].x, landmarks[LANDMARKS.SMILE_RIGHT].y)
        mouth_top = (landmarks[LANDMARKS.MOUTH_TOP].x, landmarks[LANDMARKS.MOUTH_TOP].y)
        mouth_bottom = (landmarks[LANDMARKS.MOUTH_BOTTOM].x, landmarks[LANDMARKS.MOUTH_BOTTOM].y)
        left_cheek = (landmarks[LANDMARKS.JAW_LEFT].x, landmarks[LANDMARKS.JAW_LEFT].y)
        right_cheek = (landmarks[LANDMARKS.JAW_RIGHT].x, landmarks[LANDMARKS.JAW_RIGHT].y)

        intensity = compute_smile_intensity(
            left_corner, right_corner, mouth_top, mouth_bottom, left_cheek, right_cheek
        )

        # Suppress smile score during yawning (high MAR with large vertical opening)
        if mar > settings.MAR_YAWN_THRESHOLD * 0.85:
            intensity *= 0.3

        return intensity

    def _default_result(self) -> MouthAnalysisResult:
        """Return a safe default when landmarks are insufficient."""
        return MouthAnalysisResult(
            mar=0.0,
            is_open=False,
            yawn_detected=False,
            yawn_confidence=0.0,
            smile_intensity=0.0,
            mouth_openness_percent=0.0,
        )

    def reset(self) -> None:
        """Reset accumulated state for a new session."""
        self._yawn_consec_frames = 0
        self._yawn_count = 0
        self._prev_mar = 0.0
