"""
AuthBrain AI Face Analysis Engine
Face Quality Scorer

Computes a composite quality score for each detected face based on:
- Image sharpness (Laplacian variance)
- Illumination quality
- Face bounding box size (too small = unreliable landmarks)
- Landmark confidence (from detection score)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.analysis.face_detector import RawFaceData
from app.core.logging import get_logger
from app.models.schemas import QualityResult
from app.utils.frame_utils import compute_illumination, compute_sharpness

logger = get_logger(__name__)

# Minimum face area ratio considered reliable
_MIN_FACE_AREA_RATIO = 0.02  # Face must occupy at least 2% of frame area


class QualityScorer:
    """
    Scores face and frame quality from multiple orthogonal metrics.

    Returns a composite QualityResult with:
    - Overall score [0, 1]
    - Individual component scores
    """

    def score(
        self,
        face_data: RawFaceData,
        frame_bgr: NDArray[np.uint8],
    ) -> QualityResult:
        """
        Compute quality metrics for a face detection result.

        Args:
            face_data: RawFaceData with bounding box and detection confidence
            frame_bgr: Original BGR frame for sharpness/illumination

        Returns:
            QualityResult with composite score
        """
        # ── Sharpness ──────────────────────────────────────────────────────────
        sharpness = compute_sharpness(frame_bgr)

        # ── Illumination ───────────────────────────────────────────────────────
        illumination = compute_illumination(frame_bgr)

        # ── Face Size Ratio ────────────────────────────────────────────────────
        face_area = face_data.bounding_box.width * face_data.bounding_box.height
        # Penalize very small faces; cap benefit at 25% of frame area
        face_size_ratio = float(np.clip(face_area / 0.25, 0.0, 1.0))

        # ── Landmark Confidence ────────────────────────────────────────────────
        landmark_confidence = face_data.detection_confidence

        # ── Composite Weighted Score ──────────────────────────────────────────
        weights = {
            "sharpness": 0.25,
            "illumination": 0.25,
            "face_size": 0.25,
            "confidence": 0.25,
        }
        overall = (
            weights["sharpness"]    * sharpness +
            weights["illumination"] * illumination +
            weights["face_size"]    * face_size_ratio +
            weights["confidence"]   * landmark_confidence
        )
        overall = float(np.clip(overall, 0.0, 1.0))

        return QualityResult(
            overall_score=round(overall, 3),
            sharpness=round(sharpness, 3),
            illumination=round(illumination, 3),
            face_size_ratio=round(face_size_ratio, 3),
            landmark_confidence=round(landmark_confidence, 3),
        )
