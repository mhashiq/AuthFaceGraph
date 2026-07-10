"""
AuthBrain AI Face Analysis Engine
MediaPipe 478-Landmark Named Constants

This module defines human-readable names for all key MediaPipe FaceMesh
landmark indices. Using named constants prevents "magic number" bugs and
improves code readability significantly.

Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LandmarkIndices:
    """
    Named constants for MediaPipe FaceMesh 478-landmark model.
    All indices are 0-based integers matching the mediapipe output.
    """

    # ── Face Oval / Jawline ────────────────────────────────────────────────────
    FACE_OVAL: tuple[int, ...] = (
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
    )

    # ── Left Eye (from subject's perspective) ─────────────────────────────────
    LEFT_EYE: tuple[int, ...] = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
    LEFT_EYE_UPPER: tuple[int, ...] = (386, 387, 388, 466, 263, 249, 390, 373)
    LEFT_EYE_LOWER: tuple[int, ...] = (374, 380, 381, 382, 362, 398, 384, 385)

    # ── Right Eye ─────────────────────────────────────────────────────────────
    RIGHT_EYE: tuple[int, ...] = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
    RIGHT_EYE_UPPER: tuple[int, ...] = (159, 160, 161, 246, 33, 173, 157, 158)
    RIGHT_EYE_LOWER: tuple[int, ...] = (145, 153, 154, 155, 133, 144, 163, 7)

    # ── EAR calculation points (6 points per eye) ─────────────────────────────
    # P1 (inner corner), P4 (outer corner), P2/P6 (upper/lower inner), P3/P5 (upper/lower outer)
    LEFT_EYE_EAR_POINTS: tuple[int, ...] = (362, 385, 387, 263, 373, 380)
    RIGHT_EYE_EAR_POINTS: tuple[int, ...] = (33, 160, 158, 133, 153, 144)

    # ── Iris ──────────────────────────────────────────────────────────────────
    LEFT_IRIS: tuple[int, ...] = (474, 475, 476, 477)
    RIGHT_IRIS: tuple[int, ...] = (469, 470, 471, 472)
    LEFT_IRIS_CENTER: int = 473
    RIGHT_IRIS_CENTER: int = 468

    # ── Eyebrows ──────────────────────────────────────────────────────────────
    LEFT_EYEBROW: tuple[int, ...] = (336, 296, 334, 293, 300, 276, 283, 282, 295, 285)
    RIGHT_EYEBROW: tuple[int, ...] = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)

    # ── Nose ──────────────────────────────────────────────────────────────────
    NOSE_BRIDGE: tuple[int, ...] = (6, 197, 195, 5)
    NOSE_TIP: int = 4
    NOSE_BOTTOM: int = 2
    NOSE_LEFT_WING: int = 79
    NOSE_RIGHT_WING: int = 309
    NOSE_CONTOUR: tuple[int, ...] = (129, 49, 131, 134, 51, 5, 281, 363, 360, 279)

    # ── Lips / Mouth ──────────────────────────────────────────────────────────
    LIPS_UPPER_OUTER: tuple[int, ...] = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
    LIPS_LOWER_OUTER: tuple[int, ...] = (146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
    LIPS_UPPER_INNER: tuple[int, ...] = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
    LIPS_LOWER_INNER: tuple[int, ...] = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)

    # MAR (Mouth Aspect Ratio) key points
    # Top lip center, bottom lip center, left corner, right corner
    MOUTH_TOP: int = 13       # Upper lip center inner
    MOUTH_BOTTOM: int = 14    # Lower lip center inner
    MOUTH_LEFT: int = 61      # Left lip corner
    MOUTH_RIGHT: int = 291    # Right lip corner

    # MAR 6-point calculation (matches EAR structure)
    MOUTH_MAR_POINTS: tuple[int, ...] = (61, 39, 0, 291, 321, 405)

    # Smile corners
    SMILE_LEFT: int = 61
    SMILE_RIGHT: int = 291

    # ── Chin / Jawline ────────────────────────────────────────────────────────
    CHIN_TIP: int = 152
    JAW_LEFT: int = 234
    JAW_RIGHT: int = 454

    # ── Forehead ──────────────────────────────────────────────────────────────
    FOREHEAD_CENTER: int = 10
    FOREHEAD_LEFT: int = 109
    FOREHEAD_RIGHT: int = 338

    # ── Cheeks ────────────────────────────────────────────────────────────────
    LEFT_CHEEK: int = 234
    RIGHT_CHEEK: int = 454

    # ── Head Pose 3D Reference Points ─────────────────────────────────────────
    # These 6 landmarks correspond to the canonical 3D face model points
    # used with solvePnP for head pose estimation
    HEAD_POSE_LANDMARKS: tuple[int, ...] = (
        1,    # Nose tip
        152,  # Chin
        33,   # Left eye left corner
        263,  # Right eye right corner
        61,   # Left mouth corner
        291,  # Right mouth corner
    )

    # ── Symmetry Reference Points ──────────────────────────────────────────────
    # Left-right pairs for facial symmetry calculation
    SYMMETRY_PAIRS: tuple[tuple[int, int], ...] = (
        (33, 263),    # Eye corners
        (61, 291),    # Mouth corners
        (234, 454),   # Jaw endpoints
        (70, 300),    # Eyebrow ends
        (159, 386),   # Eye tops
        (145, 374),   # Eye bottoms
    )


# Module-level singleton instance
LANDMARKS = LandmarkIndices()
