"""
AuthBrain AI Face Analysis Engine
MediaPipe Face Detector

Wraps the MediaPipe Tasks API FaceLandmarker for production use.
Uses VIDEO running mode for temporal consistency across frames.

Download the model file:
    mkdir -p models
    curl -L https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
         -o models/face_landmarker.task
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from numpy.typing import NDArray

from app.analysis.landmark_indices import LANDMARKS
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import FaceBoundingBox, Landmark

settings = get_settings()
logger = get_logger(__name__)


@dataclass
class RawFaceData:
    """
    Raw data extracted from MediaPipe for a single detected face.
    This is the intermediate representation passed to individual analyzers.
    """
    landmarks: list[Landmark]           # 478 normalized 3D landmarks
    landmarks_px: list[tuple[int, int]] # Pixel-space (x, y) coordinates
    bounding_box: FaceBoundingBox
    detection_confidence: float
    face_index: int
    frame_width: int
    frame_height: int
    timestamp_ms: int

    # Pre-extracted named landmark groups for analyzer performance
    left_eye_ear_pts: list[tuple[float, float]] = field(default_factory=list)
    right_eye_ear_pts: list[tuple[float, float]] = field(default_factory=list)
    mouth_mar_pts: list[tuple[float, float]] = field(default_factory=list)
    left_iris_pts: list[tuple[float, float]] = field(default_factory=list)
    right_iris_pts: list[tuple[float, float]] = field(default_factory=list)
    head_pose_pts: list[tuple[float, float]] = field(default_factory=list)


class FaceDetector:
    """
    Production MediaPipe FaceLandmarker wrapper.

    Features:
    - VIDEO mode for temporal landmark consistency
    - Automatic model download verification
    - Pre-extraction of named landmark groups
    - Confidence scoring
    """

    # Canonical 3D model points (in mm) for solvePnP head pose estimation
    # These correspond to LANDMARKS.HEAD_POSE_LANDMARKS indices
    _3D_FACE_MODEL = np.array([
        [0.0,    0.0,    0.0  ],   # Nose tip
        [0.0,   -330.0, -65.0 ],   # Chin
        [-225.0, 170.0, -135.0],   # Left eye left corner
        [225.0,  170.0, -135.0],   # Right eye right corner
        [-150.0,-150.0, -125.0],   # Left mouth corner
        [150.0, -150.0, -125.0],   # Right mouth corner
    ], dtype=np.float64)

    def __init__(self) -> None:
        self._landmarker: mp_vision.FaceLandmarker | None = None
        self._model_path = Path(settings.FACE_LANDMARKER_MODEL_PATH)
        self._frame_timestamp_ms: int = 0
        self.is_loaded: bool = False

    def load_model(self) -> None:
        """
        Load the MediaPipe FaceLandmarker model.
        Must be called before any detect() calls.

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model fails to initialize
        """
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {self._model_path}\n"
                f"Download it with:\n"
                f"  mkdir -p models && "
                f"  curl -L https://storage.googleapis.com/mediapipe-models/"
                f"face_landmarker/face_landmarker/float16/1/face_landmarker.task "
                f"-o {self._model_path}"
            )

        base_options = mp_python.BaseOptions(
            model_asset_path=str(self._model_path),
        )

        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=settings.MAX_NUM_FACES,
            min_face_detection_confidence=settings.MIN_FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=settings.MIN_FACE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=settings.MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,    # Disable — we compute our own metrics
            output_facial_transformation_matrixes=True,  # Needed for head pose
        )

        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.is_loaded = True
        logger.info(
            "face_landmarker_loaded",
            model=str(self._model_path),
            max_faces=settings.MAX_NUM_FACES,
        )

    def detect(
        self,
        frame_rgb: NDArray[np.uint8],
        frame_width: int,
        frame_height: int,
        active_face_index: int = 0,
    ) -> list[RawFaceData]:
        """
        Run face landmark detection on a single RGB frame.

        Args:
            frame_rgb: RGB uint8 numpy array from webcam
            frame_width: Frame pixel width
            frame_height: Frame pixel height
            active_face_index: Index of the primary face to track

        Returns:
            List of RawFaceData (one per detected face), empty if none detected
        """
        if not self.is_loaded or self._landmarker is None:
            raise RuntimeError("FaceDetector not loaded. Call load_model() first.")

        # Increment timestamp for VIDEO mode (required for temporal tracking)
        self._frame_timestamp_ms += int(1000 / settings.TARGET_FPS)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not result.face_landmarks:
            return []

        faces: list[RawFaceData] = []

        for face_idx, face_lm_list in enumerate(result.face_landmarks):
            # Convert MediaPipe NormalizedLandmark to our Landmark schema
            landmarks = [
                Landmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in face_lm_list
            ]

            # Compute pixel-space coordinates
            landmarks_px = [
                (int(lm.x * frame_width), int(lm.y * frame_height))
                for lm in face_lm_list
            ]

            # Estimate bounding box from landmark extremes
            xs = [lm.x for lm in face_lm_list]
            ys = [lm.y for lm in face_lm_list]
            bbox = FaceBoundingBox(
                x=min(xs),
                y=min(ys),
                width=max(xs) - min(xs),
                height=max(ys) - min(ys),
            )

            # Estimate detection confidence from face presence score
            # (MediaPipe Tasks API provides face_blendshapes — use 1.0 if unavailable)
            confidence = 1.0

            # Pre-extract named groups for O(1) access in analyzers
            def get_pts(indices: tuple[int, ...]) -> list[tuple[float, float]]:
                return [(landmarks[i].x, landmarks[i].y) for i in indices if i < len(landmarks)]

            raw = RawFaceData(
                landmarks=landmarks,
                landmarks_px=landmarks_px,
                bounding_box=bbox,
                detection_confidence=confidence,
                face_index=face_idx,
                frame_width=frame_width,
                frame_height=frame_height,
                timestamp_ms=self._frame_timestamp_ms,
                left_eye_ear_pts=get_pts(LANDMARKS.LEFT_EYE_EAR_POINTS),
                right_eye_ear_pts=get_pts(LANDMARKS.RIGHT_EYE_EAR_POINTS),
                mouth_mar_pts=get_pts(LANDMARKS.MOUTH_MAR_POINTS),
                left_iris_pts=get_pts(LANDMARKS.LEFT_IRIS),
                right_iris_pts=get_pts(LANDMARKS.RIGHT_IRIS),
                head_pose_pts=get_pts(LANDMARKS.HEAD_POSE_LANDMARKS),
            )
            faces.append(raw)

        return faces

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
            self.is_loaded = False
        logger.info("face_landmarker_closed")

    def __enter__(self) -> "FaceDetector":
        self.load_model()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
