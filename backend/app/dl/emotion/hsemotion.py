"""
AuthBrain DL Platform — HSEmotion Pretrained Emotion Model

Wraps the HSEmotion library for real-time facial emotion recognition.
HSEmotion (Savchenko, 2022) achieves state-of-the-art accuracy on AffectNet
with an EfficientNet-b0 backbone.

Model: EfficientNet-b0 trained on AffectNet-8 (8 classes)
Input: 224×224 cropped face image (RGB)
Output: 8 emotion probabilities

Install: pip install hsemotion-onnx

If not installed, falls back to a lightweight OpenCV-based heuristic model
to maintain functionality without the dependency.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.dl.base import EmotionModelBase, EmotionPrediction

settings = get_settings()
logger = get_logger(__name__)

EMOTION_LABELS = [
    "neutral", "happy", "sad", "surprise",
    "fear", "disgust", "anger", "contempt",
]


class HSEmotionModel(EmotionModelBase):
    """
    HSEmotion ONNX wrapper for real-time facial emotion recognition.

    Uses hsemotion-onnx for fast CPU inference (~10–20ms).
    Falls back to a heuristic model if the library is not installed.
    """

    def __init__(self) -> None:
        super().__init__()
        self._recognizer: "Any" = None
        self._mode: str = "unloaded"  # "hsemotion" | "fallback"

    @property
    def model_id(self) -> str:
        return "hsemotion"

    @property
    def emotion_classes(self) -> list[str]:
        return EMOTION_LABELS

    def _load_impl(self) -> None:
        # Try hsemotion-onnx first
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
            cache_dir = Path(settings.DL_MODEL_CACHE_DIR) / "hsemotion"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # HSEmotion will auto-download weights to its own cache
            self._recognizer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
            self._mode = "hsemotion"
            logger.info("hsemotion_loaded", mode="onnx")
            return
        except ImportError:
            logger.warning(
                "hsemotion_not_installed",
                hint="pip install hsemotion-onnx",
                fallback="Using heuristic fallback model",
            )
        except Exception as exc:
            logger.warning("hsemotion_load_failed", error=str(exc), fallback="Using heuristic")

        # Fallback: simple rule-based heuristic using facial geometry
        # Not accurate — placeholder until library is installed
        self._mode = "fallback"
        logger.info("hsemotion_fallback_loaded")

    def _predict_impl(self, face_crop_rgb: np.ndarray) -> EmotionPrediction:
        if self._mode == "hsemotion":
            return self._predict_hsemotion(face_crop_rgb)
        else:
            return self._predict_fallback(face_crop_rgb)

    def _predict_hsemotion(self, face_crop_rgb: np.ndarray) -> EmotionPrediction:
        """Run HSEmotion ONNX inference."""
        try:
            # HSEmotion expects BGR
            face_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
            emotion_label, scores = self._recognizer.predict_emotions(
                face_bgr, logits=False
            )

            hsemotion_classes = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
            hsemotion_to_internal = {
                "anger": "anger",
                "contempt": "contempt",
                "disgust": "disgust",
                "fear": "fear",
                "happiness": "happy",
                "happy": "happy",
                "neutral": "neutral",
                "sadness": "sad",
                "sad": "sad",
                "surprise": "surprise"
            }

            probs = {e: 0.0 for e in EMOTION_LABELS}

            if isinstance(scores, dict):
                for k, v in scores.items():
                    key_lower = k.lower()
                    mapped_key = hsemotion_to_internal.get(key_lower, key_lower)
                    if mapped_key in probs:
                        probs[mapped_key] = float(v)
            else:
                scores_arr = np.array(scores, dtype=np.float32)
                # Softmax normalize if not already probabilities
                if scores_arr.max() > 1.5:
                    scores_arr = np.exp(scores_arr - scores_arr.max())
                    scores_arr /= scores_arr.sum()
                for i, val in enumerate(scores_arr):
                    hs_label = hsemotion_classes[i]
                    mapped_key = hsemotion_to_internal.get(hs_label, hs_label)
                    if mapped_key in probs:
                        probs[mapped_key] = float(val)

            top_emotion = max(probs, key=probs.get)  # type: ignore[arg-type]
            top_conf = probs[top_emotion]

            return EmotionPrediction(
                emotion=top_emotion,
                confidence=top_conf,
                probabilities=probs,
                model_id=self.model_id,
            )
        except Exception as exc:
            logger.warning("hsemotion_predict_error", error=str(exc))
            return self._predict_fallback(face_crop_rgb)

    def _predict_fallback(self, face_crop_rgb: np.ndarray) -> EmotionPrediction:
        """
        Simple grayscale image statistic heuristic.
        Very rough — neutral dominant, slight variation from brightness.
        This is ONLY a placeholder until hsemotion-onnx is installed.
        """
        gray = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2GRAY)
        mean_brightness = float(np.mean(gray)) / 255.0
        std_brightness = float(np.std(gray)) / 128.0

        # Heuristic: bright+high-std → happy, dark → sad, mid → neutral
        probs = {e: 0.0 for e in EMOTION_LABELS}
        probs["neutral"] = 0.5
        probs["happy"] = 0.2 * mean_brightness
        probs["sad"] = 0.2 * (1.0 - mean_brightness)
        probs["surprise"] = 0.1 * std_brightness

        # Normalize
        total = sum(probs.values()) + 1e-9
        probs = {k: v / total for k, v in probs.items()}
        top = max(probs, key=probs.get)  # type: ignore[arg-type]

        return EmotionPrediction(
            emotion=top,
            confidence=probs[top],
            probabilities=probs,
            model_id=f"{self.model_id}_fallback",
        )


def extract_face_crop(
    frame_bgr: np.ndarray,
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
    target_size: int = 224,
    padding: float = 0.2,
) -> np.ndarray:
    """
    Extract and resize a face crop from a BGR frame.

    Args:
        frame_bgr: Full BGR frame
        bbox_{x,y,w,h}: Normalized bounding box [0, 1]
        target_size: Output square size in pixels
        padding: Extra padding fraction around the bounding box

    Returns:
        RGB uint8 array [target_size, target_size, 3]
    """
    h, w = frame_bgr.shape[:2]

    # Expand bbox with padding
    x1 = max(0, int((bbox_x - padding * bbox_w) * w))
    y1 = max(0, int((bbox_y - padding * bbox_h) * h))
    x2 = min(w, int((bbox_x + bbox_w + padding * bbox_w) * w))
    y2 = min(h, int((bbox_y + bbox_h + padding * bbox_h) * h))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    crop = frame_bgr[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
