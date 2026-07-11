"""
AuthBrain DL Platform — HSEmotion Pretrained Emotion Model

Wraps the HSEmotion library for real-time facial emotion recognition.
HSEmotion (Savchenko, 2022) achieves state-of-the-art accuracy on AffectNet
with an EfficientNet-b0 backbone.

Model: EfficientNet-b0 trained on AffectNet-8 (8 classes)
Input: 224×224 cropped face image (RGB)
Output: 8 emotion probabilities

Install: pip install hsemotion-onnx

If the model cannot be loaded, inference fails loudly so the pipeline does
not silently collapse into placeholder predictions.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import urllib.request

from app.core.config import get_settings
from app.core.logging import get_logger
from app.dl.base import EmotionModelBase, EmotionPrediction
from app.models.schemas import FaceBoundingBox

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
    """

    def __init__(self) -> None:
        super().__init__()
        self._recognizer: "Any" = None
        self._mode: str = "unloaded"

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
            self._recognizer = HSEmotionRecognizer(model_name="enet_b2_8")
            self._mode = "hsemotion"
            logger.info("hsemotion_loaded", mode="onnx")
            return
        except Exception as exc:
            logger.error("hsemotion_load_failed", error=str(exc))
            raise RuntimeError(f"HSEmotion model load failed: {exc}")

    def _predict_impl(self, face_crop_rgb: np.ndarray) -> EmotionPrediction:
        if self._mode != "hsemotion":
            raise RuntimeError("HSEmotion model is not loaded (weights failed to initialize).")
        return self._predict_hsemotion(face_crop_rgb)

    def _predict_hsemotion(self, face_crop_rgb: np.ndarray) -> EmotionPrediction:
        """Run HSEmotion ONNX inference."""
        try:
            # HSEmotion expects RGB because it uses ImageNet normalization (0.485, 0.456, 0.406) for R, G, B
            
            t_start = time.perf_counter()
            emotion_label, scores = self._recognizer.predict_emotions(face_crop_rgb, logits=True)
            latency_ms = (time.perf_counter() - t_start) * 1000.0

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
                raw_logits = [float(scores.get(cls, 0.0)) for cls in hsemotion_classes]
                logits_arr = np.array(raw_logits, dtype=np.float32)
            else:
                logits_arr = np.array(scores, dtype=np.float32)
                raw_logits = logits_arr.tolist()

            # Stable softmax normalisation
            exp_logits = np.exp(logits_arr - np.max(logits_arr))
            probs_arr = exp_logits / np.sum(exp_logits)
            
            for i, val in enumerate(probs_arr):
                hs_label = hsemotion_classes[i]
                mapped_key = hsemotion_to_internal.get(hs_label, hs_label)
                if mapped_key in probs:
                    probs[mapped_key] = float(val)

            from app.utils.calibration import temperature_scale_probs

            top_emotion = max(probs, key=probs.get)  # type: ignore[arg-type]
            raw_top_conf = probs[top_emotion]

            # Use uncalibrated probabilities directly as HSEmotion probabilities are already quite flat
            calibrated_probs = probs
            calibrated_top_emotion = top_emotion
            calibrated_top_conf = raw_top_conf

            checkpoint_path = os.path.expanduser("~/.hsemotion/enet_b2_8.onnx")

            # Detailed inference audit logging
            logger.info(
                "hsemotion_inference_details",
                input_shape=list(face_crop_rgb.shape),
                logits=raw_logits,
                probabilities=probs,
                calibrated_probabilities=calibrated_probs,
                predicted_emotion=calibrated_top_emotion,
                confidence=calibrated_top_conf,
                checkpoint=checkpoint_path,
                latency_ms=round(latency_ms, 2),
            )

            # Debug mode file saving
            if settings.DL_DEBUG_MODE:
                import json
                debug_dir = Path("/Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis/backend/debug_inference")
                debug_dir.mkdir(parents=True, exist_ok=True)
                timestamp = int(time.time() * 1000)
                
                face_bgr_debug = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(debug_dir / f"crop_{timestamp}_hsemotion.jpg"), face_bgr_debug)
                np.save(str(debug_dir / f"preprocessed_{timestamp}_hsemotion.npy"), face_crop_rgb)
                
                inf_data = {
                    "model_id": self.model_id,
                    "timestamp": timestamp,
                    "device": "cpu",
                    "checkpoint_path": checkpoint_path,
                    "evaluation_mode": True,
                    "raw_logits": raw_logits,
                    "softmax_probabilities": probs,
                    "calibrated_probabilities": calibrated_probs,
                    "predicted_emotion": calibrated_top_emotion,
                    "confidence": calibrated_top_conf,
                    "latency_ms": latency_ms,
                }
                with open(debug_dir / f"inference_{timestamp}_hsemotion.json", "w") as f:
                    json.dump(inf_data, f, indent=2)

            return EmotionPrediction(
                emotion=calibrated_top_emotion,
                confidence=calibrated_top_conf,
                probabilities=calibrated_probs,
                raw_confidence=raw_top_conf,
                raw_probabilities=probs,
                model_id=self.model_id,
            )
        except Exception as exc:
            logger.error("hsemotion_predict_failed", error=str(exc))
            raise RuntimeError(f"HSEmotion prediction failed: {exc}")


def extract_face_crop(
    frame_bgr: np.ndarray,
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
    target_size: int = 224,
    padding: float = 0.5,
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


def extract_aligned_face_crop(
    frame_bgr: np.ndarray,
    landmarks: list[Any],
    bounding_box: FaceBoundingBox | None = None,
    target_size: int = 224,
    padding: float = 0.5,
) -> np.ndarray:
    """
    Extract a normalized, aligned face crop based on eye landmarks.
    Ensures the face is rotated upright, centered, and scaled consistently.
    """
    h, w = frame_bgr.shape[:2]
    
    # Eye landmarks to align the face horizontally
    right_eye_indices = [33, 133, 157, 158, 159, 160, 161, 246]
    left_eye_indices = [263, 362, 384, 385, 386, 387, 388, 466]
    
    right_eye_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in right_eye_indices if i < len(landmarks)])
    left_eye_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in left_eye_indices if i < len(landmarks)])
    
    if len(right_eye_pts) == 0 or len(left_eye_pts) == 0:
        if bounding_box is not None:
            return extract_face_crop(frame_bgr, bounding_box.x, bounding_box.y, bounding_box.width, bounding_box.height, target_size, padding)
        # Fallback to landmark-derived crop if the box is not available.
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        return extract_face_crop(frame_bgr, min_x / w, min_y / h, (max_x - min_x) / w, (max_y - min_y) / h, target_size, padding)
        
    right_eye_center = right_eye_pts.mean(axis=0)
    left_eye_center = left_eye_pts.mean(axis=0)
    
    # Angle of rotation between the eye centers (from camera-left eye to camera-right eye)
    # camera-left eye (smaller x, left of image) is left_eye_center
    # camera-right eye (larger x, right of image) is right_eye_center
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Rotation center is the eye midpoint
    eye_center = ((right_eye_center[0] + left_eye_center[0]) / 2.0,
                  (right_eye_center[1] + left_eye_center[1]) / 2.0)
                  
    # Determine face size based on the backend face box when available.
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)

    if bounding_box is not None:
        min_x = bounding_box.x * w
        min_y = bounding_box.y * h
        max_x = (bounding_box.x + bounding_box.width) * w
        max_y = (bounding_box.y + bounding_box.height) * h
    
    face_w = max_x - min_x
    face_h = max_y - min_y
    face_size = max(face_w, face_h)
    
    # Rotate the frame
    rot_mat = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated_frame = cv2.warpAffine(frame_bgr, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
    
    # Crop the face square
    face_center_x = (min_x + max_x) / 2.0
    face_center_y = (min_y + max_y) / 2.0
    
    half_size = face_size * (0.5 + padding)
    x1 = max(0, int(face_center_x - half_size))
    y1 = max(0, int(face_center_y - half_size))
    x2 = min(w, int(face_center_x + half_size))
    y2 = min(h, int(face_center_y + half_size))
    
    if x2 <= x1 or y2 <= y1:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
    crop = rotated_frame[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
