"""
AuthBrain DL Platform — Deep Learning Pipeline Engine

Orchestrates all deep learning model inference, graph construction,
Action Unit estimation, and ensemble combination.
Integrates directly as an optional stage inside FaceAnalysisPipeline.
"""
from __future__ import annotations

import time
from collections import deque
import numpy as np
from typing import Any

from app.analysis.landmark_indices import LANDMARKS
from app.core.config import get_settings
from app.core.logging import get_logger
from app.dl.registry import registry
from app.dl.graph.constructor import GraphConstructor
from app.dl.graph.gat import FaceGAT
from app.dl.emotion.hsemotion import HSEmotionModel, extract_face_crop, extract_aligned_face_crop
from app.dl.emotion.efficientface import EfficientFaceModel
from app.dl.emotion.ensemble import EmotionEnsemble
from app.dl.emotion.xai_summary import build_emotion_explanation
from app.dl.action_units.au_estimator import GeometricAUEstimator
from app.dl.xai.gnn_explainer import GNNExplainerWrapper
from app.utils.math_utils import compute_smile_intensity

# Autoregister the plugins at import time
registry.register_emotion("hsemotion", HSEmotionModel)
registry.register_emotion("efficientface", EfficientFaceModel)
registry.register_gnn("gnn_gat", FaceGAT)

settings = get_settings()
logger = get_logger(__name__)


class DLEngine:
    """
    Main manager for Deep Learning models.
    Coordinates face cropping, emotion model ensembling, GNN landmark graphs,
    Action Unit detection, and explanation outputs.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._graph_constructor = GraphConstructor(
            edge_strategy=settings.DL_GRAPH_EDGE_STRATEGY,
            knn_k=settings.DL_GRAPH_KNN_K,
        )
        self._ensemble = EmotionEnsemble(strategy=settings.DL_ENSEMBLE_STRATEGY)
        self._au_estimator = GeometricAUEstimator()
        self._gnn_explainer = GNNExplainerWrapper()
        self._loaded = False
        history_size = int(getattr(settings, "DL_EMOTION_HISTORY_SIZE", 30))
        self._emotion_history: deque[dict[str, Any]] = deque(maxlen=history_size)

    def load(self) -> None:
        """
        Idempotent load call to fetch/initialize model weights.
        """
        if self._loaded:
            return

        # Load configured emotion models
        active_emotions = settings.DL_EMOTION_MODELS
        registry.load_emotion_models(active_emotions)

        # Load GNN models if enabled
        if settings.DL_GNN_ENABLED:
            from pathlib import Path

            gnn_checkpoint = Path(settings.DL_GNN_CHECKPOINT_PATH)
            if gnn_checkpoint.exists():
                registry.load_gnn_models(["gnn_gat"])
            else:
                logger.warning(
                    "dl_gnn_checkpoint_missing",
                    checkpoint=str(gnn_checkpoint),
                    note="Skipping GNN load until a trained checkpoint is provided.",
                )

        self._loaded = True
        logger.info("dl_engine_loaded", session_id=self.session_id)

    def process(
        self,
        face_data: Any,
        frame_bgr: np.ndarray,
        frame_width: int,
        frame_height: int,
        timestamp_ms: float,
        quality_score: float = 1.0,
    ) -> dict[str, Any] | None:
        """
        Executes Deep Learning pipeline for a single frame.

        Args:
            face_data: RawFaceData from MediaPipe face detector
            frame_bgr: Original frame in BGR
            frame_width: Frame width (pixels)
            frame_height: Frame height (pixels)
            timestamp_ms: Current elapsed frame timestamp in ms
            quality_score: Frame/face quality score from quality estimator

        Returns:
            dict containing all DL pipeline results, or None if failed
        """
        if not self._loaded:
            self.load()

        t_start = time.perf_counter()

        # Initialize tracking dictionaries for ensemble reliability weights
        if not hasattr(self, "_model_errors"):
            self._model_errors = {}
        if not hasattr(self, "_model_latencies"):
            self._model_latencies = {}

        # ── 1. Graph Construction ─────────────────────────────────────────────
        graph = self._graph_constructor.build(face_data.landmarks, timestamp_ms)

        # Save original frame in debug mode
        if settings.DL_DEBUG_MODE:
            import cv2
            from pathlib import Path
            debug_dir = Path("/Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis/backend/debug_inference")
            debug_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time() * 1000)
            cv2.imwrite(str(debug_dir / f"frame_{timestamp}.jpg"), frame_bgr)

        # ── 2. Face Crop Extraction with Horizontal Alignment & Normalization ──
        face_crop = extract_aligned_face_crop(
            frame_bgr=frame_bgr,
            landmarks=face_data.landmarks,
            bounding_box=face_data.bounding_box,
            target_size=224,
            padding=0.35,
        )

        if settings.DL_DEBUG_MODE:
            import cv2
            from pathlib import Path
            debug_dir = Path("/Users/mdmehedihassan/Desktop/Projects/AuthBrain_AI_Face_Analysis/backend/debug_inference")
            debug_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time() * 1000)
            cv2.imwrite(str(debug_dir / f"aligned_{timestamp}.jpg"), cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
            np.save(str(debug_dir / f"input_{timestamp}.npy"), face_crop)

        # ── 3. Run Emotion Model Inference ────────────────────────────────────
        predictions = []
        for model_id in registry.list_enabled_emotions():
            model = registry.get_emotion(model_id)
            if model:
                # Active health-check skip if model has crashed repeatedly
                if self._model_errors.get(model_id, 0) >= 5:
                    logger.error("dl_model_deactivated_consecutive_errors", model_id=model_id)
                    continue

                t_model_start = time.perf_counter()
                try:
                    pred = model.predict(face_crop)
                    t_model_elapsed = (time.perf_counter() - t_model_start) * 1000.0

                    if t_model_elapsed > settings.DL_INFERENCE_TIMEOUT_MS:
                        logger.warning(
                            "dl_model_timeout_skipped",
                            model_id=model_id,
                            elapsed_ms=round(t_model_elapsed, 2),
                        )
                        self._model_errors[model_id] = self._model_errors.get(model_id, 0) + 1
                    else:
                        self._model_errors[model_id] = 0  # reset errors on success
                        self._model_latencies[model_id] = t_model_elapsed
                        predictions.append(pred)
                except Exception as exc:
                    logger.error("dl_model_execution_failed", model_id=model_id, error=str(exc))
                    self._model_errors[model_id] = self._model_errors.get(model_id, 0) + 1
                    continue

        # ── 4. Run GNN Inference ──────────────────────────────────────────────
        gnn_pred_dict = None
        gnn_prediction = None
        if settings.DL_GNN_ENABLED:
            gnn_model = registry.get_gnn("gnn_gat")
            if gnn_model and gnn_model._loaded:
                gnn_prediction = gnn_model.forward(graph)

                # Update GNN health tracking
                self._model_errors["gnn_gat"] = gnn_model._error_count
                self._model_latencies["gnn_gat"] = gnn_prediction.latency_ms

                # Parse GNN Prediction into dictionary
                gnn_pred_dict = {
                    "emotion": gnn_prediction.emotion,
                    "confidence": gnn_prediction.confidence,
                    "probabilities": gnn_prediction.probabilities,
                    "raw_confidence": gnn_prediction.raw_confidence,
                    "calibrated_confidence": gnn_prediction.confidence,
                    "model_id": gnn_prediction.model_id,
                    "node_importance": gnn_prediction.node_importance,
                    "edge_attention": gnn_prediction.edge_attention,
                    "edge_index": gnn_prediction.edge_index,
                    "latency_ms": gnn_prediction.latency_ms,
                    "status": gnn_prediction.status,
                    "error": gnn_prediction.error,
                }

        # ── 5. Action Unit Estimation ─────────────────────────────────────────
        action_units_res = self._au_estimator.estimate(
            face_data.landmarks, frame_width, frame_height
        )
        action_units = [
            {
                "au_id": au.au_id,
                "name": au.name,
                "present": au.present,
                "intensity": au.intensity,
            }
            for au in action_units_res
        ]

        # ── 6. Emotion Prediction Ensemble ────────────────────────────────────
        smile_intensity = self._estimate_smile_intensity(face_data.landmarks, action_units_res)
        ensemble_inputs = list(predictions)
        if gnn_prediction and gnn_prediction.error is None:
            ensemble_inputs.append(gnn_prediction)
        try:
            ensemble_result = self._ensemble.combine(
                predictions=ensemble_inputs,
                quality_score=quality_score,
                smile_intensity=smile_intensity,
                model_latencies=self._model_latencies,
                model_errors=self._model_errors,
            )
        except Exception as exc:
            logger.warning("dl_ensemble_fallback_used", error=str(exc))
            fallback_probs = {
                "neutral": 0.125,
                "happy": 0.125,
                "sad": 0.125,
                "surprise": 0.125,
                "fear": 0.125,
                "disgust": 0.125,
                "anger": 0.125,
                "contempt": 0.125,
            }
            if predictions:
                accumulated = {emotion: 0.0 for emotion in fallback_probs}
                for prediction in predictions:
                    for emotion, probability in prediction.probabilities.items():
                        accumulated[emotion] += probability
                total = float(len(predictions))
                if total > 0.0:
                    fallback_probs = {emotion: value / total for emotion, value in accumulated.items()}

            ensemble_result = {
                "final_emotion": "unknown",
                "confidence": 0.0,
                "probabilities": fallback_probs,
                "disagreement_score": 1.0 if predictions else 0.0,
                "uncertainty": 1.0,
                "agreement_score": 0.0,
                "raw_confidence": 0.0,
                "calibrated_confidence": 0.0,
            }

        self._emotion_history.append({
            "timestamp_ms": timestamp_ms,
            "final_emotion": ensemble_result["final_emotion"],
            "confidence": ensemble_result["confidence"],
            "probabilities": ensemble_result["probabilities"],
        })

        if len(self._emotion_history) >= 3:
            history_counts: dict[str, int] = {}
            history_scores: dict[str, float] = {}
            for item in self._emotion_history:
                emotion = item["final_emotion"]
                confidence = float(item["confidence"])
                history_counts[emotion] = history_counts.get(emotion, 0) + 1
                history_scores[emotion] = history_scores.get(emotion, 0.0) + confidence

            dominant = max(history_counts, key=lambda e: (history_counts[e], history_scores.get(e, 0.0)))
            if history_counts[dominant] >= max(2, len(self._emotion_history) // 2):
                ensemble_result["final_emotion"] = dominant
                ensemble_result["confidence"] = round(history_scores[dominant] / history_counts[dominant], 4)
                ensemble_result["calibrated_confidence"] = ensemble_result["confidence"]
                ensemble_result["uncertainty"] = round(1.0 - ensemble_result["confidence"], 4)

        # ── 7. XAI Explanations ───────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        top_landmarks = []
        xai_explanations = [
            build_emotion_explanation(
                ensemble_result=ensemble_result,
                predictions=predictions,
                quality_score=quality_score,
                smile_intensity=smile_intensity,
                inference_time_ms=elapsed_ms,
            )
        ]
        if gnn_prediction:
            expl = self._gnn_explainer.explain(gnn_prediction)
            top_landmarks = expl["top_landmarks"]
            xai_explanations.append({
                "metric_name": "gnn_emotion",
                "final_value": gnn_prediction.confidence,
                "confidence": gnn_prediction.confidence,
                "attributions": expl["attributions"],
                "processing_time_ms": gnn_prediction.latency_ms,
                "landmark_quality": 0.9,
                "explanation_text": expl["explanation_text"],
            })

        # Compile final results
        # Convert predictions to schemas-compatible format
        model_predictions_list = [
            {
                "emotion": p.emotion,
                "confidence": p.confidence,
                "probabilities": p.probabilities,
                "raw_confidence": p.raw_confidence,
                "calibrated_confidence": p.confidence,
                "model_id": p.model_id,
                "latency_ms": p.latency_ms,
                "status": p.status,
                "error": p.error,
            }
            for p in predictions
        ]

        # Get health report for all models
        model_health = {}
        for m_id in registry.list_enabled_emotions():
            m = registry.get_emotion(m_id)
            if m:
                model_health[m_id] = m.health()
        if settings.DL_GNN_ENABLED:
            gm = registry.get_gnn("gnn_gat")
            if gm:
                model_health["gnn_gat"] = gm.health()

        # Convert landmarks to dict list
        landmarks_list = [
            {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
            for lm in face_data.landmarks
        ]

        return {
            "dl_enabled": True,
            "dl_inference_time_ms": round(elapsed_ms, 2),
            "emotion_ensemble": {
                "final_emotion": ensemble_result["final_emotion"],
                "confidence": ensemble_result["confidence"],
                "probabilities": ensemble_result["probabilities"],
                "model_predictions": model_predictions_list,
                "disagreement_score": ensemble_result["disagreement_score"],
                "uncertainty": ensemble_result["uncertainty"],
                "agreement_score": ensemble_result["agreement_score"],
                "raw_confidence": ensemble_result["raw_confidence"],
                "calibrated_confidence": ensemble_result["calibrated_confidence"],
            },
            "gnn_prediction": gnn_pred_dict,
            "action_units": action_units,
            "top_important_landmarks": top_landmarks,
            "models_used": [p.model_id for p in predictions] + (["gnn_gat"] if gnn_pred_dict else []),
            "xai_explanations": xai_explanations,
            "landmarks": landmarks_list,
            "model_health": model_health,
        }

    def reset(self) -> None:
        """Reset temporal graphs state."""
        self._graph_constructor.reset()

    def _estimate_smile_intensity(self, landmarks: list[Any], action_units_res: list[Any] | None = None) -> float:
        """Estimate a smile score from facial geometry and AU12 support."""
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

        geometry_smile = compute_smile_intensity(
            left_corner,
            right_corner,
            mouth_top,
            mouth_bottom,
            left_cheek,
            right_cheek,
        )

        au12_smile = 0.0
        if action_units_res:
            for au in action_units_res:
                if getattr(au, "au_id", None) == "AU12":
                    au12_smile = float(getattr(au, "intensity", 0.0)) / 5.0
                    break

        return float(max(geometry_smile, au12_smile))
