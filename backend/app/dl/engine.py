"""
AuthBrain DL Platform — Deep Learning Pipeline Engine

Orchestrates all deep learning model inference, graph construction,
Action Unit estimation, and ensemble combination.
Integrates directly as an optional stage inside FaceAnalysisPipeline.
"""
from __future__ import annotations

import time
import numpy as np
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.dl.registry import registry
from app.dl.graph.constructor import GraphConstructor
from app.dl.graph.gat import FaceGAT
from app.dl.emotion.hsemotion import HSEmotionModel, extract_face_crop
from app.dl.emotion.efficientface import EfficientFaceModel
from app.dl.emotion.ensemble import EmotionEnsemble
from app.dl.action_units.au_estimator import GeometricAUEstimator
from app.dl.xai.gnn_explainer import GNNExplainerWrapper

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
            registry.load_gnn_models(["gnn_gat"])

        self._loaded = True
        logger.info("dl_engine_loaded", session_id=self.session_id)

    def process(
        self,
        face_data: Any,
        frame_bgr: np.ndarray,
        frame_width: int,
        frame_height: int,
        timestamp_ms: float,
    ) -> dict[str, Any] | None:
        """
        Executes Deep Learning pipeline for a single frame.

        Args:
            face_data: RawFaceData from MediaPipe face detector
            frame_bgr: Original frame in BGR
            frame_width: Frame width (pixels)
            frame_height: Frame height (pixels)
            timestamp_ms: Current elapsed frame timestamp in ms

        Returns:
            dict containing all DL pipeline results, or None if failed
        """
        if not self._loaded:
            self.load()

        t_start = time.perf_counter()

        # ── 1. Graph Construction ─────────────────────────────────────────────
        graph = self._graph_constructor.build(face_data.landmarks, timestamp_ms)

        # ── 2. Face Crop Extraction for Image-based Models ────────────────────
        bbox = face_data.bounding_box
        face_crop = extract_face_crop(
            frame_bgr=frame_bgr,
            bbox_x=bbox.x,
            bbox_y=bbox.y,
            bbox_w=bbox.width,
            bbox_h=bbox.height,
        )

        # ── 3. Run Emotion Model Inference ────────────────────────────────────
        predictions = []
        for model_id in registry.list_enabled_emotions():
            model = registry.get_emotion(model_id)
            if model:
                # Timeout-safe execution
                t_model_start = time.perf_counter()
                pred = model.predict(face_crop)
                t_model_elapsed = (time.perf_counter() - t_model_start) * 1000.0

                if t_model_elapsed > settings.DL_INFERENCE_TIMEOUT_MS:
                    logger.warning(
                        "dl_model_timeout_skipped",
                        model_id=model_id,
                        elapsed_ms=round(t_model_elapsed, 2),
                    )
                else:
                    predictions.append(pred)

        # ── 4. Run GNN Inference ──────────────────────────────────────────────
        gnn_pred_dict = None
        gnn_prediction = None
        if settings.DL_GNN_ENABLED:
            gnn_model = registry.get_gnn("gnn_gat")
            if gnn_model and gnn_model._loaded:
                gnn_prediction = gnn_model.forward(graph)
                # Parse GNN Prediction into dictionary
                gnn_pred_dict = {
                    "emotion": gnn_prediction.emotion,
                    "confidence": gnn_prediction.confidence,
                    "probabilities": gnn_prediction.probabilities,
                    "model_id": gnn_prediction.model_id,
                    "node_importance": gnn_prediction.node_importance,
                    "edge_attention": gnn_prediction.edge_attention,
                }

        # ── 5. Emotion Prediction Ensemble ────────────────────────────────────
        ensemble_result = self._ensemble.combine(predictions)

        # ── 6. Action Unit Estimation ─────────────────────────────────────────
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

        # ── 7. GNN XAI Explanations ───────────────────────────────────────────
        top_landmarks = []
        xai_explanations = []
        if gnn_prediction:
            expl = self._gnn_explainer.explain(gnn_prediction)
            top_landmarks = expl["top_landmarks"]
            # Map into the format frontend expects
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
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Convert predictions to schemas-compatible format
        model_predictions_list = [
            {
                "emotion": p.emotion,
                "confidence": p.confidence,
                "probabilities": p.probabilities,
                "model_id": p.model_id,
                "latency_ms": p.latency_ms,
            }
            for p in predictions
        ]

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
            },
            "gnn_prediction": gnn_pred_dict,
            "action_units": action_units,
            "top_important_landmarks": top_landmarks,
            "models_used": [p.model_id for p in predictions] + (["gnn_gat"] if gnn_pred_dict else []),
            "xai_explanations": xai_explanations,
            "landmarks": landmarks_list,
        }

    def reset(self) -> None:
        """Reset temporal graphs state."""
        self._graph_constructor.reset()
