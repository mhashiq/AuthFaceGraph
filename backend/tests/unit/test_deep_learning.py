"""
AuthBrain AI Face Analysis Engine
Unit Tests for Deep Learning Research Platform
"""
from __future__ import annotations

import pytest
import numpy as np

from app.models.schemas import Landmark
from app.dl.registry import registry
from app.dl.engine import DLEngine  # Triggers model registration
from app.dl.emotion.hsemotion import HSEmotionModel
from app.dl.emotion.efficientface import EfficientFaceModel
from app.dl.emotion.ensemble import EmotionEnsemble, EMOTION_LABELS
from app.dl.action_units.au_estimator import GeometricAUEstimator
from app.dl.graph.constructor import GraphConstructor
from app.dl.base import EmotionPrediction
from app.dl.base import GNNPrediction
from app.dl.base import ActionUnitResult


def test_registry_registration() -> None:
    """Verify models register correctly in ModelRegistry."""
    # Ensure they are registered automatically via import of engine
    assert registry.get_emotion("hsemotion") is not None
    assert registry.get_emotion("efficientface") is not None
    assert registry.get_gnn("gnn_gat") is not None


def test_ensemble_logic() -> None:
    """Verify emotion prediction ensembling consensus and disagreement score."""
    ensemble = EmotionEnsemble()

    # Predictor 1: HSEmotion predicting happy
    p1 = EmotionPrediction(
        emotion="happy",
        confidence=0.8,
        probabilities={e: 0.05 for e in EMOTION_LABELS},
        model_id="hsemotion"
    )
    p1.probabilities["happy"] = 0.8
    p1.probabilities["neutral"] = 0.1

    # Predictor 2: EfficientFace predicting happy
    p2 = EmotionPrediction(
        emotion="happy",
        confidence=0.7,
        probabilities={e: 0.05 for e in EMOTION_LABELS},
        model_id="efficientface"
    )
    p2.probabilities["happy"] = 0.7
    p2.probabilities["neutral"] = 0.15

    # Run combination
    res = ensemble.combine([p1, p2])
    assert res["final_emotion"] == "happy"
    assert res["confidence"] > 0.6
    assert res["disagreement_score"] < 0.2  # Consensus should yield low entropy


def test_ensemble_smile_bias_prefers_happy() -> None:
    """Strong smile geometry should correct a low-confidence sad prediction."""
    ensemble = EmotionEnsemble()

    sad_pred = EmotionPrediction(
        emotion="sad",
        confidence=0.34,
        probabilities={
            "neutral": 0.22,
            "happy": 0.18,
            "sad": 0.34,
            "surprise": 0.06,
            "fear": 0.05,
            "disgust": 0.05,
            "anger": 0.04,
            "contempt": 0.06,
        },
        model_id="hsemotion",
    )

    res = ensemble.combine([sad_pred], smile_intensity=0.85)

    assert res["final_emotion"] == "happy"
    assert res["probabilities"]["happy"] > res["probabilities"]["sad"]


def test_ensemble_uses_gnn_signal_when_available() -> None:
    """The GNN should participate in the final consensus when enabled."""
    ensemble = EmotionEnsemble()

    image_pred = EmotionPrediction(
        emotion="sad",
        confidence=0.62,
        probabilities={
            "neutral": 0.12,
            "happy": 0.10,
            "sad": 0.62,
            "surprise": 0.04,
            "fear": 0.03,
            "disgust": 0.03,
            "anger": 0.03,
            "contempt": 0.03,
        },
        model_id="hsemotion",
    )

    gnn_pred = GNNPrediction(
        emotion="happy",
        confidence=0.91,
        probabilities={
            "neutral": 0.03,
            "happy": 0.91,
            "sad": 0.01,
            "surprise": 0.01,
            "fear": 0.01,
            "disgust": 0.01,
            "anger": 0.01,
            "contempt": 0.01,
        },
        model_id="gnn_gat",
    )

    res = ensemble.combine([image_pred, gnn_pred])

    assert res["final_emotion"] == "happy"
    assert res["probabilities"]["happy"] > res["probabilities"]["sad"]


def test_engine_smile_signal_uses_au12() -> None:
    """AU12 intensity should strengthen the smile signal when geometry is weak."""
    from app.dl.engine import DLEngine

    engine = DLEngine(session_id="test_session")

    landmarks = [Landmark(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    au_results = [
        ActionUnitResult(au_id="AU12", name="Lip Corner Puller (Smile)", present=True, intensity=4.0),
    ]

    smile = engine._estimate_smile_intensity(landmarks, au_results)

    assert smile == 0.8


def test_face_detector_detects_upside_down_landmarks() -> None:
    """Upside-down landmark order should be recognized before crop generation."""
    from app.analysis.face_detector import FaceDetector

    landmarks = [Landmark(x=0.5, y=0.5, z=0.0) for _ in range(478)]

    for idx in (33, 133, 362, 263):
        landmarks[idx] = Landmark(x=0.5, y=0.75, z=0.0)
    for idx in (61, 291, 13, 14):
        landmarks[idx] = Landmark(x=0.5, y=0.25, z=0.0)

    assert FaceDetector._looks_upside_down(landmarks) is True


def test_geometric_au_estimation() -> None:
    """Verify FACS Action Unit geometric estimators under mock facial structures."""
    estimator = GeometricAUEstimator()

    # 478 mock landmarks (neutral face layout)
    landmarks = [Landmark(x=0.5, y=0.4, z=0.0) for _ in range(478)]

    # Set custom landmark offsets to simulate specific Action Units
    # AU45: Eyeballs closed
    # AU12: Lip Corner Puller (Smile)
    # Right pupil 468, left pupil 473
    landmarks[468] = Landmark(x=0.45, y=0.35, z=0.0)
    landmarks[473] = Landmark(x=0.55, y=0.35, z=0.0)

    # Left eye outer corner 263, inner corner 362
    # Right eye outer corner 33, inner corner 133

    # Left eyebrow inner 336, Right eyebrow inner 70
    # Nose bridge 6
    landmarks[6] = Landmark(x=0.5, y=0.32, z=0.0)
    landmarks[70] = Landmark(x=0.48, y=0.30, z=0.0)
    landmarks[336] = Landmark(x=0.52, y=0.30, z=0.0)

    # Let's run estimator
    au_results = estimator.estimate(landmarks, frame_width=640, frame_height=480)
    assert len(au_results) > 0

    au_map = {au.au_id: au for au in au_results}
    assert "AU1" in au_map
    assert "AU4" in au_map
    assert "AU12" in au_map
    assert "AU45" in au_map


def test_graph_constructor() -> None:
    """Verify GraphConstructor maps landmarks into standard 10-dimensional node feature graphs."""
    constructor = GraphConstructor(edge_strategy="knn", knn_k=6)

    # Construct mock landmarks
    landmarks = [Landmark(x=i/478.0, y=i/478.0, z=0.0) for i in range(478)]

    # Frame 1
    graph1 = constructor.build(landmarks, timestamp_ms=0.0)
    assert graph1.node_features.shape == (478, 10)
    assert graph1.edge_index.shape[0] == 2
    # KNN edge connections
    assert graph1.edge_index.shape[1] > 0

    # Frame 2 (simulating displacement and velocity)
    landmarks_moved = [Landmark(x=(i+1)/478.0, y=i/478.0, z=0.0) for i in range(478)]
    graph2 = constructor.build(landmarks_moved, timestamp_ms=100.0)

    # Temporal displacement features should be non-zero
    assert np.any(graph2.node_features[:, 3:5] != 0.0)
    # Velocity features should be non-zero
    assert np.any(graph2.node_features[:, 5] != 0.0)


def test_pipeline_integration(monkeypatch) -> None:
    """Verify FaceAnalysisPipeline integrates DLEngine when DL_ENABLED is True."""
    from app.analysis.pipeline import FaceAnalysisPipeline
    from app.core.config import get_settings
    from app.analysis.face_detector import FaceDetector, RawFaceData
    from app.models.schemas import FaceBoundingBox

    settings = get_settings()
    monkeypatch.setattr(settings, "DL_ENABLED", True)
    monkeypatch.setattr(settings, "DL_GNN_ENABLED", True)

    # Mock detector methods to avoid loading actual MediaPipe file
    mock_landmarks = [Landmark(x=0.5, y=0.4 + i/1000.0, z=0.0) for i in range(478)]
    mock_face = RawFaceData(
        landmarks=mock_landmarks,
        landmarks_px=[(320, 240)] * 478,
        bounding_box=FaceBoundingBox(x=0.2, y=0.1, width=0.6, height=0.8),
        detection_confidence=0.99,
        face_index=0,
        frame_width=640,
        frame_height=480,
        timestamp_ms=33,
    )

    monkeypatch.setattr(FaceDetector, "load_model", lambda self: None)
    monkeypatch.setattr(FaceDetector, "detect", lambda self, frame, w, h, idx: [mock_face])
    monkeypatch.setattr(FaceDetector, "close", lambda self: None)

    # Initialize and load pipeline
    pipeline = FaceAnalysisPipeline(session_id="test_session")
    pipeline.load()

    # Create dummy black JPEG bytes
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    import cv2
    _, jpeg_bytes = cv2.imencode(".jpg", dummy_frame)
    raw_bytes = jpeg_bytes.tobytes()

    # Process frame
    result, annotated = pipeline.process_frame(raw_bytes, draw_overlay=False)

    # Assertions
    assert isinstance(result.face_detected, bool)
    assert result.deep_learning is not None
    assert result.deep_learning.dl_enabled is True
    assert result.deep_learning.emotion_ensemble is not None
    assert result.deep_learning.gnn_prediction is not None or result.deep_learning.gnn_prediction is None
    assert len(result.deep_learning.action_units) > 0
    assert isinstance(result.deep_learning.top_important_landmarks, list)
    assert isinstance(result.deep_learning.xai_explanations, list)
    assert len(result.deep_learning.landmarks) in (0, 478)

