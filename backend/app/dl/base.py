"""
AuthBrain DL Platform — Abstract Base Classes

All pluggable DL components implement these interfaces.
Dependency injection is achieved by registering implementations
in the ModelRegistry.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmotionPrediction:
    """Output from a single emotion recognition model."""
    emotion: str                          # top-1 emotion label
    confidence: float                     # top-1 calibrated confidence [0, 1]
    probabilities: dict[str, float]       # all calibrated class probabilities
    raw_confidence: float = 0.0           # raw top-1 confidence [0, 1]
    raw_probabilities: dict[str, float] = field(default_factory=dict)
    model_id: str = ""
    latency_ms: float = 0.0
    error: str | None = None              # set if inference failed
    status: str = "healthy"               # "healthy" | "error" | "disabled"


@dataclass
class GNNPrediction:
    """Output from a GNN model over the facial landmark graph."""
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    raw_confidence: float = 0.0
    raw_probabilities: dict[str, float] = field(default_factory=dict)
    node_importance: list[float] = field(default_factory=list)          # [478] per-landmark importance
    edge_attention: list[float] = field(default_factory=list)           # per-edge attention weights
    edge_index: list[list[int]] = field(default_factory=list)           # actual edge index [2, num_edges]
    model_id: str = ""
    latency_ms: float = 0.0
    error: str | None = None
    status: str = "healthy"


@dataclass
class ActionUnitResult:
    """A single Facial Action Unit detection result."""
    au_id: str          # e.g. "AU12"
    name: str           # e.g. "Lip Corner Puller"
    present: bool
    intensity: float    # 0.0–5.0 AU intensity units


@dataclass
class FaceGraph:
    """
    Graph representation of a single facial landmark frame.

    Nodes: 478 MediaPipe landmarks
    Edges: configurable (anatomical / knn / radius)
    Features: coordinates, temporal, region encoding
    """
    node_features: Any          # np.ndarray [478, feature_dim]
    edge_index: Any             # np.ndarray [2, num_edges] (source, target)
    edge_attr: Any | None       # np.ndarray [num_edges, edge_feat_dim] | None
    landmark_positions: Any     # np.ndarray [478, 3] original xyz coords
    timestamp_ms: float
    num_nodes: int = 478
    feature_dim: int = 0

    def __post_init__(self) -> None:
        if self.node_features is not None:
            self.num_nodes, self.feature_dim = self.node_features.shape[:2] if self.node_features.ndim == 2 else (self.node_features.shape[0], 0)


# ══════════════════════════════════════════════════════════════════════════════
# Latency Tracker (mixin)
# ══════════════════════════════════════════════════════════════════════════════

class LatencyMixin:
    """Rolling average latency tracker for any inference component."""

    def __init__(self) -> None:
        self._latency_window: deque[float] = deque(maxlen=30)

    def _record_latency(self, ms: float) -> None:
        self._latency_window.append(ms)

    @property
    def avg_latency_ms(self) -> float:
        if not self._latency_window:
            return 0.0
        return float(np.mean(list(self._latency_window)))

    @property
    def p95_latency_ms(self) -> float:
        if not self._latency_window:
            return 0.0
        return float(np.percentile(list(self._latency_window), 95))


# ══════════════════════════════════════════════════════════════════════════════
# Abstract Base Classes
# ══════════════════════════════════════════════════════════════════════════════

class EmotionModelBase(ABC, LatencyMixin):
    """
    Abstract base for all pretrained emotion recognition models.

    Implementations wrap third-party libraries (HSEmotion, DeepFace, etc.)
    and normalize their output to EmotionPrediction.
    """

    def __init__(self) -> None:
        LatencyMixin.__init__(self)
        self._loaded = False
        self._inference_count = 0
        self._error_count = 0

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique string identifier for this model."""
        ...

    @property
    @abstractmethod
    def emotion_classes(self) -> list[str]:
        """Ordered list of emotion class labels this model outputs."""
        ...

    @abstractmethod
    def _load_impl(self) -> None:
        """Implementation-specific model loading."""
        ...

    @abstractmethod
    def _predict_impl(self, face_crop_rgb: np.ndarray) -> EmotionPrediction:
        """Run inference on a cropped face image [H, W, 3] RGB uint8."""
        ...

    def load(self) -> None:
        """Load model weights. Safe to call multiple times (idempotent)."""
        if self._loaded:
            return
        t = time.perf_counter()
        self._load_impl()
        self._loaded = True
        elapsed = (time.perf_counter() - t) * 1000
        print(f"[{self.model_id}] loaded in {elapsed:.1f}ms")

    def predict(self, face_crop_rgb: np.ndarray) -> EmotionPrediction:
        """
        Run emotion inference with latency tracking.

        Args:
            face_crop_rgb: Cropped face [H, W, 3] RGB uint8

        Returns:
            EmotionPrediction with probabilities for all classes
        """
        if not self._loaded:
            return EmotionPrediction(
                emotion="unknown", confidence=0.0, probabilities={},
                model_id=self.model_id, error="Model not loaded",
            )
        t = time.perf_counter()
        try:
            result = self._predict_impl(face_crop_rgb)
            self._inference_count += 1
        except Exception as exc:
            self._error_count += 1
            result = EmotionPrediction(
                emotion="unknown", confidence=0.0, probabilities={},
                model_id=self.model_id, error=str(exc),
            )
        elapsed_ms = (time.perf_counter() - t) * 1000
        result.latency_ms = elapsed_ms
        self._record_latency(elapsed_ms)
        return result

    def health(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "loaded": self._loaded,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "inference_count": self._inference_count,
            "error_count": self._error_count,
            "error_rate": round(self._error_count / max(self._inference_count, 1), 4),
        }


class GNNModelBase(ABC, LatencyMixin):
    """Abstract base for Graph Neural Network emotion models."""

    def __init__(self) -> None:
        LatencyMixin.__init__(self)
        self._loaded = False
        self._inference_count = 0
        self._error_count = 0

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def _load_impl(self) -> None: ...

    @abstractmethod
    def _forward_impl(self, graph: FaceGraph) -> GNNPrediction: ...

    def load(self) -> None:
        if self._loaded:
            return
        self._load_impl()
        self._loaded = True

    def forward(self, graph: FaceGraph) -> GNNPrediction:
        if not self._loaded:
            return GNNPrediction(
                emotion="unknown", confidence=0.0, probabilities={},
                node_importance=[], edge_attention=[], edge_index=[],
                model_id=self.model_id, error="Model not loaded", status="disabled",
            )
        t = time.perf_counter()
        try:
            result = self._forward_impl(graph)
            self._inference_count += 1
        except Exception as exc:
            self._error_count += 1
            result = GNNPrediction(
                emotion="unknown", confidence=0.0, probabilities={},
                node_importance=[], edge_attention=[], edge_index=[],
                model_id=self.model_id, error=str(exc), status="error",
            )
        result.latency_ms = (time.perf_counter() - t) * 1000
        self._record_latency(result.latency_ms)
        return result

    def health(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "loaded": self._loaded,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "inference_count": self._inference_count,
            "error_count": self._error_count,
            "error_rate": round(self._error_count / max(self._inference_count, 1), 4),
        }


class TemporalModelBase(ABC, LatencyMixin):
    """Abstract base for temporal sequence models (LSTM, GRU, Temporal Transformer)."""

    def __init__(self) -> None:
        LatencyMixin.__init__(self)
        self._loaded = False

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def _load_impl(self) -> None: ...

    @abstractmethod
    def update(self, features: np.ndarray) -> dict[str, Any]:
        """
        Update internal state with new frame features.

        Args:
            features: 1D feature vector for the current frame

        Returns:
            dict with keys: 'emotion', 'confidence', 'probabilities'
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset temporal state (e.g. on new session)."""
        ...


class AUEstimatorBase(ABC):
    """Abstract base for Facial Action Unit estimators."""

    @abstractmethod
    def estimate(
        self,
        landmarks: list[Any],  # list of Landmark(x,y,z) objects
        frame_width: int,
        frame_height: int,
    ) -> list[ActionUnitResult]: ...
