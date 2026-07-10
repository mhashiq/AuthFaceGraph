"""
AuthBrain DL Platform — Model Plugin Registry

Central hub for registering, loading, and querying DL model plugins.
Models self-register by calling registry.register() at import time.

Thread-safe for concurrent frame processing.
"""
from __future__ import annotations

import threading
from typing import Any, Type

from app.core.logging import get_logger
from app.dl.base import EmotionModelBase, GNNModelBase, TemporalModelBase

logger = get_logger(__name__)


class ModelRegistry:
    """
    Singleton plugin registry for all DL models.

    Usage:
        # Registration (at module load time)
        registry.register_emotion("hsemotion", HSEmotionModel)

        # Retrieval
        model = registry.get_emotion("hsemotion")
        result = model.predict(face_crop)
    """

    _instance: ModelRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ModelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._emotion_registry: dict[str, EmotionModelBase] = {}
        self._gnn_registry: dict[str, GNNModelBase] = {}
        self._temporal_registry: dict[str, TemporalModelBase] = {}
        self._enabled_emotions: list[str] = []
        self._enabled_gnns: list[str] = []
        self._model_lock = threading.RLock()

    # ── Registration ─────────────────────────────────────────────────────────

    def register_emotion(
        self,
        model_id: str,
        model_class: Type[EmotionModelBase],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Register an emotion model class (not yet loaded)."""
        with self._model_lock:
            instance = model_class(*args, **kwargs)
            self._emotion_registry[model_id] = instance
            logger.info("emotion_model_registered", model_id=model_id)

    def register_gnn(
        self,
        model_id: str,
        model_class: Type[GNNModelBase],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Register a GNN model class."""
        with self._model_lock:
            instance = model_class(*args, **kwargs)
            self._gnn_registry[model_id] = instance
            logger.info("gnn_model_registered", model_id=model_id)

    def register_temporal(
        self,
        model_id: str,
        model_class: Type[TemporalModelBase],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Register a temporal model class."""
        with self._model_lock:
            instance = model_class(*args, **kwargs)
            self._temporal_registry[model_id] = instance
            logger.info("temporal_model_registered", model_id=model_id)

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_emotion_models(self, model_ids: list[str]) -> list[str]:
        """
        Load specified emotion models. Returns list of successfully loaded IDs.
        Models not in registry are skipped with a warning.
        """
        loaded: list[str] = []
        for mid in model_ids:
            model = self._emotion_registry.get(mid)
            if model is None:
                logger.warning("emotion_model_not_found", model_id=mid)
                continue
            try:
                model.load()
                loaded.append(mid)
                logger.info("emotion_model_loaded", model_id=mid)
            except Exception as exc:
                logger.error("emotion_model_load_failed", model_id=mid, error=str(exc))
        self._enabled_emotions = loaded
        return loaded

    def load_gnn_models(self, model_ids: list[str]) -> list[str]:
        """Load specified GNN models."""
        loaded: list[str] = []
        for mid in model_ids:
            model = self._gnn_registry.get(mid)
            if model is None:
                logger.warning("gnn_model_not_found", model_id=mid)
                continue
            try:
                model.load()
                loaded.append(mid)
                logger.info("gnn_model_loaded", model_id=mid)
            except Exception as exc:
                logger.error("gnn_model_load_failed", model_id=mid, error=str(exc))
        self._enabled_gnns = loaded
        return loaded

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_emotion(self, model_id: str) -> EmotionModelBase | None:
        return self._emotion_registry.get(model_id)

    def get_gnn(self, model_id: str) -> GNNModelBase | None:
        return self._gnn_registry.get(model_id)

    def get_temporal(self, model_id: str) -> TemporalModelBase | None:
        return self._temporal_registry.get(model_id)

    def list_enabled_emotions(self) -> list[str]:
        return list(self._enabled_emotions)

    def list_all_emotion_models(self) -> list[str]:
        return list(self._emotion_registry.keys())

    def list_all_gnn_models(self) -> list[str]:
        return list(self._gnn_registry.keys())

    # ── Health ────────────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        """Return health status for all registered models."""
        emotion_health = {}
        for mid, model in self._emotion_registry.items():
            emotion_health[mid] = model.health()

        gnn_health: dict[str, Any] = {}
        for mid, model in self._gnn_registry.items():
            gnn_health[mid] = {
                "model_id": mid,
                "loaded": model._loaded,
                "avg_latency_ms": round(model.avg_latency_ms, 2),
            }

        return {
            "emotion_models": emotion_health,
            "gnn_models": gnn_health,
            "enabled_emotions": self._enabled_emotions,
            "enabled_gnns": self._enabled_gnns,
        }


# Singleton instance
registry = ModelRegistry()
