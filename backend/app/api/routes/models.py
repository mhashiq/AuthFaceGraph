"""
AuthBrain AI Face Analysis Engine
Deep Learning Models Route

Provides endpoints to query the deep learning model registry,
model health, configuration, and enabled components.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import get_settings
from app.dl.registry import registry
from app.dl.engine import DLEngine  # Import to trigger automatic registration
from app.core.security import get_current_user  # If exists

settings = get_settings()
router = APIRouter(prefix="/api/models", tags=["deep-learning-models"])


@router.get("/", summary="List all registered DL models")
async def list_models() -> dict:
    """
    Returns list of registered image-based emotion models, GNN models,
    and currently loaded/enabled models.
    """
    return {
        "dl_enabled": settings.DL_ENABLED,
        "registered_emotions": registry.list_all_emotion_models(),
        "registered_gnns": registry.list_all_gnn_models(),
        "enabled_emotions": registry.list_enabled_emotions(),
        "enabled_gnns": registry.list_enabled_emotions(),
    }


@router.get("/health", summary="Get DL model registry health report")
async def get_models_health() -> dict:
    """
    Returns detail health status of each model (latency, loading state, error rate).
    """
    return registry.health_report()


@router.post("/{model_id}/enable", summary="Enable/disable model at runtime")
async def enable_model(model_id: str, enable: bool) -> dict:
    """
    Enables or disables a model in the active configuration registry.
    """
    emotion_models = registry.list_all_emotion_models()
    gnn_models = registry.list_all_gnn_models()

    if model_id not in emotion_models and model_id not in gnn_models:
        raise HTTPException(status_code=404, detail="Model ID not found in registry")

    if model_id in emotion_models:
        active = list(registry.list_enabled_emotions())
        if enable and model_id not in active:
            try:
                registry.load_emotion_models([model_id])
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
        elif not enable and model_id in active:
            active.remove(model_id)
            registry._enabled_emotions = active

    return {
        "model_id": model_id,
        "enabled": enable,
        "active_emotions": registry.list_enabled_emotions(),
    }
