"""
Emotion XAI summary helpers.

Provides a lightweight explanation payload for the dashboard when the graph
explainer is unavailable or disabled.
"""
from __future__ import annotations

from typing import Any

from app.models.schemas import ExplanationResult, FeatureAttribution


def build_emotion_explanation(
    ensemble_result: dict[str, Any],
    predictions: list[Any],
    quality_score: float,
    smile_intensity: float,
    inference_time_ms: float,
) -> ExplanationResult:
    """Build a DL-facing explanation so XAI is never empty."""
    top_probabilities = sorted(
        ensemble_result["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:3]

    attributions = [
        FeatureAttribution(
            feature_name=f"{prediction.model_id}_vote",
            contribution=max(float(prediction.confidence), 0.01),
            landmark_indices=[],
            value=float(prediction.confidence),
            description=f"{prediction.model_id} voted for {prediction.emotion}.",
        )
        for prediction in predictions
    ]

    if not attributions:
        attributions.append(
            FeatureAttribution(
                feature_name="fallback_uncertainty",
                contribution=1.0,
                landmark_indices=[],
                value=0.0,
                description="No valid emotion model outputs were available for this frame.",
            )
        )

    summary = ", ".join(f"{emotion}={prob:.0%}" for emotion, prob in top_probabilities)
    explanation = (
        f"Emotion ensemble selected {ensemble_result['final_emotion']} at "
        f"{ensemble_result['confidence']:.0%} confidence. "
        f"Top probabilities: {summary}. "
        f"Face quality: {quality_score:.0%}. Smile signal: {smile_intensity:.0%}."
    )

    return ExplanationResult(
        metric_name="Emotion Ensemble",
        final_value=float(ensemble_result["confidence"]),
        confidence=float(max(min(ensemble_result["confidence"], 1.0), 0.0)),
        attributions=attributions,
        processing_time_ms=round(inference_time_ms, 2),
        landmark_quality=float(max(min(quality_score, 1.0), 0.0)),
        explanation_text=explanation,
    )
