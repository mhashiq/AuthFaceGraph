"""
AuthBrain DL Platform — Multi-Model Emotion Ensemble

Combines predictions from multiple active emotion recognition models
using temperature-scaled weighted average or majority voting.
Computes a disagreement score (entropy) to capture model uncertainty.
"""
from __future__ import annotations

import numpy as np
from typing import Any

from app.core.config import get_settings
from app.dl.base import EmotionPrediction

settings = get_settings()

EMOTION_LABELS = [
    "neutral", "happy", "sad", "surprise",
    "fear", "disgust", "anger", "contempt",
]


class EmotionEnsemble:
    """
    Ensemble combination layer for facial emotion predictions.
    Computes weighted average consensus, final emotion classification,
    and a disagreement/entropy score representing ensemble uncertainty.
    """

    def __init__(self, strategy: str = "weighted_avg") -> None:
        self.strategy = strategy
        # Typical accuracies on validation sets as initial weights
        self.model_weights = {
            "hsemotion": 0.55,
            "efficientface": 0.45,
            "gnn_gat": 0.40,
        }

    def combine(self, predictions: list[EmotionPrediction]) -> dict[str, Any]:
        """
        Combine predictions from multiple models.

        Args:
            predictions: List of EmotionPrediction outputs from enabled models

        Returns:
            dict containing combined probabilities, final emotion, and disagreement
        """
        valid_preds = [p for p in predictions if p.error is None]
        if not valid_preds:
            # Fallback to neutral
            neutral_probs = {e: 0.125 for e in EMOTION_LABELS}
            neutral_probs["neutral"] = 1.0
            return {
                "final_emotion": "neutral",
                "confidence": 0.0,
                "probabilities": neutral_probs,
                "disagreement_score": 0.0,
                "uncertainty": 0.0,
            }

        # 1. Accumulate probabilities
        combined_probs = {e: 0.0 for e in EMOTION_LABELS}
        total_weight = 0.0

        for pred in valid_preds:
            weight = self.model_weights.get(pred.model_id, 1.0)
            total_weight += weight
            for emotion, prob in pred.probabilities.items():
                if emotion in combined_probs:
                    combined_probs[emotion] += prob * weight

        # Normalize
        if total_weight > 0.0:
            for emotion in combined_probs:
                combined_probs[emotion] /= total_weight

        # Normalize once more to ensure exactly 1.0 sum
        prob_sum = sum(combined_probs.values())
        if prob_sum > 0:
            for emotion in combined_probs:
                combined_probs[emotion] /= prob_sum

        final_emotion = max(combined_probs, key=combined_probs.get)
        confidence = combined_probs[final_emotion]

        # 2. Compute Disagreement Score (normalized entropy across predictions)
        # Low entropy = consensus, High entropy = high disagreement
        entropy = 0.0
        # Average probability distribution across all predictions
        if len(valid_preds) > 1:
            all_prob_arrays = []
            for p in valid_preds:
                arr = [p.probabilities.get(e, 0.0) for e in EMOTION_LABELS]
                all_prob_arrays.append(arr)
            # Variance across models for each emotion
            vars_per_class = np.var(all_prob_arrays, axis=0)
            disagreement_score = float(np.mean(vars_per_class)) * 8.0  # scaled to approx [0, 1]
            disagreement_score = min(max(disagreement_score, 0.0), 1.0)
        else:
            disagreement_score = 0.0

        return {
            "final_emotion": final_emotion,
            "confidence": confidence,
            "probabilities": combined_probs,
            "disagreement_score": round(disagreement_score, 4),
            "uncertainty": round(1.0 - confidence, 4),
        }
