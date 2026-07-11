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

    def combine(
        self,
        predictions: list[EmotionPrediction],
        quality_score: float = 1.0,
        smile_intensity: float = 0.0,
        model_latencies: dict[str, float] | None = None,
        model_errors: dict[str, int] | None = None,
        strategy: str | None = None
    ) -> dict[str, Any]:
        """
        Combine predictions from multiple models.

        Args:
            predictions: List of EmotionPrediction outputs from enabled models.
            quality_score: Face quality score [0, 1] used in dynamic reliability weighting.
            model_latencies: Optional dict of model runtimes to compute reliability.
            model_errors: Optional dict of model failure counts.
            strategy: Dynamic strategy override.

        Returns:
            dict containing combined probabilities, final emotion, agreement_score, and disagreement.
        """
        valid_preds = [p for p in predictions if p.error is None]
        if not valid_preds:
            raise RuntimeError("No valid emotion model predictions were produced for this frame.")

        active_strategy = strategy or self.strategy

        combined_probs = {emotion: 0.0 for emotion in EMOTION_LABELS}

        if active_strategy == "hard_voting":
            votes = {emotion: 0.0 for emotion in EMOTION_LABELS}
            confidence_totals = {emotion: 0.0 for emotion in EMOTION_LABELS}

            for prediction in valid_preds:
                votes[prediction.emotion] += 1.0
                confidence_totals[prediction.emotion] += prediction.confidence
                for emotion, probability in prediction.probabilities.items():
                    combined_probs[emotion] += probability

            vote_winners = [emotion for emotion, count in votes.items() if count == max(votes.values())]
            if len(vote_winners) == 1:
                final_emotion = vote_winners[0]
            else:
                final_emotion = max(
                    vote_winners,
                    key=lambda emotion: confidence_totals[emotion] / max(1.0, votes[emotion]),
                )

            confidence = confidence_totals[final_emotion] / max(1.0, votes[final_emotion])
            combined_probs = {emotion: value / len(valid_preds) for emotion, value in combined_probs.items()}
        else:
            total_weight = 0.0
            for prediction in valid_preds:
                weight = 1.0
                if active_strategy == "weighted_avg":
                    weight = self.model_weights.get(prediction.model_id, 1.0)
                elif active_strategy == "confidence_weighted":
                    weight = prediction.confidence
                elif active_strategy == "dynamic_reliability":
                    weight = self.model_weights.get(prediction.model_id, 1.0)
                    if prediction.model_id in ["hsemotion", "efficientface"]:
                        weight *= max(0.1, quality_score)
                    if model_latencies and prediction.model_id in model_latencies:
                        latency = model_latencies[prediction.model_id]
                        if latency > 100.0:
                            weight *= 100.0 / latency
                    if model_errors and prediction.model_id in model_errors:
                        errors = model_errors[prediction.model_id]
                        if errors > 0:
                            weight *= max(0.01, 1.0 / (errors + 1))

                total_weight += weight
                for emotion, probability in prediction.probabilities.items():
                    combined_probs[emotion] += probability * weight

            if total_weight > 0.0:
                combined_probs = {emotion: value / total_weight for emotion, value in combined_probs.items()}
            else:
                for prediction in valid_preds:
                    for emotion, probability in prediction.probabilities.items():
                        combined_probs[emotion] += probability / len(valid_preds)

            smile_signal = float(np.clip(smile_intensity, 0.0, 1.0))
            if smile_signal >= 0.4:
                smile_boost = min(0.35, 0.18 + (smile_signal - 0.4) * 0.5)
                sad_penalty = min(combined_probs["sad"], smile_boost * 0.9)
                combined_probs["happy"] += smile_boost
                combined_probs["sad"] = max(0.0, combined_probs["sad"] - sad_penalty)

                if smile_signal >= 0.6 and combined_probs["happy"] <= combined_probs["sad"]:
                    combined_probs["happy"] = max(combined_probs["happy"], smile_signal * 0.55)
                    combined_probs["sad"] = min(combined_probs["sad"], 1.0 - combined_probs["happy"])

            prob_sum = sum(combined_probs.values())
            if prob_sum > 0.0:
                combined_probs = {emotion: float(value / prob_sum) for emotion, value in combined_probs.items()}

            final_emotion = max(combined_probs, key=combined_probs.get)
            confidence = combined_probs[final_emotion]

        min_confidence = float(getattr(settings, "DL_EMOTION_MIN_CONFIDENCE", 0.15))
        if confidence < min_confidence:
            final_emotion = "unknown"

        # 2. Compute agreement score
        models_agreeing = sum(1 for p in valid_preds if p.emotion == final_emotion)
        agreement_score = float(models_agreeing / len(valid_preds))

        # 3. Compute Disagreement Score (normalized entropy/variance across models)
        if len(valid_preds) > 1:
            all_prob_arrays = []
            for p in valid_preds:
                arr = [p.probabilities.get(e, 0.0) for e in EMOTION_LABELS]
                all_prob_arrays.append(arr)
            vars_per_class = np.var(all_prob_arrays, axis=0)
            disagreement_score = float(np.mean(vars_per_class)) * 8.0
            disagreement_score = min(max(disagreement_score, 0.0), 1.0)
        else:
            disagreement_score = 0.0

        # Raw vs Calibrated comparison
        raw_conf_sum = sum(p.raw_confidence for p in valid_preds) / len(valid_preds)

        return {
            "final_emotion": final_emotion,
            "confidence": confidence,  # Calibrated
            "probabilities": combined_probs,
            "disagreement_score": round(disagreement_score, 4),
            "uncertainty": round(1.0 - confidence, 4),
            "agreement_score": round(agreement_score, 4),
            "raw_confidence": round(raw_conf_sum, 4),
            "calibrated_confidence": round(confidence, 4),
        }
