"""
AuthBrain DL Platform — GNN Explainer wrapper

Provides Explainable AI (XAI) feature attribution for Graph Neural Networks.
Identifies the top-20 most critical landmarks contributing to the GNN emotion prediction
and generates structured feature attribution records.
"""
from __future__ import annotations

import numpy as np
from typing import Any

from app.analysis.landmark_indices import LANDMARKS
from app.dl.base import GNNPrediction

# Map landmark indices back to human readable names for explanation text
_LANDMARK_LABELS = {
    4: "Nose Tip",
    152: "Chin",
    33: "Right Eye Outer Corner",
    133: "Right Eye Inner Corner",
    263: "Left Eye Outer Corner",
    362: "Left Eye Inner Corner",
    61: "Mouth Corner Right",
    291: "Mouth Corner Left",
    13: "Upper Lip Inner",
    14: "Lower Lip Inner",
    70: "Right Eyebrow Inner",
    336: "Left Eyebrow Inner",
}


class GNNExplainerWrapper:
    """
    Analyzes the GNN prediction node importance weights and returns
    structured feature attributions for key landmark groups.
    """

    def __init__(self) -> None:
        pass

    def explain(self, prediction: GNNPrediction) -> dict[str, Any]:
        """
        Extract the most important nodes/landmarks and compile their contributions.
        """
        importance = np.array(prediction.node_importance, dtype=np.float32)
        if len(importance) == 0:
            return {
                "top_landmarks": [],
                "attributions": [],
                "explanation_text": "GNN explanation not computed.",
            }

        # Find top 20 landmarks by GNN importance score
        top_indices = np.argsort(importance)[::-1][:20].tolist()

        # Group by facial region to show high-level attributions
        region_importance = {
            "eyes": 0.0,
            "mouth": 0.0,
            "nose": 0.0,
            "eyebrows": 0.0,
            "other": 0.0,
        }

        # Calculate region-level importance totals
        left_eye_set = set(LANDMARKS.LEFT_EYE + LANDMARKS.LEFT_IRIS)
        right_eye_set = set(LANDMARKS.RIGHT_EYE + LANDMARKS.RIGHT_IRIS)
        mouth_set = set(
            LANDMARKS.LIPS_UPPER_OUTER + LANDMARKS.LIPS_LOWER_OUTER +
            LANDMARKS.LIPS_UPPER_INNER + LANDMARKS.LIPS_LOWER_INNER
        )
        nose_set = set(LANDMARKS.NOSE_BRIDGE + LANDMARKS.NOSE_CONTOUR + (LANDMARKS.NOSE_TIP,))
        eyebrow_set = set(LANDMARKS.LEFT_EYEBROW + LANDMARKS.RIGHT_EYEBROW)

        for i, val in enumerate(importance):
            if i in left_eye_set or i in right_eye_set:
                region_importance["eyes"] += val
            elif i in mouth_set:
                region_importance["mouth"] += val
            elif i in nose_set:
                region_importance["nose"] += val
            elif i in eyebrow_set:
                region_importance["eyebrows"] += val
            else:
                region_importance["other"] += val

        # Normalize region importance
        total_region = sum(region_importance.values()) + 1e-9
        region_importance = {k: v / total_region for k, v in region_importance.items()}

        # Build feature attribution records
        attributions = []
        for region, val in region_importance.items():
            attributions.append({
                "feature_name": f"region_{region}",
                "contribution": float(val),
                "description": f"Facial region: {region.capitalize()} landmark motion and shape GNN features."
            })

        # Plain text explanation summary
        strongest_region = max(region_importance, key=region_importance.get)  # type: ignore[arg-type]
        top_named_landmarks = []
        for idx in top_indices:
            if idx in _LANDMARK_LABELS:
                top_named_landmarks.append(_LANDMARK_LABELS[idx])
            if len(top_named_landmarks) >= 3:
                break

        explanation_text = (
            f"GNN predicted '{prediction.emotion}' (confidence {prediction.confidence:.0%}). "
            f"Strongest feature activation was in the {strongest_region} region. "
            f"Key contributing landmarks: {', '.join(top_named_landmarks) if top_named_landmarks else 'Facial contour landmarks'}."
        )

        return {
            "top_landmarks": top_indices,
            "attributions": attributions,
            "explanation_text": explanation_text,
        }
