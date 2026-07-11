"""
AuthBrain AI Face Analysis Engine
Inference Confidence Calibration Utilities

Implements:
1. Temperature Scaling: Calibrates multiclass probabilities by scaling pseudo-logits.
2. Platt Scaling: Sigmoid calibration for confidence alignment.
"""

from __future__ import annotations

import numpy as np


def temperature_scale_probs(probs: dict[str, float], temperature: float) -> dict[str, float]:
    """
    Calibrate a probability distribution dictionary using Temperature Scaling.
    
    Args:
        probs: Dictionary mapping class labels to raw probabilities.
        temperature: Scaling factor (T > 1 decreases confidence peaks, T < 1 increases them).

    Returns:
        Calibrated probability dictionary.
    """
    if not probs:
        return {}
    if temperature <= 0.0:
        temperature = 1.0

    labels = list(probs.keys())
    values = np.array([probs[k] for k in labels], dtype=np.float32)
    
    # Convert probabilities to pseudo-logits (with epsilon safety)
    eps = 1e-7
    logits = np.log(values + eps)
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Stable softmax normalization
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    scaled_values = exp_logits / np.sum(exp_logits)
    
    return {labels[i]: float(scaled_values[i]) for i in range(len(labels))}


def platt_scale_confidence(conf: float, a: float = -2.5, b: float = 0.5) -> float:
    """
    Calibrate a single confidence score using Platt Scaling (sigmoid mapping).
    p = 1 / (1 + exp(a * conf + b))
    
    Args:
        conf: Raw confidence score [0, 1].
        a: Sigmoid slope parameter.
        b: Sigmoid intercept/bias parameter.
        
    Returns:
        Calibrated confidence score.
    """
    # Sigmoid function
    return float(1.0 / (1.0 + np.exp(a * conf + b)))
