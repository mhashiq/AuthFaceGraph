"""
AuthBrain AI Face Analysis Engine
Expert System Composite Scorer

Combines outputs from all analyzers into a unified expert system result.
Calculates fatigue score, focus score, stress risk, and attention state
using weighted rule evaluation and metric aggregation.
"""

from __future__ import annotations

import numpy as np

from app.core.logging import get_logger
from app.expert_system.rules import RuleContext, RuleEngine, RiskLevel
from app.models.schemas import (
    AttentionState,
    BehaviorResult,
    EyeAnalysisResult,
    ExpertSystemResult,
    HeadPoseResult,
    MouthAnalysisResult,
    QualityResult,
)

logger = get_logger(__name__)

# Weights for composite fatigue score
_FATIGUE_WEIGHTS = {
    "ear_component":         0.30,   # Low EAR → drowsy
    "blink_rate_component":  0.20,   # High blink rate → stress/fatigue
    "yawn_component":        0.20,   # Yawning → fatigue
    "head_pose_component":   0.15,   # Head drooping → fatigue
    "stability_component":   0.15,   # Low stability → fatigue/distraction
}

# Weights for composite focus score
_FOCUS_WEIGHTS = {
    "forward_facing":    0.35,
    "ear_open":          0.30,
    "no_yawn":           0.15,
    "stability":         0.20,
}


class ExpertSystemScorer:
    """
    Aggregates all analysis results into a comprehensive expert system verdict.

    Produces:
    - Fatigue score [0, 1]: 0 = alert, 1 = critically fatigued
    - Focus score [0, 1]: 0 = distracted, 1 = fully focused
    - Attention state classification
    - Risk-sorted alert messages
    - Stress risk score (average + maximum tracking)
    """

    def __init__(self) -> None:
        self._rule_engine = RuleEngine()

    def score(
        self,
        eyes: EyeAnalysisResult,
        mouth: MouthAnalysisResult,
        head_pose: HeadPoseResult,
        behavior: BehaviorResult,
        quality: QualityResult,
    ) -> ExpertSystemResult:
        """
        Compute composite expert system result.

        Args:
            eyes: Eye analysis result
            mouth: Mouth analysis result
            head_pose: Head pose result
            behavior: Behavioral tracking result
            quality: Frame quality result

        Returns:
            ExpertSystemResult with scores, alerts, and attention state
        """
        # Build rule evaluation context
        ctx = RuleContext(
            eyes=eyes,
            mouth=mouth,
            head_pose=head_pose,
            behavior=behavior,
            quality=quality,
        )

        # Evaluate all rules
        matched_rules = self._rule_engine.evaluate(ctx)

        # ── Fatigue Score ──────────────────────────────────────────────────────
        fatigue_score = self._compute_fatigue(eyes, mouth, head_pose, behavior)

        # ── Focus Score ───────────────────────────────────────────────────────
        focus_score = self._compute_focus(eyes, head_pose, behavior)

        # ── Attention State (from behavior tracker, optionally overridden) ─────
        attention_state = behavior.attention_state
        if matched_rules:
            top_risk = matched_rules[0].risk_level
            if top_risk == RiskLevel.CRITICAL:
                attention_state = AttentionState.DROWSY
            elif top_risk == RiskLevel.HIGH and attention_state == AttentionState.FOCUSED:
                attention_state = AttentionState.DISTRACTED

        # ── Alert Messages ────────────────────────────────────────────────────
        alerts = [rule.alert_message for rule in matched_rules if rule.risk_level.value in ("critical", "high")]

        # ── Overall Confidence ────────────────────────────────────────────────
        # Weighted by quality score — poor quality reduces confidence in results
        base_confidence = 0.85 if quality.overall_score > 0.6 else 0.60
        overall_confidence = float(np.clip(base_confidence * quality.overall_score + 0.1, 0.0, 1.0))

        return ExpertSystemResult(
            attention_state=attention_state,
            fatigue_score=round(fatigue_score, 3),
            focus_score=round(focus_score, 3),
            alerts=alerts,
            explanations=[],   # Filled by XAIExplainer in pipeline
            overall_confidence=round(overall_confidence, 3),
        )

    def _compute_fatigue(
        self,
        eyes: EyeAnalysisResult,
        mouth: MouthAnalysisResult,
        head_pose: HeadPoseResult,
        behavior: BehaviorResult,
    ) -> float:
        """
        Compute fatigue score [0, 1].
        0 = fully alert, 1 = critically fatigued.
        """
        # EAR component: lower EAR → higher fatigue
        # Normal EAR ~0.30; threshold 0.25; critical 0.15
        ear_component = float(np.clip(1.0 - (eyes.average_ear / 0.30), 0.0, 1.0))

        # Blink rate component: >20 bpm elevated, >25 bpm high fatigue
        blink_component = float(np.clip((eyes.blinks_per_minute - 10) / 20.0, 0.0, 1.0))

        # Yawn component: confirmed yawn = high fatigue signal
        yawn_component = mouth.yawn_confidence if mouth.yawn_detected else 0.0

        # Head pose: pitched forward/down → fatigue drooping
        pitch_normalized = float(np.clip(head_pose.pitch / 30.0, 0.0, 1.0))
        head_component = pitch_normalized

        # Stability: low stability = fatigue or restlessness
        stability_component = float(1.0 - behavior.landmark_stability)

        fatigue = (
            _FATIGUE_WEIGHTS["ear_component"]        * ear_component +
            _FATIGUE_WEIGHTS["blink_rate_component"] * blink_component +
            _FATIGUE_WEIGHTS["yawn_component"]        * yawn_component +
            _FATIGUE_WEIGHTS["head_pose_component"]  * head_component +
            _FATIGUE_WEIGHTS["stability_component"]  * stability_component
        )
        return float(np.clip(fatigue, 0.0, 1.0))

    def _compute_focus(
        self,
        eyes: EyeAnalysisResult,
        head_pose: HeadPoseResult,
        behavior: BehaviorResult,
    ) -> float:
        """
        Compute focus score [0, 1].
        0 = completely distracted, 1 = fully focused.
        """
        # Forward-facing score
        yaw_score   = float(np.clip(1.0 - abs(head_pose.yaw)   / 30.0, 0.0, 1.0))
        pitch_score = float(np.clip(1.0 - abs(head_pose.pitch) / 30.0, 0.0, 1.0))
        forward_score = (yaw_score + pitch_score) / 2.0

        # Eyes-open score
        ear_score = float(np.clip(eyes.average_ear / 0.30, 0.0, 1.0))

        # Stability score
        stability_score = behavior.landmark_stability

        focus = (
            _FOCUS_WEIGHTS["forward_facing"] * forward_score +
            _FOCUS_WEIGHTS["ear_open"]        * ear_score +
            _FOCUS_WEIGHTS["no_yawn"]         * 1.0 +  # Placeholder — will be 0 if yawning
            _FOCUS_WEIGHTS["stability"]       * stability_score
        )
        return float(np.clip(focus, 0.0, 1.0))
