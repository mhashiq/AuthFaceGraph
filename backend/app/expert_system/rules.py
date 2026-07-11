"""
AuthBrain AI Face Analysis Engine
Expert System Rule Engine

Rule-based inference engine that classifies attention, fatigue, and risk levels
from facial analysis metrics. Each rule is explicit and explainable.

Rules are evaluated in priority order. The first matching rule wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from app.core.logging import get_logger
from app.models.schemas import (
    AttentionState,
    BehaviorResult,
    EyeAnalysisResult,
    HeadPoseResult,
    MouthAnalysisResult,
    QualityResult,
)

logger = get_logger(__name__)


class RiskLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    CRITICAL = "critical"


@dataclass
class RuleContext:
    """Input data for rule evaluation."""
    eyes: EyeAnalysisResult
    mouth: MouthAnalysisResult
    head_pose: HeadPoseResult
    behavior: BehaviorResult
    quality: QualityResult


@dataclass
class RuleMatch:
    """Result of a matching rule."""
    rule_id: str
    description: str
    alert_message: str
    risk_level: RiskLevel
    confidence: float
    contributing_features: list[str]


@dataclass
class ExpertRule:
    """A single named, documented rule."""
    rule_id: str
    description: str
    alert_message: str
    risk_level: RiskLevel
    condition: Callable[[RuleContext], bool]
    confidence_fn: Callable[[RuleContext], float]
    contributing_features: list[str]

    def evaluate(self, ctx: RuleContext) -> RuleMatch | None:
        """Evaluate this rule against a context. Returns match or None."""
        try:
            if self.condition(ctx):
                confidence = self.confidence_fn(ctx)
                return RuleMatch(
                    rule_id=self.rule_id,
                    description=self.description,
                    alert_message=self.alert_message,
                    risk_level=self.risk_level,
                    confidence=confidence,
                    contributing_features=self.contributing_features,
                )
        except Exception as exc:
            logger.warning("rule_evaluation_error", rule_id=self.rule_id, error=str(exc))
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Rule Definitions
# Each rule is fully documented and independently testable.
# ══════════════════════════════════════════════════════════════════════════════

EXPERT_RULES: list[ExpertRule] = [

    # ── Critical Rules ─────────────────────────────────────────────────────────
    ExpertRule(
        rule_id="RULE_EYES_CLOSED_EXTENDED",
        description="Eyes closed for more than 2 seconds — potential microsleep",
        alert_message="⚠️ Eyes closed for extended period. Risk of microsleep.",
        risk_level=RiskLevel.CRITICAL,
        condition=lambda ctx: ctx.eyes.eye_closure_duration_ms > 2000,
        confidence_fn=lambda ctx: min(ctx.eyes.eye_closure_duration_ms / 3000.0, 1.0),
        contributing_features=["average_ear", "eye_closure_duration_ms"],
    ),

    ExpertRule(
        rule_id="RULE_EXTREMELY_LOW_EAR",
        description="EAR critically low — eyes nearly or fully closed",
        alert_message="⚠️ Critical eye closure detected.",
        risk_level=RiskLevel.CRITICAL,
        condition=lambda ctx: ctx.eyes.average_ear < 0.15 and ctx.eyes.eye_closure_duration_ms > 350,
        confidence_fn=lambda ctx: 1.0 - (ctx.eyes.average_ear / 0.15),
        contributing_features=["average_ear"],
    ),

    # ── High Risk Rules ────────────────────────────────────────────────────────
    ExpertRule(
        rule_id="RULE_YAWNING_HIGH_BLINK",
        description="Yawning combined with elevated blink rate — fatigue indicator",
        alert_message="🥱 Fatigue signs detected: yawning with high blink rate.",
        risk_level=RiskLevel.HIGH,
        condition=lambda ctx: ctx.mouth.yawn_detected and ctx.eyes.blinks_per_minute > 22,
        confidence_fn=lambda ctx: min(ctx.mouth.yawn_confidence + 0.1, 1.0),
        contributing_features=["yawn_detected", "blinks_per_minute", "mar"],
    ),

    ExpertRule(
        rule_id="RULE_SUSTAINED_HEAD_TURN",
        description="Head rotated significantly away from forward position",
        alert_message="↩️ Sustained head turn detected — distracted from screen.",
        risk_level=RiskLevel.HIGH,
        condition=lambda ctx: abs(ctx.head_pose.yaw) > 30 or abs(ctx.head_pose.pitch) > 25,
        confidence_fn=lambda ctx: min((max(abs(ctx.head_pose.yaw), abs(ctx.head_pose.pitch)) - 25) / 30.0 + 0.6, 1.0),
        contributing_features=["head_yaw", "head_pitch"],
    ),

    ExpertRule(
        rule_id="RULE_LOW_EAR_DROWSY",
        description="Persistently low EAR — signs of drowsiness",
        alert_message="😴 Drowsiness detected: reduced eye openness.",
        risk_level=RiskLevel.HIGH,
        condition=lambda ctx: 0.15 <= ctx.eyes.average_ear < 0.22 and ctx.eyes.eye_closure_duration_ms > 400,
        confidence_fn=lambda ctx: (0.22 - ctx.eyes.average_ear) / 0.07,
        contributing_features=["average_ear", "blinks_per_minute"],
    ),

    # ── Medium Risk Rules ──────────────────────────────────────────────────────
    ExpertRule(
        rule_id="RULE_FREQUENT_BLINKING",
        description="Blink rate significantly above baseline (>20 bpm)",
        alert_message="👁️ High blink frequency — possible eye strain or stress.",
        risk_level=RiskLevel.MEDIUM,
        condition=lambda ctx: ctx.eyes.blinks_per_minute > 20,
        confidence_fn=lambda ctx: min((ctx.eyes.blinks_per_minute - 20) / 10.0 + 0.5, 0.9),
        contributing_features=["blinks_per_minute"],
    ),

    ExpertRule(
        rule_id="RULE_LOW_FACE_QUALITY",
        description="Face quality below acceptable threshold for reliable analysis",
        alert_message="📷 Poor image quality — results may be unreliable.",
        risk_level=RiskLevel.MEDIUM,
        condition=lambda ctx: ctx.quality.overall_score < 0.4,
        confidence_fn=lambda ctx: 1.0 - ctx.quality.overall_score,
        contributing_features=["sharpness", "illumination", "face_size_ratio"],
    ),

    ExpertRule(
        rule_id="RULE_LOW_FACIAL_SYMMETRY",
        description="Facial symmetry dropped below normal — possible fatigue or distress",
        alert_message="↔️ Facial asymmetry detected — potential fatigue marker.",
        risk_level=RiskLevel.MEDIUM,
        condition=lambda ctx: ctx.behavior.facial_symmetry < 0.6,
        confidence_fn=lambda ctx: (0.6 - ctx.behavior.facial_symmetry) / 0.3,
        contributing_features=["facial_symmetry"],
    ),

    # ── Low Risk / Informational ───────────────────────────────────────────────
    ExpertRule(
        rule_id="RULE_YAWNING_ISOLATED",
        description="Single yawn detected — may indicate mild fatigue or boredom",
        alert_message="🥱 Yawn detected.",
        risk_level=RiskLevel.LOW,
        condition=lambda ctx: ctx.mouth.yawn_detected and ctx.eyes.blinks_per_minute <= 22,
        confidence_fn=lambda ctx: ctx.mouth.yawn_confidence,
        contributing_features=["yawn_detected", "mar"],
    ),

    ExpertRule(
        rule_id="RULE_HEAD_TILT",
        description="Head significantly tilted (high roll) — postural issue",
        alert_message="↗️ Head tilt detected. Check posture.",
        risk_level=RiskLevel.LOW,
        condition=lambda ctx: abs(ctx.head_pose.roll) > 20,
        confidence_fn=lambda ctx: min(abs(ctx.head_pose.roll) / 40.0, 0.8),
        contributing_features=["head_roll"],
    ),

    ExpertRule(
        rule_id="RULE_LOW_LANDMARK_STABILITY",
        description="Face landmarks are unstable — possibly due to movement",
        alert_message="📍 Landmark instability — please remain still.",
        risk_level=RiskLevel.LOW,
        condition=lambda ctx: ctx.behavior.landmark_stability < 0.5,
        confidence_fn=lambda ctx: 1.0 - ctx.behavior.landmark_stability,
        contributing_features=["landmark_stability"],
    ),
]


class RuleEngine:
    """
    Evaluates all expert rules against a context and returns matched rules.
    """

    def __init__(self, rules: list[ExpertRule] | None = None) -> None:
        self._rules = rules or EXPERT_RULES

    def evaluate(self, ctx: RuleContext) -> list[RuleMatch]:
        """
        Evaluate all rules against the given context.

        Returns:
            List of matched rules sorted by risk level (critical first)
        """
        matches: list[RuleMatch] = []
        for rule in self._rules:
            match = rule.evaluate(ctx)
            if match is not None:
                matches.append(match)

        # Sort by risk severity
        severity_order = {
            RiskLevel.CRITICAL: 0,
            RiskLevel.HIGH: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 3,
        }
        matches.sort(key=lambda m: (severity_order[m.risk_level], -m.confidence))
        return matches
