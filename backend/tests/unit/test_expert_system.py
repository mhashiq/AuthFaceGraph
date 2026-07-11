"""
AuthBrain AI Face Analysis Engine
Unit Tests — Expert System Rules

Tests rule evaluation, match ordering, and score computation.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.expert_system.rules import RuleEngine, RuleContext, RiskLevel
from app.models.schemas import (
    AttentionState, BehaviorResult, EyeAnalysisResult, EyeResult,
    GazeDirection, HeadPoseResult, MouthAnalysisResult, QualityResult,
)


def make_context(
    avg_ear: float = 0.30,
    eye_closure_ms: float = 0.0,
    blinks_per_min: float = 15.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    yawn_detected: bool = False,
    yawn_confidence: float = 0.0,
    quality_score: float = 0.9,
    symmetry: float = 0.9,
    stability: float = 0.9,
    mar: float = 0.1,
) -> RuleContext:
    """Helper to create a RuleContext with configurable fields."""
    eyes = EyeAnalysisResult(
        left=EyeResult(ear=avg_ear, is_open=avg_ear > 0.25),
        right=EyeResult(ear=avg_ear, is_open=avg_ear > 0.25),
        average_ear=avg_ear,
        blink_count=0,
        eye_closure_duration_ms=eye_closure_ms,
        gaze_direction=GazeDirection.CENTER,
        blinks_per_minute=blinks_per_min,
    )
    mouth = MouthAnalysisResult(
        mar=mar, is_open=mar > 0.3,
        yawn_detected=yawn_detected,
        yawn_confidence=yawn_confidence,
        smile_intensity=0.1,
        mouth_openness_percent=mar * 100,
    )
    head_pose = HeadPoseResult(pitch=pitch, yaw=yaw, roll=roll, is_facing_forward=abs(yaw) < 20)
    behavior = BehaviorResult(
        head_movement_velocity=0.01,
        facial_movement_score=0.1,
        landmark_stability=stability,
        facial_symmetry=symmetry,
        attention_state=AttentionState.FOCUSED,
    )
    quality = QualityResult(
        overall_score=quality_score,
        sharpness=quality_score,
        illumination=quality_score,
        face_size_ratio=quality_score,
        landmark_confidence=quality_score,
    )
    return RuleContext(eyes=eyes, mouth=mouth, head_pose=head_pose, behavior=behavior, quality=quality)


class TestRuleEngine:

    def test_no_alerts_for_normal_face(self):
        """No rules should match for a normal, alert, forward-facing face."""
        engine = RuleEngine()
        ctx    = make_context(avg_ear=0.30, yaw=5.0, quality_score=0.85)
        matches = engine.evaluate(ctx)
        high_risk = [m for m in matches if m.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)]
        assert len(high_risk) == 0, f"Unexpected high risk alerts: {[m.rule_id for m in high_risk]}"

    def test_critical_rule_matches_low_ear(self):
        """RULE_EXTREMELY_LOW_EAR should trigger for EAR < 0.15."""
        engine  = RuleEngine()
        ctx     = make_context(avg_ear=0.10, eye_closure_ms=500.0)
        matches = engine.evaluate(ctx)
        rule_ids = [m.rule_id for m in matches]
        assert "RULE_EXTREMELY_LOW_EAR" in rule_ids

    def test_critical_rule_matches_extended_closure(self):
        """RULE_EYES_CLOSED_EXTENDED should trigger for closure > 2000ms."""
        engine  = RuleEngine()
        ctx     = make_context(avg_ear=0.20, eye_closure_ms=2500.0)
        matches = engine.evaluate(ctx)
        rule_ids = [m.rule_id for m in matches]
        assert "RULE_EYES_CLOSED_EXTENDED" in rule_ids

    def test_high_risk_rule_yawn_with_blinks(self):
        """RULE_YAWNING_HIGH_BLINK should trigger for yawn + high blink rate."""
        engine  = RuleEngine()
        ctx     = make_context(yawn_detected=True, yawn_confidence=0.9, blinks_per_min=25.0)
        matches = engine.evaluate(ctx)
        rule_ids = [m.rule_id for m in matches]
        assert "RULE_YAWNING_HIGH_BLINK" in rule_ids

    def test_high_risk_rule_head_turn(self):
        """RULE_SUSTAINED_HEAD_TURN should trigger for yaw > 30 degrees."""
        engine  = RuleEngine()
        ctx     = make_context(yaw=40.0)
        matches = engine.evaluate(ctx)
        rule_ids = [m.rule_id for m in matches]
        assert "RULE_SUSTAINED_HEAD_TURN" in rule_ids

    def test_results_sorted_by_severity(self):
        """Matches should be sorted critical > high > medium > low."""
        engine  = RuleEngine()
        # Trigger multiple rules at once
        ctx     = make_context(avg_ear=0.10, eye_closure_ms=500.0, yaw=35.0, quality_score=0.3, stability=0.4)
        matches = engine.evaluate(ctx)

        severity = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        levels   = [severity[m.risk_level] for m in matches]
        assert levels == sorted(levels), "Results should be sorted by severity"

    def test_confidence_in_valid_range(self):
        """All rule confidences should be in [0, 1]."""
        engine  = RuleEngine()
        ctx     = make_context(avg_ear=0.10, eye_closure_ms=500.0, yaw=45.0, yawn_detected=True)
        matches = engine.evaluate(ctx)
        for m in matches:
            assert 0.0 <= m.confidence <= 1.0, f"Rule {m.rule_id} confidence {m.confidence} out of range"

    def test_quality_rule_triggers_for_low_score(self):
        """RULE_LOW_FACE_QUALITY should trigger for quality < 0.4."""
        engine  = RuleEngine()
        ctx     = make_context(quality_score=0.25)
        matches = engine.evaluate(ctx)
        rule_ids = [m.rule_id for m in matches]
        assert "RULE_LOW_FACE_QUALITY" in rule_ids

    def test_symmetry_rule_triggers_for_low_symmetry(self):
        """RULE_LOW_FACIAL_SYMMETRY should trigger for symmetry < 0.6."""
        engine  = RuleEngine()
        ctx     = make_context(symmetry=0.45)
        matches = engine.evaluate(ctx)
        rule_ids = [m.rule_id for m in matches]
        assert "RULE_LOW_FACIAL_SYMMETRY" in rule_ids


class TestExpertSystemScorer:

    def test_fatigue_score_high_for_drowsy_state(self):
        from app.expert_system.scorer import ExpertSystemScorer
        scorer = ExpertSystemScorer()
        eyes = MagicMock()
        eyes.average_ear = 0.15
        eyes.blinks_per_minute = 25.0
        mouth = MagicMock()
        mouth.yawn_detected = True
        mouth.yawn_confidence = 0.9
        head_pose = MagicMock()
        head_pose.yaw = 5.0; head_pose.pitch = 5.0; head_pose.roll = 2.0
        head_pose.is_facing_forward = True
        behavior = MagicMock()
        behavior.landmark_stability = 0.8
        behavior.facial_symmetry = 0.85
        behavior.attention_state = AttentionState.DROWSY
        quality = MagicMock()
        quality.overall_score = 0.8
        quality.landmark_confidence = 0.9

        result = scorer.score(eyes, mouth, head_pose, behavior, quality)
        assert result.fatigue_score > 0.5, "Drowsy state should yield high fatigue score"

    def test_focus_score_high_for_alert_state(self):
        from app.expert_system.scorer import ExpertSystemScorer
        scorer = ExpertSystemScorer()
        eyes = MagicMock()
        eyes.average_ear = 0.32
        eyes.blinks_per_minute = 15.0
        mouth = MagicMock()
        mouth.yawn_detected = False
        mouth.yawn_confidence = 0.0
        head_pose = MagicMock()
        head_pose.yaw = 2.0; head_pose.pitch = 1.0; head_pose.roll = 1.0
        head_pose.is_facing_forward = True
        behavior = MagicMock()
        behavior.landmark_stability = 0.95
        behavior.facial_symmetry = 0.92
        behavior.attention_state = AttentionState.FOCUSED
        quality = MagicMock()
        quality.overall_score = 0.9
        quality.landmark_confidence = 0.98

        result = scorer.score(eyes, mouth, head_pose, behavior, quality)
        assert result.focus_score > 0.6, "Alert state should yield high focus score"
