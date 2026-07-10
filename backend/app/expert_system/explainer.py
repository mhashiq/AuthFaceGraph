"""
AuthBrain AI Face Analysis Engine
XAI Explainer — Feature Attribution

Generates human-readable explanations for every metric produced by the pipeline.
For each key output (EAR, head pose, smile, behavior), explains:
 - Which facial landmarks contributed
 - Normalized contribution weight
 - Plain-English description
 - Landmark quality score

This module implements the XAI (Explainable AI) requirement.
"""

from __future__ import annotations

import time

from app.analysis.landmark_indices import LANDMARKS
from app.core.logging import get_logger
from app.models.schemas import (
    BehaviorResult,
    EyeAnalysisResult,
    ExplanationResult,
    FeatureAttribution,
    HeadPoseResult,
    MouthAnalysisResult,
    QualityResult,
)

logger = get_logger(__name__)


class XAIExplainer:
    """
    Produces feature attribution explanations for key analysis outputs.

    For each metric, describes:
    - The contributing landmark indices
    - A normalized weight (0–1) for each feature
    - Human-readable explanation text
    """

    def explain(
        self,
        eyes: EyeAnalysisResult,
        mouth: MouthAnalysisResult,
        head_pose: HeadPoseResult,
        behavior: BehaviorResult,
        quality: QualityResult,
        inference_time_ms: float,
    ) -> list[ExplanationResult]:
        """
        Generate XAI explanations for all analysis metrics.

        Returns:
            List of ExplanationResult objects, one per key metric
        """
        t_start = time.time()
        explanations: list[ExplanationResult] = []

        # ── EAR / Blink Explanation ────────────────────────────────────────────
        explanations.append(self._explain_ear(eyes, inference_time_ms))

        # ── Head Pose Explanation ─────────────────────────────────────────────
        explanations.append(self._explain_head_pose(head_pose, inference_time_ms))

        # ── Mouth / Yawn / Smile Explanation ──────────────────────────────────
        explanations.append(self._explain_mouth(mouth, inference_time_ms))

        # ── Behavior / Attention Explanation ──────────────────────────────────
        explanations.append(self._explain_behavior(behavior, quality, inference_time_ms))

        return explanations

    def _explain_ear(
        self,
        eyes: EyeAnalysisResult,
        base_time_ms: float,
    ) -> ExplanationResult:
        """Explain EAR and blink detection."""
        ear = eyes.average_ear
        confidence = min(ears_to_confidence(ear), 1.0)

        attributions = [
            FeatureAttribution(
                feature_name="left_eye_vertical_opening",
                contribution=0.40,
                landmark_indices=list(LANDMARKS.LEFT_EYE_EAR_POINTS),
                value=round(eyes.left.ear, 4),
                description="Vertical distance between upper and lower left eyelid landmarks",
            ),
            FeatureAttribution(
                feature_name="right_eye_vertical_opening",
                contribution=0.40,
                landmark_indices=list(LANDMARKS.RIGHT_EYE_EAR_POINTS),
                value=round(eyes.right.ear, 4),
                description="Vertical distance between upper and lower right eyelid landmarks",
            ),
            FeatureAttribution(
                feature_name="eye_horizontal_width",
                contribution=0.20,
                landmark_indices=[LANDMARKS.LEFT_EYE_EAR_POINTS[0], LANDMARKS.LEFT_EYE_EAR_POINTS[3]],
                value=round(ear, 4),
                description="Horizontal distance between eye corners used as denominator normalization",
            ),
        ]

        if ear < 0.25:
            state = "closed"
            explanation = (
                f"EAR is {ear:.3f} (below 0.25 threshold), indicating the eye is closed or nearly closed. "
                f"6 eyelid landmarks (3 upper, 3 lower per eye) were used to compute vertical vs. horizontal extent. "
                f"Blinks: {eyes.blink_count} | Rate: {eyes.blinks_per_minute:.1f}/min"
            )
        elif ear < 0.32:
            state = "narrowed"
            explanation = (
                f"EAR is {ear:.3f} — eyes are partially open (narrowed). "
                f"This may indicate drowsiness or squinting. "
                f"The ratio of vertical eyelid separation to horizontal eye width is reduced."
            )
        else:
            state = "open"
            explanation = (
                f"EAR is {ear:.3f} — eyes are open and alert. "
                f"The landmark geometry shows normal vertical eyelid separation relative to horizontal eye width. "
                f"Blink count: {eyes.blink_count} | Gaze: {eyes.gaze_direction.value}"
            )

        return ExplanationResult(
            metric_name="Eye Aspect Ratio (EAR)",
            final_value=round(ear, 4),
            confidence=round(confidence, 3),
            attributions=attributions,
            processing_time_ms=round(base_time_ms, 2),
            landmark_quality=round(quality_from_ear(ear), 3),
            explanation_text=explanation,
        )

    def _explain_head_pose(
        self,
        head_pose: HeadPoseResult,
        base_time_ms: float,
    ) -> ExplanationResult:
        """Explain head pose estimation via solvePnP."""
        max_angle = max(abs(head_pose.pitch), abs(head_pose.yaw), abs(head_pose.roll))
        confidence = max(1.0 - max_angle / 90.0, 0.0)

        attributions = [
            FeatureAttribution(
                feature_name="nose_tip_position",
                contribution=0.35,
                landmark_indices=[LANDMARKS.NOSE_TIP],
                value=round(head_pose.yaw, 2),
                description="Nose tip (landmark 1) is the primary anchor for 3D pose via solvePnP",
            ),
            FeatureAttribution(
                feature_name="chin_position",
                contribution=0.20,
                landmark_indices=[LANDMARKS.CHIN_TIP],
                value=round(head_pose.pitch, 2),
                description="Chin (landmark 152) defines vertical orientation (pitch)",
            ),
            FeatureAttribution(
                feature_name="eye_corner_separation",
                contribution=0.25,
                landmark_indices=[33, 263],
                value=round(head_pose.roll, 2),
                description="Left/right eye corners define lateral tilt (roll)",
            ),
            FeatureAttribution(
                feature_name="mouth_corners",
                contribution=0.20,
                landmark_indices=[61, 291],
                value=round(head_pose.yaw, 2),
                description="Mouth corners anchor the lower face for full 6-DOF estimation",
            ),
        ]

        facing = "forward" if head_pose.is_facing_forward else "away"
        explanation = (
            f"Head pose estimated using OpenCV solvePnP with 6 canonical 3D face model points. "
            f"Yaw: {head_pose.yaw:.1f}° | Pitch: {head_pose.pitch:.1f}° | Roll: {head_pose.roll:.1f}°. "
            f"Subject is facing {facing}. "
            f"Pose accuracy decreases at angles >45°."
        )

        return ExplanationResult(
            metric_name="Head Pose (Pitch/Yaw/Roll)",
            final_value=round(head_pose.yaw, 2),
            confidence=round(confidence, 3),
            attributions=attributions,
            processing_time_ms=round(base_time_ms, 2),
            landmark_quality=round(confidence, 3),
            explanation_text=explanation,
        )

    def _explain_mouth(
        self,
        mouth: MouthAnalysisResult,
        base_time_ms: float,
    ) -> ExplanationResult:
        """Explain MAR, yawn detection, and smile score."""
        mar_confidence = min(mouth.mar / 0.8 + 0.5, 1.0) if mouth.yawn_detected else 0.8

        attributions = [
            FeatureAttribution(
                feature_name="upper_lower_lip_separation",
                contribution=0.50,
                landmark_indices=list(LANDMARKS.MOUTH_MAR_POINTS[:3]),
                value=round(mouth.mar, 4),
                description="Vertical distance between upper and lower inner lip landmarks",
            ),
            FeatureAttribution(
                feature_name="mouth_horizontal_width",
                contribution=0.25,
                landmark_indices=[LANDMARKS.MOUTH_LEFT, LANDMARKS.MOUTH_RIGHT],
                value=round(mouth.mar, 4),
                description="Horizontal distance between mouth corners (denominator normalization)",
            ),
            FeatureAttribution(
                feature_name="smile_corner_elevation",
                contribution=0.25,
                landmark_indices=[LANDMARKS.SMILE_LEFT, LANDMARKS.SMILE_RIGHT],
                value=round(mouth.smile_intensity, 4),
                description="Lip corner height relative to face center — indicates smile",
            ),
        ]

        if mouth.yawn_detected:
            explanation = (
                f"Yawn detected with {mouth.yawn_confidence:.0%} confidence. "
                f"MAR is {mouth.mar:.3f} (threshold: 0.60). "
                f"6 lip landmarks show sustained large vertical mouth opening. "
                f"Smile suppressed during yawning: {mouth.smile_intensity:.0%}."
            )
        else:
            explanation = (
                f"MAR is {mouth.mar:.3f} — mouth is {'open' if mouth.is_open else 'closed'}. "
                f"No yawning detected. "
                f"Smile intensity: {mouth.smile_intensity:.0%} (based on lip corner width/height ratio)."
            )

        return ExplanationResult(
            metric_name="Mouth Aspect Ratio (MAR)",
            final_value=round(mouth.mar, 4),
            confidence=round(mar_confidence, 3),
            attributions=attributions,
            processing_time_ms=round(base_time_ms, 2),
            landmark_quality=0.9,
            explanation_text=explanation,
        )

    def _explain_behavior(
        self,
        behavior: BehaviorResult,
        quality: QualityResult,
        base_time_ms: float,
    ) -> ExplanationResult:
        """Explain behavioral tracking and attention state."""
        attributions = [
            FeatureAttribution(
                feature_name="landmark_stability",
                contribution=0.30,
                landmark_indices=[1, 152, 33, 263],  # Key reference landmarks
                value=round(behavior.landmark_stability, 4),
                description="Variance of landmark positions over last 15 frames",
            ),
            FeatureAttribution(
                feature_name="facial_symmetry",
                contribution=0.25,
                landmark_indices=list(sum(LANDMARKS.SYMMETRY_PAIRS, ())),
                value=round(behavior.facial_symmetry, 4),
                description="Similarity between mirrored left/right landmark positions",
            ),
            FeatureAttribution(
                feature_name="head_movement_velocity",
                contribution=0.25,
                landmark_indices=[LANDMARKS.NOSE_TIP],
                value=round(behavior.head_movement_velocity, 4),
                description="Nose tip displacement velocity over rolling window",
            ),
            FeatureAttribution(
                feature_name="facial_movement_score",
                contribution=0.20,
                landmark_indices=[],
                value=round(behavior.facial_movement_score, 4),
                description="Mean landmark displacement velocity across all 478 landmarks",
            ),
        ]

        explanation = (
            f"Attention state: '{behavior.attention_state.value}'. "
            f"Landmark stability: {behavior.landmark_stability:.0%} "
            f"(computed from {15} frame rolling window). "
            f"Facial symmetry: {behavior.facial_symmetry:.0%}. "
            f"Head velocity: {behavior.head_movement_velocity:.4f} units/s. "
            f"Frame quality: {quality.overall_score:.0%}."
        )

        return ExplanationResult(
            metric_name="Behavioral Analysis",
            final_value=round(behavior.landmark_stability, 4),
            confidence=round(behavior.landmark_stability * quality.overall_score, 3),
            attributions=attributions,
            processing_time_ms=round(base_time_ms, 2),
            landmark_quality=round(quality.landmark_confidence, 3),
            explanation_text=explanation,
        )


# ── Utility functions ──────────────────────────────────────────────────────────

def ears_to_confidence(ear: float) -> float:
    """Map EAR value to detection confidence."""
    if ear < 0.2:
        return 0.95   # Very confident it's closed
    elif ear > 0.35:
        return 0.90   # Very confident it's open
    else:
        return 0.7    # Ambiguous range


def quality_from_ear(ear: float) -> float:
    """Estimate landmark quality from EAR — very high/low EAR = possible occlusion."""
    if 0.15 <= ear <= 0.5:
        return 0.95
    elif ear < 0.1:
        return 0.6   # May be occluded
    else:
        return 0.85
