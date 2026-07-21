"""
Unit tests for BiometricEnrollmentEngine State Machine Rules.
"""

import pytest
import numpy as np
from app.analysis.biometric_enrollment_engine import (
    BiometricEnrollmentEngine,
    BiometricInferenceMetrics,
    PoseDegrees,
    QualityMetrics,
)


@pytest.fixture
def engine():
    return BiometricEnrollmentEngine(liveness_threshold=0.92, sharpness_threshold=100.0, warmup_frames=0)


def test_camera_warmup_and_lighting_debouncing():
    warmup_engine = BiometricEnrollmentEngine(warmup_frames=5)
    dark_metrics = BiometricInferenceMetrics(
        num_faces_detected=1,
        detection_confidence=0.95,
        bounding_box=[100, 100, 300, 300],
        pose_degrees=PoseDegrees(yaw=0.0, pitch=0.0, roll=0.0),
        quality_metrics=QualityMetrics(sharpness_laplacian=150.0, exposure_mean_brightness=20.0, occlusion_score=0.0),
        liveness_score=0.98,
    )
    
    # Frames 1..5 should return CAMERA_WARMUP
    for _ in range(5):
        res = warmup_engine.evaluate_state_machine(dark_metrics)
        assert res.state == "CAMERA_WARMUP"
        assert res.status == "WARMUP"
        assert res.message == "Initializing camera..."

    # Frames 1..4 post-warmup: consecutive dark frames 1..4 should be debounced
    for _ in range(4):
        res = warmup_engine.evaluate_state_machine(dark_metrics)
        assert res.message != "Bad lighting. Move to better light."

    # 5th consecutive dark frame post-warmup triggers Bad lighting guidance
    res = warmup_engine.evaluate_state_machine(dark_metrics)
    assert res.status == "GUIDANCE"
    assert res.message == "Bad lighting. Move to better light."


def test_state_searching_no_face(engine):
    metrics = BiometricInferenceMetrics(
        num_faces_detected=0,
        detection_confidence=0.0,
        bounding_box=[0, 0, 0, 0],
        pose_degrees=PoseDegrees(yaw=0.0, pitch=0.0, roll=0.0),
        quality_metrics=QualityMetrics(sharpness_laplacian=150.0, exposure_mean_brightness=120.0, occlusion_score=0.0),
        liveness_score=0.98,
    )
    res = engine.evaluate_state_machine(metrics)
    assert res.status == "REJECT"
    assert res.state == "SEARCHING"
    assert "No face detected" in res.message


def test_state_searching_multiple_faces(engine):
    metrics = BiometricInferenceMetrics(
        num_faces_detected=2,
        detection_confidence=0.95,
        bounding_box=[10, 10, 200, 200],
        pose_degrees=PoseDegrees(yaw=0.0, pitch=0.0, roll=0.0),
        quality_metrics=QualityMetrics(sharpness_laplacian=150.0, exposure_mean_brightness=120.0, occlusion_score=0.0),
        liveness_score=0.98,
    )
    res = engine.evaluate_state_machine(metrics)
    assert res.status == "REJECT"
    assert res.state == "SEARCHING"
    assert "Multiple faces detected" in res.message


def test_state_pose_check_turn_right(engine):
    metrics = BiometricInferenceMetrics(
        num_faces_detected=1,
        detection_confidence=0.95,
        bounding_box=[100, 100, 300, 300],
        pose_degrees=PoseDegrees(yaw=-14.5, pitch=0.0, roll=0.0),
        quality_metrics=QualityMetrics(sharpness_laplacian=150.0, exposure_mean_brightness=120.0, occlusion_score=0.0),
        liveness_score=0.98,
    )
    res = engine.evaluate_state_machine(metrics)
    assert res.status == "GUIDANCE"
    assert res.state == "QUALITY_AND_POSE_CHECK"
    assert res.message == "Turn head slightly right."


def test_state_pose_check_turn_left(engine):
    metrics = BiometricInferenceMetrics(
        num_faces_detected=1,
        detection_confidence=0.95,
        bounding_box=[100, 100, 300, 300],
        pose_degrees=PoseDegrees(yaw=15.2, pitch=0.0, roll=0.0),
        quality_metrics=QualityMetrics(sharpness_laplacian=150.0, exposure_mean_brightness=120.0, occlusion_score=0.0),
        liveness_score=0.98,
    )
    res = engine.evaluate_state_machine(metrics)
    assert res.status == "GUIDANCE"
    assert res.message == "Turn head slightly left."


def test_state_pose_check_blurry_frame(engine):
    metrics = BiometricInferenceMetrics(
        num_faces_detected=1,
        detection_confidence=0.95,
        bounding_box=[100, 100, 300, 300],
        pose_degrees=PoseDegrees(yaw=0.0, pitch=0.0, roll=0.0),
        quality_metrics=QualityMetrics(sharpness_laplacian=45.0, exposure_mean_brightness=120.0, occlusion_score=0.0),
        liveness_score=0.98,
    )
    res = engine.evaluate_state_machine(metrics)
    assert res.status == "GUIDANCE"
    assert "blurry" in res.message.lower()


def test_state_liveness_check_spoof(engine):
    metrics = BiometricInferenceMetrics(
        num_faces_detected=1,
        detection_confidence=0.95,
        bounding_box=[100, 100, 300, 300],
        pose_degrees=PoseDegrees(yaw=0.0, pitch=0.0, roll=0.0),
        quality_metrics=QualityMetrics(sharpness_laplacian=150.0, exposure_mean_brightness=120.0, occlusion_score=0.0),
        liveness_score=0.45,
    )
    res = engine.evaluate_state_machine(metrics)
    assert res.status == "REJECT"
    assert res.state == "LIVENESS_CHECK"
    assert "Spoof detected" in res.message


def test_state_stability_lock_success(engine):
    metrics = BiometricInferenceMetrics(
        num_faces_detected=1,
        detection_confidence=0.96,
        bounding_box=[100, 100, 300, 300],
        pose_degrees=PoseDegrees(yaw=1.2, pitch=-2.4, roll=0.5),
        quality_metrics=QualityMetrics(sharpness_laplacian=185.4, exposure_mean_brightness=125.0, occlusion_score=0.02),
        liveness_score=0.98,
    )
    res = engine.evaluate_state_machine(metrics)
    assert res.status == "STABILITY_LOCK"
    assert res.state == "STABILITY_LOCK"
