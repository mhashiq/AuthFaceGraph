"""
AuthBrain AI Face Analysis Engine
Pydantic Schemas — Request/Response Models

All API data contracts are defined here.
These schemas are used for both REST and WebSocket communication.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class AttentionState(str, Enum):
    """Expert system attention state classifications."""
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    DROWSY = "drowsy"
    ALERT = "alert"
    UNKNOWN = "unknown"


class GazeDirection(str, Enum):
    """Estimated gaze direction."""
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    CLOSED = "closed"


class SessionStatus(str, Enum):
    """Analysis session lifecycle states."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


# ══════════════════════════════════════════════════════════════════════════════
# Landmark Schemas
# ══════════════════════════════════════════════════════════════════════════════

class Landmark(BaseModel):
    """3D facial landmark coordinate (normalized 0–1)."""
    x: float
    y: float
    z: float = 0.0


class FaceBoundingBox(BaseModel):
    """Axis-aligned bounding box for detected face."""
    x: float      # Top-left X (normalized)
    y: float      # Top-left Y (normalized)
    width: float  # Width (normalized)
    height: float # Height (normalized)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis Result Schemas
# ══════════════════════════════════════════════════════════════════════════════

class HeadPoseResult(BaseModel):
    """6-DOF head pose estimated via solvePnP."""
    pitch: float = Field(..., description="Head pitch angle in degrees (nodding up/down)")
    yaw: float   = Field(..., description="Head yaw angle in degrees (turning left/right)")
    roll: float  = Field(..., description="Head roll angle in degrees (tilting)")
    is_facing_forward: bool = Field(True, description="True if head pose is roughly forward")


class EyeResult(BaseModel):
    """Eye analysis metrics for one eye."""
    ear: float = Field(..., description="Eye Aspect Ratio — higher = more open")
    is_open: bool
    gaze_x: float = Field(0.0, description="Horizontal gaze offset from iris center")
    gaze_y: float = Field(0.0, description="Vertical gaze offset from iris center")


class EyeAnalysisResult(BaseModel):
    """Combined left + right eye analysis."""
    left: EyeResult
    right: EyeResult
    average_ear: float = Field(..., description="Mean EAR across both eyes")
    blink_detected: bool = Field(False, description="True if a blink occurred this frame")
    blink_count: int = Field(0, description="Total blink count for this session")
    eye_closure_duration_ms: float = Field(0.0, description="Duration of current/last closure in ms")
    gaze_direction: GazeDirection = GazeDirection.CENTER
    blinks_per_minute: float = Field(0.0, description="Rolling blink frequency")


class MouthAnalysisResult(BaseModel):
    """Mouth and facial expression metrics."""
    mar: float = Field(..., description="Mouth Aspect Ratio")
    is_open: bool
    yawn_detected: bool = False
    yawn_confidence: float = Field(0.0, ge=0.0, le=1.0)
    smile_intensity: float = Field(0.0, ge=0.0, le=1.0, description="Normalized smile score")
    mouth_openness_percent: float = Field(0.0, ge=0.0, le=100.0)


class BehaviorResult(BaseModel):
    """Temporal behavioral tracking metrics."""
    head_movement_velocity: float = Field(0.0, description="Pixels/sec head movement")
    facial_movement_score: float = Field(0.0, ge=0.0, le=1.0)
    landmark_stability: float = Field(1.0, ge=0.0, le=1.0, description="1.0 = perfectly stable")
    facial_symmetry: float = Field(1.0, ge=0.0, le=1.0, description="1.0 = perfectly symmetric")
    attention_state: AttentionState = AttentionState.UNKNOWN


class QualityResult(BaseModel):
    """Frame and face quality metrics."""
    overall_score: float = Field(..., ge=0.0, le=1.0)
    sharpness: float = Field(..., ge=0.0, le=1.0, description="Laplacian variance sharpness")
    illumination: float = Field(..., ge=0.0, le=1.0)
    face_size_ratio: float = Field(..., ge=0.0, le=1.0, description="Face bounding box area ratio")
    landmark_confidence: float = Field(..., ge=0.0, le=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Expert System / XAI Schemas
# ══════════════════════════════════════════════════════════════════════════════

class FeatureAttribution(BaseModel):
    """XAI feature attribution for a single metric."""
    feature_name: str
    contribution: float = Field(..., ge=0.0, le=1.0, description="Normalized contribution weight")
    landmark_indices: list[int] = Field(default_factory=list, description="Contributing landmark IDs")
    value: float = 0.0
    description: str


class ExplanationResult(BaseModel):
    """Explainable AI result for one analysis frame."""
    metric_name: str
    final_value: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    attributions: list[FeatureAttribution] = Field(default_factory=list)
    processing_time_ms: float
    landmark_quality: float = Field(..., ge=0.0, le=1.0)
    explanation_text: str


class ExpertSystemResult(BaseModel):
    """Composite expert system output."""
    attention_state: AttentionState
    fatigue_score: float = Field(..., ge=0.0, le=1.0)
    focus_score: float = Field(..., ge=0.0, le=1.0)
    alerts: list[str] = Field(default_factory=list)
    explanations: list[ExplanationResult] = Field(default_factory=list)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Deep Learning Schemas
# ══════════════════════════════════════════════════════════════════════════════

class EmotionLabel(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISE = "surprise"
    FEAR = "fear"
    DISGUST = "disgust"
    ANGER = "anger"
    CONTEMPT = "contempt"
    UNKNOWN = "unknown"


class DLEmotionPrediction(BaseModel):
    """Output from a single image-based emotion recognition model."""
    model_config = ConfigDict(protected_namespaces=())

    emotion: EmotionLabel
    confidence: float
    probabilities: dict[str, float]
    raw_confidence: float = 0.0
    calibrated_confidence: float = 0.0
    model_id: str
    latency_ms: float = 0.0
    error: str | None = None
    status: str = "healthy"


class ActionUnit(BaseModel):
    """Facial Action Unit presence and intensity."""
    au_id: str
    name: str
    present: bool
    intensity: float  # 0.0 to 5.0 intensity units


class GNNPrediction(BaseModel):
    """Output from GNN emotion model."""
    model_config = ConfigDict(protected_namespaces=())

    emotion: EmotionLabel
    confidence: float
    probabilities: dict[str, float]
    raw_confidence: float = 0.0
    calibrated_confidence: float = 0.0
    model_id: str
    node_importance: list[float] = Field(default_factory=list)
    edge_attention: list[float] = Field(default_factory=list)
    edge_index: list[list[int]] = Field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None
    status: str = "healthy"


class EnsembleResult(BaseModel):
    """Multi-model ensemble combined verdict."""
    model_config = ConfigDict(protected_namespaces=())

    final_emotion: EmotionLabel
    confidence: float
    probabilities: dict[str, float]
    model_predictions: list[DLEmotionPrediction] = Field(default_factory=list)
    disagreement_score: float = 0.0
    uncertainty: float = 0.0
    agreement_score: float = 0.0
    raw_confidence: float = 0.0
    calibrated_confidence: float = 0.0


class DLAnalysisResult(BaseModel):
    """Aggregate result from all Deep Learning models."""
    model_config = ConfigDict(protected_namespaces=())

    dl_enabled: bool = False
    dl_inference_time_ms: float = 0.0
    emotion_ensemble: EnsembleResult | None = None
    gnn_prediction: GNNPrediction | None = None
    action_units: list[ActionUnit] = Field(default_factory=list)
    top_important_landmarks: list[int] = Field(default_factory=list)
    models_used: list[str] = Field(default_factory=list)
    xai_explanations: list[ExplanationResult] = Field(default_factory=list)
    landmarks: list[Landmark] = Field(default_factory=list)
    model_health: dict[str, Any] = Field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Main Analysis Result — Sent over WebSocket per frame
# ══════════════════════════════════════════════════════════════════════════════

class FaceAnalysisResult(BaseModel):
    """
    Complete per-frame face analysis result transmitted over WebSocket.
    This is the primary data contract between backend and frontend.
    """
    model_config = ConfigDict(protected_namespaces=())

    # Metadata
    frame_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    inference_time_ms: float

    # Detection
    face_detected: bool
    face_count: int = 0
    active_face_index: int = 0
    bounding_box: FaceBoundingBox | None = None
    landmark_count: int = 0
    landmarks: list[Landmark] = Field(default_factory=list)

    # Analysis results (None if no face detected)
    head_pose: HeadPoseResult | None = None
    eyes: EyeAnalysisResult | None = None
    mouth: MouthAnalysisResult | None = None
    behavior: BehaviorResult | None = None
    quality: QualityResult | None = None
    expert_system: ExpertSystemResult | None = None

    # Deep learning extensions (None if DL disabled or frame analysis failed)
    deep_learning: DLAnalysisResult | None = None

    # Performance metrics
    fps: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    model_confidence: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket Message Schemas
# ══════════════════════════════════════════════════════════════════════════════

class WSMessageType(str, Enum):
    ANALYSIS_RESULT = "analysis_result"
    ERROR = "error"
    STATUS = "status"
    CONFIG = "config"


class WSMessage(BaseModel):
    """Wrapper for all WebSocket JSON messages."""
    type: WSMessageType
    payload: dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WSErrorMessage(BaseModel):
    """WebSocket error payload."""
    code: str
    message: str
    detail: str | None = None


# ══════════════════════════════════════════════════════════════════════════════
# REST API Schemas
# ══════════════════════════════════════════════════════════════════════════════

class ConsentRequest(BaseModel):
    """User consent request — required before analysis begins."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_agent: str | None = None
    consent_granted: bool = Field(..., description="Must be True to start analysis")
    consent_text_version: str = "1.0"


class ConsentResponse(BaseModel):
    """Consent acknowledgment with analysis token."""
    session_id: str
    consent_granted: bool
    analysis_token: str
    expires_at: datetime
    message: str


class HealthResponse(BaseModel):
    """System health check response."""
    model_config = ConfigDict(protected_namespaces=())

    status: str
    app_name: str
    version: str
    environment: str
    database_connected: bool
    model_loaded: bool
    gpu_available: bool
    uptime_seconds: float
    active_sessions: int


class SessionSummary(BaseModel):
    """Persisted session summary in the database."""
    session_id: str
    started_at: datetime
    ended_at: datetime | None = None
    status: SessionStatus
    total_frames: int
    total_blinks: int
    avg_ear: float
    avg_head_yaw: float
    avg_head_pitch: float
    dominant_attention_state: str
    face_quality_score: float
