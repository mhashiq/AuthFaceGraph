"""
AuthBrain AI Face Analysis Engine
Behavior Tracker

Tracks temporal facial behavior metrics including:
- Head movement velocity
- Facial movement score
- Landmark stability
- Facial symmetry ratio
- Attention state classification

All metrics require at least 2 frames to compute meaningful values.
"""

from __future__ import annotations

import time
from collections import deque, Counter

import numpy as np
from numpy.typing import NDArray

from app.analysis.face_detector import RawFaceData
from app.analysis.landmark_indices import LANDMARKS
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import AttentionState, BehaviorResult
from app.utils.math_utils import (
    compute_facial_symmetry,
    compute_landmark_velocity,
    euclidean_distance,
)

settings = get_settings()
logger = get_logger(__name__)

# Rolling windows for behavior analysis
_STABILITY_WINDOW = 15   # frames
_MOVEMENT_WINDOW  = 10   # frames


class BehaviorTracker:
    """
    Tracks temporal facial behavior across frames.

    Maintains rolling buffers of landmark positions to compute
    velocity, stability, and attention state over time.
    """

    def __init__(self) -> None:
        # Stores recent nose tip positions for head movement tracking
        self._nose_positions: deque[tuple[float, float]] = deque(maxlen=_MOVEMENT_WINDOW)
        self._nose_timestamps: deque[float] = deque(maxlen=_MOVEMENT_WINDOW)

        # Rolling buffer of landmark arrays for stability computation
        self._landmark_history: deque[NDArray[np.float64]] = deque(maxlen=_STABILITY_WINDOW)

        # Rolling history of raw attention classifications to prevent flickering (majority voting)
        self._attention_history: deque[AttentionState] = deque(maxlen=15)

        # Previous frame state
        self._prev_landmarks: NDArray[np.float64] | None = None
        self._prev_timestamp: float = 0.0

    def track(
        self,
        face_data: RawFaceData,
        eye_ear: float,
        head_yaw: float,
        head_pitch: float,
        yawn_detected: bool,
        blinks_per_minute: float,
        eye_closure_duration_ms: float = 0.0,
    ) -> BehaviorResult:
        """
        Compute behavioral metrics for the current frame.

        Args:
            face_data: Current frame face data
            eye_ear: Average EAR from eye analyzer
            head_yaw: Head yaw angle in degrees
            head_pitch: Head pitch angle in degrees
            yawn_detected: Whether yawn was detected this frame
            blinks_per_minute: Rolling blink frequency
            eye_closure_duration_ms: Current sustained eye closure duration in ms

        Returns:
            BehaviorResult with all behavioral metrics
        """
        now = time.time()
        dt_ms = (now - self._prev_timestamp) * 1000.0 if self._prev_timestamp > 0 else 33.0
        self._prev_timestamp = now

        # Convert landmarks to numpy array for batch operations
        current_lm = np.array([
            [lm.x, lm.y] for lm in face_data.landmarks
        ], dtype=np.float64)

        # ── Head Movement Velocity ─────────────────────────────────────────────
        nose_tip = face_data.landmarks[LANDMARKS.NOSE_TIP]
        self._nose_positions.append((nose_tip.x, nose_tip.y))
        self._nose_timestamps.append(now)

        head_movement_velocity = self._compute_head_velocity(dt_ms)

        # ── Facial Movement Score ─────────────────────────────────────────────
        facial_movement_score = 0.0
        if self._prev_landmarks is not None and len(current_lm) == len(self._prev_landmarks):
            velocity = compute_landmark_velocity(current_lm, self._prev_landmarks, dt_ms)
            # Normalize to [0, 1] — typical range 0–0.05 landmarks/sec
            facial_movement_score = float(np.clip(velocity / 0.05, 0.0, 1.0))

        # ── Landmark Stability ─────────────────────────────────────────────────
        self._landmark_history.append(current_lm)
        landmark_stability = self._compute_stability()

        # ── Facial Symmetry ────────────────────────────────────────────────────
        facial_symmetry = self._compute_symmetry(face_data)

        # ── Attention State Classification ────────────────────────────────────
        raw_state = self._classify_attention(
            ear=eye_ear,
            yaw=head_yaw,
            pitch=head_pitch,
            yawn_detected=yawn_detected,
            blinks_per_minute=blinks_per_minute,
            landmark_stability=landmark_stability,
            eye_closure_duration_ms=eye_closure_duration_ms,
        )
        self._attention_history.append(raw_state)

        # Majority vote (mode) to prevent rapid flickering
        attention_state = Counter(self._attention_history).most_common(1)[0][0]

        # Store for next frame
        self._prev_landmarks = current_lm.copy()

        return BehaviorResult(
            head_movement_velocity=round(head_movement_velocity, 4),
            facial_movement_score=round(facial_movement_score, 4),
            landmark_stability=round(landmark_stability, 4),
            facial_symmetry=round(facial_symmetry, 4),
            attention_state=attention_state,
        )

    def _compute_head_velocity(self, dt_ms: float) -> float:
        """Compute nose tip movement velocity from rolling position buffer."""
        if len(self._nose_positions) < 2:
            return 0.0
        positions = list(self._nose_positions)
        # Average displacement over last N positions
        displacements = [
            euclidean_distance(positions[i], positions[i - 1])
            for i in range(1, len(positions))
        ]
        avg_disp = float(np.mean(displacements))
        velocity = avg_disp / max(dt_ms / 1000.0, 0.001)
        return velocity

    def _compute_stability(self) -> float:
        """
        Compute landmark stability as 1 - mean positional variance.
        High stability = consistent face position.
        """
        if len(self._landmark_history) < 3:
            return 1.0

        # Stack history: (N_frames, N_landmarks, 2)
        history_stack = np.array(list(self._landmark_history))
        # Variance per landmark per axis
        variance = float(np.mean(np.var(history_stack, axis=0)))
        # Normalize: typical variance range 0–0.001 for stable face
        stability = float(np.clip(1.0 - variance * 500.0, 0.0, 1.0))
        return stability

    def _compute_symmetry(self, face_data: RawFaceData) -> float:
        """Compute facial symmetry from mirrored landmark pairs."""
        landmarks = face_data.landmarks
        if len(landmarks) < 454:
            return 1.0

        left_pts = []
        right_pts = []

        for left_idx, right_idx in LANDMARKS.SYMMETRY_PAIRS:
            if left_idx < len(landmarks) and right_idx < len(landmarks):
                left_pts.append([landmarks[left_idx].x, landmarks[left_idx].y])
                right_pts.append([landmarks[right_idx].x, landmarks[right_idx].y])

        if not left_pts:
            return 1.0

        left_arr = np.array(left_pts, dtype=np.float64)
        right_arr = np.array(right_pts, dtype=np.float64)

        # Mirror right side (1 - x) to compare against left
        right_arr[:, 0] = 1.0 - right_arr[:, 0]

        face_width = face_data.bounding_box.width
        return compute_facial_symmetry(left_arr, right_arr, max(face_width, 0.01))

    def _classify_attention(
        self,
        ear: float,
        yaw: float,
        pitch: float,
        yawn_detected: bool,
        blinks_per_minute: float,
        landmark_stability: float,
        eye_closure_duration_ms: float = 0.0,
    ) -> AttentionState:
        """
        Rule-based attention state classification.

        Rules (in priority order):
        1. DROWSY: Low EAR (extended eye closure > 1s) or yawning with high blink rate
        2. DISTRACTED: High head rotation (looking away)
        3. FOCUSED: Forward-facing, normal EAR, normal blink rate
        4. ALERT: Forward-facing, above-normal blink rate or jitter
        5. UNKNOWN: Low landmark stability
        """
        if landmark_stability < 0.3:
            return AttentionState.UNKNOWN

        # Drowsiness: extended eye closed (>1.0s) or yawning with elevated blink rate
        is_drowsy_closure = ear < settings.EAR_CLOSURE_THRESHOLD and eye_closure_duration_ms > 1000.0
        if is_drowsy_closure or (yawn_detected and blinks_per_minute > 20):
            return AttentionState.DROWSY

        # Distracted: significant head rotation (post-calibration offsets)
        # Thresholds are generous because calibration already removed baseline posture bias
        if abs(yaw) > 35 or abs(pitch) > 30:
            return AttentionState.DISTRACTED

        # Alert: slight elevation in blink rate, still forward-facing
        if blinks_per_minute > 20 and abs(yaw) < 20:
            return AttentionState.ALERT

        # Default: focused
        return AttentionState.FOCUSED

    def reset(self) -> None:
        """Reset all accumulated state for a new session."""
        self._nose_positions.clear()
        self._nose_timestamps.clear()
        self._landmark_history.clear()
        self._attention_history.clear()
        self._prev_landmarks = None
        self._prev_timestamp = 0.0
