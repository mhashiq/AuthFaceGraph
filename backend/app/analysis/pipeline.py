"""
AuthBrain AI Face Analysis Engine
Analysis Pipeline Orchestrator

Coordinates all analyzers and produces a unified FaceAnalysisResult per frame.
Designed for execution in a thread pool (non-blocking from FastAPI event loop).

Performance target: < 30ms per frame on CPU (33ms = 30 FPS budget)
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from app.analysis.behavior_tracker import BehaviorTracker
from app.analysis.eye_analyzer import EyeAnalyzer
from app.analysis.face_detector import FaceDetector, RawFaceData
from app.analysis.head_pose import HeadPoseEstimator
from app.analysis.mouth_analyzer import MouthAnalyzer
from app.analysis.quality_scorer import QualityScorer
from app.core.config import get_settings
from app.core.logging import get_logger
from app.expert_system.explainer import XAIExplainer
from app.expert_system.scorer import ExpertSystemScorer
from app.analysis.identity_verifier import IdentityVerifier
from app.models.schemas import (
    FaceAnalysisResult,
    FaceBoundingBox,
    QualityResult,
    IdentityVerificationResult,
)
from app.utils.frame_utils import (
    bgr_to_rgb,
    draw_bounding_box,
    draw_face_mesh_overlay,
    frame_to_jpeg_bytes,
    jpeg_bytes_to_frame,
    resize_for_inference,
)

settings = get_settings()
logger = get_logger(__name__)

# MediaPipe face mesh connection pairs for visualization
# Subset of FACEMESH_TESSELATION for clean overlay
_MESH_CONNECTIONS: list[tuple[int, int]] = [
    # Face oval
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251),
    (251, 389), (389, 356), (356, 454), (454, 323), (323, 361),
    (361, 288), (288, 397), (397, 365), (365, 379), (379, 378),
    (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172), (172, 58),
    (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
    # Eyes
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (133, 173), (173, 157), (157, 158),
    (158, 159), (159, 160), (160, 161), (161, 246), (246, 33),
    (362, 382), (382, 381), (381, 380), (380, 374), (374, 373),
    (373, 390), (390, 249), (249, 263), (263, 466), (466, 388),
    (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362),
    # Nose
    (6, 197), (197, 195), (195, 5),
    (129, 49), (49, 131), (131, 134), (134, 51), (51, 5),
    (281, 363), (363, 360), (360, 279),
    # Lips
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
    (267, 269), (269, 270), (270, 409), (409, 291),
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291),
]


class FaceAnalysisPipeline:
    """
    Main orchestrator for the entire face analysis pipeline.

    Manages analyzer lifecycle, processes frames, and produces
    complete FaceAnalysisResult objects ready for WebSocket transmission.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

        # Initialize all analysis components
        self._detector = FaceDetector()
        self._head_pose = HeadPoseEstimator()
        self._eye_analyzer = EyeAnalyzer()
        self._mouth_analyzer = MouthAnalyzer()
        self._behavior_tracker = BehaviorTracker()
        self._quality_scorer = QualityScorer()
        self._expert_scorer = ExpertSystemScorer()
        self._xai_explainer = XAIExplainer()
        self._identity_verifier = IdentityVerifier()
        self._dl_engine = None

        # Performance tracking
        self._frame_times: list[float] = []
        self._frame_count: int = 0
        self._session_start: float = time.time()

        # Running metric logs for session-end database summary
        self._ears: list[float] = []
        self._fatigues: list[float] = []
        self._focuses: list[float] = []
        self._qualities: list[float] = []
        self._yaws: list[float] = []
        self._pitches: list[float] = []
        self._dominant_attentions: list[str] = []

    def load(self) -> None:
        """Load all ML models. Must be called before process_frame()."""
        self._detector.load_model()
        logger.info("pipeline_loaded", session_id=self.session_id)

    def process_frame(
        self,
        frame_bytes: bytes,
        active_face_index: int = 0,
        draw_overlay: bool = True,
    ) -> tuple[FaceAnalysisResult, bytes]:
        """
        Process a single JPEG frame and return analysis results + annotated frame.

        Args:
            frame_bytes: Raw JPEG bytes from WebSocket
            active_face_index: Index of the face to analyze (0 = first/largest)
            draw_overlay: If True, draw face mesh on the returned frame

        Returns:
            (FaceAnalysisResult, annotated_frame_bytes)
        """
        t_start = time.time()

        # ── 1. Decode frame ────────────────────────────────────────────────────
        frame_bgr = jpeg_bytes_to_frame(frame_bytes)
        if frame_bgr is None:
            return self._no_face_result(), frame_bytes

        frame_bgr, scale = resize_for_inference(frame_bgr)
        frame_rgb = bgr_to_rgb(frame_bgr)
        h, w = frame_bgr.shape[:2]

        # ── 2. Face Detection ──────────────────────────────────────────────────
        t_detect = time.time()
        try:
            all_faces: list[RawFaceData] = self._detector.detect(
                frame_rgb, w, h, active_face_index
            )
        except Exception as exc:
            logger.error("detection_failed", error=str(exc))
            return self._no_face_result(), frame_bytes

        self._frame_count += 1
        face_count = len(all_faces)

        if face_count == 0:
            # Draw "searching" indicator and return
            annotated = frame_bgr.copy()
            annotated_bytes = frame_to_jpeg_bytes(annotated)
            return self._no_face_result(frame_w=w, frame_h=h), annotated_bytes

        # ── 3. Select active face ──────────────────────────────────────────────
        active_idx = min(active_face_index, face_count - 1)
        face = all_faces[active_idx]

        # ── 4. Real-Time Identity Verification Layer ─────────────────────────
        lm_dicts = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in face.landmarks]
        id_eval = self._identity_verifier.verify_frame(lm_dicts, frame_bgr)
        id_result = IdentityVerificationResult(
            status=id_eval["status"],
            enrolled_user_name=id_eval["enrolled_user_name"],
            match_confidence=id_eval["match_confidence"],
            liveness_score=id_eval["liveness_score"],
            is_live=id_eval["is_live"],
            is_enrolled=id_eval["is_enrolled"],
            is_paused=id_eval["is_paused"],
        )

        # ── 5. Run all analyzers in sequence ───────────────────────────────────
        head_pose = self._head_pose.estimate(face)
        eyes = self._eye_analyzer.analyze(face)
        mouth = self._mouth_analyzer.analyze(face)
        quality = self._quality_scorer.score(face, frame_bgr)
        behavior = self._behavior_tracker.track(
            face_data=face,
            eye_ear=eyes.average_ear,
            head_yaw=head_pose.yaw,
            head_pitch=head_pose.pitch,
            yawn_detected=mouth.yawn_detected,
            blinks_per_minute=eyes.blinks_per_minute,
            eye_closure_duration_ms=eyes.eye_closure_duration_ms,
        )

        # ── 6. Expert system ───────────────────────────────────────────────────
        expert_result = self._expert_scorer.score(
            eyes=eyes,
            mouth=mouth,
            head_pose=head_pose,
            behavior=behavior,
            quality=quality,
        )

        # If identity verification is paused (mismatch or liveness failure), pause tracking!
        if id_eval["is_paused"]:
            behavior.attention_state = "paused"  # type: ignore[assignment]
            expert_result.attention_state = "paused"  # type: ignore[assignment]

        # ── 7. XAI Explanations ────────────────────────────────────────────────
        explanations = self._xai_explainer.explain(
            eyes=eyes,
            mouth=mouth,
            head_pose=head_pose,
            behavior=behavior,
            quality=quality,
            inference_time_ms=(time.time() - t_detect) * 1000.0,
        )
        expert_result.explanations = explanations

        # ── 8. Deep Learning (optional) ───────────────────────────────────────
        dl_result = None
        if settings.DL_ENABLED and not id_eval["is_paused"]:
            if self._dl_engine is None:
                try:
                    from app.dl.engine import DLEngine
                    self._dl_engine = DLEngine(self.session_id)
                    self._dl_engine.load()
                except Exception as exc:
                    logger.error("dl_engine_load_failed", error=str(exc))

            if self._dl_engine is not None:
                try:
                    elapsed_ms = (time.time() - self._session_start) * 1000.0
                    dl_res_dict = self._dl_engine.process(
                        face_data=face,
                        frame_bgr=frame_bgr,
                        frame_width=w,
                        frame_height=h,
                        timestamp_ms=elapsed_ms,
                        quality_score=quality.overall_score,
                    )
                    if dl_res_dict:
                        from app.models.schemas import DLAnalysisResult
                        dl_result = DLAnalysisResult(**dl_res_dict)
                except Exception as exc:
                    logger.error("dl_inference_failed", error=str(exc))

        # Log metrics for session-end database summary (only when not paused)
        if not id_eval["is_paused"]:
            self._ears.append(eyes.average_ear)
            self._fatigues.append(expert_result.fatigue_score)
            self._focuses.append(expert_result.focus_score)
            self._qualities.append(quality.overall_score)
            self._yaws.append(head_pose.yaw)
            self._pitches.append(head_pose.pitch)
            self._dominant_attentions.append(behavior.attention_state.value if hasattr(behavior.attention_state, "value") else str(behavior.attention_state))

        # ── 9. Draw overlay ────────────────────────────────────────────────────
        if draw_overlay:
            landmark_xy = [(lm.x, lm.y) for lm in face.landmarks]
            annotated = draw_face_mesh_overlay(
                frame_bgr,
                landmark_xy,
                connections=_MESH_CONNECTIONS,
                landmark_color=(0, 255, 120),
                connection_color=(0, 180, 80),
                landmark_radius=1,
                connection_thickness=1,
            )
            box_color = (0, 255, 120) if not id_eval["is_paused"] else (0, 0, 255)
            box_label = f"{id_eval['enrolled_user_name']} | {id_eval['match_confidence']:.1%}" if not id_eval["is_paused"] else "MISMATCH / PAUSED"
            annotated = draw_bounding_box(
                annotated,
                face.bounding_box.x,
                face.bounding_box.y,
                face.bounding_box.width,
                face.bounding_box.height,
                label=box_label,
                color=box_color,
            )
        else:
            annotated = frame_bgr

        annotated_bytes = frame_to_jpeg_bytes(annotated)

        # ── 10. Compute FPS ────────────────────────────────────────────────────
        t_total = (time.time() - t_start) * 1000.0
        self._frame_times.append(t_total)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        fps = 1000.0 / max(float(np.mean(self._frame_times)), 1.0)

        result = FaceAnalysisResult(
            frame_id=str(uuid.uuid4()),
            session_id=self.session_id,
            timestamp=datetime.now(timezone.utc),
            inference_time_ms=round(t_total, 2),
            face_detected=True,
            face_count=face_count,
            active_face_index=active_idx,
            bounding_box=face.bounding_box,
            landmark_count=len(face.landmarks),
            landmarks=face.landmarks,
            head_pose=head_pose,
            eyes=eyes,
            mouth=mouth,
            behavior=behavior,
            quality=quality,
            expert_system=expert_result,
            identity_verification=id_result,
            deep_learning=dl_result,
            fps=round(fps, 1),
            frame_width=w,
            frame_height=h,
            model_confidence=quality.landmark_confidence,
        )

        return result, annotated_bytes

    def _no_face_result(
        self,
        frame_w: int = 0,
        frame_h: int = 0,
    ) -> FaceAnalysisResult:
        """Return a minimal result when no face is detected."""
        return FaceAnalysisResult(
            frame_id=str(uuid.uuid4()),
            session_id=self.session_id,
            timestamp=datetime.now(timezone.utc),
            inference_time_ms=0.0,
            face_detected=False,
            face_count=0,
            identity_verification=IdentityVerificationResult(
                status="no_face",
                enrolled_user_name=self._identity_verifier.enrolled_user_name,
                match_confidence=0.0,
                liveness_score=0.0,
                is_live=False,
                is_enrolled=self._identity_verifier.enrolled_embedding is not None,
                is_paused=True,
            ),
            fps=0.0,
            frame_width=frame_w,
            frame_height=frame_h,
        )

    def get_session_stats(self) -> dict:
        """Return aggregate session performance statistics."""
        from collections import Counter
        import numpy as np

        dom_attn = "unknown"
        if self._dominant_attentions:
            dom_attn = Counter(self._dominant_attentions).most_common(1)[0][0]

        return {
            "session_id": self.session_id,
            "total_frames": self._frame_count,
            "uptime_seconds": round(time.time() - self._session_start, 1),
            "avg_inference_ms": round(float(np.mean(self._frame_times)), 2) if self._frame_times else 0.0,
            "total_blinks": self._eye_analyzer._blink_count,
            "avg_ear": round(float(np.mean(self._ears)), 4) if self._ears else 0.0,
            "max_fatigue": round(float(np.max(self._fatigues)), 4) if self._fatigues else 0.0,
            "avg_fatigue": round(float(np.mean(self._fatigues)), 4) if self._fatigues else 0.0,
            "avg_focus": round(float(np.mean(self._focuses)), 4) if self._focuses else 0.0,
            "avg_quality": round(float(np.mean(self._qualities)), 4) if self._qualities else 0.0,
            "avg_yaw": round(float(np.mean(self._yaws)), 2) if self._yaws else 0.0,
            "avg_pitch": round(float(np.mean(self._pitches)), 2) if self._pitches else 0.0,
            "dominant_attention": dom_attn,
        }

    def reset_analyzers(self) -> None:
        """Reset all analyzers (call when user reconnects)."""
        self._head_pose.reset()
        self._eye_analyzer.reset()
        self._mouth_analyzer.reset()
        self._behavior_tracker.reset()

    def close(self) -> None:
        """Release all resources."""
        self._detector.close()
        logger.info("pipeline_closed", session_id=self.session_id)
