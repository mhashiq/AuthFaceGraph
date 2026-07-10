"""
AuthBrain DL Platform — Facial Action Unit (AU) Geometry Estimator

Estimates Facial Action Units (FACS) directly from the geometry of 478
MediaPipe FaceMesh landmarks.
Provides presence (0 or 1) and intensity (0.0 to 5.0) for key action units.

This implementation acts as an auditable, geometric detector (akin to OpenFace
geometric AUs) which is extremely fast (<1ms CPU) and fits real-time pipeline requirements.
"""
from __future__ import annotations

import numpy as np
from typing import Any

from app.analysis.landmark_indices import LANDMARKS
from app.dl.base import AUEstimatorBase, ActionUnitResult
from app.utils.math_utils import euclidean_distance


class GeometricAUEstimator(AUEstimatorBase):
    """
    Computes key FACS Action Unit presence and intensity by analyzing
    spatial distance relations between specific FaceMesh landmarks.
    """

    def __init__(self) -> None:
        pass

    def estimate(
        self,
        landmarks: list[Any],
        frame_width: int,
        frame_height: int,
    ) -> list[ActionUnitResult]:
        """
        Estimate AUs using facial distance thresholds normalized by face scale.
        """
        if not landmarks or len(landmarks) < 468:
            return []

        # ── 1. Calculate Face Scale Normalization ────────────────────────────────
        # Use pupil distance or face height as a normalization scale
        # Pupil centers: LEFT_IRIS_CENTER (473), RIGHT_IRIS_CENTER (468)
        left_pupil = landmarks[LANDMARKS.LEFT_IRIS_CENTER]
        right_pupil = landmarks[LANDMARKS.RIGHT_IRIS_CENTER]
        pd_norm = euclidean_distance(
            (left_pupil.x * frame_width, left_pupil.y * frame_height),
            (right_pupil.x * frame_width, right_pupil.y * frame_height)
        )
        if pd_norm < 1.0:
            pd_norm = 1.0  # Avoid division by zero

        # Convert landmarks to pixel space for distance calculations
        def dist(i1: int, i2: int) -> float:
            p1 = (landmarks[i1].x * frame_width, landmarks[i1].y * frame_height)
            p2 = (landmarks[i2].x * frame_width, landmarks[i2].y * frame_height)
            return euclidean_distance(p1, p2) / pd_norm

        results: list[ActionUnitResult] = []

        # ── AU1: Inner Brow Raiser ────────────────────────────────────────────────
        # Distance from inner eyebrow to nose bridge
        # Eyebrows: 70 (right inner), 336 (left inner). Nose bridge: 6
        au1_dist = (dist(70, 6) + dist(336, 6)) / 2.0
        # Baseline inner brow to nose bridge ratio is roughly 0.65.
        au1_intensity = np.clip((au1_dist - 0.62) * 15.0, 0.0, 5.0)
        results.append(ActionUnitResult(
            au_id="AU1",
            name="Inner Brow Raiser",
            present=bool(au1_intensity > 1.5),
            intensity=round(float(au1_intensity), 2)
        ))

        # ── AU2: Outer Brow Raiser ────────────────────────────────────────────────
        # Distance from outer eyebrows (107, 337) to outer eyes (130, 359)
        # Using general outer eyebrow to outer eye corner landmarks
        au2_dist = (dist(107, 33) + dist(336, 263)) / 2.0
        au2_intensity = np.clip((au2_dist - 0.70) * 12.0, 0.0, 5.0)
        results.append(ActionUnitResult(
            au_id="AU2",
            name="Outer Brow Raiser",
            present=bool(au2_intensity > 1.5),
            intensity=round(float(au2_intensity), 2)
        ))

        # ── AU4: Brow Lowerer (Frown / Anger) ─────────────────────────────────────
        # Distance between the two inner eyebrows (70 to 336)
        # Decreased distance = brow lowerer
        au4_dist = dist(70, 336)
        au4_intensity = np.clip((0.40 - au4_dist) * 20.0, 0.0, 5.0)
        results.append(ActionUnitResult(
            au_id="AU4",
            name="Brow Lowerer",
            present=bool(au4_intensity > 1.5),
            intensity=round(float(au4_intensity), 2)
        ))

        # ── AU12: Lip Corner Puller (Smile) ───────────────────────────────────────
        # Distance of mouth corners from lip center, normalized
        # Left corner 61, right corner 291, nose bottom 2
        au12_dist = (dist(61, 2) + dist(291, 2)) / 2.0
        au12_intensity = np.clip((au12_dist - 0.85) * 18.0, 0.0, 5.0)
        results.append(ActionUnitResult(
            au_id="AU12",
            name="Lip Corner Puller (Smile)",
            present=bool(au12_intensity > 1.0),
            intensity=round(float(au12_intensity), 2)
        ))

        # ── AU15: Lip Corner Depressor (Frown) ─────────────────────────────────────
        # Drop of lip corners (61, 291) relative to lower lip center (17)
        # Higher vertical offset of corners below center = depressor
        au15_dist = (dist(61, 17) + dist(291, 17)) / 2.0
        au15_intensity = np.clip((au15_dist - 0.45) * 15.0, 0.0, 5.0)
        results.append(ActionUnitResult(
            au_id="AU15",
            name="Lip Corner Depressor",
            present=bool(au15_intensity > 1.5),
            intensity=round(float(au15_intensity), 2)
        ))

        # ── AU25: Lips Part ───────────────────────────────────────────────────────
        # Inner vertical distance of mouth: MOUTH_TOP (13) to MOUTH_BOTTOM (14)
        au25_dist = dist(13, 14)
        au25_intensity = np.clip((au25_dist - 0.05) * 25.0, 0.0, 5.0)
        results.append(ActionUnitResult(
            au_id="AU25",
            name="Lips Part",
            present=bool(au25_intensity > 0.8),
            intensity=round(float(au25_intensity), 2)
        ))

        # ── AU26: Jaw Drop ────────────────────────────────────────────────────────
        # Lower face height stretching (distance from nose tip 4 to chin 152)
        au26_dist = dist(4, 152)
        au26_intensity = np.clip((au26_dist - 1.8) * 8.0, 0.0, 5.0)
        results.append(ActionUnitResult(
            au_id="AU26",
            name="Jaw Drop",
            present=bool(au26_intensity > 1.5),
            intensity=round(float(au26_intensity), 2)
        ))

        # ── AU45: Blink ───────────────────────────────────────────────────────────
        # Approximate check based on vertical eye lid distances (e.g. EAR points)
        # Using EAR values instead of re-calculating distances or mapping them
        # Let's map it directly to eye closeness
        left_eye_h = (dist(385, 373) + dist(387, 380)) / 2.0
        right_eye_h = (dist(160, 153) + dist(158, 144)) / 2.0
        avg_eye_h = (left_eye_h + right_eye_h) / 2.0
        au45_intensity = np.clip((0.15 - avg_eye_h) * 40.0, 0.0, 5.0)
        results.append(ActionUnitResult(
            au_id="AU45",
            name="Blink/Eye Closure",
            present=bool(au45_intensity > 2.0),
            intensity=round(float(au45_intensity), 2)
        ))

        return results
