"""
AuthBrain AI Face Analysis Engine
Unit Tests — Continuous Real-Time Identity Verification Layer
"""

import pytest
import numpy as np
from app.analysis.identity_verifier import (
    IdentityVerifier,
    cosine_similarity,
    similarity_to_confidence,
)


def test_cosine_similarity_identical():
    v1 = [0.1] * 512
    v2 = [0.1] * 512
    sim = cosine_similarity(v1, v2)
    assert abs(sim - 1.0) < 1e-4
    conf = similarity_to_confidence(sim)
    assert conf >= 0.99


def test_cosine_similarity_orthogonal():
    v1 = [1.0] + [0.0] * 511
    v2 = [0.0] * 511 + [1.0]
    sim = cosine_similarity(v1, v2)
    assert abs(sim - 0.0) < 1e-4
    conf = similarity_to_confidence(sim)
    assert conf <= 0.30


def test_identity_verifier_enrollment_and_matching():
    landmarks = [{"x": float(i), "y": float(i * 1.5), "z": float(i * 0.1)} for i in range(68)]
    verifier = IdentityVerifier(enrolled_user_name="John Smith")

    # Frame 1: Auto enrollment
    res1 = verifier.verify_frame(landmarks, force_check=True)
    assert res1["status"] == "verified"
    assert res1["enrolled_user_name"] == "John Smith"
    assert res1["match_confidence"] >= 0.95
    assert res1["is_paused"] is False

    # Frame 2: Same landmarks -> Should match
    res2 = verifier.verify_frame(landmarks, force_check=True)
    assert res2["status"] == "verified"
    assert res2["is_paused"] is False


def test_identity_verifier_mismatch_pauses_tracking():
    landmarks1 = [{"x": float(i), "y": float(i * 1.5), "z": float(i * 0.1)} for i in range(68)]
    # Use random/orthogonal landmark pattern
    np.random.seed(42)
    rand_coords = np.random.randn(68, 3)
    landmarks2 = [{"x": float(rand_coords[i, 0]), "y": float(rand_coords[i, 1]), "z": float(rand_coords[i, 2])} for i in range(68)]

    verifier = IdentityVerifier(enrolled_user_name="John Smith")
    verifier.verify_frame(landmarks1, force_check=True)

    # Different face landmarks -> Match fail twice
    verifier.verify_frame(landmarks2, force_check=True)
    res_mismatch = verifier.verify_frame(landmarks2, force_check=True)

    assert res_mismatch["status"] in ("mismatch", "liveness_failed")
    assert res_mismatch["is_paused"] is True
    assert res_mismatch["match_confidence"] <= 0.65
