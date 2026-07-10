"""
AuthBrain DL Platform — Facial Landmark Graph Constructor

Converts MediaPipe 478-landmark data into a graph for GNN processing.

Node features per landmark (10-dimensional):
  [0]   x coordinate (normalized 0–1)
  [1]   y coordinate (normalized 0–1)
  [2]   z depth (normalized)
  [3]   Δx from previous frame (temporal displacement)
  [4]   Δy from previous frame
  [5]   velocity magnitude (Euclidean Δ / Δt)
  [6-9] region one-hot: [eye, nose, mouth, other]

Edge strategies:
  - "knn"        : k-nearest neighbours in 3D space (default)
  - "anatomical" : MediaPipe FACEMESH_TESSELATION subset (~1400 edges)
  - "radius"     : connect all landmarks within Euclidean radius r
  - "combined"   : knn + anatomical union
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from app.analysis.landmark_indices import LANDMARKS
from app.dl.base import FaceGraph


# ── Region label assignment (landmark index → region id) ─────────────────────
# 0=other, 1=eye, 2=nose, 3=mouth
_REGION_MAP: dict[int, int] = {}

def _build_region_map() -> dict[int, int]:
    m: dict[int, int] = {}
    for idx in LANDMARKS.LEFT_EYE + LANDMARKS.RIGHT_EYE + LANDMARKS.LEFT_IRIS + LANDMARKS.RIGHT_IRIS:
        m[idx] = 1
    for idx in LANDMARKS.NOSE_BRIDGE + LANDMARKS.NOSE_CONTOUR + (LANDMARKS.NOSE_TIP,):
        m[idx] = 2
    for idx in (
        LANDMARKS.LIPS_UPPER_OUTER + LANDMARKS.LIPS_LOWER_OUTER +
        LANDMARKS.LIPS_UPPER_INNER + LANDMARKS.LIPS_LOWER_INNER
    ):
        m[idx] = 3
    return m

_REGION_MAP = _build_region_map()

# Pre-encoded one-hot region vectors [4-dim]
_REGION_ONEHOT: NDArray[np.float32] = np.eye(4, dtype=np.float32)

# ── Anatomical edges (subset of FACEMESH_TESSELATION for efficiency) ─────────
# These match the mesh connections used in pipeline.py overlay drawing
_ANATOMICAL_EDGES: list[tuple[int, int]] = [
    # Face oval
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),
    (356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),
    (379,378),(378,400),(400,377),(377,152),(152,148),(148,176),(176,149),
    (149,150),(150,136),(136,172),(172,58),(58,132),(132,93),(93,234),
    (234,127),(127,162),(162,21),(21,54),(54,103),(103,67),(67,109),(109,10),
    # Eyes
    (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),(155,133),
    (133,173),(173,157),(157,158),(158,159),(159,160),(160,161),(161,246),(246,33),
    (362,382),(382,381),(381,380),(380,374),(374,373),(373,390),(390,249),
    (249,263),(263,466),(466,388),(388,387),(387,386),(386,385),(385,384),
    (384,398),(398,362),
    # Nose
    (6,197),(197,195),(195,5),(129,49),(49,131),(131,134),(134,51),(51,5),
    (281,363),(363,360),(360,279),
    # Lips
    (61,185),(185,40),(40,39),(39,37),(37,0),(0,267),(267,269),(269,270),
    (270,409),(409,291),(61,146),(146,91),(91,181),(181,84),(84,17),(17,314),
    (314,405),(405,321),(321,375),(375,291),
    # Eyebrows
    (336,296),(296,334),(334,293),(293,300),(300,276),(276,283),(283,282),
    (282,295),(295,285),(70,63),(63,105),(105,66),(66,107),(107,55),(55,65),
    (65,52),(52,53),(53,46),
]

# Make edges bidirectional
_ANATOMICAL_EDGE_INDEX: NDArray[np.int64] = np.array(
    [(a, b) for a, b in _ANATOMICAL_EDGES] + [(b, a) for a, b in _ANATOMICAL_EDGES],
    dtype=np.int64,
).T  # [2, num_edges]


class GraphConstructor:
    """
    Converts per-frame MediaPipe landmark data into FaceGraph objects
    for GNN consumption.

    Maintains temporal state (previous frame positions) across frames
    to compute displacement and velocity features.
    """

    NUM_NODES: int = 478
    FEATURE_DIM: int = 10  # x,y,z + Δx,Δy + vel + region_onehot[4]

    def __init__(
        self,
        edge_strategy: str = "knn",
        knn_k: int = 6,
        radius: float = 0.08,
    ) -> None:
        """
        Args:
            edge_strategy: "knn" | "anatomical" | "radius" | "combined"
            knn_k: Number of neighbours for KNN edges
            radius: Radius for radius-based edges (normalized coords)
        """
        self._strategy = edge_strategy
        self._k = knn_k
        self._radius = radius

        # Temporal state
        self._prev_positions: NDArray[np.float32] | None = None
        self._prev_timestamp: float = 0.0

        # Cache static edge index (anatomical never changes)
        self._static_edge_index: NDArray[np.int64] | None = (
            _ANATOMICAL_EDGE_INDEX if edge_strategy == "anatomical" else None
        )

    def build(
        self,
        landmarks: list,  # list of Landmark(x,y,z)
        timestamp_ms: float,
    ) -> FaceGraph:
        """
        Build a FaceGraph from a list of 478 Landmark objects.

        Args:
            landmarks: 478 Landmark objects from MediaPipe
            timestamp_ms: Current frame timestamp

        Returns:
            FaceGraph with node features and edge index
        """
        n = len(landmarks)

        # ── 1. Extract raw positions ──────────────────────────────────────────
        positions = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks],
            dtype=np.float32,
        )  # [478, 3]

        # ── 2. Temporal features ──────────────────────────────────────────────
        dt = max((timestamp_ms - self._prev_timestamp) / 1000.0, 1e-6)  # seconds

        if self._prev_positions is not None and self._prev_positions.shape[0] == n:
            delta = positions - self._prev_positions          # [478, 3]
            velocity = np.linalg.norm(delta[:, :2], axis=1, keepdims=True) / dt  # [478, 1]
            delta_xy = delta[:, :2]                           # [478, 2]
        else:
            delta_xy = np.zeros((n, 2), dtype=np.float32)
            velocity = np.zeros((n, 1), dtype=np.float32)

        # ── 3. Region one-hot encoding ────────────────────────────────────────
        region_ids = np.array(
            [_REGION_MAP.get(i, 0) for i in range(n)], dtype=np.int32
        )
        region_onehot = _REGION_ONEHOT[region_ids]  # [478, 4]

        # ── 4. Normalize velocity to [0, 1] ───────────────────────────────────
        vel_norm = np.clip(velocity / 100.0, 0.0, 1.0)  # 100 px/s = max

        # ── 5. Assemble node features [478, 10] ───────────────────────────────
        node_features = np.concatenate(
            [positions, delta_xy, vel_norm, region_onehot], axis=1
        ).astype(np.float32)

        # ── 6. Build edge index ───────────────────────────────────────────────
        edge_index = self._build_edges(positions)

        # Update temporal state
        self._prev_positions = positions.copy()
        self._prev_timestamp = timestamp_ms

        return FaceGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=None,
            landmark_positions=positions,
            timestamp_ms=timestamp_ms,
            num_nodes=n,
            feature_dim=self.FEATURE_DIM,
        )

    def _build_edges(self, positions: NDArray[np.float32]) -> NDArray[np.int64]:
        """Build edge index according to configured strategy."""
        if self._strategy == "anatomical":
            return _ANATOMICAL_EDGE_INDEX
        elif self._strategy == "knn":
            return self._knn_edges(positions)
        elif self._strategy == "radius":
            return self._radius_edges(positions)
        elif self._strategy == "combined":
            knn = self._knn_edges(positions)
            combined = np.concatenate([_ANATOMICAL_EDGE_INDEX, knn], axis=1)
            # Deduplicate
            edge_set = np.unique(combined, axis=1)
            return edge_set
        else:
            return self._knn_edges(positions)

    def _knn_edges(self, positions: NDArray[np.float32]) -> NDArray[np.int64]:
        """k-NN graph in 3D landmark space."""
        n = positions.shape[0]
        k = min(self._k, n - 1)

        # Pairwise squared distances
        diff = positions[:, None, :] - positions[None, :, :]  # [N, N, 3]
        dist_sq = np.sum(diff ** 2, axis=-1)                  # [N, N]
        np.fill_diagonal(dist_sq, np.inf)

        # k nearest for each node
        knn_indices = np.argpartition(dist_sq, k, axis=1)[:, :k]  # [N, k]
        src = np.repeat(np.arange(n), k)
        dst = knn_indices.flatten()

        # Bidirectional
        edge_index = np.stack([
            np.concatenate([src, dst]),
            np.concatenate([dst, src]),
        ], axis=0).astype(np.int64)

        return edge_index

    def _radius_edges(self, positions: NDArray[np.float32]) -> NDArray[np.int64]:
        """Connect all landmarks within Euclidean radius."""
        diff = positions[:, None, :] - positions[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        src, dst = np.where(dist < self._radius)
        return np.stack([src, dst], axis=0).astype(np.int64)

    def reset(self) -> None:
        """Reset temporal state for a new session."""
        self._prev_positions = None
        self._prev_timestamp = 0.0
