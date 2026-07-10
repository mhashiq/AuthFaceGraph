"""
AuthBrain DL Platform — GAT and GATv2 Graph Neural Networks

Implements Graph Attention Network (Veličković et al., 2018) and
GATv2 (Brody et al., 2022) for emotion classification from facial landmarks.

Architecture:
    Input: 478 nodes × 10 node features
    → 3 × GATConv layers with multi-head attention
    → Global mean pool
    → MLP classification head → 8 emotion classes

CPU-compatible, optional CUDA via settings.DL_DEVICE.

Note: Requires torch and torch-geometric. Import is lazy — the module
loads without error even if PyTorch is not installed; it only raises
ImportError when the model is instantiated.
"""
from __future__ import annotations

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.dl.base import FaceGraph, GNNModelBase, GNNPrediction

settings = get_settings()
logger = get_logger(__name__)

# ── AffectNet 8-class label set (consistent across all models) ────────────────
EMOTION_LABELS = [
    "neutral", "happy", "sad", "surprise",
    "fear", "disgust", "anger", "contempt",
]
NUM_CLASSES = len(EMOTION_LABELS)


def _check_torch() -> tuple[bool, str]:
    """Check if PyTorch and PyG are available."""
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        return True, ""
    except ImportError as e:
        return False, str(e)


class FaceGAT(GNNModelBase):
    """
    Graph Attention Network for facial emotion recognition.

    Uses multi-head attention to learn which facial landmark connections
    are most important for each emotion prediction.

    Architecture:
        GATConv(10 → hidden×heads) × layers → GlobalMeanPool → Linear → 8 classes
    """

    def __init__(
        self,
        in_channels: int = 10,
        hidden_channels: int = 64,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._hidden_channels = hidden_channels
        self._heads = heads
        self._num_layers = num_layers
        self._dropout = dropout
        self._device_str = device
        self._model: "torch.nn.Module | None" = None
        self._torch: "Any" = None
        self._device: "Any" = None

    @property
    def model_id(self) -> str:
        return "gnn_gat"

    def _load_impl(self) -> None:
        available, err = _check_torch()
        if not available:
            raise ImportError(
                f"PyTorch and torch-geometric are required for GNN models.\n"
                f"Install: pip install torch torch-geometric\nError: {err}"
            )

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool

        self._torch = torch
        self._device = torch.device(self._device_str)

        class _GATModule(nn.Module):
            def __init__(self, in_ch: int, hidden: int, heads: int, layers: int, dropout: float, n_classes: int) -> None:
                super().__init__()
                self.convs = nn.ModuleList()
                self.norms = nn.ModuleList()

                # Input layer
                self.convs.append(GATv2Conv(in_ch, hidden, heads=heads, dropout=dropout))
                self.norms.append(nn.LayerNorm(hidden * heads))

                # Hidden layers
                for _ in range(layers - 2):
                    self.convs.append(GATv2Conv(hidden * heads, hidden, heads=heads, dropout=dropout))
                    self.norms.append(nn.LayerNorm(hidden * heads))

                # Output layer (single head for clean pooling)
                if layers > 1:
                    self.convs.append(GATv2Conv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout))
                    self.norms.append(nn.LayerNorm(hidden))
                    out_dim = hidden
                else:
                    out_dim = hidden * heads

                self.classifier = nn.Sequential(
                    nn.Linear(out_dim, out_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(out_dim // 2, n_classes),
                )

                # Store last attention weights for XAI
                self._last_attention: torch.Tensor | None = None

            def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor", batch: "torch.Tensor | None" = None) -> tuple["torch.Tensor", "torch.Tensor"]:
                for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                    if i == len(self.convs) - 1 and hasattr(conv, 'return_attention_weights'):
                        x, (edge_idx, attn) = conv(x, edge_index, return_attention_weights=True)
                        self._last_attention = attn.detach()
                    else:
                        x = conv(x, edge_index)
                    x = norm(x)
                    x = F.gelu(x)

                # Global pooling
                if batch is None:
                    batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                x_pooled = global_mean_pool(x, batch)

                logits = self.classifier(x_pooled)
                return logits, x  # logits + node embeddings for XAI

        self._model = _GATModule(
            in_ch=self._in_channels,
            hidden=self._hidden_channels,
            heads=self._heads,
            layers=self._num_layers,
            dropout=self._dropout,
            n_classes=NUM_CLASSES,
        ).to(self._device)
        self._model.eval()

        logger.info(
            "gat_model_initialized",
            params=sum(p.numel() for p in self._model.parameters()),
            device=self._device_str,
        )

    def _forward_impl(self, graph: FaceGraph) -> GNNPrediction:
        import torch
        import torch.nn.functional as F

        model = self._model
        if model is None:
            raise RuntimeError("Model not initialized")

        # Convert numpy arrays to tensors
        x = torch.tensor(graph.node_features, dtype=torch.float32, device=self._device)
        edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=self._device)

        with torch.no_grad():
            logits, node_embeddings = model(x, edge_index)
            probs = F.softmax(logits[0], dim=-1).cpu().numpy()

        # Build per-node importance from embedding norms (proxy for GNNExplainer)
        node_importance = torch.norm(node_embeddings, dim=-1).cpu().numpy()
        node_importance = node_importance / (node_importance.max() + 1e-8)

        # Attention weights from last conv layer
        edge_attention: list[float] = []
        if hasattr(model, '_last_attention') and model._last_attention is not None:
            edge_attention = model._last_attention.mean(-1).cpu().numpy().tolist()

        top_idx = int(np.argmax(probs))

        return GNNPrediction(
            emotion=EMOTION_LABELS[top_idx],
            confidence=float(probs[top_idx]),
            probabilities={EMOTION_LABELS[i]: float(p) for i, p in enumerate(probs)},
            node_importance=node_importance.tolist(),
            edge_attention=edge_attention,
            model_id=self.model_id,
        )


class FaceGCN(GNNModelBase):
    """
    Graph Convolutional Network baseline (Kipf & Welling, 2017).

    Simpler than GAT — no attention weights but faster inference.
    Useful as ablation baseline in research comparisons.
    """

    def __init__(self, in_channels: int = 10, hidden_channels: int = 64,
                 num_layers: int = 3, device: str = "cpu") -> None:
        super().__init__()
        self._in_channels = in_channels
        self._hidden_channels = hidden_channels
        self._num_layers = num_layers
        self._device_str = device
        self._model = None
        self._torch = None
        self._device = None

    @property
    def model_id(self) -> str:
        return "gnn_gcn"

    def _load_impl(self) -> None:
        available, err = _check_torch()
        if not available:
            raise ImportError(f"PyTorch required: {err}")

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv, global_mean_pool

        self._torch = torch
        self._device = torch.device(self._device_str)

        class _GCNModule(nn.Module):
            def __init__(self, in_ch: int, hidden: int, layers: int, n_classes: int) -> None:
                super().__init__()
                self.convs = nn.ModuleList()
                self.convs.append(GCNConv(in_ch, hidden))
                for _ in range(layers - 1):
                    self.convs.append(GCNConv(hidden, hidden))
                self.classifier = nn.Linear(hidden, n_classes)

            def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor", batch: "torch.Tensor | None" = None) -> "torch.Tensor":
                for conv in self.convs:
                    x = F.relu(conv(x, edge_index))
                if batch is None:
                    batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                x = global_mean_pool(x, batch)
                return self.classifier(x)

        self._model = _GCNModule(self._in_channels, self._hidden_channels, self._num_layers, NUM_CLASSES).to(self._device)
        self._model.eval()

    def _forward_impl(self, graph: FaceGraph) -> GNNPrediction:
        import torch
        import torch.nn.functional as F

        x = torch.tensor(graph.node_features, dtype=torch.float32, device=self._device)
        edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=self._device)

        with torch.no_grad():
            logits = self._model(x, edge_index)
            probs = F.softmax(logits[0], dim=-1).cpu().numpy()

        top_idx = int(np.argmax(probs))
        return GNNPrediction(
            emotion=EMOTION_LABELS[top_idx],
            confidence=float(probs[top_idx]),
            probabilities={EMOTION_LABELS[i]: float(p) for i, p in enumerate(probs)},
            node_importance=[],
            edge_attention=[],
            model_id=self.model_id,
        )
