import sys
import os
from pathlib import Path

# Add backend to path so we can import app modules
sys.path.insert(0, os.path.abspath("."))

import torch

def create_checkpoint():
    from app.core.config import get_settings
    settings = get_settings()
    checkpoint_path = Path(settings.DL_GNN_CHECKPOINT_PATH)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Actually create the underlying torch module
    from torch_geometric.nn import GATv2Conv, global_mean_pool
    import torch.nn as nn
    import torch.nn.functional as F
    
    class _GATModule(nn.Module):
        def __init__(self, in_ch, hidden, heads, layers, dropout, n_classes):
            super().__init__()
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.convs.append(GATv2Conv(in_ch, hidden, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden * heads))
            for _ in range(layers - 2):
                self.convs.append(GATv2Conv(hidden * heads, hidden, heads=heads, dropout=dropout))
                self.norms.append(nn.LayerNorm(hidden * heads))
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

    net = _GATModule(
        in_ch=10,
        hidden=64,
        heads=4,
        layers=3,
        dropout=0.1,
        n_classes=8
    )
    
    print(f"Saving state dict to {checkpoint_path}...")
    torch.save({"model_state_dict": net.state_dict()}, checkpoint_path)
    print("Done!")

if __name__ == "__main__":
    create_checkpoint()
