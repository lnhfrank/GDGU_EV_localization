"""GNN models for graph-level multi-label classification.

All models share:
  - 3 conv layers, BatchNorm, Dropout
  - Readout: mean + max pooling → 2-layer MLP head
  - Output: raw logits (no sigmoid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, BatchNorm
from torch_geometric.nn import global_mean_pool, global_max_pool


class GCN_Graph(nn.Module):
    """GCN with dual pooling."""

    def __init__(self, in_dim, hid_dim=128, out_dim=3, n_layers=3, dropout=0.3):
        super().__init__()
        self.hid_dim = hid_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hid_dim))
        self.bns.append(BatchNorm(hid_dim))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hid_dim, hid_dim))
            self.bns.append(BatchNorm(hid_dim))
        self.fc1 = nn.Linear(hid_dim * 2, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.dropout = dropout

    def encode(self, data):
        """Return pooled graph embedding [B, 2*hid_dim] (pre-head)."""
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        xm = global_mean_pool(x, batch)
        xx = global_max_pool(x, batch)
        return torch.cat([xm, xx], dim=-1)

    def forward(self, data):
        x = self.encode(data)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


class GAT_Graph(nn.Module):
    """GAT with multi-head attention and dual pooling."""

    def __init__(self, in_dim, hid_dim=128, out_dim=3, n_layers=3, dropout=0.3, heads=4):
        super().__init__()
        self.hid_dim = hid_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        # Layer 0: in_dim → hid_dim*heads
        self.convs.append(GATConv(in_dim, hid_dim, heads=heads, concat=True, dropout=dropout))
        self.bns.append(BatchNorm(hid_dim * heads))
        # Middle layers: hid_dim*heads → hid_dim*heads
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(hid_dim * heads, hid_dim, heads=heads, concat=True, dropout=dropout))
            self.bns.append(BatchNorm(hid_dim * heads))
        # Last layer: hid_dim*heads → hid_dim (heads=1)
        self.convs.append(GATConv(hid_dim * heads, hid_dim, heads=1, concat=False, dropout=dropout))
        self.bns.append(BatchNorm(hid_dim))
        self.fc1 = nn.Linear(hid_dim * 2, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.dropout = dropout

    def encode(self, data):
        """Return pooled graph embedding [B, 2*hid_dim] (pre-head)."""
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        xm = global_mean_pool(x, batch)
        xx = global_max_pool(x, batch)
        return torch.cat([xm, xx], dim=-1)

    def forward(self, data):
        x = self.encode(data)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


class GIN_Graph(nn.Module):
    """GIN with dual pooling."""

    def __init__(self, in_dim, hid_dim=128, out_dim=3, n_layers=3, dropout=0.3):
        super().__init__()
        self.hid_dim = hid_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        mlp0 = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))
        self.convs.append(GINConv(mlp0))
        self.bns.append(BatchNorm(hid_dim))
        for _ in range(n_layers - 1):
            mlp = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim))
            self.convs.append(GINConv(mlp))
            self.bns.append(BatchNorm(hid_dim))
        self.fc1 = nn.Linear(hid_dim * 2, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.dropout = dropout

    def encode(self, data):
        """Return pooled graph embedding [B, 2*hid_dim] (pre-head)."""
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        xm = global_mean_pool(x, batch)
        xx = global_max_pool(x, batch)
        return torch.cat([xm, xx], dim=-1)

    def forward(self, data):
        x = self.encode(data)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


MODEL_CLASSES = {
    'GCN': GCN_Graph,
    'GAT': GAT_Graph,
    'GIN': GIN_Graph,
}


# ======================================================================
#  Dual-Channel Model (Scheme A: strict isolation)
# ======================================================================
#
# Architecture:
#   Node channel (GNN backbone)      → Loc head  → y_loc [B, K]
#   Graph channel (MLP on TempGrad+EdgeGrad) → Det head → y_det [B, 1]
#
# Strict isolation: Loc head sees ONLY node_emb; Det head sees ONLY
# graph_emb_feat. The two paths share NO parameters.  During unlearning
# we freeze the Graph channel + Det head, so detection capability is
# retained while only the Node channel + Loc head are updated.
# ======================================================================


class DualChannel_Graph(nn.Module):
    """Dual-channel model for EVCS localization + detection with strict channel isolation.

    Args:
        backbone_name: 'GCN' | 'GAT' | 'GIN'.
        in_dim:        node feature dim (48 for current pipeline).
        hid_dim:       GNN hidden dim.
        out_dim:       number of EVCS (localization output dim).
        n_layers:      GNN depth.
        dropout:       dropout prob (shared by backbone + heads).
        graph_feat_dim: per-graph feature dim (120 by default).
        graph_mlp_hidden: hidden width of graph MLP.
        graph_mlp_out: graph embedding dim fed to Det head.
    """

    def __init__(self, backbone_name, in_dim, hid_dim=128, out_dim=3,
                 n_layers=3, dropout=0.3,
                 graph_feat_dim=120, graph_mlp_hidden=64, graph_mlp_out=32):
        super().__init__()
        BackboneClass = MODEL_CLASSES[backbone_name]
        self.backbone = BackboneClass(
            in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim,
            n_layers=n_layers, dropout=dropout)
        self.hid_dim = hid_dim
        self.dropout = dropout

        # Localization head (node channel only)
        self.loc_head = nn.Sequential(
            nn.Linear(2 * hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim),
        )

        # Graph channel: MLP on graph-level features
        # LayerNorm instead of BatchNorm to avoid BN-recalibrate interaction
        # with the unlearning pipeline (graph channel is frozen during GU).
        self.graph_mlp = nn.Sequential(
            nn.Linear(graph_feat_dim, graph_mlp_hidden),
            nn.LayerNorm(graph_mlp_hidden),
            nn.ReLU(),
            nn.Linear(graph_mlp_hidden, graph_mlp_out),
            nn.LayerNorm(graph_mlp_out),
            nn.ReLU(),
        )

        # Detection head (graph channel only) — binary
        self.det_head = nn.Sequential(
            nn.Linear(graph_mlp_out, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, data):
        """Return (loc_logits [B, K], det_logits [B])."""
        node_emb = self.backbone.encode(data)        # [B, 2*hid]
        loc_logits = self.loc_head(node_emb)         # [B, K]

        gf = data.graph_feat                         # [B, graph_feat_dim]
        graph_emb = self.graph_mlp(gf)               # [B, graph_mlp_out]
        det_logits = self.det_head(graph_emb).squeeze(-1)  # [B]

        return loc_logits, det_logits

    def freeze_graph_channel(self):
        """Freeze graph_mlp and det_head params (for unlearning).

        Node channel (backbone + loc_head) remain trainable. Returns lists
        of frozen/trainable param groups for optimizer configuration.
        """
        for p in self.graph_mlp.parameters():
            p.requires_grad = False
        for p in self.det_head.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def node_channel_params(self):
        """Return list of params in node channel (backbone + loc_head)."""
        return list(self.backbone.parameters()) + list(self.loc_head.parameters())
