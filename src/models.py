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

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        xm = global_mean_pool(x, batch)
        xx = global_max_pool(x, batch)
        x = torch.cat([xm, xx], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


class GAT_Graph(nn.Module):
    """GAT with multi-head attention and dual pooling."""

    def __init__(self, in_dim, hid_dim=128, out_dim=3, n_layers=3, dropout=0.3, heads=4):
        super().__init__()
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

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        xm = global_mean_pool(x, batch)
        xx = global_max_pool(x, batch)
        x = torch.cat([xm, xx], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


class GIN_Graph(nn.Module):
    """GIN with dual pooling."""

    def __init__(self, in_dim, hid_dim=128, out_dim=3, n_layers=3, dropout=0.3):
        super().__init__()
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

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, ei)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        xm = global_mean_pool(x, batch)
        xx = global_max_pool(x, batch)
        x = torch.cat([xm, xx], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


MODEL_CLASSES = {
    'GCN': GCN_Graph,
    'GAT': GAT_Graph,
    'GIN': GIN_Graph,
}
