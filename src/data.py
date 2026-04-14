"""Data loading, graph construction, and splitting for EVCS localization."""

import pickle
import gzip
import numpy as np
import torch
import networkx as nx
from collections import Counter
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Pre-defined EVCS-to-bus mappings
EVCS_PRESETS = {
    '34bus': {
        'EVCS 1': '814',
        'EVCS 2': '852',
        'EVCS 3': '890',
    },
    '123bus': {
        'EVCS 1': '25',
        'EVCS 2': '40',
        'EVCS 3': '54',
        'EVCS 4': '62',
        'EVCS 5': '76',
    },
}


def load_evcs_data(pkl_paths, gml_path, evcs_map=None, bus_system='34bus'):
    """Load raw EVCS attack data and build node features, multi-hot labels, edge_index.

    Args:
        pkl_paths: str or list of str. Single .pkl file or list of .pkl.gz files.
        gml_path:  str. Path to GML topology file.
        evcs_map:  dict or None. {'EVCS 1': '814', ...}. If None, uses preset.
        bus_system: str. '34bus' or '123bus', used for preset lookup and returned tag.

    Returns dict with keys:
        all_x, all_y, edge_index, n_nodes, n_feat, n_evcs,
        bus_names, bus_to_idx, evcs_buses, evcs_map, evcs_names, bus_system
    """
    # Resolve EVCS mapping
    if evcs_map is None:
        evcs_map = EVCS_PRESETS[bus_system]
    evcs_names_ordered = sorted(evcs_map.keys(), key=lambda k: int(k.split()[-1]))
    n_evcs = len(evcs_names_ordered)

    # Load raw data (single pkl or multiple pkl.gz)
    if isinstance(pkl_paths, str):
        pkl_paths = [pkl_paths]
    raw_data = []
    for p in pkl_paths:
        if p.endswith('.gz'):
            with gzip.open(p, 'rb') as f:
                raw_data.extend(pickle.load(f))
        else:
            with open(p, 'rb') as f:
                raw_data.extend(pickle.load(f))

    # Bus names and index mapping
    sample0 = raw_data[0]
    bus_names = list(sample0['BusVoltage series'].keys())
    n_nodes = len(bus_names)
    bus_to_idx = {name: i for i, name in enumerate(bus_names)}

    # EVCS bus indices
    evcs_buses = {}
    for evcs_name in evcs_names_ordered:
        bus_id = evcs_map[evcs_name]
        evcs_buses[int(bus_id)] = bus_to_idx[bus_id]

    # Node features: 24 hourly peak voltages (mean across 3 phases)
    n_graphs = len(raw_data)
    all_x = np.zeros((n_graphs, n_nodes, 24), dtype=np.float32)
    all_y = np.zeros((n_graphs, n_evcs), dtype=np.float32)

    for gi, scenario in enumerate(raw_data):
        bv = scenario['BusVoltage series']
        for ni, bus in enumerate(bus_names):
            voltages = np.array(bv[bus])  # (288, 3)
            mean_phase = voltages.mean(axis=1)  # (288,)
            all_x[gi, ni, :] = mean_phase.reshape(24, 12).max(axis=1)

        targeted = scenario['Targeted Stations']
        for evcs_idx, evcs_name in enumerate(evcs_names_ordered):
            if evcs_name in targeted:
                all_y[gi, evcs_idx] = 1.0

    n_feat = all_x.shape[2]

    # Edge index from GML topology (bidirectional)
    G = nx.read_gml(gml_path)
    edges_src, edges_dst = [], []
    gml_to_idx = {n: bus_to_idx.get(n, -1) for n in G.nodes()}
    for u, v in G.edges():
        ui, vi = gml_to_idx.get(u, -1), gml_to_idx.get(v, -1)
        if ui >= 0 and vi >= 0:
            edges_src.extend([ui, vi])
            edges_dst.extend([vi, ui])
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    # Summary
    n_normal = int((all_y.sum(axis=1) == 0).sum())
    print(f"[{bus_system}] Graphs: {n_graphs}")
    print(f"  Nodes  : {n_nodes}, Features: {n_feat}")
    print(f"  Edges  : {edge_index.shape[1]} (bidirectional)")
    print(f"  Labels : Normal={n_normal}, Attack={n_graphs - n_normal}")
    print(f"  EVCS   : {n_evcs} stations")
    print(f"\n  Per-EVCS attack frequency:")
    for i, evcs_name in enumerate(evcs_names_ordered):
        bus_id = evcs_map[evcs_name]
        print(f"    {evcs_name} (Bus {bus_id}): {int(all_y[:, i].sum())}/{n_graphs}")
    combos = Counter([tuple(row) for row in all_y.astype(int)])
    print(f"\n  Multi-label distribution:")
    for combo, cnt in sorted(combos.items(), key=lambda x: -x[1]):
        print(f"    {list(combo)}: {cnt}")

    return {
        'all_x': all_x,
        'all_y': all_y,
        'edge_index': edge_index,
        'n_graphs': n_graphs,
        'n_nodes': n_nodes,
        'n_feat': n_feat,
        'n_evcs': n_evcs,
        'bus_names': bus_names,
        'bus_to_idx': bus_to_idx,
        'evcs_buses': evcs_buses,
        'evcs_map': evcs_map,
        'evcs_names': evcs_names_ordered,
        'bus_system': bus_system,
    }


def build_graphs(x_np, y_np, edge_idx, n_nodes, n_feat,
                 scaler=None, forget_indices=None):
    """Build list of PyG Data objects with optional edge masking + feature zeroing.

    When forget_indices is provided:
      1. Remove all edges touching forget nodes (edge masking)
      2. Scale features first, then zero forget nodes
         (zeroing after scaling ensures 0.0 in standardized space = neutral signal,
          avoiding the extreme negative values from scaling raw zeros)
    """
    # Precompute masked edge_index: drop all edges where src or dst is a forget node
    if forget_indices is not None and len(forget_indices) > 0:
        forget_t = torch.tensor(forget_indices, dtype=torch.long)
        mask = ~torch.isin(edge_idx[0], forget_t) & ~torch.isin(edge_idx[1], forget_t)
        edge_idx_use = edge_idx[:, mask]
    else:
        edge_idx_use = edge_idx

    graphs = []
    for i in range(len(x_np)):
        xi = x_np[i].copy()
        if scaler is not None:
            xi = scaler.transform(xi.reshape(1, -1)).reshape(n_nodes, n_feat)
        if forget_indices is not None:
            xi[forget_indices, :] = 0.0
        x_t = torch.tensor(xi, dtype=torch.float)
        y_t = torch.tensor(y_np[i], dtype=torch.float).unsqueeze(0)
        graphs.append(Data(x=x_t, edge_index=edge_idx_use.clone(), y=y_t))
    return graphs


def stratified_split_multilabel(y, test_size, val_ratio, seed):
    """Stratified split for multi-label by hashing each label vector."""
    label_hash = np.array([hash(tuple(row)) for row in y])
    idx_all = np.arange(len(y))
    idx_train, idx_temp = train_test_split(
        idx_all, test_size=test_size, stratify=label_hash, random_state=seed)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=val_ratio, stratify=label_hash[idx_temp], random_state=seed)
    return idx_train, idx_val, idx_test


def fit_scaler(all_x, idx_train):
    """Fit StandardScaler on training data (flattened node features)."""
    x_train_flat = all_x[idx_train].reshape(len(idx_train), -1)
    return StandardScaler().fit(x_train_flat)
