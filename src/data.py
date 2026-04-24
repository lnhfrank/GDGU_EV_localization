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

    # Node features: 48-dim voltage (24 hourly mean + 24 hourly std)
    # Phase-averaged, then split each hour (12 steps × 5 min) into mean & std.
    # mean captures magnitude changes (Type 1/3 attacks); std captures temporal
    # volatility within each hour (Type 2/4 time-shift and surge attacks).
    # Replaces previous 24-dim hourly-max which discarded sub-hourly dynamics.
    # Also retains all_V (phase-averaged raw time series) for downstream
    # graph-level feature extraction in the dual-channel model.
    n_graphs = len(raw_data)
    all_x = np.zeros((n_graphs, n_nodes, 48), dtype=np.float32)
    all_V = np.zeros((n_graphs, n_nodes, 288), dtype=np.float32)
    all_y = np.zeros((n_graphs, n_evcs), dtype=np.float32)

    for gi, scenario in enumerate(raw_data):
        bv = scenario['BusVoltage series']
        for ni, bus in enumerate(bus_names):
            voltages = np.array(bv[bus])  # (288, 3)
            # Use only active phases (>0.1 p.u.) to avoid 3x scaling artifact
            # on single-phase buses (810, 818, 820, 822, 826, 856, 864, 838).
            active = voltages.mean(axis=0) > 0.1
            if active.any():
                mean_phase = voltages[:, active].mean(axis=1)  # (288,)
            else:
                mean_phase = voltages.mean(axis=1)
            all_V[gi, ni] = mean_phase
            hourly = mean_phase.reshape(24, 12)  # (24 hours, 12 steps)
            all_x[gi, ni, :24] = hourly.mean(axis=1)
            all_x[gi, ni, 24:] = hourly.std(axis=1)

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
        'all_V': all_V,
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


# ======================================================================
#  V6.0 Route A: A-append data pipeline
# ======================================================================

TYPE_MAP = {"Nil": 0, "Type 1": 1, "Type 2": 2, "Type 3": 3, "Type 4": 4}


def _load_raw_pkl(pkl_paths):
    """Load raw scenario list from pkl/pkl.gz file(s)."""
    if isinstance(pkl_paths, str):
        pkl_paths = [pkl_paths]
    raw = []
    for p in pkl_paths:
        if p.endswith('.gz'):
            with gzip.open(p, 'rb') as f:
                raw.extend(pickle.load(f))
        else:
            with open(p, 'rb') as f:
                raw.extend(pickle.load(f))
    return raw


def augment_route_a(data_dict, pkl_paths):
    """Extend load_evcs_data output with P features and attack-type labels.

    Adds keys: all_x_P [G, N, 48], all_attack_type [G], evcs_node_indices [K].
    """
    raw = _load_raw_pkl(pkl_paths)
    evcs_node_indices = [
        data_dict['bus_to_idx'][data_dict['evcs_map'][nm]]
        for nm in data_dict['evcs_names']
    ]
    G = len(raw)
    n_nodes = data_dict['n_nodes']

    all_x_P = np.zeros((G, n_nodes, 48), dtype=np.float32)
    for gi, s in enumerate(raw):
        for evcs_1based, p_series in s["EVCS power series"].items():
            idx0 = evcs_1based - 1
            node_idx = evcs_node_indices[idx0]
            p = np.asarray(p_series, dtype=np.float32).reshape(24, 12)
            all_x_P[gi, node_idx, :24] = p.mean(axis=1)
            all_x_P[gi, node_idx, 24:] = p.std(axis=1)

    all_attack_type = np.array(
        [TYPE_MAP[s["Attack Type"]] for s in raw], dtype=np.int64)

    data_dict['all_x_P'] = all_x_P
    data_dict['all_attack_type'] = all_attack_type
    data_dict['evcs_node_indices'] = evcs_node_indices
    return data_dict


def build_graphs_route_a(all_x_V, all_x_P, all_y, all_attack_type,
                          edge_index, n_nodes, idx,
                          scaler_V, scaler_P,
                          forget_P_at_node=None):
    """Build PyG Data list for Route A (A-append [V_48|P_48], optional P zeroing).

    Args:
        forget_P_at_node: if int, zero dims 48:96 at this node for every graph.
    """
    n = len(idx)
    xV = all_x_V[idx]
    xP = all_x_P[idx]
    v = scaler_V.transform(xV.reshape(n, -1)).reshape(n, n_nodes, 48).astype(np.float32)
    p = scaler_P.transform(xP.reshape(n, -1)).reshape(n, n_nodes, 48).astype(np.float32)
    x = np.concatenate([v, p], axis=2)

    if forget_P_at_node is not None:
        x[:, forget_P_at_node, 48:] = 0.0

    ds = []
    for i in range(n):
        ds.append(Data(
            x=torch.tensor(x[i], dtype=torch.float),
            edge_index=edge_index.clone(),
            y=torch.tensor(all_y[idx[i]], dtype=torch.float).unsqueeze(0),
            y_type=torch.tensor([all_attack_type[idx[i]]], dtype=torch.long),
        ))
    return ds
