"""
Smoke test for V5.1 Route A — Two-controller modality-level unlearning.

Quick-and-dirty single-seed / single-scenario validation of L1 + L2 claims:
  L1 (Utility Preservation): forgetting P at EVCS1 should NOT harm
                             localization F1 / ROC-AUC (DSO's V data intact).
  L2 (Modality Erasure):    P-specific privacy metrics should drop to
                             baseline — gradient on P-channel -> 0,
                             occlusion Delta-AUC -> 0, reconstruction
                             MSE rises to random-baseline level.

Setup:
  Backbone : GIN (fastest, most stable on 34-bus)
  Scenario : S1-0 — forget EVCS 1 (bus 814)
  Seed     : 42
  Device   : cuda:0 (as explicitly requested for this run; GPU 1 occupied)

A-append data structure:
  node_feat[i] = [V_48 | P_48]  (dim = 96)
    - V part: voltage features for all nodes (DSO-owned, never masked)
    - P part: charging power features for EVCS nodes only
             (EVCS-owned, zeroed at forget target during unlearning)
"""

import os
import sys
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# Add project root to path
PROJ = '/home/Nanhong147/1P_WTT_NVD/Projects/4-GU_EV_loc'
sys.path.insert(0, PROJ)
from src.data import (
    load_evcs_data, build_graph_features, fit_scaler, fit_graph_feat_scaler,
    stratified_split_multilabel,
)
from src.models import DualChannel_Graph
from src.training import (
    train_model_dual, evaluate_model_dual, get_pos_weights,
)
from src.unlearning import (
    recalibrate_batchnorm, _finetune_after_gdgu_dual,
)

from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

# ================================================================
#  Config
# ================================================================
SEED = 42
DEVICE = torch.device('cuda:0')
BACKBONE = 'GIN'
N_EPOCHS = 150      # reduced from 300 for smoke speed
PATIENCE = 30
BATCH_SIZE = 32
LR = 5e-4
HID_DIM = 128

OUT_DIR = os.path.join(PROJ, 'results', 'smoke_A_' + time.strftime('%Y-%m-%d_%H'))
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[SMOKE-A] Output dir: {OUT_DIR}")
print(f"[SMOKE-A] Device: {DEVICE}   Backbone: {BACKBONE}   Seed: {SEED}")

SOURCE = '/home/Nanhong147/1P_WTT_NVD/Projects/Source/PB_data/3_EVCS Attacks/34_bus'
PKL = os.path.join(SOURCE, 'EVCSAttacks_34.pkl')
GML = os.path.join(SOURCE, '34busEx.gml')

torch.manual_seed(SEED)
np.random.seed(SEED)


# ================================================================
#  1. Build A-append features: [V_48 | P_48]
# ================================================================
print("\n[1/6] Loading data and building V+P features ...")

# Load V features (reuses existing pipeline) and raw V time series
d = load_evcs_data(PKL, GML, bus_system='34bus')
all_x_V = d['all_x']          # [G, N, 48] voltage mean+std
all_V   = d['all_V']           # [G, N, 288] raw V time series
all_y   = d['all_y']           # [G, K] multi-hot
edge_idx = d['edge_index']
n_nodes  = d['n_nodes']
n_evcs   = d['n_evcs']
bus_to_idx = d['bus_to_idx']
evcs_names_ordered = d['evcs_names']
evcs_map = d['evcs_map']

# Reload raw pkl to extract P time series (not in load_evcs_data)
import pickle
with open(PKL, 'rb') as f:
    raw = pickle.load(f)
n_graphs = len(raw)

# EVCS bus indices (order matches evcs_names_ordered)
evcs_node_indices = [bus_to_idx[evcs_map[name]] for name in evcs_names_ordered]
print(f"  EVCS nodes: {list(zip(evcs_names_ordered, evcs_node_indices))}")

# Build P features: [G, N, 48] = hourly mean(24d) + hourly std(24d) per EVCS
# P raw is shape (288,) keyed by EVCS index 1/2/3
all_x_P = np.zeros((n_graphs, n_nodes, 48), dtype=np.float32)
for gi, s in enumerate(raw):
    ps = s['EVCS power series']   # dict: {1: (288,), 2: (288,), 3: (288,)}
    for evcs_i_1based, p_series in ps.items():
        evcs_idx_0based = evcs_i_1based - 1           # 1→0, 2→1, 3→2
        node_idx = evcs_node_indices[evcs_idx_0based]
        p = np.asarray(p_series, dtype=np.float32)
        hourly = p.reshape(24, 12)
        all_x_P[gi, node_idx, :24] = hourly.mean(axis=1)
        all_x_P[gi, node_idx, 24:] = hourly.std(axis=1)

# Concatenate: node feat = [V_48 | P_48], 96d
all_x_VP = np.concatenate([all_x_V, all_x_P], axis=2)   # [G, N, 96]
print(f"  Node feat shape: V+P = {all_x_VP.shape}")

# Graph-level features (same as V5.0, purely V-derived, DSO-owned)
graph_feat = build_graph_features(all_V, edge_idx)       # [G, 120]


# ================================================================
#  2. Split
# ================================================================
print("\n[2/6] Stratified split ...")
idx_tr, idx_va, idx_te = stratified_split_multilabel(
    all_y, test_size=0.30, val_ratio=0.50, seed=SEED)
print(f"  Train/Val/Test = {len(idx_tr)}/{len(idx_va)}/{len(idx_te)}")

# Fit scalers on training subset
# IMPORTANT: scale V and P separately so P scaling doesn't get mixed with V
scaler_V = StandardScaler().fit(all_x_V[idx_tr].reshape(len(idx_tr), -1))
scaler_P = StandardScaler().fit(all_x_P[idx_tr].reshape(len(idx_tr), -1))
gf_scaler = StandardScaler().fit(graph_feat[idx_tr])


def scale_node_feat(x_V, x_P):
    """Apply per-modality StandardScaler, return concatenated [G, N, 96]."""
    G = x_V.shape[0]
    xV = scaler_V.transform(x_V.reshape(G, -1)).reshape(G, n_nodes, 48).astype(np.float32)
    xP = scaler_P.transform(x_P.reshape(G, -1)).reshape(G, n_nodes, 48).astype(np.float32)
    return np.concatenate([xV, xP], axis=2)


# ================================================================
#  3. Build graph datasets — 3 versions for 3 conditions
# ================================================================
def build_pyg_dataset(x_V, x_P, y, graph_feat_arr, forget_node_idx=None,
                      forget_P_only=False):
    """Build list of PyG Data objects.

    forget_node_idx:  int or None — node to "forget" (EVCS bus index).
    forget_P_only:    if True, zero ONLY the P-channel (dims 48:96) at forget node
                      (two-controller setup: V stays, P removed).
                      if False (and forget_node_idx is set), zero the full feature
                      row at forget node (legacy single-controller behavior).
    """
    G = x_V.shape[0]
    x_full = scale_node_feat(x_V, x_P)                # [G, N, 96]
    if forget_node_idx is not None:
        if forget_P_only:
            x_full[:, forget_node_idx, 48:] = 0.0     # zero P-channel only
        else:
            x_full[:, forget_node_idx, :]  = 0.0      # zero everything

    gf = gf_scaler.transform(graph_feat_arr).astype(np.float32)

    out = []
    # When we forget P only, edges are NOT masked (V at the node is still valid)
    # When we forget full (legacy), edges ARE masked (matches V5.0 behavior)
    if forget_node_idx is not None and not forget_P_only:
        mask = (edge_idx[0] != forget_node_idx) & (edge_idx[1] != forget_node_idx)
        ei_use = edge_idx[:, mask]
    else:
        ei_use = edge_idx

    for i in range(G):
        y_det = float(y[i].sum() > 0)
        out.append(Data(
            x=torch.tensor(x_full[i], dtype=torch.float),
            edge_index=ei_use.clone(),
            y=torch.tensor(y[i], dtype=torch.float).unsqueeze(0),
            y_det=torch.tensor([[y_det]], dtype=torch.float),
            graph_feat=torch.tensor(gf[i:i+1], dtype=torch.float),
        ))
    return out


# Scenario S1-0: forget EVCS 1 (bus 814)
FORGET_EVCS_IDX = 0                                   # index into evcs_names_ordered
FORGET_NODE_IDX = evcs_node_indices[FORGET_EVCS_IDX]
FORGET_BUS = evcs_map[evcs_names_ordered[FORGET_EVCS_IDX]]
print(f"\n[3/6] Forget target: EVCS {FORGET_EVCS_IDX+1} = Bus {FORGET_BUS} "
      f"(node idx {FORGET_NODE_IDX})")

# ---- three training datasets ----
#  ds_orig     : no forgetting, full V+P (used for 'Original' and as reference)
#  ds_forget_P : forget P-channel at EVCS1 (Two-controller Route A)
#  ds_forget_full : forget full features + edges at EVCS1 (legacy single-controller)
print("  Building datasets ...")
ds_orig    = build_pyg_dataset(all_x_V, all_x_P, all_y, graph_feat,
                               forget_node_idx=None)
ds_forgetP = build_pyg_dataset(all_x_V, all_x_P, all_y, graph_feat,
                               forget_node_idx=FORGET_NODE_IDX, forget_P_only=True)

# Split by index
def subset(ds, idx): return [ds[i] for i in idx]
tr_orig, va_orig, te_orig = subset(ds_orig, idx_tr), subset(ds_orig, idx_va), subset(ds_orig, idx_te)
tr_fP,   va_fP,   te_fP   = subset(ds_forgetP, idx_tr), subset(ds_forgetP, idx_va), subset(ds_forgetP, idx_te)

def mk_loaders(train, val, test):
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val,   batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test,  batch_size=BATCH_SIZE, shuffle=False),
    )

tr_orig_ld, va_orig_ld, te_orig_ld = mk_loaders(tr_orig, va_orig, te_orig)
tr_fP_ld,   va_fP_ld,   te_fP_ld   = mk_loaders(tr_fP,   va_fP,   te_fP)

# pos_weights computed from training labels only
pos_w = get_pos_weights(all_y[idx_tr], DEVICE)


def build_model():
    torch.manual_seed(SEED)
    return DualChannel_Graph(
        BACKBONE, in_dim=96, hid_dim=HID_DIM, out_dim=n_evcs,
        n_layers=3, dropout=0.3, graph_feat_dim=120,
        graph_mlp_hidden=64, graph_mlp_out=32,
    ).to(DEVICE)


# ================================================================
#  4. Train 3 conditions
# ================================================================
results = {}

print("\n[4/6] === Condition 1: Original (V+P, no forgetting) ===")
t0 = time.time()
model_orig = build_model()
model_orig, best_va, _ = train_model_dual(
    model_orig, tr_orig_ld, va_orig_ld, DEVICE,
    epochs=N_EPOCHS, lr=LR, weight_decay=1e-4, patience=PATIENCE,
    pos_weights=pos_w, alpha=1.0, beta=1.0, verbose=True)
t_orig = time.time() - t0
m = evaluate_model_dual(model_orig, te_orig_ld, DEVICE)
results['Original_VP'] = dict(m, train_time=t_orig)
print(f"  Time: {t_orig:.1f}s  Macro-F1={m['macro_f1']:.4f}  Macro-ROC={m['macro_roc']:.4f} "
      f"DetAUC={m['det_auc']:.4f}")

# Save Original state for recon attack + GDGU init
orig_state = copy.deepcopy(model_orig.state_dict())


print("\n[5/6] === Condition 2: GDGU-dual-A (forget P-channel at EVCS 1) ===")
# Load fresh copy of Original weights, then run GDGU targeting P-only forgetting
t0 = time.time()
model_gdgu = build_model()
model_gdgu.load_state_dict(orig_state)

model_gdgu.freeze_graph_channel()
crit_loc = nn.BCEWithLogitsLoss(pos_weight=pos_w)

# Gradient difference on loc loss
def _batch_grad_loc(model, loader):
    from src.unlearning import _compute_batch_gradient_loc
    return _compute_batch_gradient_loc(model, loader, crit_loc, DEVICE)

g_mod = _batch_grad_loc(model_gdgu, tr_fP_ld)
g_org = _batch_grad_loc(model_gdgu, tr_orig_ld)
delta = [a - b for a, b in zip(g_mod, g_org)]
dn = sum(di.norm().item()**2 for di in delta) ** 0.5
print(f"  Delta gradient norm: {dn:.6f}")

damp = 0.1
max_norm = 1.0
update = [d / damp for d in delta]
un = sum(u.norm().item()**2 for u in update) ** 0.5
clip = min(1.0, max_norm / (un + 1e-8))
print(f"  Update norm: {un:.6f}, clip_coef: {clip:.4f}")
params_trainable = [p for p in model_gdgu.parameters() if p.requires_grad]
with torch.no_grad():
    for p, u in zip(params_trainable, update):
        if torch.isnan(u).any(): continue
        p.data.add_(u, alpha=clip)

recalibrate_batchnorm(model_gdgu, tr_fP_ld, DEVICE)
model_gdgu = _finetune_after_gdgu_dual(
    model_gdgu, tr_fP_ld, va_fP_ld, DEVICE,
    epochs=25, lr=1e-4, alpha=1.0, beta=1.0, pos_weights=pos_w)
t_gdgu = time.time() - t0
m = evaluate_model_dual(model_gdgu, te_fP_ld, DEVICE)
results['GDGU_dual_A'] = dict(m, train_time=t_gdgu)
print(f"  Time: {t_gdgu:.1f}s  Macro-F1={m['macro_f1']:.4f}  Macro-ROC={m['macro_roc']:.4f} "
      f"DetAUC={m['det_auc']:.4f}")


print("\n[5b/6] === Condition 3: Retrain (P_814 zeroed from scratch) ===")
t0 = time.time()
model_rt = build_model()
model_rt, _, _ = train_model_dual(
    model_rt, tr_fP_ld, va_fP_ld, DEVICE,
    epochs=N_EPOCHS, lr=LR, weight_decay=1e-4, patience=PATIENCE,
    pos_weights=pos_w, alpha=1.0, beta=1.0, verbose=True)
t_rt = time.time() - t0
m = evaluate_model_dual(model_rt, te_fP_ld, DEVICE)
results['Retrain_A'] = dict(m, train_time=t_rt)
print(f"  Time: {t_rt:.1f}s  Macro-F1={m['macro_f1']:.4f}  Macro-ROC={m['macro_roc']:.4f} "
      f"DetAUC={m['det_auc']:.4f}")


# ================================================================
#  6. L2 modality-erasure evaluation
# ================================================================
print("\n[6/6] === L2 modality-specific privacy metrics ===")

def p_gradient_norm(model, loader):
    """Mean ||d loc_loss / d x_P_at_forget_node|| across batches.

    Measures how much the model *uses* the forgotten EVCS's P input.
    Post-unlearn should approach 0.
    """
    model.eval()
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    norms = []
    for batch in loader:
        batch = batch.to(DEVICE)
        batch.x.requires_grad_(True)
        loc_logits, _ = model(batch)
        loss = crit(loc_logits, batch.y)
        grads = torch.autograd.grad(loss, batch.x, retain_graph=False,
                                    create_graph=False)[0]     # [sumN, 96]
        # Pick gradient rows for the forget node across all graphs in batch
        # batch.batch tells us which graph each row belongs to
        # Rows where node_idx_within_graph == FORGET_NODE_IDX
        # With simple PyG batching, forget node's position within each graph
        # is the same (n_nodes per graph is constant for dense 34-bus data).
        num_graphs = batch.num_graphs
        # reshape to [B, N, 96]
        g_3d = grads.view(num_graphs, n_nodes, 96)
        # P-channel at forget node: dims 48:96
        p_grad_at_forget = g_3d[:, FORGET_NODE_IDX, 48:]
        norms.append(p_grad_at_forget.norm(dim=1).detach().cpu().numpy())
        batch.x.requires_grad_(False)
    return float(np.concatenate(norms).mean())


def occlusion_delta_auc(model, test_P_orig, test_P_zeroed):
    """Macro-ROC drop when P at forget node is zeroed at eval time.

    test_P_orig  : DataLoader where P at forget node is PRESENT
    test_P_zeroed: DataLoader where P at forget node is ZEROED
    Small Delta = model doesn't depend on forgotten P (good unlearning).
    """
    m_orig = evaluate_model_dual(model, test_P_orig, DEVICE)
    m_zero = evaluate_model_dual(model, test_P_zeroed, DEVICE)
    return {
        'auc_P_present': m_orig['macro_roc'],
        'auc_P_occluded': m_zero['macro_roc'],
        'delta_auc': m_orig['macro_roc'] - m_zero['macro_roc'],
    }


def reconstruction_attack(model, train_loader_with_P, test_loader_with_P,
                          epochs=40):
    """Train MLP decoder: node_emb_at_forget -> P_48_at_forget.

    Low MSE on test = model leaks forgotten P info.
    Expected: Original < GDGU ≈ Retrain.
    """
    # Get embeddings and target P for every graph
    def extract(model, loader):
        model.eval()
        embs, targets = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                # Get per-node embedding (pre-pool) via backbone's internal layers
                # For simplicity use pooled encode output; decoder learns from graph emb
                node_emb_pooled = model.backbone.encode(batch)     # [B, 2*hid]
                embs.append(node_emb_pooled.cpu())
                # Target: P_48 at forget node (unscaled ground truth)
                g_3d = batch.x.view(batch.num_graphs, n_nodes, 96)
                p_target = g_3d[:, FORGET_NODE_IDX, 48:].cpu()     # scaled
                targets.append(p_target)
        return torch.cat(embs), torch.cat(targets)

    emb_tr, p_tr = extract(model, train_loader_with_P)
    emb_te, p_te = extract(model, test_loader_with_P)

    dec = nn.Sequential(
        nn.Linear(emb_tr.shape[1], 128), nn.ReLU(),
        nn.Linear(128, 48),
    ).to(DEVICE)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    emb_tr_g, p_tr_g = emb_tr.to(DEVICE), p_tr.to(DEVICE)
    for ep in range(epochs):
        dec.train()
        opt.zero_grad()
        pred = dec(emb_tr_g)
        loss = loss_fn(pred, p_tr_g)
        loss.backward()
        opt.step()
    dec.eval()
    with torch.no_grad():
        pred_te = dec(emb_te.to(DEVICE)).cpu()
        mse_te = loss_fn(pred_te, p_te).item()
        # Population baseline: predict with training mean
        p_mean = p_tr.mean(dim=0, keepdim=True).expand_as(p_te)
        mse_baseline = loss_fn(p_mean, p_te).item()
    return {'mse_test': mse_te, 'mse_baseline': mse_baseline,
            'mse_ratio_to_baseline': mse_te / mse_baseline}


# --- Run L2 evals on all 3 models ---
# For attribution + occlusion, we need the eval loader with P still present
# (for Original/Retrain eval baseline). For GDGU, eval on P-occluded data
# (since that's the "post-forget" state).
# We compare all three models on the SAME test data (P at 814 zeroed)
# to be fair.

print("\n  -- Attribution (P-channel gradient norm at forget node) --")
# Eval on ds_orig (P present) to measure what the model WANTS to use;
# post-unlearn model should show ~0 gradient even when P is present
attr_orig = p_gradient_norm(model_orig, te_orig_ld)
attr_gdgu = p_gradient_norm(model_gdgu, te_orig_ld)
attr_rt   = p_gradient_norm(model_rt,   te_orig_ld)
print(f"  Original  : {attr_orig:.6f}")
print(f"  GDGU-dual : {attr_gdgu:.6f}")
print(f"  Retrain   : {attr_rt:.6f}")
results['L2_attribution'] = {
    'Original': attr_orig, 'GDGU_dual_A': attr_gdgu, 'Retrain_A': attr_rt,
}

print("\n  -- Occlusion Delta-AUC --")
occ_orig = occlusion_delta_auc(model_orig, te_orig_ld, te_fP_ld)
occ_gdgu = occlusion_delta_auc(model_gdgu, te_orig_ld, te_fP_ld)
occ_rt   = occlusion_delta_auc(model_rt,   te_orig_ld, te_fP_ld)
for name, r in [('Original', occ_orig), ('GDGU_dual', occ_gdgu), ('Retrain', occ_rt)]:
    print(f"  {name:10s}: AUC(P present)={r['auc_P_present']:.4f} "
          f"AUC(P occluded)={r['auc_P_occluded']:.4f} "
          f"Delta={r['delta_auc']:+.4f}")
results['L2_occlusion'] = {
    'Original': occ_orig, 'GDGU_dual_A': occ_gdgu, 'Retrain_A': occ_rt,
}

print("\n  -- Reconstruction attack (embed -> P_48) --")
# IMPORTANT: train decoder on ds_orig (P present). This measures whether
# the model's embedding carries info about true P even when we feed it
# graphs whose P is NOT zeroed.
r_orig = reconstruction_attack(model_orig, tr_orig_ld, te_orig_ld)
r_gdgu = reconstruction_attack(model_gdgu, tr_orig_ld, te_orig_ld)
r_rt   = reconstruction_attack(model_rt,   tr_orig_ld, te_orig_ld)
for name, r in [('Original', r_orig), ('GDGU_dual', r_gdgu), ('Retrain', r_rt)]:
    print(f"  {name:10s}: MSE_test={r['mse_test']:.4f}  "
          f"MSE_baseline={r['mse_baseline']:.4f}  "
          f"ratio={r['mse_ratio_to_baseline']:.3f}")
results['L2_reconstruction'] = {
    'Original': r_orig, 'GDGU_dual_A': r_gdgu, 'Retrain_A': r_rt,
}

# ================================================================
#  Save summary
# ================================================================
with open(os.path.join(OUT_DIR, 'smoke_A_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: (
        float(x) if hasattr(x, 'item') else
        x.tolist() if hasattr(x, 'tolist') else str(x)))
print(f"\n[SMOKE-A] Results saved to {OUT_DIR}/smoke_A_results.json")
print("[SMOKE-A] Done.")
