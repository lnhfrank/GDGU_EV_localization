"""Experiment runner: single trial (one backbone x scenario x seed)."""

import copy
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.models import MODEL_CLASSES, DualChannel_Graph, AuxWrapper
from src.data import (build_graphs, build_graphs_route_a,
                      stratified_split_multilabel, fit_scaler,
                      fit_graph_feat_scaler)
from src.training import (kaiming_init, get_pos_weights, train_model,
                          train_model_joint, evaluate_model,
                          evaluate_aux_acc, compute_mia_auc,
                          train_model_dual, evaluate_model_dual,
                          compute_mia_auc_dual)
from src.unlearning import (gdgu_feature_unlearn, gif_unlearn, idea_unlearn,
                            gdgu_dual_unlearn, gif_dual_unlearn,
                            idea_dual_unlearn)
from src.privacy import (L2_a_integrated_gradients,
                         L2_b_occlusion_delta_auc)


def _peak_memory_mb(device):
    """Return peak GPU memory in MB since last reset, or 0 for CPU."""
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return 0.0


def run_single_trial(backbone_name, scen_key, scen_val, seed,
                     all_x, all_y, edge_index_t, config, device):
    """Run Original + GDGU + GIF + IDEA + Retrain for one (backbone, scenario, seed).

    Returns:
        list of 5 result dicts,
        dict of epoch_logs keyed by method name.
    """
    forget_idx = scen_val['forget_indices']
    scen_label = scen_val['label']
    forget_label_idx = scen_val.get('forget_label_indices')  # [0], [0,1], or [0,1,2]
    n_nodes, n_feat = all_x.shape[1], all_x.shape[2]

    print(f"\n{'='*70}")
    print(f"  {backbone_name} | {scen_label} | seed={seed}")
    print(f"{'='*70}")

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Split
    idx_train, idx_val, idx_test = stratified_split_multilabel(
        all_y, config['test_size'], config['val_ratio'], seed)
    print(f"  Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    # pos_weight
    pw = get_pos_weights(all_y[idx_train], device)
    criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=pw)

    # Scaler
    scaler = fit_scaler(all_x, idx_train)

    # Build original graphs
    bs = config['batch_size']
    train_graphs = build_graphs(all_x[idx_train], all_y[idx_train], edge_index_t, n_nodes, n_feat, scaler)
    val_graphs = build_graphs(all_x[idx_val], all_y[idx_val], edge_index_t, n_nodes, n_feat, scaler)
    test_graphs = build_graphs(all_x[idx_test], all_y[idx_test], edge_index_t, n_nodes, n_feat, scaler)
    train_loader = DataLoader(train_graphs, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=bs)
    test_loader = DataLoader(test_graphs, batch_size=bs)

    # Build unlearned graphs (forget nodes zeroed)
    train_graphs_unl = build_graphs(all_x[idx_train], all_y[idx_train], edge_index_t, n_nodes, n_feat, scaler, forget_idx)
    val_graphs_unl = build_graphs(all_x[idx_val], all_y[idx_val], edge_index_t, n_nodes, n_feat, scaler, forget_idx)
    test_graphs_unl = build_graphs(all_x[idx_test], all_y[idx_test], edge_index_t, n_nodes, n_feat, scaler, forget_idx)
    train_loader_unl = DataLoader(train_graphs_unl, batch_size=bs, shuffle=True)
    val_loader_unl = DataLoader(val_graphs_unl, batch_size=bs)
    test_loader_unl = DataLoader(test_graphs_unl, batch_size=bs)

    # Non-shuffled loaders for gradient computation
    train_loader_noshuffle = DataLoader(train_graphs, batch_size=bs, shuffle=False)
    train_loader_unl_noshuffle = DataLoader(train_graphs_unl, batch_size=bs, shuffle=False)

    results = []
    all_epoch_logs = {}

    # ── (A) Original Model ──
    ModelClass = MODEL_CLASSES[backbone_name]
    model_orig = ModelClass(in_dim=n_feat, hid_dim=config['hidden_dim'],
                            out_dim=config['out_dim'], n_layers=config['n_layers'],
                            dropout=config['dropout']).to(device)
    kaiming_init(model_orig)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model_orig, val_metric_orig, logs_orig = train_model(
        model_orig, train_loader, val_loader, device,
        epochs=config['epochs'], lr=config['lr'],
        weight_decay=config['weight_decay'], patience=config['patience'],
        pos_weights=pw)
    time_orig = time.time() - t0
    mem_orig = _peak_memory_mb(device)
    all_epoch_logs['Original'] = logs_orig

    res_o = evaluate_model(model_orig, test_loader, device)
    print(f"  [Original]  ExMatch={res_o['exact_match']:.4f}  "
          f"MacroROC={res_o['macro_roc']:.4f}  MacroF1={res_o['macro_f1']:.4f}  "
          f"Time={time_orig:.1f}s  Mem={mem_orig:.1f}MB")

    mia_orig = {'mia_forget': np.nan, 'mia_retain': np.nan, 'mia_overall': np.nan}
    results.append(_build_result(backbone_name, scen_key, seed, 'Original',
                                 res_o, mia_orig, time_orig, mem_orig,
                                 forget_label_idx=forget_label_idx))

    # ── (B) GDGU ──
    model_gdgu = copy.deepcopy(model_orig)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model_gdgu = gdgu_feature_unlearn(
        model_gdgu, train_loader_noshuffle, train_loader_unl_noshuffle,
        val_loader_unl, criterion_weighted, device,
        damp=config['gdgu_damp'], max_norm=config['gdgu_max_norm'],
        finetune_epochs=config['gdgu_finetune'], pos_weights=pw)
    time_gdgu = time.time() - t0
    mem_gdgu = _peak_memory_mb(device)

    res_g = evaluate_model(model_gdgu, test_loader_unl, device)
    mia_gdgu = compute_mia_auc(model_gdgu, train_loader_unl, test_loader_unl,
                                device, pw, forget_label_idx=forget_label_idx)
    print(f"  [GDGU]      ExMatch={res_g['exact_match']:.4f}  "
          f"MacroROC={res_g['macro_roc']:.4f}  MacroF1={res_g['macro_f1']:.4f}  "
          f"MIA_f={mia_gdgu['mia_forget']:.4f}  MIA_r={mia_gdgu['mia_retain']:.4f}  "
          f"Time={time_gdgu:.1f}s  Mem={mem_gdgu:.1f}MB")

    results.append(_build_result(backbone_name, scen_key, seed, 'GDGU',
                                 res_g, mia_gdgu, time_gdgu, mem_gdgu,
                                 forget_label_idx=forget_label_idx))

    # ── (C) GIF ──
    model_gif = copy.deepcopy(model_orig)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model_gif = gif_unlearn(
        model_gif, train_loader_noshuffle, train_loader_unl_noshuffle,
        criterion_weighted, device,
        iteration=config.get('gif_iteration', 50),
        damp=config.get('gif_damp', 0.01),
        scale=config.get('gif_scale', 50.0),
        max_batches=config.get('gif_max_batches'))
    time_gif = time.time() - t0
    mem_gif = _peak_memory_mb(device)

    res_gif = evaluate_model(model_gif, test_loader_unl, device)
    mia_gif = compute_mia_auc(model_gif, train_loader_unl, test_loader_unl,
                               device, pw, forget_label_idx=forget_label_idx)
    print(f"  [GIF]       ExMatch={res_gif['exact_match']:.4f}  "
          f"MacroROC={res_gif['macro_roc']:.4f}  MacroF1={res_gif['macro_f1']:.4f}  "
          f"MIA_f={mia_gif['mia_forget']:.4f}  MIA_r={mia_gif['mia_retain']:.4f}  "
          f"Time={time_gif:.1f}s  Mem={mem_gif:.1f}MB")

    results.append(_build_result(backbone_name, scen_key, seed, 'GIF',
                                 res_gif, mia_gif, time_gif, mem_gif,
                                 forget_label_idx=forget_label_idx))

    # ── (D) IDEA ──
    model_idea = copy.deepcopy(model_orig)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model_idea = idea_unlearn(
        model_idea, train_loader_noshuffle, train_loader_unl_noshuffle,
        val_loader_unl, criterion_weighted, device,
        iteration=config.get('gif_iteration', 50),
        damp=config.get('gif_damp', 0.01),
        scale=config.get('gif_scale', 50.0),
        finetune_epochs=config.get('idea_finetune', 25),
        pos_weights=pw,
        max_batches=config.get('gif_max_batches'))
    time_idea = time.time() - t0
    mem_idea = _peak_memory_mb(device)

    res_idea = evaluate_model(model_idea, test_loader_unl, device)
    mia_idea = compute_mia_auc(model_idea, train_loader_unl, test_loader_unl,
                                device, pw, forget_label_idx=forget_label_idx)
    print(f"  [IDEA]      ExMatch={res_idea['exact_match']:.4f}  "
          f"MacroROC={res_idea['macro_roc']:.4f}  MacroF1={res_idea['macro_f1']:.4f}  "
          f"MIA_f={mia_idea['mia_forget']:.4f}  MIA_r={mia_idea['mia_retain']:.4f}  "
          f"Time={time_idea:.1f}s  Mem={mem_idea:.1f}MB")

    results.append(_build_result(backbone_name, scen_key, seed, 'IDEA',
                                 res_idea, mia_idea, time_idea, mem_idea,
                                 forget_label_idx=forget_label_idx))

    # ── (E) Retrain from scratch ──
    # Re-seed so Retrain gets the same kaiming_init as Original.
    # Without this, the RNG state has drifted through Original training + GDGU,
    # giving Retrain an uncontrolled initialization that often collapses.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_retrain = ModelClass(in_dim=n_feat, hid_dim=config['hidden_dim'],
                               out_dim=config['out_dim'], n_layers=config['n_layers'],
                               dropout=config['dropout']).to(device)
    kaiming_init(model_retrain)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model_retrain, val_metric_rt, logs_retrain = train_model(
        model_retrain, train_loader_unl, val_loader_unl, device,
        epochs=config['epochs'], lr=config['lr'],
        weight_decay=config['weight_decay'], patience=config['patience'],
        pos_weights=pw)
    time_retrain = time.time() - t0
    mem_retrain = _peak_memory_mb(device)
    all_epoch_logs['Retrain'] = logs_retrain

    res_r = evaluate_model(model_retrain, test_loader_unl, device)
    mia_retrain = compute_mia_auc(model_retrain, train_loader_unl, test_loader_unl,
                                   device, pw, forget_label_idx=forget_label_idx)
    print(f"  [Retrain]   ExMatch={res_r['exact_match']:.4f}  "
          f"MacroROC={res_r['macro_roc']:.4f}  MacroF1={res_r['macro_f1']:.4f}  "
          f"MIA_f={mia_retrain['mia_forget']:.4f}  MIA_r={mia_retrain['mia_retain']:.4f}  "
          f"Time={time_retrain:.1f}s  Mem={mem_retrain:.1f}MB")
    print(f"  Speedup vs Retrain: "
          f"GDGU {time_retrain / max(time_gdgu, 1e-6):.1f}x  "
          f"GIF {time_retrain / max(time_gif, 1e-6):.1f}x  "
          f"IDEA {time_retrain / max(time_idea, 1e-6):.1f}x")

    results.append(_build_result(backbone_name, scen_key, seed, 'Retrain',
                                 res_r, mia_retrain, time_retrain, mem_retrain,
                                 forget_label_idx=forget_label_idx))

    # Cleanup
    del model_orig, model_gdgu, model_gif, model_idea, model_retrain
    torch.cuda.empty_cache()

    return results, all_epoch_logs


def _build_result(backbone, scenario, seed, method, eval_dict, mia_dict,
                   elapsed, mem_mb, forget_label_idx=None):
    """Build a flat result dict from evaluation output (dynamic EVCS count)."""
    row = {
        'Backbone': backbone, 'Scenario': scenario, 'Seed': seed,
        'Method': method,
        'ExMatch': eval_dict['exact_match'],
        'Hamming_Acc': eval_dict['hamming_acc'],
        'Macro_ROC': eval_dict['macro_roc'],
        'Macro_F1': eval_dict['macro_f1'],
    }
    per_roc = eval_dict['per_roc']
    per_f1 = eval_dict['per_f1']
    for i, roc in enumerate(per_roc):
        row[f'ROC_EVCS{i+1}'] = roc
    for i, f1 in enumerate(per_f1):
        row[f'F1_EVCS{i+1}'] = f1

    # Forget / retain grouped metrics
    if forget_label_idx is not None:
        n_labels = len(per_f1)
        retain_idx = [i for i in range(n_labels) if i not in forget_label_idx]
        row['F1_forget'] = float(np.mean([per_f1[i] for i in forget_label_idx]))
        row['ROC_forget'] = float(np.mean([per_roc[i] for i in forget_label_idx]))
        if retain_idx:
            row['F1_retain'] = float(np.mean([per_f1[i] for i in retain_idx]))
            row['ROC_retain'] = float(np.mean([per_roc[i] for i in retain_idx]))
        else:
            row['F1_retain'] = np.nan
            row['ROC_retain'] = np.nan
    else:
        row['F1_forget'] = np.nan
        row['ROC_forget'] = np.nan
        row['F1_retain'] = np.nan
        row['ROC_retain'] = np.nan

    # MIA split (OpenGU-aligned)
    row['MIA_forget'] = mia_dict['mia_forget']
    row['MIA_retain'] = mia_dict['mia_retain']
    row['MIA_AUC'] = mia_dict['mia_overall']  # backward compatible
    row['Time'] = elapsed
    row['Peak_Memory_MB'] = mem_mb
    return row


# ======================================================================
#  Route A trial runner (V6.0: single-head + aux attack-type head)
# ======================================================================


def run_single_trial_route_a(backbone_name, scen_key, scen_val, seed,
                              data_dict, config, device):
    """Route A trial: Original + GDGU + GIF + IDEA + Retrain-A with L2 privacy.

    Uses AuxWrapper for joint loc + attack-type training. Unlearning methods
    operate on the loc head only (aux head is frozen). Privacy evaluation
    includes L1 (utility), L2-a (IG), L2-b (occlusion delta-AUC), L2-e
    (aux acc), and MIA.

    Args:
        data_dict: output of load_evcs_data + augment_route_a.
        config: flat dict with training/gdgu/gif/route_a parameters.

    Returns:
        list of 5 result dicts, dict of epoch_logs.
    """
    forget_node_indices = scen_val['forget_node_indices']
    forget_label_idx = scen_val['forget_label_indices']
    scen_label = scen_val['label']

    all_x_V = data_dict['all_x']
    all_x_P = data_dict['all_x_P']
    all_y = data_dict['all_y']
    all_attack_type = data_dict['all_attack_type']
    edge_index = data_dict['edge_index']
    n_nodes = data_dict['n_nodes']
    n_evcs = data_dict['n_evcs']

    ra = config.get('route_a', {})
    gamma = ra.get('gamma', 0.5)
    n_attack_types = ra.get('n_attack_types', 5)
    aux_hidden = ra.get('aux_hidden', 64)
    ig_steps = ra.get('ig_steps', 50)
    lr_gat = ra.get('lr_gat', 1e-4)

    in_dim = 96
    lr = lr_gat if backbone_name == 'GAT' else config['lr']

    print(f"\n{'='*70}")
    print(f"  [Route A] {backbone_name} | {scen_label} | seed={seed}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    idx_train, idx_val, idx_test = stratified_split_multilabel(
        all_y, config['test_size'], config['val_ratio'], seed)
    print(f"  Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    scaler_V = fit_scaler(all_x_V, idx_train)
    scaler_P = fit_scaler(all_x_P, idx_train)

    pw = get_pos_weights(all_y[idx_train], device)
    criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=pw)

    type_counts = Counter(all_attack_type[idx_train])
    n_train = len(idx_train)
    tw = torch.tensor([n_train / (n_attack_types * max(type_counts.get(c, 1), 1))
                        for c in range(n_attack_types)], dtype=torch.float).to(device)

    bs = config['batch_size']

    def _bg(idx, forget=None):
        return build_graphs_route_a(
            all_x_V, all_x_P, all_y, all_attack_type,
            edge_index, n_nodes, idx, scaler_V, scaler_P,
            forget_P_at_node=forget)

    train_full = _bg(idx_train)
    val_full = _bg(idx_val)
    test_full = _bg(idx_test)
    train_loader = DataLoader(train_full, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_full, batch_size=bs)
    test_loader = DataLoader(test_full, batch_size=bs)

    train_occ = _bg(idx_train, forget=forget_node_indices)
    val_occ = _bg(idx_val, forget=forget_node_indices)
    test_occ = _bg(idx_test, forget=forget_node_indices)
    train_loader_occ = DataLoader(train_occ, batch_size=bs, shuffle=True)
    val_loader_occ = DataLoader(val_occ, batch_size=bs)
    test_loader_occ = DataLoader(test_occ, batch_size=bs)

    train_loader_noshuffle = DataLoader(train_full, batch_size=bs, shuffle=False)
    train_loader_occ_noshuffle = DataLoader(train_occ, batch_size=bs, shuffle=False)

    results = []
    all_epoch_logs = {}

    def _make_model():
        BackboneClass = MODEL_CLASSES[backbone_name]
        backbone = BackboneClass(
            in_dim=in_dim, hid_dim=config['hidden_dim'],
            out_dim=n_evcs, n_layers=config['n_layers'],
            dropout=config['dropout'])
        model = AuxWrapper(backbone, n_types=n_attack_types,
                           aux_hidden=aux_hidden).to(device)
        kaiming_init(model)
        return model

    def _eval_l2(model, native_test_loader):
        """Compute L1 + L2 metrics for a given model."""
        res = evaluate_model(model, native_test_loader, device)
        aux = evaluate_aux_acc(model, native_test_loader, device)
        l2a = L2_a_integrated_gradients(
            model, test_loader, forget_node_indices, forget_label_idx,
            n_nodes, device, n_steps=ig_steps)
        l2b = L2_b_occlusion_delta_auc(
            model, test_loader, test_loader_occ, device)
        return res, aux, l2a, l2b

    # ── (A) Original: joint training on full data ──
    model_orig = _make_model()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model_orig, _, logs_orig = train_model_joint(
        model_orig, train_loader, val_loader, device,
        epochs=config['epochs'], lr=lr, weight_decay=config['weight_decay'],
        patience=config['patience'], pos_weights=pw, type_weights=tw,
        gamma=gamma, scheduler_patience=config['scheduler_patience'])
    time_orig = time.time() - t0
    mem_orig = _peak_memory_mb(device)
    all_epoch_logs['Original'] = logs_orig

    res_o, aux_o, l2a_o, l2b_o = _eval_l2(model_orig, test_loader)
    mia_orig = {'mia_forget': np.nan, 'mia_retain': np.nan, 'mia_overall': np.nan}
    print(f"  [Original]  ExMatch={res_o['exact_match']:.4f}  "
          f"MacroROC={res_o['macro_roc']:.4f}  MacroF1={res_o['macro_f1']:.4f}  "
          f"AuxAcc={aux_o:.4f}  L2b={l2b_o['delta_auc']:.4f}  "
          f"Time={time_orig:.1f}s  Mem={mem_orig:.1f}MB")
    results.append(_build_result_route_a(
        backbone_name, scen_key, seed, 'Original',
        res_o, mia_orig, time_orig, mem_orig,
        forget_label_idx=forget_label_idx,
        aux_acc=aux_o, l2a=l2a_o, l2b=l2b_o))

    # ── (B) GDGU ──
    model_gdgu = copy.deepcopy(model_orig)
    model_gdgu.freeze_aux()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model_gdgu = gdgu_feature_unlearn(
        model_gdgu, train_loader_noshuffle, train_loader_occ_noshuffle,
        val_loader_occ, criterion_weighted, device,
        damp=config['gdgu_damp'], max_norm=config['gdgu_max_norm'],
        finetune_epochs=config['gdgu_finetune'], pos_weights=pw)
    time_gdgu = time.time() - t0
    mem_gdgu = _peak_memory_mb(device)

    res_g, aux_g, l2a_g, l2b_g = _eval_l2(model_gdgu, test_loader_occ)
    mia_gdgu = compute_mia_auc(model_gdgu, train_loader_occ, test_loader_occ,
                                device, pw, forget_label_idx=forget_label_idx)
    print(f"  [GDGU]      ExMatch={res_g['exact_match']:.4f}  "
          f"MacroROC={res_g['macro_roc']:.4f}  MacroF1={res_g['macro_f1']:.4f}  "
          f"AuxAcc={aux_g:.4f}  L2b={l2b_g['delta_auc']:.4f}  "
          f"MIA_f={mia_gdgu['mia_forget']:.4f}  "
          f"Time={time_gdgu:.1f}s  Mem={mem_gdgu:.1f}MB")
    results.append(_build_result_route_a(
        backbone_name, scen_key, seed, 'GDGU',
        res_g, mia_gdgu, time_gdgu, mem_gdgu,
        forget_label_idx=forget_label_idx,
        aux_acc=aux_g, l2a=l2a_g, l2b=l2b_g))

    # ── (C) GIF ──
    model_gif = copy.deepcopy(model_orig)
    model_gif.freeze_aux()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model_gif = gif_unlearn(
        model_gif, train_loader_noshuffle, train_loader_occ_noshuffle,
        criterion_weighted, device,
        iteration=config.get('gif_iteration', 50),
        damp=config.get('gif_damp', 0.01),
        scale=config.get('gif_scale', 50.0),
        max_batches=config.get('gif_max_batches'))
    time_gif = time.time() - t0
    mem_gif = _peak_memory_mb(device)

    res_gif, aux_gif, l2a_gif, l2b_gif = _eval_l2(model_gif, test_loader_occ)
    mia_gif = compute_mia_auc(model_gif, train_loader_occ, test_loader_occ,
                               device, pw, forget_label_idx=forget_label_idx)
    print(f"  [GIF]       ExMatch={res_gif['exact_match']:.4f}  "
          f"MacroROC={res_gif['macro_roc']:.4f}  MacroF1={res_gif['macro_f1']:.4f}  "
          f"AuxAcc={aux_gif:.4f}  L2b={l2b_gif['delta_auc']:.4f}  "
          f"MIA_f={mia_gif['mia_forget']:.4f}  "
          f"Time={time_gif:.1f}s  Mem={mem_gif:.1f}MB")
    results.append(_build_result_route_a(
        backbone_name, scen_key, seed, 'GIF',
        res_gif, mia_gif, time_gif, mem_gif,
        forget_label_idx=forget_label_idx,
        aux_acc=aux_gif, l2a=l2a_gif, l2b=l2b_gif))

    # ── (D) IDEA ──
    model_idea = copy.deepcopy(model_orig)
    model_idea.freeze_aux()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model_idea = idea_unlearn(
        model_idea, train_loader_noshuffle, train_loader_occ_noshuffle,
        val_loader_occ, criterion_weighted, device,
        iteration=config.get('gif_iteration', 50),
        damp=config.get('gif_damp', 0.01),
        scale=config.get('gif_scale', 50.0),
        finetune_epochs=config.get('idea_finetune', 25),
        pos_weights=pw,
        max_batches=config.get('gif_max_batches'))
    time_idea = time.time() - t0
    mem_idea = _peak_memory_mb(device)

    res_idea, aux_idea, l2a_idea, l2b_idea = _eval_l2(model_idea, test_loader_occ)
    mia_idea = compute_mia_auc(model_idea, train_loader_occ, test_loader_occ,
                                device, pw, forget_label_idx=forget_label_idx)
    print(f"  [IDEA]      ExMatch={res_idea['exact_match']:.4f}  "
          f"MacroROC={res_idea['macro_roc']:.4f}  MacroF1={res_idea['macro_f1']:.4f}  "
          f"AuxAcc={aux_idea:.4f}  L2b={l2b_idea['delta_auc']:.4f}  "
          f"MIA_f={mia_idea['mia_forget']:.4f}  "
          f"Time={time_idea:.1f}s  Mem={mem_idea:.1f}MB")
    results.append(_build_result_route_a(
        backbone_name, scen_key, seed, 'IDEA',
        res_idea, mia_idea, time_idea, mem_idea,
        forget_label_idx=forget_label_idx,
        aux_acc=aux_idea, l2a=l2a_idea, l2b=l2b_idea))

    # ── (E) Retrain-A: joint training on occluded data ──
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_retrain = _make_model()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model_retrain, _, logs_rt = train_model_joint(
        model_retrain, train_loader_occ, val_loader_occ, device,
        epochs=config['epochs'], lr=lr, weight_decay=config['weight_decay'],
        patience=config['patience'], pos_weights=pw, type_weights=tw,
        gamma=gamma, scheduler_patience=config['scheduler_patience'])
    time_retrain = time.time() - t0
    mem_retrain = _peak_memory_mb(device)
    all_epoch_logs['Retrain'] = logs_rt

    res_r, aux_r, l2a_r, l2b_r = _eval_l2(model_retrain, test_loader_occ)
    mia_retrain = compute_mia_auc(model_retrain, train_loader_occ, test_loader_occ,
                                   device, pw, forget_label_idx=forget_label_idx)
    print(f"  [Retrain-A] ExMatch={res_r['exact_match']:.4f}  "
          f"MacroROC={res_r['macro_roc']:.4f}  MacroF1={res_r['macro_f1']:.4f}  "
          f"AuxAcc={aux_r:.4f}  L2b={l2b_r['delta_auc']:.4f}  "
          f"MIA_f={mia_retrain['mia_forget']:.4f}  "
          f"Time={time_retrain:.1f}s  Mem={mem_retrain:.1f}MB")
    print(f"  Speedup vs Retrain: "
          f"GDGU {time_retrain / max(time_gdgu, 1e-6):.1f}x  "
          f"GIF {time_retrain / max(time_gif, 1e-6):.1f}x  "
          f"IDEA {time_retrain / max(time_idea, 1e-6):.1f}x")
    results.append(_build_result_route_a(
        backbone_name, scen_key, seed, 'Retrain',
        res_r, mia_retrain, time_retrain, mem_retrain,
        forget_label_idx=forget_label_idx,
        aux_acc=aux_r, l2a=l2a_r, l2b=l2b_r))

    del model_orig, model_gdgu, model_gif, model_idea, model_retrain
    torch.cuda.empty_cache()

    return results, all_epoch_logs


def _build_result_route_a(backbone, scenario, seed, method, eval_dict, mia_dict,
                           elapsed, mem_mb, forget_label_idx=None,
                           aux_acc=None, l2a=None, l2b=None):
    """Flat result dict for Route A trial (adds L2 privacy metrics)."""
    row = _build_result(backbone, scenario, seed, method, eval_dict, mia_dict,
                        elapsed, mem_mb, forget_label_idx=forget_label_idx)
    row['Aux_Acc'] = float(aux_acc) if aux_acc is not None else np.nan
    if l2a is not None:
        row['L2a_IG_mean'] = l2a['mean']
        row['L2a_IG_std'] = l2a['std']
    else:
        row['L2a_IG_mean'] = np.nan
        row['L2a_IG_std'] = np.nan
    if l2b is not None:
        row['L2b_delta_auc'] = l2b['delta_auc']
        row['L2b_auc_P_present'] = l2b['auc_P_present']
        row['L2b_auc_P_occluded'] = l2b['auc_P_occluded']
    else:
        row['L2b_delta_auc'] = np.nan
        row['L2b_auc_P_present'] = np.nan
        row['L2b_auc_P_occluded'] = np.nan
    return row


# ======================================================================
#  Dual-Channel trial runner (Scheme A, V5.1: full 5-method matrix)
# ======================================================================
#
# Runs Original + GDGU-dual + GIF-dual + IDEA-dual + Retrain-dual with
# strict channel isolation.  All unlearning methods freeze the graph
# channel (graph_mlp + det_head) so that only the node channel is
# updated — detection capability is preserved by construction.
# ======================================================================


def run_single_trial_dual(backbone_name, scen_key, scen_val, seed,
                          all_x, all_V, all_y, edge_index_t, config, device):
    """Dual-channel trial: Original + GDGU + GIF + IDEA + Retrain.

    Uses DualChannel_Graph with strict channel isolation. Retrain baseline
    also uses dual-channel architecture for fair comparison.
    """
    from src.data import build_graph_features

    forget_idx = scen_val['forget_indices']
    scen_label = scen_val['label']
    forget_label_idx = scen_val.get('forget_label_indices')
    n_nodes, n_feat = all_x.shape[1], all_x.shape[2]

    print(f"\n{'='*70}")
    print(f"  [DUAL] {backbone_name} | {scen_label} | seed={seed}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Split
    idx_train, idx_val, idx_test = stratified_split_multilabel(
        all_y, config['test_size'], config['val_ratio'], seed)
    print(f"  Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    # pos_weight for localization (multi-label)
    pw = get_pos_weights(all_y[idx_train], device)
    criterion_loc = nn.BCEWithLogitsLoss(pos_weight=pw)

    # Scalers: node features + graph features
    scaler = fit_scaler(all_x, idx_train)
    graph_feat_all = build_graph_features(all_V, edge_index_t)  # [G, 120]
    gf_scaler = fit_graph_feat_scaler(graph_feat_all, idx_train)
    gf_dim = graph_feat_all.shape[1]
    print(f"  Graph-level feature dim: {gf_dim}")

    # Build graphs (original + unlearned), with graph_feat attached
    bs = config['batch_size']
    def _bg(idx, forget=None):
        return build_graphs(
            all_x[idx], all_y[idx], edge_index_t, n_nodes, n_feat,
            scaler=scaler, forget_indices=forget,
            graph_feat=graph_feat_all[idx], graph_feat_scaler=gf_scaler)

    train_graphs = _bg(idx_train)
    val_graphs = _bg(idx_val)
    test_graphs = _bg(idx_test)
    train_loader = DataLoader(train_graphs, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=bs)
    test_loader = DataLoader(test_graphs, batch_size=bs)

    train_graphs_unl = _bg(idx_train, forget=forget_idx)
    val_graphs_unl = _bg(idx_val, forget=forget_idx)
    test_graphs_unl = _bg(idx_test, forget=forget_idx)
    train_loader_unl = DataLoader(train_graphs_unl, batch_size=bs, shuffle=True)
    val_loader_unl = DataLoader(val_graphs_unl, batch_size=bs)
    test_loader_unl = DataLoader(test_graphs_unl, batch_size=bs)

    # Non-shuffled for gradient computation
    train_loader_noshuffle = DataLoader(train_graphs, batch_size=bs, shuffle=False)
    train_loader_unl_noshuffle = DataLoader(train_graphs_unl, batch_size=bs, shuffle=False)

    dc = config.get('dual_channel', {})
    alpha = dc.get('alpha', 1.0)
    beta = dc.get('beta', 1.0)
    gh = dc.get('graph_mlp_hidden', 64)
    go = dc.get('graph_mlp_out', 32)

    results = []
    all_epoch_logs = {}

    def _new_model():
        m = DualChannel_Graph(
            backbone_name=backbone_name,
            in_dim=n_feat, hid_dim=config['hidden_dim'],
            out_dim=config['out_dim'], n_layers=config['n_layers'],
            dropout=config['dropout'],
            graph_feat_dim=gf_dim,
            graph_mlp_hidden=gh, graph_mlp_out=go,
        ).to(device)
        kaiming_init(m)
        return m

    # ── (A) Original ──
    model_orig = _new_model()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model_orig, val_metric_orig, logs_orig = train_model_dual(
        model_orig, train_loader, val_loader, device,
        epochs=config['epochs'], lr=config['lr'],
        weight_decay=config['weight_decay'], patience=config['patience'],
        pos_weights=pw, alpha=alpha, beta=beta,
        scheduler_patience=config['scheduler_patience'])
    time_orig = time.time() - t0
    mem_orig = _peak_memory_mb(device)
    all_epoch_logs['Original'] = logs_orig

    res_o = evaluate_model_dual(model_orig, test_loader, device)
    print(f"  [Original]  ExMatch={res_o['exact_match']:.4f}  "
          f"MacroROC={res_o['macro_roc']:.4f}  MacroF1={res_o['macro_f1']:.4f}  "
          f"DetAcc={res_o['det_acc']:.4f}  DetAUC={res_o['det_auc']:.4f}  "
          f"Time={time_orig:.1f}s  Mem={mem_orig:.1f}MB")

    mia_orig = {'mia_forget': np.nan, 'mia_retain': np.nan, 'mia_overall': np.nan}
    results.append(_build_result_dual(backbone_name, scen_key, seed, 'Original',
                                       res_o, mia_orig, time_orig, mem_orig,
                                       forget_label_idx=forget_label_idx))

    # ── (B) GDGU-dual ──
    model_gdgu = copy.deepcopy(model_orig)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model_gdgu = gdgu_dual_unlearn(
        model_gdgu, train_loader_noshuffle, train_loader_unl_noshuffle,
        val_loader_unl, criterion_loc, device,
        damp=config['gdgu_damp'], max_norm=config['gdgu_max_norm'],
        finetune_epochs=config['gdgu_finetune'],
        alpha=alpha, beta=beta, pos_weights=pw)
    time_gdgu = time.time() - t0
    mem_gdgu = _peak_memory_mb(device)

    res_g = evaluate_model_dual(model_gdgu, test_loader_unl, device)
    mia_gdgu = compute_mia_auc_dual(model_gdgu, train_loader_unl, test_loader_unl,
                                     device, pw, forget_label_idx=forget_label_idx)
    print(f"  [GDGU-dual] ExMatch={res_g['exact_match']:.4f}  "
          f"MacroROC={res_g['macro_roc']:.4f}  MacroF1={res_g['macro_f1']:.4f}  "
          f"DetAcc={res_g['det_acc']:.4f}  "
          f"MIA_f={mia_gdgu['mia_forget']:.4f}  MIA_r={mia_gdgu['mia_retain']:.4f}  "
          f"Time={time_gdgu:.1f}s  Mem={mem_gdgu:.1f}MB")

    results.append(_build_result_dual(backbone_name, scen_key, seed, 'GDGU',
                                       res_g, mia_gdgu, time_gdgu, mem_gdgu,
                                       forget_label_idx=forget_label_idx))

    # ── (C) GIF-dual ──
    model_gif = copy.deepcopy(model_orig)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model_gif = gif_dual_unlearn(
        model_gif, train_loader_noshuffle, train_loader_unl_noshuffle,
        criterion_loc, device,
        iteration=config.get('gif_iteration', 50),
        damp=config.get('gif_damp', 0.01),
        scale=config.get('gif_scale', 50.0),
        max_batches=config.get('gif_max_batches'))
    time_gif = time.time() - t0
    mem_gif = _peak_memory_mb(device)

    res_gif = evaluate_model_dual(model_gif, test_loader_unl, device)
    mia_gif = compute_mia_auc_dual(model_gif, train_loader_unl, test_loader_unl,
                                    device, pw, forget_label_idx=forget_label_idx)
    print(f"  [GIF-dual]  ExMatch={res_gif['exact_match']:.4f}  "
          f"MacroROC={res_gif['macro_roc']:.4f}  MacroF1={res_gif['macro_f1']:.4f}  "
          f"DetAcc={res_gif['det_acc']:.4f}  "
          f"MIA_f={mia_gif['mia_forget']:.4f}  MIA_r={mia_gif['mia_retain']:.4f}  "
          f"Time={time_gif:.1f}s  Mem={mem_gif:.1f}MB")

    results.append(_build_result_dual(backbone_name, scen_key, seed, 'GIF',
                                       res_gif, mia_gif, time_gif, mem_gif,
                                       forget_label_idx=forget_label_idx))

    # ── (D) IDEA-dual ──
    model_idea = copy.deepcopy(model_orig)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model_idea = idea_dual_unlearn(
        model_idea, train_loader_noshuffle, train_loader_unl_noshuffle,
        val_loader_unl, criterion_loc, device,
        iteration=config.get('gif_iteration', 50),
        damp=config.get('gif_damp', 0.01),
        scale=config.get('gif_scale', 50.0),
        finetune_epochs=config.get('idea_finetune', 25),
        alpha=alpha, beta=beta, pos_weights=pw,
        max_batches=config.get('gif_max_batches'))
    time_idea = time.time() - t0
    mem_idea = _peak_memory_mb(device)

    res_idea = evaluate_model_dual(model_idea, test_loader_unl, device)
    mia_idea = compute_mia_auc_dual(model_idea, train_loader_unl, test_loader_unl,
                                     device, pw, forget_label_idx=forget_label_idx)
    print(f"  [IDEA-dual] ExMatch={res_idea['exact_match']:.4f}  "
          f"MacroROC={res_idea['macro_roc']:.4f}  MacroF1={res_idea['macro_f1']:.4f}  "
          f"DetAcc={res_idea['det_acc']:.4f}  "
          f"MIA_f={mia_idea['mia_forget']:.4f}  MIA_r={mia_idea['mia_retain']:.4f}  "
          f"Time={time_idea:.1f}s  Mem={mem_idea:.1f}MB")

    results.append(_build_result_dual(backbone_name, scen_key, seed, 'IDEA',
                                       res_idea, mia_idea, time_idea, mem_idea,
                                       forget_label_idx=forget_label_idx))

    # ── (E) Retrain from scratch (dual-channel) ──
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_retrain = _new_model()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    model_retrain, val_metric_rt, logs_retrain = train_model_dual(
        model_retrain, train_loader_unl, val_loader_unl, device,
        epochs=config['epochs'], lr=config['lr'],
        weight_decay=config['weight_decay'], patience=config['patience'],
        pos_weights=pw, alpha=alpha, beta=beta,
        scheduler_patience=config['scheduler_patience'])
    time_retrain = time.time() - t0
    mem_retrain = _peak_memory_mb(device)
    all_epoch_logs['Retrain'] = logs_retrain

    res_r = evaluate_model_dual(model_retrain, test_loader_unl, device)
    mia_retrain = compute_mia_auc_dual(model_retrain, train_loader_unl, test_loader_unl,
                                        device, pw, forget_label_idx=forget_label_idx)
    print(f"  [Retrain]   ExMatch={res_r['exact_match']:.4f}  "
          f"MacroROC={res_r['macro_roc']:.4f}  MacroF1={res_r['macro_f1']:.4f}  "
          f"DetAcc={res_r['det_acc']:.4f}  "
          f"MIA_f={mia_retrain['mia_forget']:.4f}  MIA_r={mia_retrain['mia_retain']:.4f}  "
          f"Time={time_retrain:.1f}s  Mem={mem_retrain:.1f}MB")
    print(f"  Speedup vs Retrain: GDGU {time_retrain / max(time_gdgu, 1e-6):.1f}x  "
          f"GIF {time_retrain / max(time_gif, 1e-6):.1f}x  "
          f"IDEA {time_retrain / max(time_idea, 1e-6):.1f}x")

    results.append(_build_result_dual(backbone_name, scen_key, seed, 'Retrain',
                                       res_r, mia_retrain, time_retrain, mem_retrain,
                                       forget_label_idx=forget_label_idx))

    del model_orig, model_gdgu, model_gif, model_idea, model_retrain
    torch.cuda.empty_cache()

    return results, all_epoch_logs


def _build_result_dual(backbone, scenario, seed, method, eval_dict, mia_dict,
                       elapsed, mem_mb, forget_label_idx=None):
    """Flat result dict for dual-channel trial (adds Det metrics)."""
    row = _build_result(backbone, scenario, seed, method, eval_dict, mia_dict,
                        elapsed, mem_mb, forget_label_idx=forget_label_idx)
    row['Det_Acc'] = eval_dict.get('det_acc', np.nan)
    row['Det_AUC'] = eval_dict.get('det_auc', np.nan)
    row['Det_F1'] = eval_dict.get('det_f1', np.nan)
    return row
