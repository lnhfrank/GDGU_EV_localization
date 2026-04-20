"""Experiment runner: single trial (one backbone x scenario x seed)."""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.models import MODEL_CLASSES
from src.data import build_graphs, stratified_split_multilabel, fit_scaler
from src.training import (kaiming_init, get_pos_weights, train_model,
                          evaluate_model, compute_mia_auc)
from src.unlearning import gdgu_feature_unlearn, gif_unlearn, idea_unlearn


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
