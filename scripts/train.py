#!/usr/bin/env python3
"""Non-interactive training script for HPC batch jobs.

Hyperparameters are loaded from config/<bus_system>.yaml (same as run_experiments.py).
All outputs (CSV, JSON logs) are saved to results/<YYYY-MM-DD_HH>/.
Visualization is handled separately via notebooks/Viz_GDGU_loc.ipynb.

Usage:
    python train.py --bus 34bus
    python train.py --bus 123bus
    python train.py --bus 34bus --backbone GCN --scenario S1 --seed 42
    python train.py --bus 34bus --route A          # V6.0 Route A
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / 'config'

sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_evcs_data, expand_forget_khop, augment_route_a
from src.experiment import run_single_trial, run_single_trial_dual, run_single_trial_route_a
from src.models import MODEL_CLASSES


def load_experiment(bus_system: str, source_data: Path) -> dict:
    """Load experiment definition from config/<bus_system>.yaml.

    Same logic as run_experiments.py to guarantee parameter consistency.
    """
    cfg_path = CONFIG_DIR / f'{bus_system}.yaml'
    if not cfg_path.exists():
        raise FileNotFoundError(
            f'Config not found: {cfg_path}\n'
            f'Available configs: {list(CONFIG_DIR.glob("*.yaml"))}'
        )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    d = cfg['data']
    t = cfg['training']
    g = cfg['gdgu']
    gi = cfg.get('gif', {})
    e = cfg['experiment']
    dc = cfg.get('dual_channel', {})
    ra = cfg.get('route_a', {})

    # Resolve pkl_paths: support glob patterns (123-bus) and plain filenames
    raw_paths = d['pkl_paths']
    if '*' in raw_paths:
        pkl_paths = sorted(str(p) for p in source_data.glob(raw_paths))
    else:
        pkl_paths = str(source_data / raw_paths)

    return {
        'pkl_paths':    pkl_paths,
        'gml_path':     str(source_data / d['gml_path']),
        'evcs_bus_ids': d['evcs_bus_ids'],
        'config': {
            # ── data split ──
            'test_size':          t['test_size'],
            'val_ratio':          t['val_ratio'],
            # ── training ──
            'epochs':             t['epochs'],
            'batch_size':         t['batch_size'],
            'lr':                 t['lr'],
            'weight_decay':       t['weight_decay'],
            'patience':           t['patience'],
            'scheduler_patience': t.get('scheduler_patience', 20),
            # ── model architecture ──
            'hidden_dim':         t['hidden_dim'],
            'n_layers':           t['n_layers'],
            'dropout':            t['dropout'],
            # ── gdgu ──
            'gdgu_damp':          g['damp'],
            'gdgu_max_norm':      g['max_norm'],
            'gdgu_finetune':      g['finetune_epochs'],
            'gdgu_finetune_lr':   g.get('finetune_lr', 1e-4),
            # ── gif / idea ──
            'gif_iteration':      gi.get('iteration', 50),
            'gif_damp':           gi.get('damp', 0.01),
            'gif_scale':          gi.get('scale', 50.0),
            'idea_finetune':      gi.get('idea_finetune', 25),
            'gif_max_batches':    gi.get('max_batches'),
            # ── experiment ──
            'seeds':              e['seeds'],
            'backbones':          e['backbones'],
            'k_hops':             e.get('k_hops', [0]),
            # ── dual-channel (V5.0 archived) ──
            'dual_channel': {
                'graph_feat_dim':   dc.get('graph_feat_dim', 120),
                'graph_mlp_hidden': dc.get('graph_mlp_hidden', 64),
                'graph_mlp_out':    dc.get('graph_mlp_out', 32),
                'alpha':            dc.get('alpha', 1.0),
                'beta':             dc.get('beta', 1.0),
            },
            # ── route_a (V6.0) ──
            'route_a': {
                'gamma':          ra.get('gamma', 0.5),
                'n_attack_types': ra.get('n_attack_types', 5),
                'aux_hidden':     ra.get('aux_hidden', 64),
                'ig_steps':       ra.get('ig_steps', 50),
                'lr_gat':         ra.get('lr_gat', 1e-4),
            },
        },
    }


def build_scenarios(evcs_bus_ids, evcs_buses, n_nodes, edge_index, k_hops=None):
    """Build cumulative unlearning scenarios with k-hop neighbor expansion.

    Naming: S{n}-{k}, e.g. S1-0 (EVCS only), S1-1 (+1-hop neighbors), S1-2 (+2-hop).

    Args:
        evcs_bus_ids: list of EVCS bus IDs in cumulative forget order.
        evcs_buses:   dict {bus_id: node_index}.
        n_nodes:      total number of nodes in the graph.
        edge_index:   [2, E] tensor for topology.
        k_hops:       list of k values, e.g. [0, 1, 2]. Defaults to [0].
    """
    if k_hops is None:
        k_hops = [0]

    scenarios = {}
    for i in range(1, len(evcs_bus_ids) + 1):
        forget_buses = evcs_bus_ids[:i]
        evcs_indices = [evcs_buses[b] for b in forget_buses]
        names_str = '+'.join(str(b) for b in forget_buses)

        for k in k_hops:
            expanded = expand_forget_khop(evcs_indices, edge_index, k=k)
            n_forget = len(expanded)
            pct = n_forget / n_nodes * 100
            scen_key = f'S{i}-{k}'
            scenarios[scen_key] = {
                'label': f'S{i}-{k}: Bus {names_str} '
                         f'(k={k}, {n_forget} node{"s" if n_forget > 1 else ""}, {pct:.1f}%)',
                'forget_indices': expanded,
                'forget_label_indices': list(range(i)),
                'evcs_indices': evcs_indices,
                'k_hop': k,
                'n_evcs_forget': i,
            }
    return scenarios


def build_scenarios_route_a(evcs_bus_ids, evcs_node_indices):
    """Build cumulative unlearning scenarios for Route A (no k-hop).

    Naming: S{n}, e.g. S1, S2, S3.
    """
    scenarios = {}
    for i in range(1, len(evcs_bus_ids) + 1):
        forget_buses = evcs_bus_ids[:i]
        forget_nodes = evcs_node_indices[:i]
        names_str = '+'.join(str(b) for b in forget_buses)
        scen_key = f'S{i}'
        scenarios[scen_key] = {
            'label': f'S{i}: Bus {names_str} ({i} EVCS, P zeroed)',
            'forget_node_indices': forget_nodes,
            'forget_label_indices': list(range(i)),
            'n_evcs_forget': i,
        }
    return scenarios


def save_results(results_all, all_logs, bus_system, data, config,
                 scenarios, backbones, output_dir, device, tag):
    """Save CSV + JSON results to output_dir."""
    df = pd.DataFrame(results_all)

    # ── Raw CSV ──
    raw_csv = output_dir / f'{tag}_results_raw.csv'
    df.to_csv(raw_csv, index=False)
    print(f'Raw results saved to {raw_csv}  ({len(df)} rows)')

    # ── Summary CSV ──
    roc_cols = sorted([c for c in df.columns if c.startswith('ROC_EVCS')])
    f1_cols = sorted([c for c in df.columns if c.startswith('F1_EVCS')])
    metric_cols = ['ExMatch', 'Hamming_Acc', 'Macro_ROC', 'Macro_F1'] \
                + roc_cols + f1_cols \
                + ['F1_forget', 'F1_retain', 'ROC_forget', 'ROC_retain'] \
                + ['MIA_forget', 'MIA_retain', 'MIA_AUC', 'Time', 'Peak_Memory_MB'] \
                + ['Aux_Acc', 'L2a_IG_mean', 'L2a_IG_std',
                   'L2b_delta_auc', 'L2b_auc_P_present', 'L2b_auc_P_occluded']
    metric_cols = [c for c in metric_cols if c in df.columns]
    summary = df.groupby(['Backbone', 'Scenario', 'Method'])[metric_cols].agg(['mean', 'std']).round(4)
    sum_csv = output_dir / f'{tag}_results_summary.csv'
    summary.to_csv(sum_csv)
    print(f'Summary saved to {sum_csv}')

    # ── Epoch logs JSON (with metadata) ──
    meta = {
        'bus_system': bus_system,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'data': {
            'n_graphs': int(data['n_graphs']),
            'n_nodes': int(data['n_nodes']),
            'n_feat': int(data['n_feat']),
            'n_edges': int(data['edge_index'].shape[1]),
            'n_evcs': int(data['n_evcs']),
        },
        'config': {k: v for k, v in config.items()},
        'model_params': {},
        'scenarios': {k: v['label'] for k, v in scenarios.items()},
    }
    for name in backbones:
        m = MODEL_CLASSES[name](in_dim=data['n_feat'], hid_dim=config['hidden_dim'],
                                out_dim=config['out_dim'], n_layers=config['n_layers'],
                                dropout=config['dropout'])
        meta['model_params'][name] = sum(p.numel() for p in m.parameters())

    log_out = {'_metadata': meta, **all_logs}
    log_json = output_dir / f'{tag}_epoch_logs.json'
    with open(log_json, 'w') as f:
        json.dump(log_out, f, indent=2)
    print(f'Epoch logs saved to {log_json}')

    # ── Print summary table ──
    for bb in backbones:
        print(f'\n{"─"*80}\n  Backbone: {bb}\n{"─"*80}')
        for scen in scenarios:
            print(f'\n  {scen}:')
            for method in ['Original', 'GDGU', 'GIF', 'IDEA', 'Retrain']:
                sub = df[(df.Backbone == bb) & (df.Scenario == scen) & (df.Method == method)]
                if len(sub) == 0:
                    continue
                em = f"{sub['ExMatch'].mean():.3f}±{sub['ExMatch'].std():.3f}"
                mr = f"{sub['Macro_ROC'].mean():.3f}±{sub['Macro_ROC'].std():.3f}"
                mf = f"{sub['Macro_F1'].mean():.3f}±{sub['Macro_F1'].std():.3f}"
                mia_f = f"{sub['MIA_forget'].mean():.3f}" if sub['MIA_forget'].notna().any() else '—'
                mia_r = f"{sub['MIA_retain'].mean():.3f}" if sub['MIA_retain'].notna().any() else '—'
                t = f"{sub['Time'].mean():.1f}s"
                line = (f'    {method:10s}  ExMatch={em}  MacroROC={mr}  MacroF1={mf}  '
                        f'MIA_f={mia_f}  MIA_r={mia_r}  Time={t}')
                if 'Aux_Acc' in sub.columns and sub['Aux_Acc'].notna().any():
                    line += f"  AuxAcc={sub['Aux_Acc'].mean():.3f}"
                if 'L2b_delta_auc' in sub.columns and sub['L2b_delta_auc'].notna().any():
                    line += f"  L2b={sub['L2b_delta_auc'].mean():.4f}"
                print(line)

    return df


def main():
    parser = argparse.ArgumentParser(description='GDGU EVCS Localization Training')
    parser.add_argument('--bus', type=str, required=True, choices=['34bus', '123bus'],
                        help='Bus system to run')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Single backbone to run (GCN/GAT/GIN). Default: all three')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Single scenario to run (S1/S2/S1-0/...). Default: all')
    parser.add_argument('--khop', type=int, default=None,
                        help='Single k-hop value to run (0/1/2). Default: all from config')
    parser.add_argument('--seed', type=int, default=None,
                        help='Single seed to run. Default: all seeds from config')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Multiple seeds to run, e.g. --seeds 42 77 88. '
                             'Overrides config["seeds"]; ignored if --seed is set.')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override source data directory')
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU device index (default: 1)')
    parser.add_argument('--dual', action='store_true',
                        help='Use dual-channel model (V5.0 archived). '
                             'Runs Original + GDGU + GIF + IDEA + Retrain.')
    parser.add_argument('--route', type=str, default='A', choices=['A'],
                        help='V6.0 Route A (default): single-head + aux attack-type head, '
                             'modality-level unlearning (P-channel only). '
                             'Use --no-route for legacy V2.0 pipeline.')
    parser.add_argument('--no-route', dest='route', action='store_const', const=None,
                        help='Disable Route A; use legacy V2.0 pipeline.')
    args = parser.parse_args()

    if args.dual and args.route:
        parser.error('--dual and --route are mutually exclusive')

    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_properties(device).name}')

    # Paths
    bus_system = args.bus
    if args.data_dir:
        source_data = Path(args.data_dir)
    else:
        source_data = PROJECT_ROOT.parent / 'Source' / 'PB_data' / '3_EVCS Attacks'

    # Load config from YAML (same as run_experiments.py)
    exp = load_experiment(bus_system, source_data)
    config = exp['config']

    # Single output directory — everything goes here
    today = datetime.now().strftime('%Y-%m-%d_%H')
    output_dir = PROJECT_ROOT / 'results' / today
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Bus    : {bus_system}')
    print(f'Config : {CONFIG_DIR / f"{bus_system}.yaml"}')
    print(f'Output : {output_dir}')

    # Load data
    data = load_evcs_data(exp['pkl_paths'], exp['gml_path'], bus_system=bus_system)
    config['out_dim'] = data['n_evcs']

    # ── Route A (V6.0) ──
    if args.route == 'A':
        data = augment_route_a(data, exp['pkl_paths'])
        scenarios = build_scenarios_route_a(exp['evcs_bus_ids'],
                                             data['evcs_node_indices'])
    else:
        # k-hop values: CLI override or config
        k_hops = [args.khop] if args.khop is not None else config.get('k_hops', [0])
        scenarios = build_scenarios(exp['evcs_bus_ids'], data['evcs_buses'],
                                    data['n_nodes'], data['edge_index'], k_hops=k_hops)

    # Print scenario summary
    print(f'\nScenarios ({len(scenarios)}):')
    for sk, sv in scenarios.items():
        print(f'  {sk:8s}  {sv["label"]}')

    # Filter by CLI args
    backbones = [args.backbone] if args.backbone else config['backbones']
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = config['seeds']
    scen_filter = {args.scenario: scenarios[args.scenario]} if args.scenario else scenarios

    # Run
    results_all = []
    all_logs = {}
    total = len(backbones) * len(scen_filter) * len(seeds)
    count = 0
    t_start = time.time()

    for backbone in backbones:
        for scen_key, scen_val in scen_filter.items():
            for seed in seeds:
                count += 1
                print(f'\n[{count}/{total}]')
                if args.route == 'A':
                    trial_results, trial_logs = run_single_trial_route_a(
                        backbone, scen_key, scen_val, seed,
                        data, config, device)
                elif args.dual:
                    trial_results, trial_logs = run_single_trial_dual(
                        backbone, scen_key, scen_val, seed,
                        data['all_x'], data['all_V'], data['all_y'],
                        data['edge_index'], config, device)
                else:
                    trial_results, trial_logs = run_single_trial(
                        backbone, scen_key, scen_val, seed,
                        data['all_x'], data['all_y'], data['edge_index'],
                        config, device)
                results_all.extend(trial_results)
                all_logs[f'{backbone}_{scen_key}_{seed}'] = trial_logs
                print(f'  Finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    elapsed = time.time() - t_start
    print(f'\n{"="*70}')
    print(f'All {total} runs completed in {elapsed:.1f}s ({elapsed/60:.1f}min)')
    print(f'Finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*70}')

    # Tag for partial runs
    tag = bus_system
    if args.route == 'A':
        tag += '_routeA'
    elif args.dual:
        tag += '_dual'
    if args.backbone:
        tag += f'_{args.backbone}'
    if args.scenario:
        tag += f'_{args.scenario}'

    # Save all outputs to the single date folder
    save_results(results_all, all_logs, bus_system, data, config,
                 scen_filter, backbones, output_dir, device, tag)

    print(f'\nAll outputs in: {output_dir}')
    if args.route == 'A':
        print(f'Visualize via:  notebooks/Viz_V6.ipynb')
    else:
        print(f'Visualize via:  notebooks/Viz_GDGU_loc.ipynb')


if __name__ == '__main__':
    main()
