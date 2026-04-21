"""Visualization functions for GDGU/GIF/IDEA localization experiments.

All style constants are grouped at the top for easy customization.
Call `apply_style()` once before plotting to set Times New Roman globally.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================================================
#  Style defaults — override via apply_style(overrides={...})
# ============================================================
STYLE = {
    # Font
    'font_family':   'Times New Roman',
    'fs_label':      26,
    'fs_tick':       24,
    'fs_legend':     20,
    'fs_subtitle':   22,
    'fs_annotation': 18,

    # Colors — 5 methods
    'colors': {
        'Original': '#2196F3',
        'GDGU':     '#FF9800',
        'GIF':      '#9C27B0',
        'IDEA':     '#E91E63',
        'Retrain':  '#4CAF50',
    },
    'markers': {
        'Original': 'o',
        'GDGU':     's',
        'GIF':      'D',
        'IDEA':     'P',
        'Retrain':  '^',
    },
    'ideal_line_color': 'red',
    'grid_alpha':       0.3,
    'bar_alpha':        0.85,
    'bar_edge_color':   'black',
    'bar_edge_width':   0.5,
    'fill_alpha':       0.15,

    # Save
    'save_fmt': 'pdf',
    'save_dpi': 300,
}

METHODS_ORDER = ['Original', 'GDGU', 'GIF', 'IDEA', 'Retrain']
GU_METHODS = ['GDGU', 'GIF', 'IDEA']


def apply_style(overrides=None):
    """Apply matplotlib rcParams for Times New Roman. Call once before plotting.

    Args:
        overrides: dict to merge into STYLE, e.g. {'fs_label': 24, 'save_fmt': 'png'}.
    """
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(STYLE.get(k), dict):
                STYLE[k].update(v)
            else:
                STYLE[k] = v

    font = STYLE['font_family']
    mpl.rcParams.update({
        'font.family':      'serif',
        'font.serif':       [font],
        'mathtext.fontset':  'custom',
        'mathtext.rm':       font,
        'mathtext.it':       f'{font}:italic',
        'mathtext.bf':       f'{font}:bold',
    })


# ============================================================
#  Data loading
# ============================================================
def _scenario_sort_key(s):
    """Sort key for scenario names like 'S1', 'S2-0', 'S3-1'.

    Returns (n, k) tuple so S1-0 < S1-1 < S1-2 < S2-0 < ...
    Legacy names without '-k' suffix get k=-1 so they sort first.
    """
    if '-' in s:
        parts = s.split('-')
        return (int(parts[0][1:]), int(parts[1]))
    return (int(s[1:]), -1)


def load_results(results_dir, bus_system='123bus', tag=None):
    """Load raw CSV from a date-named results folder.

    Args:
        results_dir: path to date folder, e.g. '.../results/2026-04-07'.
        bus_system:  '34bus' or '123bus'.
        tag:         Optional filename tag appended to bus_system.  For
                     example tag='dual' loads '{bus_system}_dual_results_raw.csv'.
                     Default None loads '{bus_system}_results_raw.csv'.

    Returns:
        df, scenarios, backbones
    """
    prefix = bus_system if tag is None else f'{bus_system}_{tag}'
    csv_path = os.path.join(results_dir, f'{prefix}_results_raw.csv')
    df = pd.read_csv(csv_path)

    scen_keys = sorted(df.Scenario.unique(), key=_scenario_sort_key)
    scenarios = {sk: {'label': sk} for sk in scen_keys}
    backbones = sorted(df.Backbone.unique())

    print(f'Loaded {len(df)} rows from {csv_path}')
    print(f'  Backbones : {backbones}')
    print(f'  Scenarios : {scen_keys}')
    print(f'  Methods   : {sorted(df.Method.unique())}')
    return df, scenarios, backbones


# ============================================================
#  Internal helpers
# ============================================================
def _savefig(fig, filepath):
    fig.savefig(filepath, bbox_inches='tight', dpi=STYLE['save_dpi'])


def _scen_labels(scenarios):
    """Return short display labels for scenarios.

    For Sn-k format (e.g. 'S1-0'), use the key directly.
    For legacy format with full label, take the part before ':'.
    """
    labels = []
    for s in scenarios:
        lbl = scenarios[s]['label']
        labels.append(lbl.split(':')[0] if ':' in lbl else s)
    return labels


def _available_methods(df):
    """Return methods present in df, in METHODS_ORDER order."""
    present = set(df.Method.unique())
    return [m for m in METHODS_ORDER if m in present]


# ============================================================
#  Plot functions
# ============================================================
def plot_metric_bars(df, metric_col, ylabel, ylim, filepath,
                     scenarios, backbones):
    """Grouped bar chart (1x3 subplots, one per backbone)."""
    S = STYLE
    methods = _available_methods(df)
    n_methods = len(methods)
    width = 0.8 / n_methods

    fig, axes = plt.subplots(1, len(backbones), figsize=(18, 5), sharey=True)
    scen_keys = list(scenarios.keys())
    labels = _scen_labels(scenarios)
    for ax_idx, bb in enumerate(backbones):
        ax = axes[ax_idx]
        x = np.arange(len(scen_keys))
        for m_idx, method in enumerate(methods):
            offset = (m_idx - (n_methods - 1) / 2) * width
            means, stds = [], []
            for sk in scen_keys:
                sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                means.append(sub[metric_col].mean())
                stds.append(sub[metric_col].std())
            ax.bar(x + offset, means, width, yerr=stds,
                   label=method, color=S['colors'][method], alpha=S['bar_alpha'],
                   capsize=3, edgecolor=S['bar_edge_color'], linewidth=S['bar_edge_width'])
        ax.set_title(bb, fontsize=S['fs_subtitle'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=S['fs_tick'])
        ax.tick_params(axis='y', labelsize=S['fs_tick'])
        if ax_idx == 0:
            ax.set_ylabel(ylabel, fontsize=S['fs_label'])
        ax.set_ylim(ylim)
        ax.grid(axis='y', alpha=S['grid_alpha'])
    axes[-1].legend(fontsize=S['fs_legend'])
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def plot_metric_lines(df, metric_col, ylabel, ylim, filepath,
                      scenarios, backbones):
    """Line chart with shaded std (1x3 subplots, one per backbone)."""
    S = STYLE
    methods = _available_methods(df)
    fig, axes = plt.subplots(1, len(backbones), figsize=(18, 5), sharey=True)
    scen_keys = list(scenarios.keys())
    labels = _scen_labels(scenarios)
    x = np.arange(len(scen_keys))
    for ax_idx, bb in enumerate(backbones):
        ax = axes[ax_idx]
        for method in methods:
            means, stds = [], []
            for sk in scen_keys:
                sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                means.append(sub[metric_col].mean())
                stds.append(sub[metric_col].std())
            means, stds = np.array(means), np.array(stds)
            ax.plot(x, means, marker=S['markers'][method], color=S['colors'][method],
                    label=method, linewidth=2, markersize=8)
            ax.fill_between(x, means - stds, means + stds,
                            color=S['colors'][method], alpha=S['fill_alpha'])
        ax.set_title(bb, fontsize=S['fs_subtitle'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=S['fs_tick'])
        ax.tick_params(axis='y', labelsize=S['fs_tick'])
        ax.set_xlabel('Scenario', fontsize=S['fs_label'])
        if ax_idx == 0:
            ax.set_ylabel(ylabel, fontsize=S['fs_label'])
        ax.set_ylim(ylim)
        ax.grid(True, alpha=S['grid_alpha'])
    axes[-1].legend(fontsize=S['fs_legend'])
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def plot_per_evcs_roc(df, filepath, scenarios, backbones, evcs_names=None):
    """backbone (row) x EVCS (col) grid, grouped bars per scenario."""
    S = STYLE
    methods = _available_methods(df)
    n_methods = len(methods)
    width = 0.8 / n_methods

    roc_cols = sorted([c for c in df.columns if c.startswith('ROC_EVCS')])
    n_evcs = len(roc_cols)
    n_bb = len(backbones)

    if evcs_names is None:
        evcs_names = [f'EVCS {i+1}' for i in range(n_evcs)]

    scen_keys = list(scenarios.keys())
    labels = _scen_labels(scenarios)
    fig, axes = plt.subplots(n_bb, n_evcs, figsize=(6 * n_evcs, 5 * n_bb), sharey=True)
    if n_bb == 1:
        axes = [axes]
    if n_evcs == 1:
        axes = [[ax] for ax in axes]

    for row, bb in enumerate(backbones):
        for col, (roc_col, evcs_name) in enumerate(zip(roc_cols, evcs_names)):
            ax = axes[row][col]
            x = np.arange(len(scen_keys))
            for m_idx, method in enumerate(methods):
                offset = (m_idx - (n_methods - 1) / 2) * width
                means, stds = [], []
                for sk in scen_keys:
                    sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                    means.append(sub[roc_col].mean())
                    stds.append(sub[roc_col].std())
                ax.bar(x + offset, means, width, yerr=stds,
                       label=method, color=S['colors'][method], alpha=S['bar_alpha'],
                       capsize=3, edgecolor=S['bar_edge_color'], linewidth=S['bar_edge_width'])
            if row == 0:
                ax.set_title(evcs_name, fontsize=S['fs_subtitle'] - 2, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=S['fs_tick'] - 2)
            ax.tick_params(axis='y', labelsize=S['fs_tick'] - 2)
            if col == 0:
                ax.set_ylabel(f'{bb}\nROC-AUC', fontsize=S['fs_label'] - 2)
            ax.set_ylim(0.4, 1.0)
            ax.grid(axis='y', alpha=S['grid_alpha'])
            if row == 0 and col == n_evcs - 1:
                ax.legend(fontsize=S['fs_legend'] - 2)
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def plot_mia_auc(df, filepath, scenarios, backbones):
    """MIA-AUC bar chart (all GU methods + Retrain) with ideal=0.5 line."""
    S = STYLE
    mia_methods = [m for m in ['GDGU', 'GIF', 'IDEA', 'Retrain']
                   if m in df.Method.unique()]
    n_methods = len(mia_methods)
    width = 0.8 / n_methods

    fig, axes = plt.subplots(1, len(backbones), figsize=(18, 5), sharey=True)
    scen_keys = list(scenarios.keys())
    labels = _scen_labels(scenarios)

    for ax_idx, bb in enumerate(backbones):
        ax = axes[ax_idx]
        x = np.arange(len(scen_keys))
        for m_idx, method in enumerate(mia_methods):
            offset = (m_idx - (n_methods - 1) / 2) * width
            means, stds = [], []
            for sk in scen_keys:
                sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                means.append(sub['MIA_AUC'].mean())
                stds.append(sub['MIA_AUC'].std())
            ax.bar(x + offset, means, width, yerr=stds,
                   label=method, color=S['colors'][method], alpha=S['bar_alpha'],
                   capsize=3, edgecolor=S['bar_edge_color'], linewidth=S['bar_edge_width'])
        ax.axhline(y=0.5, color=S['ideal_line_color'], linestyle='--', alpha=0.7,
                   label='Ideal (0.5)')
        ax.set_title(bb, fontsize=S['fs_subtitle'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=S['fs_tick'])
        ax.tick_params(axis='y', labelsize=S['fs_tick'])
        if ax_idx == 0:
            ax.set_ylabel('MIA-AUC', fontsize=S['fs_label'])
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis='y', alpha=S['grid_alpha'])
    axes[-1].legend(fontsize=S['fs_legend'])
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def plot_time_comparison(df, filepath, scenarios, backbones):
    """Time bar chart for all GU methods + Retrain, with speedup annotations."""
    S = STYLE
    gu_methods = [m for m in ['GDGU', 'GIF', 'IDEA', 'Retrain']
                  if m in df.Method.unique()]
    n_methods = len(gu_methods)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(14, 7))
    scen_keys = list(scenarios.keys())

    groups = []
    for bb in backbones:
        for sk in scen_keys:
            groups.append(f'{bb}\n{sk}')

    x = np.arange(len(groups))
    for m_idx, method in enumerate(gu_methods):
        offset = (m_idx - (n_methods - 1) / 2) * width
        means = []
        for bb in backbones:
            for sk in scen_keys:
                sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                means.append(sub['Time'].mean())
        ax.bar(x + offset, means, width,
               label=method, color=S['colors'][method],
               edgecolor=S['bar_edge_color'], linewidth=S['bar_edge_width'])

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=S['fs_tick'] - 4)
    ax.tick_params(axis='y', labelsize=S['fs_tick'])
    ax.set_ylabel('Time (seconds)', fontsize=S['fs_label'])
    ax.legend(fontsize=S['fs_legend'])
    ax.grid(axis='y', alpha=S['grid_alpha'])
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def plot_memory_usage(df, filepath, scenarios, backbones):
    """Peak GPU memory bar chart, one group per backbone x scenario."""
    S = STYLE
    if 'Peak_Memory_MB' not in df.columns:
        print('  [skip] Peak_Memory_MB column not found — skipping memory plot')
        return

    methods = _available_methods(df)
    n_methods = len(methods)
    width = 0.8 / n_methods

    fig, axes = plt.subplots(1, len(backbones), figsize=(18, 5), sharey=True)
    if len(backbones) == 1:
        axes = [axes]
    scen_keys = list(scenarios.keys())
    labels = _scen_labels(scenarios)

    for ax_idx, bb in enumerate(backbones):
        ax = axes[ax_idx]
        x = np.arange(len(scen_keys))
        for m_idx, method in enumerate(methods):
            offset = (m_idx - (n_methods - 1) / 2) * width
            means, stds = [], []
            for sk in scen_keys:
                sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                means.append(sub['Peak_Memory_MB'].mean())
                stds.append(sub['Peak_Memory_MB'].std())
            ax.bar(x + offset, means, width, yerr=stds,
                   label=method, color=S['colors'][method], alpha=S['bar_alpha'],
                   capsize=3, edgecolor=S['bar_edge_color'], linewidth=S['bar_edge_width'])
        ax.set_title(bb, fontsize=S['fs_subtitle'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=S['fs_tick'])
        ax.tick_params(axis='y', labelsize=S['fs_tick'])
        if ax_idx == 0:
            ax.set_ylabel('Peak Memory (MB)', fontsize=S['fs_label'])
        ax.grid(axis='y', alpha=S['grid_alpha'])
    axes[-1].legend(fontsize=S['fs_legend'])
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def plot_f1_vs_mia(df, filepath, scenarios, backbones):
    """Forgetting-Reasoning trade-off scatter: Macro F1 (y) vs MIA-AUC (x).

    One subplot per backbone. Each point = one scenario (mean over seeds).
    All GU methods + Retrain shown. Ideal: MIA close to 0.5, F1 high.
    """
    S = STYLE
    mia_methods = [m for m in ['GDGU', 'GIF', 'IDEA', 'Retrain']
                   if m in df.Method.unique()]
    scen_keys = list(scenarios.keys())

    fig, axes = plt.subplots(1, len(backbones), figsize=(7 * len(backbones), 5.5),
                             sharey=True, sharex=True)
    if len(backbones) == 1:
        axes = [axes]

    for ax_idx, bb in enumerate(backbones):
        ax = axes[ax_idx]
        for method in mia_methods:
            mia_means, f1_means = [], []
            for sk in scen_keys:
                sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                mia_means.append(sub['MIA_AUC'].mean())
                f1_means.append(sub['Macro_F1'].mean())
            ax.scatter(mia_means, f1_means,
                       color=S['colors'][method], marker=S['markers'][method],
                       s=120, edgecolors='black', linewidths=0.6,
                       label=method, zorder=3)
            for i, sk in enumerate(scen_keys):
                ax.annotate(scenarios[sk]['label'].split(':')[0],
                            (mia_means[i], f1_means[i]),
                            textcoords='offset points', xytext=(6, 4),
                            fontsize=S['fs_annotation'], fontweight='bold')

        ax.axvline(x=0.5, color=S['ideal_line_color'], linestyle='--', alpha=0.6,
                   label='Ideal MIA (0.5)')
        ax.set_title(bb, fontsize=S['fs_subtitle'], fontweight='bold')
        ax.set_xlabel('MIA-AUC', fontsize=S['fs_label'])
        ax.tick_params(axis='both', labelsize=S['fs_tick'])
        if ax_idx == 0:
            ax.set_ylabel('Macro F1', fontsize=S['fs_label'])
        ax.grid(True, alpha=S['grid_alpha'])
    axes[-1].legend(fontsize=S['fs_legend'], loc='best')
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def plot_gu_comparison(df, filepath, scenarios, backbones):
    """GU method comparison: GDGU vs GIF vs IDEA across scenarios.

    One subplot per backbone. Lines for each GU method + dashed Retrain
    as gold-standard reference. Y-axis = Macro ROC-AUC.
    """
    S = STYLE
    gu_plus = [m for m in ['GDGU', 'GIF', 'IDEA', 'Retrain']
               if m in df.Method.unique()]
    scen_keys = list(scenarios.keys())
    labels = _scen_labels(scenarios)
    x = np.arange(len(scen_keys))

    fig, axes = plt.subplots(1, len(backbones), figsize=(7 * len(backbones), 5.5),
                             sharey=True)
    if len(backbones) == 1:
        axes = [axes]

    for ax_idx, bb in enumerate(backbones):
        ax = axes[ax_idx]
        for method in gu_plus:
            means, stds = [], []
            for sk in scen_keys:
                sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                means.append(sub['Macro_ROC'].mean())
                stds.append(sub['Macro_ROC'].std())
            means, stds = np.array(means), np.array(stds)
            ls = '--' if method == 'Retrain' else '-'
            lw = 1.5 if method == 'Retrain' else 2.5
            ax.plot(x, means, marker=S['markers'][method], color=S['colors'][method],
                    label=method, linewidth=lw, linestyle=ls, markersize=9)
            ax.fill_between(x, means - stds, means + stds,
                            color=S['colors'][method], alpha=S['fill_alpha'])

        ax.set_title(bb, fontsize=S['fs_subtitle'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=S['fs_tick'])
        ax.tick_params(axis='y', labelsize=S['fs_tick'])
        ax.set_xlabel('Scenario', fontsize=S['fs_label'])
        if ax_idx == 0:
            ax.set_ylabel('Macro ROC-AUC', fontsize=S['fs_label'])
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=S['grid_alpha'])
    axes[-1].legend(fontsize=S['fs_legend'], loc='best')
    plt.suptitle('GU Method Comparison', fontsize=S['fs_label'] + 2, fontweight='bold', y=1.02)
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def plot_khop_comparison(df, filepath, backbones, metric_col='MIA_AUC',
                         ylabel='MIA-AUC', ylim=None):
    """Compare k-hop expansion levels for each base scenario (Sn).

    For each backbone subplot, X-axis = k values, one line per GU method + Retrain.
    Each figure covers one base scenario Sn (all k values).
    Produces one figure per Sn, with 1×len(backbones) subplots.

    Args:
        df:         DataFrame with Scenario column in 'Sn-k' format.
        filepath:   base filepath — Sn is appended, e.g. '...khop_MIA_S1.pdf'.
        backbones:  list of backbone names.
        metric_col: column to plot.
        ylabel:     Y-axis label.
        ylim:       (ymin, ymax) or None for auto.
    """
    S = STYLE
    methods = [m for m in ['GDGU', 'GIF', 'IDEA', 'Retrain']
               if m in df.Method.unique()]

    # Parse Sn-k structure from scenario names
    scen_names = sorted(df.Scenario.unique(), key=_scenario_sort_key)
    # Group by base scenario number
    base_groups = {}
    for s in scen_names:
        if '-' not in s:
            continue
        n, k = s.split('-')
        base_groups.setdefault(n, []).append((int(k), s))

    if not base_groups:
        print(f'  [skip] No Sn-k format scenarios found — skipping k-hop comparison')
        return

    for base_sn, k_list in sorted(base_groups.items()):
        k_list.sort()
        k_vals = [kv[0] for kv in k_list]
        scen_keys = [kv[1] for kv in k_list]

        n_bb = len(backbones)
        fig, axes = plt.subplots(1, n_bb, figsize=(7 * n_bb, 5.5), sharey=True)
        if n_bb == 1:
            axes = [axes]

        for ax_idx, bb in enumerate(backbones):
            ax = axes[ax_idx]
            for method in methods:
                means, stds = [], []
                for sk in scen_keys:
                    sub = df[(df.Backbone == bb) & (df.Scenario == sk) & (df.Method == method)]
                    means.append(sub[metric_col].mean() if len(sub) > 0 else np.nan)
                    stds.append(sub[metric_col].std() if len(sub) > 0 else 0)
                means, stds = np.array(means), np.array(stds)
                ls = '--' if method == 'Retrain' else '-'
                lw = 1.5 if method == 'Retrain' else 2.5
                ax.plot(k_vals, means, marker=S['markers'][method],
                        color=S['colors'][method], label=method,
                        linewidth=lw, linestyle=ls, markersize=9)
                ax.fill_between(k_vals, means - stds, means + stds,
                                color=S['colors'][method], alpha=S['fill_alpha'])

            if metric_col == 'MIA_AUC':
                ax.axhline(y=0.5, color=S['ideal_line_color'], linestyle='--',
                           alpha=0.6, label='Ideal (0.5)')

            ax.set_title(bb, fontsize=S['fs_subtitle'], fontweight='bold')
            ax.set_xticks(k_vals)
            ax.set_xticklabels([f'k={k}' for k in k_vals], fontsize=S['fs_tick'])
            ax.tick_params(axis='y', labelsize=S['fs_tick'])
            ax.set_xlabel('k-hop expansion', fontsize=S['fs_label'])
            if ax_idx == 0:
                ax.set_ylabel(ylabel, fontsize=S['fs_label'])
            if ylim:
                ax.set_ylim(ylim)
            ax.grid(True, alpha=S['grid_alpha'])
        axes[-1].legend(fontsize=S['fs_legend'], loc='best')
        plt.suptitle(f'{base_sn}: {ylabel} vs k-hop',
                     fontsize=S['fs_label'] + 2, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Construct per-Sn filepath
        base, ext = os.path.splitext(filepath)
        out_path = f'{base}_{base_sn}{ext}'
        _savefig(fig, out_path)
        plt.show()


# ============================================================
#  Dual-channel (Scheme A) plots
# ============================================================
def plot_det_preservation(df_dual, filepath, scenarios, backbones,
                          metric='Det_Acc', ylabel='Detection Accuracy',
                          ylim=(0.0, 1.0)):
    """Detection-metric preservation bar chart for dual-channel experiments.

    Shows that Det_Acc / Det_AUC / Det_F1 stay stable across
    Original / GDGU / GIF / IDEA / Retrain, confirming that the graph
    channel is correctly frozen during unlearning.
    """
    S = STYLE
    if metric not in df_dual.columns:
        print(f'[plot_det_preservation] {metric} not found in df — skipping.')
        return
    methods = _available_methods(df_dual)
    n_methods = len(methods)
    width = 0.8 / n_methods

    fig, axes = plt.subplots(1, len(backbones), figsize=(18, 5), sharey=True)
    if len(backbones) == 1:
        axes = [axes]
    scen_keys = list(scenarios.keys())
    labels = _scen_labels(scenarios)

    for ax_idx, bb in enumerate(backbones):
        ax = axes[ax_idx]
        x = np.arange(len(scen_keys))
        for m_idx, method in enumerate(methods):
            offset = (m_idx - (n_methods - 1) / 2) * width
            means, stds = [], []
            for sk in scen_keys:
                sub = df_dual[(df_dual.Backbone == bb) & (df_dual.Scenario == sk)
                              & (df_dual.Method == method)]
                means.append(sub[metric].mean() if len(sub) > 0 else np.nan)
                stds.append(sub[metric].std() if len(sub) > 0 else 0)
            ax.bar(x + offset, means, width, yerr=stds,
                   label=method, color=S['colors'][method],
                   alpha=S['bar_alpha'], capsize=3,
                   edgecolor=S['bar_edge_color'], linewidth=S['bar_edge_width'])
        ax.set_title(bb, fontsize=S['fs_subtitle'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=S['fs_tick'])
        ax.tick_params(axis='y', labelsize=S['fs_tick'])
        if ax_idx == 0:
            ax.set_ylabel(ylabel, fontsize=S['fs_label'])
        ax.set_ylim(ylim)
        ax.grid(axis='y', alpha=S['grid_alpha'])
    axes[-1].legend(fontsize=S['fs_legend'])
    plt.suptitle(f'Dual-channel: {ylabel} preserved across methods',
                 fontsize=S['fs_label'] + 2, fontweight='bold', y=1.02)
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def plot_dual_vs_nondual_mia(df_nondual, df_dual, filepath, scenarios, backbones,
                             mia_col='MIA_AUC'):
    """Side-by-side MIA_forget comparison: single-channel vs dual-channel.

    Each backbone gets one subplot.  X-axis = methods (GDGU/GIF/IDEA/Retrain),
    grouped bars for non-dual vs dual.  Ideal=0.5 reference line shown.

    This is the direct visualization of the Scheme A hypothesis:
    dual-channel isolation should push MIA_forget closer to 0.5.
    """
    S = STYLE
    methods = [m for m in ['GDGU', 'GIF', 'IDEA', 'Retrain']
               if m in df_nondual.Method.unique() and m in df_dual.Method.unique()]

    fig, axes = plt.subplots(1, len(backbones), figsize=(18, 5), sharey=True)
    if len(backbones) == 1:
        axes = [axes]
    scen_keys = list(scenarios.keys())

    for ax_idx, bb in enumerate(backbones):
        ax = axes[ax_idx]
        x = np.arange(len(methods))
        width = 0.38

        nd_means, nd_stds = [], []
        d_means, d_stds = [], []
        for method in methods:
            nd_sub = df_nondual[(df_nondual.Backbone == bb)
                                & (df_nondual.Method == method)
                                & df_nondual.Scenario.isin(scen_keys)]
            d_sub = df_dual[(df_dual.Backbone == bb)
                            & (df_dual.Method == method)
                            & df_dual.Scenario.isin(scen_keys)]
            nd_means.append(nd_sub[mia_col].mean())
            nd_stds.append(nd_sub[mia_col].std())
            d_means.append(d_sub[mia_col].mean())
            d_stds.append(d_sub[mia_col].std())

        ax.bar(x - width / 2, nd_means, width, yerr=nd_stds,
               label='Single-channel', color='#8D8D8D',
               alpha=S['bar_alpha'], capsize=3,
               edgecolor=S['bar_edge_color'], linewidth=S['bar_edge_width'])
        ax.bar(x + width / 2, d_means, width, yerr=d_stds,
               label='Dual-channel (Scheme A)', color='#1976D2',
               alpha=S['bar_alpha'], capsize=3,
               edgecolor=S['bar_edge_color'], linewidth=S['bar_edge_width'])

        ax.axhline(y=0.5, color=S['ideal_line_color'], linestyle='--',
                   alpha=0.7, label='Ideal (0.5)')
        ax.set_title(bb, fontsize=S['fs_subtitle'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=S['fs_tick'])
        ax.tick_params(axis='y', labelsize=S['fs_tick'])
        if ax_idx == 0:
            ax.set_ylabel('MIA-AUC on forget set', fontsize=S['fs_label'])
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis='y', alpha=S['grid_alpha'])
    axes[-1].legend(fontsize=S['fs_legend'])
    plt.suptitle('Scheme A: Dual-channel isolation vs single-channel baseline',
                 fontsize=S['fs_label'] + 2, fontweight='bold', y=1.02)
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


def summarize_dual_vs_nondual(df_nondual, df_dual, metrics=None):
    """Return a side-by-side mean±std DataFrame for Dual vs Non-dual comparison.

    Grouped by (Backbone, Method).  Useful for quick numerical readout
    alongside the bar chart.
    """
    metrics = metrics or ['MIA_AUC', 'MIA_forget', 'ExMatch', 'Macro_ROC',
                           'Macro_F1', 'Time']
    available = [m for m in metrics if m in df_nondual.columns
                 and m in df_dual.columns]
    rows = []
    for bb in sorted(set(df_nondual.Backbone.unique())
                     & set(df_dual.Backbone.unique())):
        for method in [m for m in METHODS_ORDER
                       if m in df_nondual.Method.unique()
                       and m in df_dual.Method.unique()]:
            nd = df_nondual[(df_nondual.Backbone == bb)
                            & (df_nondual.Method == method)]
            d = df_dual[(df_dual.Backbone == bb) & (df_dual.Method == method)]
            row = {'Backbone': bb, 'Method': method, 'n_nondual': len(nd),
                   'n_dual': len(d)}
            for m in available:
                row[f'{m}_nondual'] = f'{nd[m].mean():.3f}±{nd[m].std():.3f}' \
                    if len(nd) > 0 else 'n/a'
                row[f'{m}_dual'] = f'{d[m].mean():.3f}±{d[m].std():.3f}' \
                    if len(d) > 0 else 'n/a'
            rows.append(row)
    return pd.DataFrame(rows)


def plot_khop_forget_size(df, filepath, backbones, scenarios):
    """Bar chart: number of forget nodes per scenario, colored by k value.

    Quick overview of how much data each Sn-k scenario removes.
    """
    S = STYLE
    scen_names = sorted(scenarios.keys(), key=_scenario_sort_key)

    # Extract forget node counts from scenario labels
    # Label format: 'S1-0: Bus 814 (k=0, 1 node, 2.7%)'
    import re
    scen_data = []
    for sk in scen_names:
        if '-' not in sk:
            continue
        lbl = scenarios[sk]['label']
        m = re.search(r'(\d+) nodes?', lbl)
        n_forget = int(m.group(1)) if m else 0
        parts = sk.split('-')
        scen_data.append({'scen': sk, 'base': parts[0], 'k': int(parts[1]),
                          'n_forget': n_forget})

    if not scen_data:
        return

    k_colors = {0: '#2196F3', 1: '#FF9800', 2: '#E91E63'}
    fig, ax = plt.subplots(figsize=(max(12, len(scen_data) * 0.8), 5))
    x = np.arange(len(scen_data))
    colors = [k_colors.get(d['k'], '#999') for d in scen_data]
    bars = ax.bar(x, [d['n_forget'] for d in scen_data], color=colors,
                  edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([d['scen'] for d in scen_data],
                       fontsize=S['fs_tick'] - 4, rotation=45, ha='right')
    ax.set_ylabel('Forget set size (nodes)', fontsize=S['fs_label'])
    ax.tick_params(axis='y', labelsize=S['fs_tick'])
    ax.grid(axis='y', alpha=S['grid_alpha'])

    # Legend for k values
    from matplotlib.patches import Patch
    k_vals = sorted(set(d['k'] for d in scen_data))
    legend_patches = [Patch(facecolor=k_colors.get(k, '#999'), label=f'k={k}')
                      for k in k_vals]
    ax.legend(handles=legend_patches, fontsize=S['fs_legend'])
    plt.tight_layout()
    _savefig(fig, filepath)
    plt.show()


# ============================================================
#  Convenience: generate all figures at once
# ============================================================
def plot_all(df, output_dir, scenarios, backbones, bus_system='34bus'):
    """Generate all standard figures and save to output_dir.

    Args:
        df:         DataFrame (raw results or loaded via load_results).
        output_dir: directory to save figures into.
        scenarios:  dict, e.g. {'S1': {'label': 'S1'}, ...}.
        backbones:  list, e.g. ['GCN', 'GAT', 'GIN'].
        bus_system: '34bus' or '123bus', used as filename prefix.
    """
    os.makedirs(output_dir, exist_ok=True)
    j = lambda name: os.path.join(output_dir, f'{bus_system}_{name}.{STYLE["save_fmt"]}')

    plot_metric_bars(df, 'ExMatch', 'Exact Match Accuracy', (0.0, 1.0),
                     j('ExMatch_comparison'), scenarios, backbones)
    plot_metric_lines(df, 'ExMatch', 'Exact Match Accuracy', (0.0, 1.0),
                      j('ExMatch_trend'), scenarios, backbones)
    plot_metric_bars(df, 'Macro_ROC', 'Macro ROC-AUC', (0.4, 1.0),
                     j('MacroROC_comparison'), scenarios, backbones)
    plot_metric_lines(df, 'Macro_ROC', 'Macro ROC-AUC', (0.4, 1.0),
                      j('MacroROC_trend'), scenarios, backbones)
    plot_metric_bars(df, 'Macro_F1', 'Macro F1', (0.0, 1.0),
                     j('MacroF1_comparison'), scenarios, backbones)
    plot_metric_lines(df, 'Macro_F1', 'Macro F1', (0.0, 1.0),
                      j('MacroF1_trend'), scenarios, backbones)
    plot_metric_bars(df, 'Hamming_Acc', 'Hamming Accuracy', (0.4, 1.0),
                     j('HammingAcc_comparison'), scenarios, backbones)
    plot_metric_lines(df, 'Hamming_Acc', 'Hamming Accuracy', (0.4, 1.0),
                      j('HammingAcc_trend'), scenarios, backbones)
    plot_per_evcs_roc(df, j('PerEVCS_ROC_breakdown'), scenarios, backbones)
    plot_mia_auc(df, j('MIA_AUC_comparison'), scenarios, backbones)
    plot_f1_vs_mia(df, j('F1_vs_MIA_tradeoff'), scenarios, backbones)
    plot_time_comparison(df, j('Time_comparison'), scenarios, backbones)
    plot_memory_usage(df, j('Memory_usage'), scenarios, backbones)
    plot_gu_comparison(df, j('GU_method_comparison'), scenarios, backbones)

    # k-hop comparison plots (only when Sn-k scenarios exist)
    has_khop = any('-' in s for s in df.Scenario.unique())
    if has_khop:
        # plot_khop_comparison appends _Sn to the base path internally
        fmt = STYLE['save_fmt']
        khop_base = lambda name: os.path.join(output_dir, f'{bus_system}_{name}.{fmt}')
        plot_khop_comparison(df, khop_base('khop_MIA'),
                             backbones, 'MIA_AUC', 'MIA-AUC', (0.3, 1.0))
        plot_khop_comparison(df, khop_base('khop_MacroF1'),
                             backbones, 'Macro_F1', 'Macro F1', (0.0, 1.0))
        plot_khop_comparison(df, khop_base('khop_MacroROC'),
                             backbones, 'Macro_ROC', 'Macro ROC-AUC', (0.4, 1.0))
        plot_khop_comparison(df, khop_base('khop_ExMatch'),
                             backbones, 'ExMatch', 'Exact Match', (0.0, 1.0))
        plot_khop_forget_size(df, j('khop_forget_size'), backbones, scenarios)

    print(f"\nAll figures saved to {output_dir}/ (prefix: {bus_system}_)")
