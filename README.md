# Graph Unlearning for EVCS Cyber Attack Localization

**Nanhong Liu**  
Department of Mechanical Engineering  
The University of Texas at Dallas  
*(Extended from IDETC2026-193882: detection → localization)*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Environment Setup](#3-environment-setup)
4. [Data](#4-data)
5. [Quick Start](#5-quick-start)
6. [Experiment Design](#6-experiment-design)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Results Output](#8-results-output)
9. [References](#9-references)

---

## 1. Project Overview

**Background.** Charging Manipulation Attacks (CMAs) on electric vehicle charging stations (EVCSs) alter charging profiles to induce voltage violations and grid instability. While [Project 3](../3-GDGU_EV_det) addresses **detection** (is there an attack?), this project tackles the harder follow-up problem: **localization** — identifying *which specific EVCS buses* are under attack.

**Task.** Given a set of 24-hour bus voltage snapshots on an IEEE distribution feeder, predict a **multi-hot binary label vector** indicating which EVCS stations are currently under attack. This is a **graph-level multi-label classification** task.

- Each EVCS station corresponds to one output bit (label).
- A scenario may have zero, one, or multiple EVCS buses simultaneously attacked.
- 34-bus: 3 EVCS → output dim = 3
- 123-bus: 5 EVCS → output dim = 5

**Unlearning Problem.** When an EVCS bus owner requests data deletion (GDPR/CCPA right to be forgotten), its voltage measurements must be removed from the trained GNN without full retraining. We formulate this as **node feature unlearning with edge isolation**: the forget bus's features are zeroed and all edges incident to it are removed across all graph instances, then the model parameters are updated via approximate unlearning algorithms.

> **Design note — edge masking.** Edges connected to forget nodes are removed (*edge isolation*) in addition to zeroing features. This prevents the forget node from participating in message passing (neither as sender nor receiver), achieving true node-level information removal and avoiding GNN representation collapse that would occur with feature-zeroing alone.

**Unlearning Algorithms (three, plus gold-standard Retrain):**
1. **GDGU** — First-order gradient-difference update (Δg/λ, clipped) + BatchNorm recalibration + recovery fine-tuning.
2. **GIF** — Second-order Neumann-series H⁻¹Δ∇ parameter correction + BatchNorm recalibration.
3. **IDEA** — GIF update + recovery fine-tuning (same stage as GDGU step 3).

**Graph Unlearning Methods.** Three approximate unlearning algorithms are implemented and compared:

- **GDGU** (Gradient Difference-based Graph Unlearning): First-order gradient-difference update + BatchNorm recalibration + recovery fine-tuning. Lightweight and memory-efficient.
- **GIF** (Graph Influence Function): Second-order Neumann-series approximation of H⁻¹Δ∇ to estimate parameter shift without full retraining. Uses `create_graph=True` HVP; memory-controlled via `max_batches`.
- **IDEA** (GIF + Recovery Fine-tuning): Extends GIF with the same recovery fine-tuning stage used in GDGU, improving post-unlearning utility.

**Comparison.** Each experimental group runs five methods:
- **Original** — trained GNN on D_tr, evaluated on D'_test (original features, no unlearning).
- **GDGU** — approximate unlearning from Original via gradient difference.
- **GIF** — approximate unlearning from Original via influence function (second-order).
- **IDEA** — GIF followed by recovery fine-tuning.
- **Retrain** — full retraining from scratch on D'_tr (forget nodes zeroed), gold-standard reference.

**GNN Backbones.** GCN, GAT, GIN — 3 convolutional layers + BatchNorm + dropout, mean+max dual pooling, 2-layer MLP head.

---

## 2. Directory Structure

```
4-GU_EV_loc/
├── README.md
├── Version.md                    # Version changelog
├── requirements.txt              # Pinned Python dependencies
│
├── config/                       # YAML hyperparameter configs
│   ├── 34bus.yaml                # Hyperparameters for 34-bus experiments
│   └── 123bus.yaml               # Hyperparameters for 123-bus experiments
│
├── src/                          # Reusable Python modules
│   ├── __init__.py               # Public API with __all__
│   ├── models.py                 # GCN / GAT / GIN multi-label classifiers (out_dim=3 or 5)
│   ├── data.py                   # Data loading, graph construction, edge masking
│   ├── training.py               # Train / evaluate (multi-label) / MIA / checkpoint
│   ├── unlearning.py             # GDGU, GIF, IDEA: unlearning implementations
│   ├── experiment.py             # Single-trial runner (Original → GDGU → GIF → IDEA → Retrain)
│   └── visualization.py          # Publication-quality result plots (Times New Roman)
│
├── notebooks/                    # Interactive experiment notebooks
│   ├── GDGU_EV_loc_34.ipynb      # 34-bus standard (3 EVCS, S1–S3)
│   ├── GDGU_EV_loc_123.ipynb     # 123-bus standard (5 EVCS, S1–S5)
│   └── Viz_GDGU_loc.ipynb        # Post-experiment visualization (loads from results/)
│
├── scripts/                      # Non-interactive batch scripts
│   ├── run_experiments.py        # Full experiment runner (parallel seeds via mp.Pool)
│   ├── train.py                  # HPC CLI entry point (loads YAML, single-process)
│   ├── run_juno.sbatch           # SLURM script for UTD Juno (h100 partition)
│   └── run_g2.sbatch             # SLURM script for UTD Ganymede2 (g1mig partition)
│
├── Paper/                        # Paper drafts and figures
│
└── results/                      # Experiment outputs (git-ignored)
    └── YYYY-MM-DD_HH/
        ├── {bus}_results_raw.csv
        ├── {bus}_results_summary.csv
        ├── {bus}_epoch_logs.json  # Includes _metadata block (config, device, timestamp)
        └── *.pdf
```

---

## 3. Environment Setup

**Requirements:** Python 3.10, CUDA 12.1, conda.

```bash
# Step 1 — Create and activate conda environment
conda create -n evcs_gnn python=3.10 -y
conda activate evcs_gnn

# Step 2 — Install PyTorch (CUDA 12.1 build)
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Step 3 — Install PyG C++ extensions
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
            torch-cluster==1.6.3 torch-spline-conv==1.2.2 \
            -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# Step 4 — Install remaining dependencies
pip install -r requirements.txt
```

> **Lab GPU note:** Default device is `cuda:1` (cuda:0 is reserved). Change `DEVICE` in `scripts/run_experiments.py` or pass `--gpu` if using `train.py`.

---

## 4. Data

Raw data is **not tracked by Git** and must be sourced from the PowerBench benchmark (Jacob et al., 2025).

| File | Description |
|---|---|
| `.../3_EVCS Attacks/34_bus/EVCSAttacks_34.pkl` | 2,000 scenarios, 34-bus |
| `.../3_EVCS Attacks/34_bus/34busEx.gml` | 34-bus network topology |
| `.../3_EVCS Attacks/123_bus/EVCSAttacks_123_job*_merged.pkl.gz` | 4,000 scenarios, 123-bus (chunked) |
| `.../3_EVCS Attacks/123_bus/123busEx.gml` | 123-bus network topology |

### Dataset Properties

| | IEEE 34-bus | IEEE 123-bus |
|---|---|---|
| Buses / nodes (N) | 37 | 132 |
| Branches / edges | 36 | 131 |
| Node feature dim (F) | 24 | 24 |
| Total graph instances | 2,000 | 4,000 |
| EVCS stations tracked | 3 | 5 |
| Multi-label output dim | 3 | 5 |
| Unlearning scenarios | S1–S3 | S1–S5 |

### EVCS Bus Mapping

| System | EVCS label | Bus ID |
|---|---|---|
| 34-bus | EVCS 1 | 814 |
| 34-bus | EVCS 2 | 852 |
| 34-bus | EVCS 3 | 890 |
| 123-bus | EVCS 1 | 25 |
| 123-bus | EVCS 2 | 40 |
| 123-bus | EVCS 3 | 54 |
| 123-bus | EVCS 4 | 62 |
| 123-bus | EVCS 5 | 76 |

### Feature Construction

For each of the 288 five-minute snapshots per daily scenario:
1. Compute three-phase mean voltage magnitude per bus: shape `(288,)`.
2. Reshape to `(24, 12)` and take the **max** across each hour → shape `(24,)`.
3. Node feature matrix: `X ∈ R^{N×24}`, one row per bus.

Features are **standardized per experiment**: `StandardScaler` is fit on the training split (flattened node features) and applied to all sets.

### Label Construction

Multi-hot label vector `y ∈ {0,1}^K` (K = number of tracked EVCS). For scenario `i`, `y[k] = 1` if `EVCS k` is listed in `scenario['Targeted Stations']`, otherwise `0`. A scenario with `y = [0,0,...,0]` is a normal (no-attack) scenario.

---

## 5. Quick Start

### Local (Notebook)

```bash
conda activate evcs_gnn
# Open notebooks/GDGU_EV_loc_34.ipynb or GDGU_EV_loc_123.ipynb in JupyterLab / Cursor
```

After any `src/` module change: **Kernel → Restart & Run All** (or use `%autoreload 2` at the top).

### Local (Script — parallel seeds)

```bash
conda activate torch-gpu
cd Projects/4-GU_EV_loc

# Run 34-bus (3 backbones × 3 scenarios × 10 seeds, seeds in parallel via mp.Pool)
python scripts/run_experiments.py 34bus

# Run 123-bus only
python scripts/run_experiments.py 123bus

# Run both sequentially
python scripts/run_experiments.py 34bus 123bus
```

### Local (Script — single process, for debugging / memory control)

```bash
conda activate torch-gpu
cd Projects/4-GU_EV_loc

# 34-bus, all backbones (sequential, reads config/34bus.yaml)
python scripts/train.py --bus 34bus

# 123-bus, GAT first
python scripts/train.py --bus 123bus --backbone GAT
```

> Outputs saved to `results/YYYY-MM-DD_HH/` automatically.

### HPC (SLURM — Juno / Ganymede2)

```bash
# UTD Juno (h100 partition)
sbatch scripts/run_juno.sbatch

# UTD Ganymede2 G2 (g1mig partition)
sbatch scripts/run_g2.sbatch
```

---

## 6. Experiment Design

### Cumulative Unlearning Scenarios

EVCS buses are forgotten cumulatively, one at a time (cumulative forget set):

**34-bus (S1–S3):**

| Scenario | Forget buses | # Forget nodes | % of total nodes |
|---|---|---|---|
| S1 | 814 | 1 | 2.7% |
| S2 | 814, 852 | 2 | 5.4% |
| S3 | 814, 852, 890 | 3 (all EVCS) | 8.1% |

**123-bus (S1–S5):**

| Scenario | Forget buses | # Forget nodes | % of total nodes |
|---|---|---|---|
| S1 | 25 | 1 | 0.8% |
| S2 | 25, 40 | 2 | 1.5% |
| S3 | 25, 40, 54 | 3 | 2.3% |
| S4 | 25, 40, 54, 62 | 4 | 3.0% |
| S5 | 25, 40, 54, 62, 76 | 5 (all EVCS) | 3.8% |

### Training Configuration

| Parameter | Value |
|---|---|
| Dataset split | 70% train / 15% val / 15% test (stratified by multi-label hash) |
| Epochs | up to 300 |
| Early stopping patience | 50 (on val Macro ROC-AUC) |
| Batch size | 32 |
| Hidden dimension | 128 |
| GNN layers | 3 |
| Dropout | 0.3 |
| Pooling | Mean + Max (dual) |
| Initialization | Kaiming uniform (compensates ReLU ~50% zeroing) |
| Optimizer | Adam (η = 5×10⁻⁴, weight decay = 10⁻⁴) |
| LR scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=20) |
| Loss | `BCEWithLogitsLoss(pos_weight=pw)` |
| `pos_weight` | per-label: `neg_count / pos_count` (computed on training split) |
| Gradient clipping | max_norm = 5.0 (during training) |
| Seeds | 10 (42, 77, 88, 124, 137, 226, 347, 499, 666, 999) |

> **Note on pos_weight:** Unlike binary cross-entropy with a single class weight, `BCEWithLogitsLoss(pos_weight=pw)` assigns an independent positive-class weight to each EVCS label, accounting for per-station attack frequency variation.

### GDGU Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `gdgu_damp` λ | 0.1 | Damping: `Δθ = Δg / λ` |
| `gdgu_max_norm` ρ | 1.0 | Gradient clipping norm after damping |
| `gdgu_finetune` E_ft | 25 | Recovery fine-tuning epochs |
| Fine-tune LR η_ft | 1×10⁻⁴ | Reduced LR for fine-tune stage |
| Fine-tune optimizer | Adam (weight_decay=10⁻⁴) | Same as original training |
| BN recalibration | 1 forward pass, no grad | Reset all running stats then recompute |
| Checkpoint selection | Best val Macro ROC-AUC | Over E_ft epochs |

### GIF / IDEA Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `gif_iteration` | 50 | Neumann series iterations for H⁻¹ approximation |
| `gif_damp` | 0.01 | Damping coefficient in Neumann update |
| `gif_scale` | 50.0 | Scale factor for Neumann series stability |
| `idea_finetune` E_ft | 25 | IDEA recovery fine-tuning epochs (same as GDGU) |
| `max_batches` (34-bus) | 22 | HVP sub-sampling: 22/44 batches ≈ 50% of train data |
| `max_batches` (123-bus) | 30 | HVP sub-sampling: 30/88 batches ≈ 34% of train data |

> **`max_batches` rationale.** The HVP computation in `_compute_grad_for_hvp` accumulates the full autograd graph with `create_graph=True`. For GAT on 123-bus (multi-head attention, 2800 training samples), the full 88-batch pass causes OOM on a 24 GB GPU because two computation graphs (original + modified loader) must coexist simultaneously. Sub-sampling to 30 batches reduces peak memory from ~43 GB (full) to ~20 GB while maintaining a statistically representative gradient estimate.

### GAT Architecture Note

GAT uses multi-head attention (`heads=4`) with concatenation for layers 1 and 2, and a single head (averaged) for the final convolutional layer, resulting in:
- Layer 0: `in_dim → 128×4 = 512`
- Layer 1: `512 → 512`
- Layer 2 (final conv): `512 → 128` (heads=1, concat=False)
- Pooling + MLP head: `256 → 128 → out_dim`

Activation in GAT conv layers is ELU (others use ReLU).

---

## 7. Evaluation Metrics

### Multi-label Utility Metrics (on D'_test)

| Metric | Description | Target |
|---|---|---|
| **Exact Match** | Fraction of samples where all K labels are predicted correctly | ↑ |
| **Hamming Accuracy** | `1 - Hamming Loss`; fraction of individual labels correct (averaged over K) | ↑ |
| **Macro F1** | Unweighted average F1 across K EVCS labels (threshold = 0.5) | ↑ |
| **Macro ROC-AUC** | Unweighted average ROC-AUC across K EVCS labels (threshold-free) | ↑ |
| **Per-EVCS ROC-AUC** | Individual ROC-AUC for each EVCS station | ↑ |
| **Per-EVCS F1** | Individual F1 for each EVCS station | ↑ |

> Macro averages skip any label column where only one class is present in the test set (e.g., all-zero in a small split), replacing it with 0.5 (ROC) or 0.0 (F1).

### Unlearning Metrics

| Metric | Description | Target |
|---|---|---|
| **MIA-AUC** | Loss-based Membership Inference Attack AUC (Wu et al., ICDM 2021). Higher loss on members vs. non-members → high AUC = data still memorized. | → **0.5** = perfect unlearning |
| **Time (s)** | Wall-clock time for the unlearning step or full retraining | ↓ |
| **Peak Memory (MB)** | Peak GPU memory allocation during the method | ↓ |
| **Speedup** | T_Retrain / T_method | ↑ |

**MIA implementation.** Uses `BCEWithLogitsLoss(reduction='none', pos_weight=pw)` averaged across labels per sample. Scores are negated losses (higher score = model thinks sample is a member). AUC is computed between forget-set train split (members) and test-set (non-members) on the unlearned model.

> **Physical correlation note.** In IEEE distribution feeders, voltage disturbances at an EVCS bus propagate to physically adjacent buses via power flow (Kirchhoff's laws). As a result, even after zeroing the EVCS bus features and isolating its edges, neighboring buses retain residual attack signatures. This explains why MIA-AUC remains elevated (~0.62–0.82) even after full Retrain — not a flaw in the unlearning method, but an inherent property of the grid topology. Extending the forget set to k-hop neighbors would reduce MIA further at the cost of lower utility.

---

## 8. Results Output

All outputs are saved to a single `results/YYYY-MM-DD_HH/` folder (no separate `checkpoints/` or `logs/` sub-folders).

| File | Contents |
|---|---|
| `{bus}_results_raw.csv` | One row per (backbone, scenario, seed, method): ExMatch, Hamming_Acc, Macro_ROC, Macro_F1, ROC_EVCS1..K, F1_EVCS1..K, MIA_AUC, Time, Peak_Memory_MB |
| `{bus}_results_summary.csv` | Mean ± std grouped by backbone × scenario × method |
| `{bus}_epoch_logs.json` | Per-epoch `train_loss`, `val_roc`; includes `_metadata` block (bus_system, timestamp, device, data dimensions, config, model_params, scenarios) |
| `{bus}_ExMatch_comparison.pdf` | Bar chart: Exact Match across scenarios (5 methods) |
| `{bus}_ExMatch_trend.pdf` | Line chart: Exact Match vs. cumulative forget set size |
| `{bus}_HammingAcc_*.pdf` | Hamming Accuracy comparison and trend |
| `{bus}_MacroF1_*.pdf` | Macro F1 comparison and trend |
| `{bus}_MacroROC_*.pdf` | Macro ROC-AUC comparison and trend |
| `{bus}_MIA_AUC_comparison.pdf` | MIA-AUC with ideal 0.5 reference line (5 methods) |
| `{bus}_GU_method_comparison.pdf` | Side-by-side comparison of all GU methods |
| `{bus}_PerEVCS_ROC_breakdown.pdf` | Per-station ROC-AUC breakdown heatmap |
| `{bus}_F1_vs_MIA_tradeoff.pdf` | Utility–forgetting tradeoff scatter plot |
| `{bus}_Time_comparison.pdf` | Wall-clock time + speedup bar chart |
| `{bus}_Memory_usage.pdf` | Peak GPU memory comparison |

> Visualization is decoupled from training: `train.py` / `run_experiments.py` only save CSVs and JSONs. Run `notebooks/Viz_GDGU_loc.ipynb` afterwards to generate all PDF figures from a chosen results folder.

---

## 9. References

**Detection (predecessor project):**

> N. Liu, R. A. Jacob, and J. Zhang. "Graph Unlearning for Cyber-Resilient Electric Vehicle Charging Networks." *ASME IDETC-CIE 2026*, Paper IDETC2026-193882, Houston, TX, August 2026.

**Dataset:**

- R. A. Jacob, M. J. Uddin, D. R. Olojede, B. Coskunuzer, and J. Zhang. "PowerBench Dataset – Part 3: Cyber Attacks on EVCS." 2025. DOI: [10.5281/zenodo.15401290](https://zenodo.org/records/15401290)

- R. A. Jacob, M. J. Uddin, J. Wang, B. Coskunuzer, and J. Zhang. "Cyber Attack Detection in Electric Vehicle Charging Stations Using Topological Data Aided Learning." *IEEE PESGM*, 2025. DOI: 10.1109/PESGM52009.2025.11225416

**Graph Unlearning:**

- M. Chen, Z. Zhang, T. Wang, M. Backes, M. Humbert, and Y. Zhang. "Graph Unlearning." *ACM CCS 2022*. DOI: 10.1145/3548606.3559352

- J. Wu, Y. Yang, Y. Qian, Y. Sui, X. Wang, and X. He. "GIF: A General Graph Unlearning Strategy via Influence Function." *WWW 2023*. DOI: 10.1145/3543507.3583521

- Y. Dong, B. Zhang, Z. Lei, N. Zou, and J. Li. "IDEA: A Flexible Framework of Certified Unlearning for Graph Neural Networks." *ACM KDD 2024*. DOI: 10.1145/3637528.3671744

- J. Cheng, G. Dasoulas, H. He, C. Agarwal, and M. Zitnik. "GNNDelete: A General Strategy for Unlearning in Graph Neural Networks." *ICLR 2023*.

- C. Pan, E. Chien, and O. Milenkovic. "Unlearning Graph Classifiers with Limited Data Resources." *WWW 2023*. DOI: 10.1145/3543507.3583547

**GNN Backbones:**

- T. N. Kipf and M. Welling. "Semi-Supervised Classification with Graph Convolutional Networks." *ICLR 2017*. (GCN)

- P. Veličković et al. "Graph Attention Networks." *ICLR 2018*. (GAT)

- K. Xu et al. "How Powerful are Graph Neural Networks?" *ICLR 2019*. (GIN)

**Membership Inference Attack:**

- B. Wu, X. Yang, S. Pan, and X. Yuan. "Adapting Membership Inference Attacks to GNN for Graph Classification: Approaches and Implications." *IEEE ICDM 2021*.

**Multi-label Classification:**

- B. Xu, P. Wang, Z. Zhao, B. Wang, X. Wang, and Y. Wang. "When Imbalance Meets Imbalance: Structure-Driven Learning for Imbalanced Graph Classification." *WWW 2024*.
