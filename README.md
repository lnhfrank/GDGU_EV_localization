# Graph Unlearning for EVCS Cyber Attack Localization

**Nanhong Liu** — Department of Mechanical Engineering, The University of Texas at Dallas  
*(Extended from IDETC2026-193882: binary detection → multi-label localization)*

---

## Overview

Charging Manipulation Attacks (CMAs) on EV charging stations (EVCSs) alter charging profiles to cause voltage violations. This project addresses **attack localization**: given 24-hour bus voltage snapshots on an IEEE distribution feeder, predict a **multi-hot label vector** indicating which EVCS buses are under attack — a **graph-level multi-label classification** task.

When an EVCS owner requests data deletion (GDPR right to be forgotten), its voltage measurements must be removed from the trained GNN without full retraining. We formulate this as **node feature unlearning with edge isolation**: forget-node features are zeroed and incident edges removed, then model parameters are updated via three approximate unlearning algorithms:

| Method | Approach |
|---|---|
| **GDGU** | First-order gradient-difference update + BatchNorm recalibration + recovery fine-tuning |
| **GIF** | Second-order Neumann-series H⁻¹Δ∇ correction + BatchNorm recalibration |
| **IDEA** | GIF + recovery fine-tuning |
| **Retrain** | Full retraining from scratch on masked data (gold-standard reference) |

**GNN Backbones:** GCN, GAT, GIN — 3 conv layers + BatchNorm + dropout, mean+max pooling, 2-layer MLP head.

---

## Repository Structure

```
4-GU_EV_loc/
├── README.md
├── Version.md                    # Changelog
├── requirements.txt
├── config/
│   ├── 34bus.yaml                # Hyperparameters for 34-bus
│   └── 123bus.yaml               # Hyperparameters for 123-bus
├── src/
│   ├── __init__.py
│   ├── models.py                 # GCN / GAT / GIN classifiers
│   ├── data.py                   # Data loading, graph construction, edge masking
│   ├── training.py               # Train / evaluate / MIA
│   ├── unlearning.py             # GDGU, GIF, IDEA implementations
│   ├── experiment.py             # Single-trial runner (5 methods)
│   └── visualization.py          # Result plots (Times New Roman)
├── scripts/
│   └── train.py                  # CLI entry point (loads YAML)
├── notebooks/
│   └── Viz_GDGU_loc.ipynb        # Post-experiment visualization
└── results/                      # Experiment outputs — git-ignored
    └── YYYY-MM-DD_HH/
        ├── {bus}_results_raw.csv
        ├── {bus}_results_summary.csv
        ├── {bus}_epoch_logs.json
        └── *.pdf
```

---

## Environment Setup

Python 3.10, CUDA 12.1.

```bash
conda create -n evcs_gnn python=3.10 -y && conda activate evcs_gnn

pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
            torch-cluster==1.6.3 torch-spline-conv==1.2.2 \
            -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

pip install -r requirements.txt
```

> Default device is `cuda:1`. Change via `--gpu` flag in `train.py`.

---

## Data

Raw data is **not tracked by Git**. Source from the [PowerBench benchmark](https://zenodo.org/records/15401290) (Jacob et al., 2025).

| File | Description |
|---|---|
| `.../34_bus/EVCSAttacks_34.pkl` | 2,000 scenarios, 34-bus |
| `.../34_bus/34busEx.gml` | 34-bus topology |
| `.../123_bus/EVCSAttacks_123_job*_merged.pkl.gz` | 4,000 scenarios, 123-bus |
| `.../123_bus/123busEx.gml` | 123-bus topology |

| | 34-bus | 123-bus |
|---|---|---|
| Nodes / Edges | 37 / 36 | 132 / 131 |
| Node features | 24 (hourly peak voltage) | 24 |
| Graph instances | 2,000 | 4,000 |
| EVCS tracked (output dim) | 3 (Bus 814, 852, 890) | 5 (Bus 25, 40, 54, 62, 76) |
| Unlearning scenarios | S1–S3 | S1–S5 |

**Feature construction:** For each 5-min snapshot, compute 3-phase mean voltage per bus → reshape `(288,)` to `(24, 12)` → take hourly max → `X ∈ R^{N×24}`. Standardized via `StandardScaler` fit on training split.

---

## Quick Start

```bash
conda activate evcs_gnn
cd Projects/4-GU_EV_loc

# Run 34-bus (all backbones, reads config/34bus.yaml)
python scripts/train.py --bus 34bus

# Run 123-bus, GAT only
python scripts/train.py --bus 123bus --backbone GAT
```

Outputs saved automatically to `results/YYYY-MM-DD_HH/`. Then open `notebooks/Viz_GDGU_loc.ipynb` to generate plots.

---

## Experiment Design

### Unlearning Scenarios (cumulative forget set)

| | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| **34-bus** | Bus 814 | +852 | +890 | — | — |
| **123-bus** | Bus 25 | +40 | +54 | +62 | +76 |

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Split | 70/15/15 (stratified by multi-label hash) |
| Epochs / patience | 300 / 50 (val Macro ROC-AUC) |
| Hidden dim / layers | 128 / 3 |
| Optimizer | Adam (lr=5e-4, wd=1e-4) + ReduceLROnPlateau |
| Loss | `BCEWithLogitsLoss(pos_weight=pw)` per-label |
| Seeds | 10 (42, 77, 88, 124, 137, 226, 347, 499, 666, 999) |
| GDGU damp / max_norm / finetune | 0.1 / 1.0 / 25 epochs |
| GIF iterations / damp / scale | 50 / 0.01 / 50.0 |
| `max_batches` (34 / 123-bus) | 22 / 30 (HVP sub-sampling to avoid OOM) |

> **`max_batches`:** GIF builds two autograd graphs simultaneously with `create_graph=True`. For GAT on 123-bus, full 88-batch pass exceeds 24 GB GPU memory. Sub-sampling to 30 batches reduces peak to ~20 GB.

---

## Evaluation Metrics

**Utility** (on masked test set D′_test): Exact Match, Hamming Accuracy, Macro F1, Macro ROC-AUC, Per-EVCS F1/ROC-AUC.

**Unlearning**: MIA-AUC (loss-based membership inference; target → 0.5), Time (s), Peak Memory (MB), Speedup vs. Retrain.

> MIA-AUC remains elevated (~0.62–0.82) even after full Retrain due to physical power-flow correlations: neighboring buses retain residual attack signatures from the forget node, an inherent property of the grid topology.

---

## References

- N. Liu, R. A. Jacob, J. Zhang. "Graph Unlearning for Cyber-Resilient EV Charging Networks." *ASME IDETC-CIE 2026*, IDETC2026-193882.
- R. A. Jacob et al. "PowerBench Dataset – Part 3." 2025. DOI: [10.5281/zenodo.15401290](https://zenodo.org/records/15401290)
- M. Chen et al. "Graph Unlearning." *ACM CCS 2022*.
- J. Wu et al. "GIF: A General Graph Unlearning Strategy via Influence Function." *WWW 2023*.
- Y. Dong et al. "IDEA: Certified Unlearning for GNNs." *ACM KDD 2024*.
- T. N. Kipf, M. Welling. "GCN." *ICLR 2017*. | P. Veličković et al. "GAT." *ICLR 2018*. | K. Xu et al. "GIN." *ICLR 2019*.
- B. Wu et al. "MIA on GNN." *IEEE ICDM 2021*.
