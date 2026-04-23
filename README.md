# Graph Unlearning for EVCS Cyber Attack Localization

**Nanhong Liu** — Department of Mechanical Engineering, The University of Texas at Dallas
*(Extended from IDETC2026-193882: cyber attack detection → modality-level unlearning for localization)*

---

## Overview

Charging Manipulation Attacks (CMAs) on EV charging stations (EVCSs) alter charging profiles to cause voltage violations. This project addresses **attack localization**: given 24-hour bus voltage and charging-power snapshots on an IEEE distribution feeder, predict a **multi-hot label vector** indicating which EVCS buses are under attack — a **graph-level multi-label classification** task.

**Unlearning is reformulated under a Two-Controller data-ownership model:**

- **DSO** (Distribution System Operator) owns bus voltage $V$ and topology — never revoked
- **EVCS operator** owns charging power $P$ at its PCC meter — the only modality that can be revoked under GDPR/CCPA right-to-erasure

Unlearning is therefore **modality-level**: only the $P$-channel at the revoking station is zeroed; $V$ and edges remain intact. This matches the actual data-control boundaries in cyber-physical distribution systems.

| Method | Approach |
|---|---|
| **GDGU** | First-order gradient-difference update + BatchNorm recalibration + recovery fine-tuning |
| **GIF** | Second-order Neumann-series $H^{-1}\Delta\nabla$ correction |
| **IDEA** | Certified IF-based unlearning with bounded-norm update |
| **Retrain-A** | Full retraining from scratch on $P$-zeroed data (physical-leakage floor) |

**Model:** Single-head GNN + 5-way attack-type auxiliary head (`AuxWrapper`). Detection is derived post-hoc as $y_{\text{det}} = \max(\sigma(y_{\text{loc}}))$.

**GNN Backbones:** GCN, GAT, GIN — 3 conv layers + BatchNorm + dropout, mean+max pooling, 2-layer MLP head; 96-d input ($[V_{48}\,\|\,P_{48}]$).

---

## Repository Structure

```
4-GU_EV_loc/
├── README.md
├── Version.md                    # Changelog
├── requirements.txt
├── config/
│   ├── 34bus.yaml                # Hyperparameters + route_a: section
│   └── 123bus.yaml
├── src/
│   ├── __init__.py
│   ├── models.py                 # GCN / GAT / GIN + AuxWrapper
│   ├── data.py                   # A-append features, modality-forget, PyG dataset
│   ├── training.py               # train_model, train_model_joint, evaluate_aux_acc, MIA
│   ├── unlearning.py             # GDGU, GIF, IDEA (share single-head backbone)
│   ├── privacy.py                # L2-a IG, L2-b ΔAUC, L2-c Recon, L2-e attack-type
│   ├── experiment.py             # run_single_trial_route_a (5 methods)
│   └── visualization.py          # Plots (L1 utility + L2 privacy + MIA breakdown)
├── scripts/
│   └── train.py                  # CLI entry point (loads YAML)
├── notebooks/
│   └── Viz_V6.ipynb              # Post-experiment visualization
└── results/                      # Experiment outputs — git-ignored
    └── YYYY-MM-DD_HH/
        ├── {bus}_routeA_results_raw.csv
        ├── {bus}_routeA_results_summary.csv
        ├── {bus}_routeA_epoch_logs.json
        └── *.pdf
```

---

## Environment Setup

Python 3.10, CUDA 12.1.

```bash
conda create -n torch-gpu python=3.10 -y && conda activate torch-gpu

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
| Node features | **96 = V_48 + P_48** | 96 |
| Graph instances | 2,000 | 4,000 |
| EVCS tracked (output dim $K$) | 3 (Bus 814, 852, 890) | 5 (Bus 25, 40, 54, 62, 76) |
| Unlearning scenarios | S1–S3 | S1–S5 |

**A-append feature construction:** For each node, hourly (mean, std) over 24 hours gives 48-d per modality:

$$
(288 \text{ steps}) \xrightarrow{\text{hourly bin}} 24 \xrightarrow{\text{(mean,std)}} 48\text{ dims}, \quad
\mathbf{x} = [V_{48}\,\|\,P_{48}] \in \mathbb{R}^{96}
$$

$V$ from 3-phase-mean bus voltage (all nodes); $P$ from EVCS charging power series (EVCS nodes only, zero elsewhere). $V$ and $P$ are standardized with **separate** `StandardScaler`s.

---

## Quick Start

```bash
conda activate torch-gpu
cd ~/1P_WTT_NVD/Projects/4-GU_EV_loc

python scripts/train.py --bus 34bus                    # cuda:1, all 3 backbones
python scripts/train.py --bus 123bus --gpu 0           # cuda:0 (RTX 4090 #0)
python scripts/train.py --bus 34bus --backbone GAT     # single-backbone run
```

Outputs saved automatically to `results/YYYY-MM-DD_HH/`. Then open `notebooks/Viz_V6.ipynb` (edit Cell 1 to set `DATE_FOLDER`) to generate all plots.

---

## Experiment Design

### Scenarios (cumulative P-channel forget)

| | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| **34-bus** | Bus 814 | +852 | +890 | — | — |
| **123-bus** | Bus 25 | +40 | +54 | +62 | +76 |

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Split | 80 / 10 / 10 (stratified by multi-label hash) |
| Epochs / patience | 300 / 50 (val Macro ROC-AUC) |
| Hidden dim / layers | 128 / 3 |
| Optimizer | Adam (lr=5e-4, wd=1e-4) + ReduceLROnPlateau (GAT lr=1e-4) |
| Loss (joint) | BCE(loc) + γ·CE(attack_type), γ=1.0 |
| Seeds | 10 (42, 77, 88, 124, 137, 226, 347, 499, 666, 999) |
| GDGU damp / max_norm / finetune | 0.1 / 1.0 / 25 epochs |
| GIF iterations / damp / scale | 50 / 0.01 / 50.0 |
| `max_batches` (34 / 123-bus) | 22 / 30 (HVP sub-sampling to avoid OOM) |
| Aux head hidden / n_types | 64 / 5 |
| IG steps | 20 |

---

## Evaluation Metrics

**L1 Utility** (on full test set): Exact Match, Hamming Accuracy, Macro F1, Macro ROC-AUC, Per-EVCS F1/ROC-AUC, F1/ROC split by forget vs retain EVCS.

**L2 Privacy** (verifies the $P$-channel was actually erased):

- **L2-b Occlusion ΔAUC** — *PRIMARY.* $\text{AUC}(P \text{ present}) - \text{AUC}(P \text{ occluded at test time})$. Retrain-A sets the floor ≈ 0; Original sits at +0.17; GDGU/IDEA reach +0.04.
- **L2-a Integrated Gradients** — per-step attribution of logits to $P$-dims at the forget node.
- **L2-c Reconstruction** — MLP decoder from forget-node embedding to low-dim $P$ properties.
- **L2-e Attack-type Accuracy** — 5-way linear probe on the aux head (Nil + 4 attack types).

**MIA (reference only):** MIA_forget / MIA_retain / MIA_AUC. MIA_forget remains above 0.5 even for Retrain-A because physical $V\!\to\!P$ coupling carries residual signature — an inherent grid property, not a method defect.

**Efficiency:** wall-clock Time (s), Peak Memory (MB), Speedup vs Retrain-A.

---

## Results Headline (2026-04-22 run, mean over 10 seeds, all scenarios & backbones)

| Method | 34-bus Macro-ROC | 34-bus ΔAUC | 123-bus Macro-ROC | 123-bus ΔAUC | Speedup |
|---|---|---|---|---|---|
| Original | 1.000 | +0.167 | 0.999 | +0.165 | — |
| **GDGU** | **0.944** | **+0.054** | **0.913** | **+0.068** | **10–11×** |
| GIF | 0.844 | +0.117 | 0.782 | +0.101 | 14–21× (utility collapse) |
| **IDEA** | **0.950** | **+0.042** | **0.906** | **+0.043** | 6–8× |
| Retrain-A | 0.986 | −0.011 | 0.942 | −0.031 | 1× (baseline) |

Full per-scenario / per-backbone breakdowns: see `Version.md` and `results/2026-04-22_11/`.

---

## References

- N. Liu, R. A. Jacob, J. Zhang. "Graph Unlearning for Cyber-Resilient EV Charging Networks." *ASME IDETC-CIE 2026*, IDETC2026-193882.
- R. A. Jacob et al. "PowerBench Dataset – Part 3." 2025. DOI: [10.5281/zenodo.15401290](https://zenodo.org/records/15401290)
- B. Fan et al. "OpenGU: A Comprehensive Benchmark for Graph Unlearning." *PVLDB 2025* — feature-level unlearning taxonomy.
- M. Chen et al. "Graph Unlearning." *ACM CCS 2022*.
- J. Wu et al. "GIF: A General Graph Unlearning Strategy via Influence Function." *WWW 2023*.
- Y. Dong et al. "IDEA: Certified Unlearning for GNNs." *ACM KDD 2024*.
- T. N. Kipf, M. Welling. "GCN." *ICLR 2017*. | P. Veličković et al. "GAT." *ICLR 2018*. | K. Xu et al. "GIN." *ICLR 2019*.
- B. Wu et al. "MIA on GNN." *IEEE ICDM 2021*.
