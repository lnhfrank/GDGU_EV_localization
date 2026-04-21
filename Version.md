# Version History — GU_EV_loc

---

## V5.3 — GU_EV_loc.20260421 *(not committed — exploratory)*

**Date:** 2026-04-21
**Branch:** `feat/two-controller-unlearning` (planned)
**Status:** Conceptual design + single-seed smoke test passed; L2 attack protocol needs refinement
**Results:** `results/smoke_A_2026-04-21_15/`

### Motivation

V5.0's dual-channel architecture resolved the detection/localization leakage problem but inherited a deeper semantic issue: **who actually owns which data in the EVCS–grid system?**

- **DSO (Distribution System Operator)** owns bus voltage (V) and branch flow measurements. V is collected via SCADA/PMU infrastructure.
- **EVCS operator** owns charging power (P) at each station's PCC (Point of Common Coupling) meter. Under GDPR/CCPA right-to-erasure, **P is the only variable the EVCS can legitimately revoke**.

V5.0 and earlier versions implicitly "forgot an EVCS" by zeroing the voltage features at its bus and masking incident edges — but **those are DSO-owned quantities the EVCS has no claim over**. This is not a right-to-be-forgotten scenario; it is "DSO unlearns its own bus."

The Jacob 2025 EVCS TDA paper (same group) explicitly frames EVCS attack detection as "DSO-level, from the grid operator's perspective" — under that framing the unlearning question is semantically ill-defined.

V5.3 reformulates the problem as a **two-controller data-ownership model** and implements **modality-level unlearning**: P is exposed as an EVCS-contributed input channel, and unlearning removes only the P channel of the revoking station while V (DSO-owned) stays intact.

### Key Updates

1. **Two-controller input design (A-append)** (`scripts/smoke_A/smoke_dualch_A.py`, prototype)
   - Node feature dim extended from 48 to **96 = [V_48 | P_48]**
   - For EVCS nodes: V-half = hourly mean+std of voltage; P-half = hourly mean+std of charging power (from `EVCSAttacks_34.pkl['EVCS power series']`)
   - For non-EVCS nodes: V-half populated, P-half zeroed (binary mask implicit in layout)
   - V and P scaled by **separate StandardScalers** fit on the training subset (prevents cross-station P-magnitude leakage of identity)

2. **Modality-level unlearning semantics** (`scripts/smoke_A/` prototype)
   - New `build_pyg_dataset(..., forget_P_only=True)`: zeros dims 48:96 at the forget node; **V stays intact, edges are NOT masked** (contrast with V5.0's `build_graphs()` which zeros all 48 V-dims and removes incident edges)
   - Rationale: only the EVCS-owned modality is revoked; DSO-owned V remains usable by the grid operator for legitimate detection/localization
   - The V5.0 GDGU-dual pipeline applies unchanged; the difference is only in what gets zeroed in the training data

3. **L1/L2 privacy evaluation matrix** (new, prototype)
   - **L1 — Utility Preservation**: Macro-F1 / Macro-ROC (localization), DetAUC (detection)
   - **L2 — Modality Erasure** (4 sub-metrics):
     - **L2-a Attribution**: gradient norm on P-channel at the forget node
     - **L2-b Occlusion ΔAUC**: AUC(P present) − AUC(P occluded)
     - **L2-c Reconstruction attack**: MLP decoder from graph embedding to P_48; MSE vs population baseline
     - **L2-d Property inference** (planned, not yet implemented): 3-way EVCS-ID classifier on graph embedding

### Smoke-test results (GIN × S1-0 × seed 42, single trial)

| Method | Macro-F1 | Macro-ROC | DetAUC | Time |
|---|---|---|---|---|
| Original (V+P)              | 1.0000 | 1.0000 | 0.9074 | 11.8s |
| GDGU-dual-A (forget P_814)  | 0.9769 | 0.9912 | 0.9074 | 6.4s  |
| Retrain-A   (P_814 zeroed)  | 0.9861 | 0.9952 | 0.9247 | 17.5s |

| L2 metric | Original | GDGU-dual-A | Retrain-A | Verdict |
|---|---|---|---|---|
| Occlusion ΔAUC         | +0.0883  | −0.0088  | −0.0510  | ✅ strong erasure signal |
| P-channel grad norm    | 0.000107 | 0.003658 | 0.008213 | ❌ loss-saturation confound |
| Recon MSE / baseline   | 0.502    | 0.420    | 0.510    | ❌ V→P physical bypass |

**L1 findings**: Original F1=1.0 reflects task trivialization — seeing P directly makes attack detection near-deterministic (Algorithm 1 in Jacob 2025 attacks P, not V). Once the P shortcut is removed, GDGU-dual-A retains F1=0.977 via V-only physical propagation. Detection AUC is identical to Original by construction (channel isolation).

**L2-b works** (Occlusion ΔAUC): Original drops 8.8 AUC points when P is hidden at test time; unlearned models show zero dependence on P — the cleanest evidence of successful modality erasure.

**L2-a fails** (Attribution): Original's loss ≈ 0 at F1=1.0 saturates all gradients to ≈ 0, inverting the metric direction. Needs replacement with logit-gradient or Integrated Gradients.

**L2-c fails** (Reconstruction): the pooled graph embedding still contains V at bus 814 (DSO-owned, never masked), and V is physically determined by P, so the decoder bypasses the model's memory and reconstructs P through the physical path. Needs replacement with node-level embedding + property-level attack targets.

### Pending work

**1. L2 attack protocol 2.0** (immediate next step — refine + expand):
- **L2-a Attribution (fix)**: replace loss-gradient with **Integrated Gradients on logits** at the forget node (solves F1=1.0 saturation that collapsed gradients to ≈0)
- **L2-c Reconstruction (redesign)**: swap pooled graph embedding for **node-level embedding at the forget node**; targets shift from raw P_48 to **weakly-V-correlated properties** (reduces V→P physical bypass)
- **L2-d Property Inference (implement)**: frozen-backbone linear probe for **EVCS-ID 3-way** on forget-node embedding
- **L2-e Attack-Type Inference (NEW — promoted to primary metric)**: frozen-backbone linear probe for **5-way attack-type** (nil + 4 attack types) on forget-node embedding. Three-point comparison:
  - Original (V+P) — attack-type acc = ceiling (model sees P directly)
  - GDGU-dual-A (P→0) — expected to drop if modality erasure works
  - Retrain-A (trained without P at forget node) — **physical-leakage floor** (what V alone leaks)
  - Verdict rules: `GDGU ≈ Retrain` ✅ effective erasure; `GDGU >> Retrain` ❌ residual P-memory; `Retrain itself high` → report as inherent grid-physics limit (same framing as V2.0 MIA-AUC narrative)

**2. Efficiency evaluation** (new, required for paper narrative):
- **Wall-clock time** per unlearning run, median ± std over 10 seeds on cuda:1
- **Speedup ratio** = `t_retrain / t_{GDGU, GIF, IDEA}`
- **Gradient-step count**: Retrain-A (~300 epochs × full loader) vs GDGU/GIF/IDEA (25 fine-tune epochs + one-shot gradient/HVP op)
- **Quality–Efficiency Pareto**: x=time, y=Macro-F1 on retain set — expected: GDGU top-left, Retrain top-right; re-uses V2.0 Table pattern

**3. Framing of novelty** (documentation, no code):
- Primary contribution: the **two-controller + modality-level forgetting** problem formulation (Route A)
- Secondary: Route A is demonstrated across **GDGU / GIF / IDEA / Retrain** to establish the formulation as an **algorithm-agnostic plug-in framework** (mirrors task-aware MU paper's positioning)
- Implication: all V2.0 unlearning infrastructure is reusable; V5.3 changes only the data-path semantics

**4. Production integration** (planned — files to touch):
- `src/data.py`: add `build_P_features()` (hourly mean+std = 48d from `EVCSAttacks_34.pkl['EVCS power series']`); extend `load_evcs_data()` return; add `fit_separate_scalers()` for V/P; modify `build_graphs()` to accept `forget_P_only=True` (zero dims 48:96 only, keep edges)
- `src/models.py`: bump DualChannel_Graph `in_dim` 48 → 96
- `src/experiment.py`: new `run_single_trial_dual_A()` wrapping A-append + modality-forget + 4 methods
- `src/privacy.py` **(new module)**: `L2_a_integrated_gradients()`, `L2_b_occlusion_delta_auc()`, `L2_c_reconstruction_node_emb()`, `L2_d_property_inference()`, `L2_e_attack_type_inference()`, `measure_unlearn_efficiency()`
- `config/34bus.yaml`: add `route_a:` (v_dim=48, p_dim=48, node_feat_dim=96) and `privacy:` (probe epochs, IG steps)
- `scripts/train.py`: add `--route A` flag

**5. Multi-seed validation**: 10 seeds × {GCN, GAT, GIN} × {S1-0, S2-0, S3-0} × {Original, GDGU, GIF, IDEA, Retrain}

**6. A1 ablation**: also forget V_at_PCC alongside P (PCC meter is EVCS-owned per Jacob Fig.1b)

### Files changed (vs V5.0) — so far

- `scripts/smoke_A/smoke_dualch_A.py` — new standalone smoke-test prototype (data augmentation + L1/L2 evaluation, single seed)
- `scripts/smoke_A/smoke_A.log` — full smoke-test run log
- `results/smoke_A_2026-04-21_15/` — smoke-test outputs
- `Version.md` — V5.3 entry added

**Note**: V5.3 changes are currently confined to `scripts/smoke_A/` as a prototype. Production integration into `src/` is deferred until the L2 attack protocol 2.0 (including L2-e) is refined and smoke-test results are consistent across seeds.

---

## V5.0 — GU_EV_loc.20260420 *(not committed — exploratory)*

**Date:** 2026-04-20
**Branch:** `feat/dual-channel-gu`
**Status:** Implementation complete, smoke-test passed; awaiting multi-seed validation
**Results:** `results/2026-04-20_19/` (single-seed smoke test)

### Motivation

V2.0–V4.0 experiments consistently showed MIA-AUC well above 0.5 even after full Retrain. Root cause analysis (see V4.0 Privacy Radius Analysis and `/tmp/graph_level_feat.py` eval) identified a previously unaddressed leakage channel: **graph-level voltage dynamics** (e.g., system-wide temporal/spatial gradients) carry strong attack signatures that **cannot be masked by node-level forgetting**. Non-EVCS graph-level features achieve detection AUC nearly equal to EVCS-node features (EVCS 3 Type 3: 0.888 vs 0.887), proving attack signals propagate system-wide.

V5.0 introduces a **dual-channel architecture with strict channel isolation** (Scheme A) to physically separate detection-relevant graph-level signals from localization-relevant node-level signals. This enables **selective unlearning** — forget only the localization capability for target EVCS, while preserving the detection capability that is not privacy-sensitive.

### Key Updates

1. **Dual-channel model architecture** (`src/models.py`)
   - Refactored `GCN_Graph` / `GAT_Graph` / `GIN_Graph` to expose `encode(data)` method returning pooled graph embedding `[B, 2*hid_dim]` (backward-compatible; `forward(data)` unchanged)
   - New `DualChannel_Graph` class wraps any backbone with strict isolation:
     - **Node channel**: GNN backbone → Loc head → `y_loc [B, K]`
     - **Graph channel**: MLP on graph-level features → Det head → `y_det [B]`
     - The two heads share NO parameters; `loc_head` never sees `graph_feat`
   - Graph MLP uses LayerNorm (not BN) to avoid interaction with BN-recalibrate during unlearning
   - `freeze_graph_channel()` / `unfreeze_all()` methods for unlearning control

2. **Graph-level feature extraction** (`src/data.py`)
   - New `build_graph_features(all_V, edge_index)` → 120-dim per graph:
     - **TempGrad (72d)**: system-wide temporal |dV/dt| — mean, per-hour max-avg, per-hour std-avg across nodes
     - **EdgeGrad-light (48d)**: spatial |V_u - V_v| across edges — mean and std per hour
   - `load_evcs_data()` now returns `all_V` (phase-averaged raw time series `[G, N, 288]`) for downstream feature extraction
   - `build_graphs()` accepts optional `graph_feat` and auto-derives `y_det = 1(any EVCS attacked)`; `graph_feat` is NOT masked by `forget_indices` (preserves detection capability)
   - New `fit_graph_feat_scaler()` for StandardScaler on training subset

3. **Multi-task training** (`src/training.py`)
   - `train_model_dual()`: loss = α · BCE(det) + β · BCE(loc). Default α=β=1 (strict isolation: the two channels don't share params, so no cross-contamination risk)
   - `evaluate_model_dual()`: localization metrics + `det_acc` / `det_auc` / `det_f1`
   - `compute_mia_auc_dual()`: MIA uses ONLY localization loss (the privacy-protected channel)

4. **Dual-channel GDGU** (`src/unlearning.py`)
   - `gdgu_dual_unlearn()`: freezes `graph_mlp` + `det_head` before gradient computation; only Node channel (backbone + loc_head) receives the GDGU update
   - Gradient difference computed from localization loss only
   - `_finetune_after_gdgu_dual()` continues optimizing only unfrozen params

5. **Experiment pipeline** (`src/experiment.py`, `scripts/train.py`)
   - New `run_single_trial_dual()` runs Original → GDGU-dual → Retrain (3 methods; GIF/IDEA deferred to V5.1)
   - Retrain baseline also uses dual-channel architecture for fair comparison
   - `--dual` CLI flag on `scripts/train.py`; output files tagged `34bus_dual_*`

6. **Config** (`config/34bus.yaml`)
   - New `dual_channel:` section with `graph_feat_dim`, `graph_mlp_hidden`, `graph_mlp_out`, `alpha`, `beta`

### Architecture — Scheme A (Strict Isolation)

```
Input:  x [N, 48] + graph_feat [120]

Node Channel (GNN, BN):                  Graph Channel (MLP, LayerNorm):
  GCN/GAT/GIN backbone                     120 → 64 → 32
    ↓ encode() → node_emb [B, 2*hid]        ↓ graph_emb_feat [B, 32]
    ↓                                        ↓
  Loc Head  → y_loc [B, K]                Det Head → y_det [B]
  (localization only)                     (detection only)

Unlearning: freeze graph_mlp + det_head → only Node Channel + Loc Head
are updated by GDGU. Detection capability preserved by construction.
```

### Smoke-test results (GIN × S1-0 × seed 42, single trial)

| Method | ExMatch | Macro ROC | Macro F1 | Det Acc | MIA_forget | Time |
|---|---|---|---|---|---|---|
| Original   | 0.840 | 0.978 | 0.931 | 0.883 | — | 33.9s |
| GDGU-dual  | 0.850 | 0.983 | 0.942 | 0.883 | **0.621** | 7.7s |
| Retrain    | 0.840 | 0.986 | 0.939 | 0.883 | 0.638 | 56.8s |

Observations:
- Pipeline runs end-to-end without errors
- Detection accuracy identical across Original / GDGU / Retrain (0.883) — confirms graph channel is correctly preserved
- GDGU-dual's MIA_forget (0.621) below Retrain's (0.638) on this single seed — trend supports hypothesis but needs multi-seed validation
- GDGU 7.3× speedup vs Retrain

### Pending validation

- Multi-seed (10 seeds) run on GIN × S1-0 for statistically meaningful MIA comparison
- Side-by-side with V2.0 single-channel (same seed 42) to isolate dual-channel contribution
- Extension to all 3 backbones × 4 scenarios × 4 k-hops if hypothesis holds

### Files changed (vs V4.0)

- `src/models.py` — added `encode()` method to GCN/GAT/GIN; added `DualChannel_Graph` class
- `src/data.py` — added `build_graph_features()`, `fit_graph_feat_scaler()`; extended `build_graphs()` with graph_feat/y_det support; `load_evcs_data()` now returns `all_V`
- `src/training.py` — added `train_model_dual()`, `evaluate_model_dual()`, `compute_mia_auc_dual()`
- `src/unlearning.py` — added `gdgu_dual_unlearn()` with graph-channel freezing
- `src/experiment.py` — added `run_single_trial_dual()` and `_build_result_dual()`
- `config/34bus.yaml` — added `dual_channel:` section
- `scripts/train.py` — added `--dual` flag

---

## V4.0 — GU_EV_loc.20260416

**Date:** 2026-04-16
**Branch:** `feat/extended-forget-set`
**Status:** Phase fix applied, pending re-training to validate impact

### Motivation

V3.0 feature extraction used 3-phase averaging (`voltages.mean(axis=1)`) for all buses. However, IEEE 34-bus has 8 single-phase buses and IEEE 123-bus has 61 non-3-phase buses (57 single-phase + 4 two-phase). For these buses, the two/one inactive phases are zero, diluting the true voltage by 3× (single-phase) or 1.5× (two-phase). The GNN saw single-phase buses at ~0.34 p.u. while neighboring 3-phase buses sit at ~1.03 p.u. — a 3× feature gap that is a phase artifact, not an attack signal.

Separately, a Privacy Radius Analysis was conducted to understand how attack voltage signatures propagate through the network, providing physical grounding for feature engineering decisions.

### Key Updates

1. **Critical bug fix: active-phase voltage averaging**
   - `src/data.py` line 87: `voltages.mean(axis=1)` → averages only phases with mean > 0.1 p.u.
   - 34-bus: 8/37 nodes corrected (810, 818, 820, 822, 826, 856, 864, 838) — features shift from ~0.34 to ~1.03
   - 123-bus: 61/132 nodes corrected (57 single-phase × 3.0x, 4 two-phase × 1.5x)
   - All prior V2.0/V3.0 experiment results were trained on incorrect features for these nodes

2. **Privacy Radius Analysis notebook** (`notebooks/Privacy_Radius_Analysis.ipynb`)
   - 33-cell analysis notebook with sequential `[N]` numbering
   - Hop-based AUC decay analysis for both 34-bus and 123-bus
   - Section 10 (Cells 29–33): Matched-pair controlled analysis
     - Matches each attack scenario to closest normal by source branch flow (Euclidean distance)
     - Isolates causal attack effect on per-bus voltage
     - Quantifies voltage regulator attenuation (36–51% for Type 2/3/4 across 814r, 852r)
   - Outputs: `matched_pair_feeder_path.pdf`, `matched_pair_topology.pdf`, plus 10+ other PDFs
   - Same active-phase fix applied to notebook's `_active_phase_mean()` helper

3. **Feature engineering analysis** (analytical results, no model code changes yet)
   - Compared per-bus AUC across 5 feature representations:

     | Feature          | Dim | Type 1 | Type 2 | Type 3 | Type 4 |
     | ---------------- | --- | ------ | ------ | ------ | ------ |
     | V3.0 mean+std    | 48  | 0.524  | 0.738  | 0.688  | 0.693  |
     | 1st-order diff   | 287 | 0.913  | 0.978  | 0.974  | 0.971  |
     | Hourly peaks     | 24  | 0.430  | 0.585  | 0.628  | 0.638  |
     | Raw 288-step     | 288 | 0.541  | 0.848  | 0.837  | 0.827  |

   - Key finding: V3.0's hourly mean+std loses 5-min temporal position — time-shift attacks (Type 2/3/4) are invisible at remote buses
   - Proposed roadmap: V4.0 features (96d: +gradient stats) → V5.0 features (287d: raw diff)

### Impact assessment

| Dataset | Affected nodes | % of network | Feature error |
| ------- | -------------- | ------------ | ------------- |
| 34-bus  | 8 / 37         | 21.6%        | 3.0× underestimated |
| 123-bus | 61 / 132       | 46.2%        | 3.0× or 1.5× underestimated |

### Files changed (vs V3.0)

- `src/data.py` — active-phase averaging in `load_evcs_data()` (bug fix)
- `notebooks/Privacy_Radius_Analysis.ipynb` — new 33-cell analysis notebook with matched-pair analysis

---

## V3.0 — GU_EV_loc.20260414 *(not committed — exploratory)*

**Date:** 2026-04-14
**Branch:** `feat/extended-forget-set`
**Status:** Implementation complete, pending experiment results

### Motivation

V2.0 experiments revealed that MIA-AUC remains well above 0.5 even for Retrain (34-bus: 0.73–0.83, 123-bus: 0.59–0.65). Root cause: physical power flow propagates attack signatures to neighboring buses. Forgetting only the EVCS bus itself leaves "second-hand evidence" intact, allowing the model to reconstruct the removed information via neighbor voltages.

### Key Updates

1. **k-hop neighbor expansion for forget sets**
  - New `expand_forget_khop(forget_indices, edge_index, k)` in `src/data.py`
  - BFS-based expansion: k=0 (EVCS only), k=1 (+1-hop neighbors), k=2 (+2-hop neighbors)
  - Edge masking + feature zeroing applied to the expanded set
2. **Scenario naming: Sn → Sn-k**
  - `build_scenarios()` in `scripts/train.py` now generates `Sn-k` format
  - Example: S1-0 (1 node, 2.7%), S1-1 (3 nodes, 8.1%), S1-2 (5 nodes, 13.5%) for 34-bus
  - CLI: `--khop 1` runs only k=1; `--scenario S2-1` runs a specific scenario
3. **YAML config: `k_hops` parameter**
  - `config/34bus.yaml` and `config/123bus.yaml`: `k_hops: [0, 1, 2]`
  - Default `[0]` when absent (backward-compatible)
4. **Cross-k visualization**
  - `plot_khop_comparison()`: line chart showing metric vs k for each base scenario Sn, per backbone
  - `plot_khop_forget_size()`: bar chart showing forget set size across all Sn-k scenarios
  - `plot_all()` auto-detects Sn-k format and generates k-hop figures
  - `Viz_GDGU_loc.ipynb` Cell 4–5: dedicated k-hop comparison section
5. **Scenario sort updated**
  - `_scenario_sort_key()` handles both legacy `S1` and new `S1-0` formats
  - Ordering: S1-0 < S1-1 < S1-2 < S2-0 < ...

### Forget set sizes


| Scenario                            | k=0      | k=1        | k=2        |
| ----------------------------------- | -------- | ---------- | ---------- |
| **34-bus S1** (Bus 814)             | 1 (2.7%) | 3 (8.1%)   | 5 (13.5%)  |
| **34-bus S3** (Bus 814+852+890)     | 3 (8.1%) | 8 (21.6%)  | 13 (35.1%) |
| **123-bus S1** (Bus 25)             | 1 (0.8%) | 4 (3.0%)   | 8 (6.1%)   |
| **123-bus S5** (Bus 25+40+54+62+76) | 5 (3.8%) | 20 (15.2%) | 38 (28.8%) |


### Expected outcomes (hypothesis)

- k=0 → k=1: MIA-AUC should decrease noticeably (cutting first-order power flow leakage)
- k=1 → k=2: diminishing returns; F1/ROC may drop significantly as too much information is removed
- Trade-off: larger k improves forgetting (lower MIA) but degrades utility (lower F1/ROC)

### Files changed (vs V2.0)

- `src/data.py` — `expand_forget_khop()`
- `src/__init__.py` — export `expand_forget_khop`
- `scripts/train.py` — `build_scenarios()` with k-hop; `--khop` CLI arg
- `config/34bus.yaml`, `config/123bus.yaml` — `k_hops: [0, 1, 2]`
- `src/visualization.py` — `plot_khop_comparison()`, `plot_khop_forget_size()`, `_scenario_sort_key()`
- `notebooks/Viz_GDGU_loc.ipynb` — Cell 4–5 for k-hop plots

---

## V2.0 — GU_EV_loc.20260413

**Date:** 2026-04-13
**Status:** Experiment complete — both 34-bus and 123-bus, all 5 methods, 10 seeds
**Results:** `results/2026-04-13_01/`

### Key Updates

1. **Added GIF and IDEA unlearning methods**
  - Implemented `gif_unlearn`, `idea_unlearn`, `_compute_grad_for_hvp`, and `_hvp` in `src/unlearning.py`
  - Experiment pipeline expanded from 3 methods to 5: Original → GDGU → GIF → IDEA → Retrain
  - `src/experiment.py` fully integrates all 5 methods with training, evaluation, and MIA computation
2. `**max_batches` fix for GAT × 123-bus OOM**
  - GIF's `_compute_grad_for_hvp` builds two full autograd graphs simultaneously with `create_graph=True`, causing OOM on a 24 GB RTX 4090 for GAT on 123-bus
  - Introduced `max_batches` sub-sampling for HVP: 34-bus → 22 batches (50%), 123-bus → 30 batches (~34%)
  - Parameters stored in `config/34bus.yaml` and `config/123bus.yaml`; not hardcoded
3. **YAML hyperparameter configuration system**
  - Added `config/` directory with `34bus.yaml` and `123bus.yaml`
  - Both `scripts/train.py` and `scripts/run_experiments.py` load from YAML; CLI arguments override YAML defaults
  - Added dependency: `pyyaml==6.0.2`
4. **Edge masking in data preprocessing**
  - `build_graphs` in `src/data.py` removes all edges incident to forget nodes when `forget_indices` is provided
  - Features are StandardScaler-normalized before zeroing (avoids extreme negative values from scaling raw zeros)
  - Edge masking is intentional: achieves true node-level information removal and prevents GNN representation collapse
5. **Unified output folder structure**
  - All outputs (CSV, JSON, PDF) written to a single `results/YYYY-MM-DD_HH/` folder; no separate `checkpoints/` or `logs/` subdirectories
  - `{bus}_epoch_logs.json` includes a `_metadata` block (bus_system, timestamp, device, data dimensions, full config, model_params, scenarios)
  - Visualization decoupled from training: `train.py` / `run_experiments.py` save data only; `notebooks/Viz_GDGU_loc.ipynb` generates PDF figures post-hoc
  - Completion timestamp printed after each trial for runtime estimation
6. `**src/__init__.py` completed**
  - `__all_`_ fully declared, explicitly exporting all public API including `gif_unlearn` and `idea_unlearn`
7. **SLURM batch scripts**
  - `scripts/run_juno.sbatch`: UTD Juno cluster (h100 partition), data staged from `/groups/jzhang/nliu/` to `~/scratch`
  - `scripts/run_g2.sbatch`: UTD Ganymede2 G2 (g1mig partition), data/code staged from `/mfs/io/groups/jzhang/nliu/` to WekaFS `/scratch/$USER`
8. **Experiment results summary (results/2026-04-13_01)**

  | Metric                  | 34-bus (avg, all backbones) | 123-bus (avg, all backbones) |
  | ----------------------- | --------------------------- | ---------------------------- |
  | Macro ROC-AUC           | ~0.755–0.763                | ~0.695–0.710                 |
  | Macro F1                | ~0.678–0.686                | ~0.607–0.620                 |
  | Exact Match             | ~0.384–0.397                | ~0.248–0.257                 |
  | MIA-AUC (GDGU)          | ~0.748–0.822                | ~0.610–0.642                 |
  | MIA-AUC (Retrain)       | ~0.737–0.819                | ~0.603–0.634                 |
  | GDGU speedup vs Retrain | ~4–5×                       | ~5–6×                        |
  | GIF speedup vs Retrain  | ~7–8×                       | ~9–10×                       |

  > **Note:** MIA-AUC remains well above 0.5 even for Retrain. Root cause: physical power flow in the distribution grid propagates voltage disturbances from attacked EVCS buses to neighboring buses. Neighbor nodes retain residual attack signatures that cannot be erased by forgetting the EVCS bus alone.

---

## V1.3 — GU_EV_loc.20260409

**Date:** 2026-04-09
**Status:** Complete 3-method version (Original / GDGU / Retrain); GIF and IDEA not yet implemented
**Results:** `results/2026-04-09/`
*(Note: V1.1 and V1.2 were intermediate runs on 2026-04-08 and 2026-04-10 respectively, not formally versioned)*

### Key Updates

1. First fully complete dual-dataset experiment: 34-bus (S1–S3) and 123-bus (S1–S5)
2. Dual-prefix CSV/JSON outputs per bus system (`34bus_`* and `123bus_*`)
3. `Viz_GDGU_loc.ipynb` introduced: loads results from folder and generates all PDF figures
4. `visualization.py` standardized with Times New Roman font throughout

---

## V1.0 — GU_EV_loc.20260407

**Date:** 2026-04-07
**Status:** Initial modular version; 34-bus complete, 123-bus partial
**Results:** `results/2026-04-07/`

### Key Updates

1. `src/` modular refactor completed: models / data / training / unlearning / experiment / visualization
2. Initial `run_experiments.py` with parallelization via `mp.Pool` (10 seeds in parallel)
3. Stratified multi-label data split established (70/15/15, hash-based stratification)
4. `StandardScaler` + `kaiming_init` adopted as standard training pipeline

---

---

## V5.3 — GU_EV_loc.20260421 *（未提交——探索性）*

**日期：** 2026-04-21
**分支：** `feat/two-controller-unlearning`（规划中）
**状态：** 概念设计 + 单 seed smoke test 通过；L2 攻击协议待修正
**结果目录：** `results/smoke_A_2026-04-21_15/`

### 动机

V5.0 的双通道架构解决了检测/定位泄漏问题，但遗留了一个更深层的语义问题：**EVCS—电网系统里，哪部分数据到底归谁？**

- **DSO（配电系统运营商）** 拥有 bus 电压 (V) 和支路潮流测量。V 来自 SCADA/PMU 基础设施。
- **EVCS 运营商** 拥有每个站 PCC（公共接入点）计量表的充电功率 (P)。按 GDPR/CCPA 被遗忘权，**P 是 EVCS 唯一有资格召回的变量**。

V5.0 及之前的版本在"遗忘 EVCS"时抹的是该 bus 的电压特征和相邻边——**但这些是 DSO 所有的量，EVCS 根本无权召回**。这不是 right-to-be-forgotten 情境，而是"DSO 遗忘自己的 bus"。

Jacob 2025 EVCS TDA 论文（同组）明确把 EVCS 攻击检测定位为 "DSO-level, from the grid operator's perspective"——在该框架下 unlearning 的概念**在语义层面就不成立**。

V5.3 将问题重构为**双数据控制者（Two-Controller）所有权模型**，实现**模态级遗忘**：P 作为 EVCS 贡献的输入通道被显式暴露；遗忘时只移除召回方的 P 通道，DSO 所有的 V 保持完整。

### 重要更新

1. **双控制者输入设计（A-append）**（`scripts/smoke_A/smoke_dualch_A.py`，原型）
   - 节点特征维度从 48 扩展到 **96 = [V_48 | P_48]**
   - EVCS 节点：V 半 = 电压的小时 mean+std；P 半 = 充电功率的小时 mean+std（来自 `EVCSAttacks_34.pkl['EVCS power series']`）
   - 非 EVCS 节点：V 半填入，P 半置零（通过特征排布隐式表达二值 mask）
   - V 和 P 使用**独立的 StandardScaler**（训练集拟合），防止不同站的 P 幅值差异泄露身份

2. **模态级遗忘语义**（`scripts/smoke_A/` 原型）
   - 新增 `build_pyg_dataset(..., forget_P_only=True)`：在 forget 节点抹掉 dims 48:96；**V 保持完整，边不被 mask**（对比 V5.0 的 `build_graphs()` 抹掉全部 48 V-dims 并移除相邻边）
   - 理由：只召回 EVCS 自有的模态；DSO 的 V 保留供电网运营商合法检测/定位
   - V5.0 的 GDGU-dual pipeline 无需改动即可使用；区别只在训练数据里被抹掉的内容

3. **L1/L2 隐私评估矩阵**（新增，原型）
   - **L1 —— 功能保留**：Macro-F1 / Macro-ROC（定位）、DetAUC（检测）
   - **L2 —— 模态消除**（4 个子指标）：
     - **L2-a 归因**：forget 节点上 P 通道的梯度范数
     - **L2-b 遮挡 ΔAUC**：AUC(P 在) − AUC(P 被抹)
     - **L2-c 重构攻击**：从图嵌入到 P_48 的 MLP decoder；MSE 对比 population baseline
     - **L2-d 属性推断**（规划中，未实现）：图嵌入上 3-way EVCS-ID 分类器

### Smoke-test 结果（GIN × S1-0 × seed 42，单 trial）

| 方法 | Macro-F1 | Macro-ROC | DetAUC | 时间 |
|---|---|---|---|---|
| Original (V+P)                | 1.0000 | 1.0000 | 0.9074 | 11.8s |
| GDGU-dual-A（遗忘 P_814）     | 0.9769 | 0.9912 | 0.9074 | 6.4s  |
| Retrain-A（P_814 置零重训）   | 0.9861 | 0.9952 | 0.9247 | 17.5s |

| L2 指标 | Original | GDGU-dual-A | Retrain-A | 判定 |
|---|---|---|---|---|
| Occlusion ΔAUC         | +0.0883  | −0.0088  | −0.0510  | ✅ 强消除信号 |
| P 通道梯度范数         | 0.000107 | 0.003658 | 0.008213 | ❌ loss 饱和伪影 |
| Recon MSE / baseline   | 0.502    | 0.420    | 0.510    | ❌ V→P 物理泄露旁路 |

**L1 发现**：Original F1=1.0 反映任务被简化 —— 直接观测 P 让攻击检测几乎是确定性的（Jacob 2025 Algorithm 1 攻击的是 P 不是 V）。遗忘 P 这条捷径后，GDGU-dual-A 仍靠 V 的物理传播保持 F1=0.977。Detection AUC 由架构保证与 Original 一致（通道隔离）。

**L2-b 有效**（遮挡 ΔAUC）：Original 在测试时抹掉 P 降 8.8 个 AUC 点；遗忘后的模型对 P 零依赖 —— 最干净的模态消除证据。

**L2-a 失败**（归因）：Original 在 F1=1.0 时 loss≈0，所有梯度被饱和到 ≈0，导致指标方向反转。需改为 logit 梯度或 Integrated Gradients。

**L2-c 失败**（重构）：池化后的图嵌入仍含 bus 814 的 V 信号（DSO 所有，永不 mask），而 V 由 P 物理决定，所以 decoder 绕过模型记忆，直接通过物理路径还原 P。需改为 node-level 嵌入 + 属性级攻击目标。

### 待办

**1. L2 攻击协议 2.0**（下一步立即要做——修正 + 扩展）：
- **L2-a 归因（修复）**：把 loss 梯度换成 forget 节点 logit 的 **Integrated Gradients**（解决 F1=1.0 时梯度饱和到 ≈0 的问题）
- **L2-c 重构（重设计）**：把池化图嵌入换成 **forget 节点的 node-level 嵌入**；目标从原始 P_48 换成**与 V 弱相关的属性**（降低 V→P 物理旁路）
- **L2-d Property Inference（实现）**：冻结 backbone，在 forget 节点嵌入上做 **EVCS-ID 3-way 线性探针**
- **L2-e Attack-Type Inference（新——升为主指标）**：冻结 backbone，在 forget 节点嵌入上做 **5-way attack-type**（nil + 4 种攻击）线性探针。三点对照：
  - Original (V+P) —— attack-type acc = 上限（模型直接看到 P）
  - GDGU-dual-A（P→0）—— 遗忘生效则应下降
  - Retrain-A（训练时就没见过 forget 节点的 P）—— **物理泄露地板**（V 单独能泄漏多少）
  - 判据：`GDGU ≈ Retrain` ✅ 有效遗忘；`GDGU >> Retrain` ❌ 残留 P 记忆；`Retrain 本身就高` → 写成电网物理固有限制（与 V2.0 的 MIA-AUC 叙事一致）

**2. 效率评估**（新——论文叙事必需）：
- **Wall-clock 时间**：每次遗忘运行的 cuda:1 耗时，10 seeds 的中位数 ± std
- **加速比** = `t_retrain / t_{GDGU, GIF, IDEA}`
- **梯度步数对比**：Retrain-A（约 300 epoch × 完整 loader）vs GDGU/GIF/IDEA（25 fine-tune epoch + 单次 gradient/HVP 算子）
- **质量—效率 Pareto 图**：横轴 time、纵轴 retain set Macro-F1，预期 GDGU 左上、Retrain 右上；复用 V2.0 的表格结构

**3. Novelty 的叙事定位**（文档层面，不改代码）：
- 主贡献：**两控制者 + 模态级遗忘** 的问题定义本身（Route A）
- 次贡献：Route A 在 **GDGU / GIF / IDEA / Retrain** 四方法上均可运行，建立该语义为**算法无关的 plug-in 框架**（对齐 task-aware MU 论文的定位方式）
- 含义：V2.0 所有 unlearning 基础设施都可复用；V5.3 只改数据路径的语义

**4. 生产化集成**（规划——待动文件清单）：
- `src/data.py`：新增 `build_P_features()`（从 `EVCSAttacks_34.pkl['EVCS power series']` 抽 hourly mean+std = 48d）；扩展 `load_evcs_data()` 返回值；新增 `fit_separate_scalers()`（V/P 独立 scaler）；修改 `build_graphs()` 支持 `forget_P_only=True`（仅抹 48:96,保留边）
- `src/models.py`：DualChannel_Graph 的 `in_dim` 从 48 提到 96
- `src/experiment.py`：新增 `run_single_trial_dual_A()`，封装 A-append + 模态遗忘 + 4 方法
- `src/privacy.py` **（新模块）**：`L2_a_integrated_gradients()`、`L2_b_occlusion_delta_auc()`、`L2_c_reconstruction_node_emb()`、`L2_d_property_inference()`、`L2_e_attack_type_inference()`、`measure_unlearn_efficiency()`
- `config/34bus.yaml`：新增 `route_a:`（v_dim=48, p_dim=48, node_feat_dim=96）和 `privacy:`（probe epochs、IG steps）两段
- `scripts/train.py`：新增 `--route A` 开关

**5. 多 seed 验证**：10 seeds × {GCN, GAT, GIN} × {S1-0, S2-0, S3-0} × {Original, GDGU, GIF, IDEA, Retrain}

**6. A1 消融**：加上 V_at_PCC 一起遗忘（按 Jacob Fig.1b，PCC meter 归 EVCS）

### 文件变更（相对于 V5.0）—— 目前已改

- `scripts/smoke_A/smoke_dualch_A.py` —— 新独立 smoke-test 原型（数据增强 + L1/L2 评估，单 seed）
- `scripts/smoke_A/smoke_A.log` —— 完整 smoke-test 运行日志
- `results/smoke_A_2026-04-21_15/` —— smoke-test 输出
- `Version.md` —— 新增 V5.3 条目

**注**：V5.3 改动当前限于 `scripts/smoke_A/` 原型；生产化集成到 `src/` 将推迟到 L2 攻击协议 2.0（含 L2-e）修正完成且 smoke-test 跨 seed 结果一致后进行。

---

## V5.0 — GU_EV_loc.20260420 *（未提交——探索性）*

**日期：** 2026-04-20
**分支：** `feat/dual-channel-gu`
**状态：** 代码实现完成，Smoke test 通过；待多 seed 验证
**结果目录：** `results/2026-04-20_19/`（单 seed smoke test）

### 动机

V2.0–V4.0 实验表明 MIA-AUC 即使 Retrain 也远高于 0.5。根因分析（见 V4.0 隐私半径分析及 `/tmp/graph_level_feat.py` 评估）识别出一条此前未处理的泄漏通道：**全图级电压动态**（如系统级时间/空间梯度）含有强攻击信号，**节点级遗忘无法屏蔽**。非 EVCS 全图特征的检测 AUC 几乎等于 EVCS 节点特征（EVCS 3 Type 3：0.888 vs 0.887），证明攻击信号系统级传播。

V5.0 引入**严格隔离的双通道架构**（方案 A），在架构层面物理切断"检测相关的全图信号"与"定位相关的节点信号"。实现**选择性遗忘**——只遗忘目标 EVCS 的定位能力，保留非隐私敏感的检测能力。

### 重要更新

1. **双通道模型架构**（`src/models.py`）
   - 重构 `GCN_Graph` / `GAT_Graph` / `GIN_Graph`，新增 `encode(data)` 方法返回池化后的图嵌入 `[B, 2*hid_dim]`（向后兼容，`forward(data)` 行为不变）
   - 新增 `DualChannel_Graph` 类，严格隔离包装任意 backbone：
     - **Node 通道**：GNN backbone → Loc head → `y_loc [B, K]`
     - **Graph 通道**：MLP on 全图级特征 → Det head → `y_det [B]`
     - 两个 head **不共享任何参数**；`loc_head` 永远看不到 `graph_feat`
   - Graph MLP 使用 LayerNorm（非 BN），避免遗忘时 BN-recalibrate 对 graph 通道的干扰
   - `freeze_graph_channel()` / `unfreeze_all()` 方法用于遗忘流程控制

2. **全图级特征提取**（`src/data.py`）
   - 新增 `build_graph_features(all_V, edge_index)` → 每图 120 维：
     - **TempGrad (72d)**：系统级时间梯度 |dV/dt| —— 跨节点/分时段的 mean、per-hour max-avg、per-hour std-avg
     - **EdgeGrad-light (48d)**：空间梯度 |V_u - V_v| —— 每小时跨边 mean 和 std
   - `load_evcs_data()` 新增返回 `all_V`（相均值原始时间序列 `[G, N, 288]`），供下游特征提取使用
   - `build_graphs()` 接受可选 `graph_feat` 并自动派生 `y_det = 1(任一 EVCS 被攻击)`；`graph_feat` **不被 `forget_indices` 屏蔽**（保护检测能力）
   - 新增 `fit_graph_feat_scaler()`（训练集 StandardScaler）

3. **多任务训练**（`src/training.py`）
   - `train_model_dual()`：loss = α · BCE(det) + β · BCE(loc)。默认 α=β=1（严格隔离下两通道无参数共享，不存在交叉污染风险）
   - `evaluate_model_dual()`：定位指标 + `det_acc` / `det_auc` / `det_f1`
   - `compute_mia_auc_dual()`：MIA **只用定位 loss**（隐私保护的那部分）

4. **双通道 GDGU**（`src/unlearning.py`）
   - `gdgu_dual_unlearn()`：梯度计算前冻结 `graph_mlp` + `det_head`，只有 Node 通道（backbone + loc_head）接收 GDGU 更新
   - 梯度差只基于定位 loss 计算
   - `_finetune_after_gdgu_dual()` 继续只优化未冻结参数

5. **实验流程**（`src/experiment.py`, `scripts/train.py`）
   - 新增 `run_single_trial_dual()`：Original → GDGU-dual → Retrain（3 方法，GIF/IDEA 延后至 V5.1）
   - Retrain baseline 也使用双通道架构，保证公平对比
   - `scripts/train.py` 新增 `--dual` 开关；输出文件前缀 `34bus_dual_*`

6. **配置文件**（`config/34bus.yaml`）
   - 新增 `dual_channel:` 段：`graph_feat_dim`、`graph_mlp_hidden`、`graph_mlp_out`、`alpha`、`beta`

### 架构 —— 方案 A（严格隔离）

```
输入：x [N, 48] + graph_feat [120]

Node 通道 (GNN, BN)：                    Graph 通道 (MLP, LayerNorm)：
  GCN/GAT/GIN backbone                     120 → 64 → 32
    ↓ encode() → node_emb [B, 2*hid]        ↓ graph_emb_feat [B, 32]
    ↓                                        ↓
  Loc Head  → y_loc [B, K]                Det Head → y_det [B]
  （只服务定位）                           （只服务检测）

遗忘时：冻结 graph_mlp + det_head → 只有 Node 通道 + Loc Head 被 GDGU 更新。
检测能力由架构保证不被破坏。
```

### Smoke-test 结果（GIN × S1-0 × seed 42，单 trial）

| 方法 | ExMatch | Macro ROC | Macro F1 | Det Acc | MIA_forget | 时间 |
|---|---|---|---|---|---|---|
| Original   | 0.840 | 0.978 | 0.931 | 0.883 | — | 33.9s |
| GDGU-dual  | 0.850 | 0.983 | 0.942 | 0.883 | **0.621** | 7.7s |
| Retrain    | 0.840 | 0.986 | 0.939 | 0.883 | 0.638 | 56.8s |

观察：
- Pipeline 端到端打通，无报错
- Detection Acc 在 Original / GDGU / Retrain 三者间完全相同（0.883）—— 证明 graph 通道被正确保留
- GDGU-dual 的 MIA_forget（0.621）在此单 seed 低于 Retrain（0.638）—— 趋势支持假设，但需多 seed 验证
- GDGU 相对 Retrain 加速 7.3×

### 待验证项

- GIN × S1-0 跑 10 seeds，获得统计显著的 MIA 对比
- 与 V2.0 单通道并行对照（同 seed 42），以分离双通道架构贡献
- 假设成立后扩展到 3 backbones × 4 scenarios × 4 k-hops

### 文件变更（相对于 V4.0）

- `src/models.py` —— GCN/GAT/GIN 三类增加 `encode()` 方法；新增 `DualChannel_Graph` 类
- `src/data.py` —— 新增 `build_graph_features()`、`fit_graph_feat_scaler()`；`build_graphs()` 支持 graph_feat/y_det；`load_evcs_data()` 返回 `all_V`
- `src/training.py` —— 新增 `train_model_dual()`、`evaluate_model_dual()`、`compute_mia_auc_dual()`
- `src/unlearning.py` —— 新增 `gdgu_dual_unlearn()`（冻结 graph 通道）
- `src/experiment.py` —— 新增 `run_single_trial_dual()` 和 `_build_result_dual()`
- `config/34bus.yaml` —— 新增 `dual_channel:` 段
- `scripts/train.py` —— 新增 `--dual` 开关

---

## V4.0 — GU_EV_loc.20260416

**日期：** 2026-04-16
**分支：** `feat/extended-forget-set`
**状态：** 相位修复已完成，待重训验证影响

### 动机

V3.0 的特征提取对所有 bus 使用三相均值（`voltages.mean(axis=1)`）。然而，IEEE 34-bus 有 8 个单相 bus，IEEE 123-bus 有 61 个非三相 bus（57 单相 + 4 两相）。对这些 bus，不活跃相位为零，将真实电压稀释 3×（单相）或 1.5×（两相）。GNN 看到的单相 bus 特征约 0.34 p.u.，而邻居三相 bus 约 1.03 p.u. —— 3 倍特征差异为相位伪影，非攻击信号。

此外，开展了隐私半径（Privacy Radius）分析，定量研究攻击电压信号在配电网中的传播规律，为后续特征工程决策提供物理依据。

### 重要更新

1. **关键 Bug 修复：有效相均压**
   - `src/data.py` 第 87 行：`voltages.mean(axis=1)` → 仅平均有效相（相均值 > 0.1 p.u.）
   - 34-bus：8/37 节点修正（810, 818, 820, 822, 826, 856, 864, 838）—— 特征从 ~0.34 修正至 ~1.03
   - 123-bus：61/132 节点修正（57 单相 × 3.0x, 4 两相 × 1.5x）
   - 所有先前 V2.0/V3.0 实验结果均在这些节点的错误特征上训练

2. **隐私半径分析笔记本**（`notebooks/Privacy_Radius_Analysis.ipynb`）
   - 33 个 cell 的分析笔记本，带顺序 `[N]` 编号
   - Hop-based AUC 衰减分析（34-bus 和 123-bus）
   - 第 10 节（Cell 29–33）：Matched-pair 控制变量分析
     - 按源支路潮流（欧氏距离）将每个攻击场景匹配到最接近的正常场景
     - 分离攻击对各 bus 电压的因果效应
     - 量化电压调节器衰减率（814r/852r 处 Type 2/3/4 衰减 36–51%）
   - 输出：`matched_pair_feeder_path.pdf`、`matched_pair_topology.pdf` 等 10+ PDF 文件
   - 笔记本中同步应用有效相修复（`_active_phase_mean()` 辅助函数）

3. **特征工程分析**（分析结论，未修改模型代码）
   - 对比 5 种特征表示的 per-bus AUC：

     | 特征             | 维度  | Type 1 | Type 2 | Type 3 | Type 4 |
     | ---------------- | --- | ------ | ------ | ------ | ------ |
     | V3.0 均值+标准差   | 48  | 0.524  | 0.738  | 0.688  | 0.693  |
     | 一阶差分           | 287 | 0.913  | 0.978  | 0.974  | 0.971  |
     | 小时峰值           | 24  | 0.430  | 0.585  | 0.628  | 0.638  |
     | 原始 288 步        | 288 | 0.541  | 0.848  | 0.837  | 0.827  |

   - 核心发现：V3.0 的小时均值+标准差丢失了 5 分钟级时间位置信息 —— 时移攻击（Type 2/3/4）在远端 bus 几乎不可见
   - 后续路线：V4.0 特征（96d: +梯度统计）→ V5.0 特征（287d: 原始差分）

### 影响评估

| 数据集   | 受影响节点    | 占比    | 特征误差            |
| ------- | ----------- | ------ | ------------------ |
| 34-bus  | 8 / 37      | 21.6%  | 3.0× 低估          |
| 123-bus | 61 / 132    | 46.2%  | 3.0× 或 1.5× 低估  |

### 文件变更（相对于 V3.0）

- `src/data.py` —— `load_evcs_data()` 中有效相均压（Bug 修复）
- `notebooks/Privacy_Radius_Analysis.ipynb` —— 新建 33-cell 分析笔记本，含 matched-pair 分析

---

## V3.0 — GU_EV_loc.20260414 *（未提交——探索性）*

**日期：** 2026-04-14
**分支：** `feat/extended-forget-set`
**状态：** 代码实现完成，待实验验证

### 动机

V2.0 实验揭示 MIA-AUC 即使 Retrain 也远高于 0.5（34-bus: 0.73–0.83, 123-bus: 0.59–0.65）。根因：配电网物理潮流将攻击签名传播至相邻 bus。仅遗忘 EVCS bus 本身留下了"二手证据"，模型可通过邻居电压重建被遗忘信息。

### 重要更新

1. **k-hop 邻居扩展遗忘集**
   - `src/data.py` 新增 `expand_forget_khop(forget_indices, edge_index, k)`
   - BFS 扩展：k=0（仅 EVCS）、k=1（+1-hop 邻居）、k=2（+2-hop 邻居）
   - 扩展集应用边遮蔽 + 特征置零
2. **场景命名：Sn → Sn-k**
   - `scripts/train.py` 的 `build_scenarios()` 生成 `Sn-k` 格式
   - 示例：S1-0（1 节点, 2.7%）、S1-1（3 节点, 8.1%）、S1-2（5 节点, 13.5%）—— 34-bus
   - CLI：`--khop 1` 仅跑 k=1；`--scenario S2-1` 跑指定场景
3. **YAML 配置：`k_hops` 参数**
   - `config/34bus.yaml` 和 `config/123bus.yaml`：`k_hops: [0, 1, 2]`
   - 缺省值 `[0]`（向后兼容）
4. **跨 k 可视化**
   - `plot_khop_comparison()`：折线图展示各 Sn 的指标随 k 变化
   - `plot_khop_forget_size()`：柱状图展示各 Sn-k 遗忘集大小
   - `Viz_GDGU_loc.ipynb` Cell 4–5：k-hop 对比专区
5. **场景排序更新**
   - `_scenario_sort_key()` 兼容旧格式 `S1` 和新格式 `S1-0`

### 遗忘集大小

| 场景                                  | k=0      | k=1        | k=2        |
| ------------------------------------- | -------- | ---------- | ---------- |
| **34-bus S1**（Bus 814）               | 1 (2.7%) | 3 (8.1%)   | 5 (13.5%)  |
| **34-bus S3**（Bus 814+852+890）       | 3 (8.1%) | 8 (21.6%)  | 13 (35.1%) |
| **123-bus S1**（Bus 25）               | 1 (0.8%) | 4 (3.0%)   | 8 (6.1%)   |
| **123-bus S5**（Bus 25+40+54+62+76）   | 5 (3.8%) | 20 (15.2%) | 38 (28.8%) |

### 预期结果（假设）

- k=0 → k=1：MIA-AUC 应明显下降（切断一阶潮流泄漏）
- k=1 → k=2：收益递减；F1/ROC 可能显著下降（移除信息过多）
- 权衡：更大 k 改善遗忘效果（更低 MIA）但损害效用（更低 F1/ROC）

### 文件变更（相对于 V2.0）

- `src/data.py` —— `expand_forget_khop()`
- `src/__init__.py` —— 导出 `expand_forget_khop`
- `scripts/train.py` —— `build_scenarios()` 增加 k-hop；`--khop` CLI 参数
- `config/34bus.yaml`、`config/123bus.yaml` —— `k_hops: [0, 1, 2]`
- `src/visualization.py` —— `plot_khop_comparison()`、`plot_khop_forget_size()`、`_scenario_sort_key()`
- `notebooks/Viz_GDGU_loc.ipynb` —— Cell 4–5 k-hop 对比图

---

## V2.0 — GU_EV_loc.20260413

**日期：** 2026-04-13
**状态：** 实验完成——34-bus 和 123-bus 均完成，5 种方法，10 个随机种子
**结果目录：** `results/2026-04-13_01/`

### 重要更新

1. **新增 GIF 和 IDEA 遗忘方法**
  - `src/unlearning.py` 实现 `gif_unlearn`、`idea_unlearn` 及辅助函数 `_compute_grad_for_hvp`、`_hvp`
  - 实验流程由 3 方法扩展为 5 方法：Original → GDGU → GIF → IDEA → Retrain
  - `src/experiment.py` 完整集成 5 方法的训练、评估与 MIA 计算
2. `**max_batches` OOM 修复（GAT × 123-bus × GIF/IDEA）**
  - GIF 的 `_compute_grad_for_hvp` 使用 `create_graph=True` 同时维持两个完整计算图，GAT 在 123-bus 上会触发 24 GB RTX 4090 显存溢出
  - 引入 `max_batches` 参数对 HVP 进行批次子采样：34-bus 取 22 批（50%），123-bus 取 30 批（约 34%）
  - 参数写入 `config/34bus.yaml` 和 `config/123bus.yaml`，不在代码中硬编码
3. **YAML 超参配置系统**
  - 新增 `config/` 目录，包含 `34bus.yaml` 和 `123bus.yaml`
  - `scripts/train.py` 和 `scripts/run_experiments.py` 均从 YAML 加载配置，CLI 参数可覆盖 YAML 默认值
  - 新增依赖：`pyyaml==6.0.2`
4. **数据预处理：边遮蔽（Edge Masking）**
  - `src/data.py` 的 `build_graphs` 在 `forget_indices` 非空时删除所有与 forget 节点相连的边
  - 特征先经 StandardScaler 缩放再置零（避免原始零值在标准化空间中产生极端负值）
  - 边遮蔽为有意设计：实现真正的节点级信息删除，防止 GNN 表征崩塌
5. **实验输出统一至单一日期文件夹**
  - 所有输出（CSV、JSON、PDF）存入 `results/YYYY-MM-DD_HH/`，不再有 `checkpoints/` 和 `logs/` 子目录
  - `{bus}_epoch_logs.json` 头部包含 `_metadata` 块（bus_system、timestamp、device、数据维度、完整 config、model_params、scenarios）
  - 可视化与训练解耦：`train.py` / `run_experiments.py` 只保存数据，图表由 `notebooks/Viz_GDGU_loc.ipynb` 后处理生成
  - 每个 trial 完成后打印当前时间，便于估算剩余运行时间
6. `**src/__init__.py` 完善**
  - 补全 `__all_`_ 声明，显式导出所有公开 API，包括新增的 `gif_unlearn`、`idea_unlearn`
7. **SLURM 批处理脚本**
  - `scripts/run_juno.sbatch`：UTD Juno 集群（h100 分区），数据从 `/groups/jzhang/nliu/` 复制至 `~/scratch`
  - `scripts/run_g2.sbatch`：UTD Ganymede2 G2（g1mig 分区），数据/代码从 `/mfs/io/groups/jzhang/nliu/` 复制至 WekaFS `/scratch/$USER`
8. **实验结果摘要（results/2026-04-13_01）**

  | 指标                   | 34-bus（全骨干平均） | 123-bus（全骨干平均） |
  | -------------------- | ------------- | -------------- |
  | Macro ROC-AUC        | ~0.755–0.763  | ~0.695–0.710   |
  | Macro F1             | ~0.678–0.686  | ~0.607–0.620   |
  | Exact Match          | ~0.384–0.397  | ~0.248–0.257   |
  | MIA-AUC（GDGU）        | ~0.748–0.822  | ~0.610–0.642   |
  | MIA-AUC（Retrain）     | ~0.737–0.819  | ~0.603–0.634   |
  | GDGU 加速比（vs Retrain） | ~4–5×         | ~5–6×          |
  | GIF 加速比（vs Retrain）  | ~7–8×         | ~9–10×         |

  > **注：** MIA 未能接近 0.5 的根本原因——配电网物理潮流使邻近 bus 也携带被攻击 EVCS 的电压异常信号，仅遗忘 EVCS bus 本身无法消除邻居节点的残留记忆，即使完整 Retrain 也如此。

---

## V1.3 — GU_EV_loc.20260409

**日期：** 2026-04-09
**状态：** 3 方法完整版本（Original / GDGU / Retrain），尚未实现 GIF 和 IDEA
**结果目录：** `results/2026-04-09/`
*（注：V1.1 和 V1.2 为 04-08 和 04-10 的中间运行，未正式打版本号）*

### 重要更新

1. 首次完整双数据集实验：34-bus（S1–S3）和 123-bus（S1–S5）全部完成
2. 输出文件按数据集加前缀（`34bus_`* 和 `123bus_*`）
3. 引入 `Viz_GDGU_loc.ipynb`：从 results 目录加载数据并生成 PDF 图表
4. `visualization.py` 全面统一使用 Times New Roman 字体

---

## V1.0 — GU_EV_loc.20260407

**日期：** 2026-04-07
**状态：** 初始模块化版本；34-bus 完整，123-bus 部分完成
**结果目录：** `results/2026-04-07/`

### 重要更新

1. `src/` 模块化重构完成：models / data / training / unlearning / experiment / visualization
2. 初始 `run_experiments.py` 并行化实验（`mp.Pool`，10 seeds 并行）
3. 确立标准化多标签数据切分方案（70/15/15，基于标签哈希分层）
4. `StandardScaler` + `kaiming_init` 确立为训练标准流程

