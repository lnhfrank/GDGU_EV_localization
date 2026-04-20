# Version History — GU_EV_loc

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

