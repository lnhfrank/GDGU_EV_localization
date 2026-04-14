# Version History — GU_EV_loc

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

2. **`max_batches` fix for GAT × 123-bus OOM**
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

6. **`src/__init__.py` completed**
   - `__all__` fully declared, explicitly exporting all public API including `gif_unlearn` and `idea_unlearn`

7. **SLURM batch scripts**
   - `scripts/run_juno.sbatch`: UTD Juno cluster (h100 partition), data staged from `/groups/jzhang/nliu/` to `~/scratch`
   - `scripts/run_g2.sbatch`: UTD Ganymede2 G2 (g1mig partition), data/code staged from `/mfs/io/groups/jzhang/nliu/` to WekaFS `/scratch/$USER`

8. **Experiment results summary (results/2026-04-13_01)**

   | Metric | 34-bus (avg, all backbones) | 123-bus (avg, all backbones) |
   |---|---|---|
   | Macro ROC-AUC | ~0.755–0.763 | ~0.695–0.710 |
   | Macro F1 | ~0.678–0.686 | ~0.607–0.620 |
   | Exact Match | ~0.384–0.397 | ~0.248–0.257 |
   | MIA-AUC (GDGU) | ~0.748–0.822 | ~0.610–0.642 |
   | MIA-AUC (Retrain) | ~0.737–0.819 | ~0.603–0.634 |
   | GDGU speedup vs Retrain | ~4–5× | ~5–6× |
   | GIF speedup vs Retrain | ~7–8× | ~9–10× |

   > **Note:** MIA-AUC remains well above 0.5 even for Retrain. Root cause: physical power flow in the distribution grid propagates voltage disturbances from attacked EVCS buses to neighboring buses. Neighbor nodes retain residual attack signatures that cannot be erased by forgetting the EVCS bus alone.

---

## V1.3 — GU_EV_loc.20260409

**Date:** 2026-04-09
**Status:** Complete 3-method version (Original / GDGU / Retrain); GIF and IDEA not yet implemented
**Results:** `results/2026-04-09/`
*(Note: V1.1 and V1.2 were intermediate runs on 2026-04-08 and 2026-04-10 respectively, not formally versioned)*

### Key Updates

1. First fully complete dual-dataset experiment: 34-bus (S1–S3) and 123-bus (S1–S5)
2. Dual-prefix CSV/JSON outputs per bus system (`34bus_*` and `123bus_*`)
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

## V2.0 — GU_EV_loc.20260413

**日期：** 2026-04-13
**状态：** 实验完成——34-bus 和 123-bus 均完成，5 种方法，10 个随机种子
**结果目录：** `results/2026-04-13_01/`

### 重要更新

1. **新增 GIF 和 IDEA 遗忘方法**
   - `src/unlearning.py` 实现 `gif_unlearn`、`idea_unlearn` 及辅助函数 `_compute_grad_for_hvp`、`_hvp`
   - 实验流程由 3 方法扩展为 5 方法：Original → GDGU → GIF → IDEA → Retrain
   - `src/experiment.py` 完整集成 5 方法的训练、评估与 MIA 计算

2. **`max_batches` OOM 修复（GAT × 123-bus × GIF/IDEA）**
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

6. **`src/__init__.py` 完善**
   - 补全 `__all__` 声明，显式导出所有公开 API，包括新增的 `gif_unlearn`、`idea_unlearn`

7. **SLURM 批处理脚本**
   - `scripts/run_juno.sbatch`：UTD Juno 集群（h100 分区），数据从 `/groups/jzhang/nliu/` 复制至 `~/scratch`
   - `scripts/run_g2.sbatch`：UTD Ganymede2 G2（g1mig 分区），数据/代码从 `/mfs/io/groups/jzhang/nliu/` 复制至 WekaFS `/scratch/$USER`

8. **实验结果摘要（results/2026-04-13_01）**

   | 指标 | 34-bus（全骨干平均） | 123-bus（全骨干平均） |
   |---|---|---|
   | Macro ROC-AUC | ~0.755–0.763 | ~0.695–0.710 |
   | Macro F1 | ~0.678–0.686 | ~0.607–0.620 |
   | Exact Match | ~0.384–0.397 | ~0.248–0.257 |
   | MIA-AUC（GDGU） | ~0.748–0.822 | ~0.610–0.642 |
   | MIA-AUC（Retrain） | ~0.737–0.819 | ~0.603–0.634 |
   | GDGU 加速比（vs Retrain） | ~4–5× | ~5–6× |
   | GIF 加速比（vs Retrain） | ~7–8× | ~9–10× |

   > **注：** MIA 未能接近 0.5 的根本原因——配电网物理潮流使邻近 bus 也携带被攻击 EVCS 的电压异常信号，仅遗忘 EVCS bus 本身无法消除邻居节点的残留记忆，即使完整 Retrain 也如此。

---

## V1.3 — GU_EV_loc.20260409

**日期：** 2026-04-09
**状态：** 3 方法完整版本（Original / GDGU / Retrain），尚未实现 GIF 和 IDEA
**结果目录：** `results/2026-04-09/`
*（注：V1.1 和 V1.2 为 04-08 和 04-10 的中间运行，未正式打版本号）*

### 重要更新

1. 首次完整双数据集实验：34-bus（S1–S3）和 123-bus（S1–S5）全部完成
2. 输出文件按数据集加前缀（`34bus_*` 和 `123bus_*`）
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
