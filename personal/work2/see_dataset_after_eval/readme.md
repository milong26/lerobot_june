# Ours / DynamicAnchor V2 Episode Selection

## 本次修复说明

### 路径修复
- `train_and_eval_v2.sh`：修复 `PROJECT_ROOT` 路径错误，改用 `WORK2_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"`，所有路径基于 `WORK2_ROOT` 而非写死绝对路径

### Bootstrap 扩展
- `random_bootstrap_analysis()` 现在分别对 global/wrist/combined 三种 representation 运行
- Bootstrap 同时覆盖 coverage (mean/p95/max)、redundancy、fixed-universe SIC
- 使用向量化计算 (`dist_matrix[:, sample_indices].min(axis=1)`) 提升性能
- 所有 bootstrap subset 复用同一 dbar/kernel matrix，确保公平比较

### 假设评估逻辑重构
- H1/H2/H3 由 `evaluate_h1()`/`evaluate_h2()`/`evaluate_h3()` 独立函数计算，不硬编码在 `generate_report()` 中
- H1 基于 embedding validity、position probe vs shuffled、neighbor overlap vs random 三种证据
- H2 基于 Spearman + permutation test、position probe、neighbor overlap，不再使用固定 rho 阈值
- H3 基于 bootstrap better-than-random fractions，50% random-level 结果判为 NOT SUPPORTED 而非 WEAK

### Duplicate 检测
- `load_embeddings()` 和 `load_episode_metadata()` 检测并记录重复 episode_index
- `sic_v2.py` 的 exact duplicate 改用 `np.unique(phi, axis=0)` 严格比较
- Exact duplicate 改为 warning 而非 error，不阻止分析继续

### Grid Sparse CV 修复
- `grid_classifiability()` 当最小类别样本数 <2 时返回 `status="insufficient_samples"` 而非 accuracy=0
- 使用 `StratifiedKFold(n_splits=min(5, min_class_count))` 适配稀疏类别

### 新增分析
- 增加 `global_normalized` 和 `wrist_normalized` representation 诊断
- 增加 Ours vs Uniform overlap 与 coverage delta 分析
- `compute_subset_coverage()` 和 `compute_workspace_coverage()` 同时输出 all-reference 和 unselected-only 两组指标

### 报告字段修复
- `generate_report()` Table 2 现在读取 `coverage_global`/`redundancy_global` 等正确字段，不再显示 N/A

## 新增文件清单

| 文件 | 说明 |
|------|------|
| `sic_v2.py` | 向量化、固定 reference anchor universe 的 SIC 计算器 |
| `iterative_select_episodes_v2.py` | V2 episode selection 主脚本 |
| `compare_selection_v1_v2.py` | V1 vs V2 offline comparison 脚本 |
| `test_sic_v2.py` | 10 项回归测试 |
| `sanity_check_corner3.py` | Corner3 数据集 sanity check 脚本 |
| `train_and_eval_v2.sh` | 支持 `dynamicanchor_v2` 模式的训练脚本 |

## 为什么 V1 candidate scores 不严格可比

V1 的 `compute_sic_for_candidate()` 对每个候选 episode 都：

1. 构建 `AnchorSystem(selected_episodes + candidate)`
2. 调用 `compute_dbar()` 重新计算 dbar_global / dbar_wrist
3. 用新的 anchor system 计算 SIC

这导致 candidate A 和 candidate B 被评价时：
- **anchor set 不一样**（A 加入后的 anchor set vs B 加入后的 anchor set）
- **dbar 不一样**（每个 candidate 触发一次独立的 dbar 计算）
- **SIC 的求和 reference universe 不一样**（anchor 数量从 |selected| 变成 |selected|+1）

因此它们的 absolute SIC score 不是在同一个目标函数下比较的。

## V2 如何固定 reference universe

V2 在 selection 开始之前，从所有可用 embeddings 构建一次固定的 reference anchor universe A：

```python
all_episode_indices = sorted(embeddings.keys())
A = all available episodes  # 整个 selection 过程中不变
```

`FixedAnchorSIC` 类明确区分：
- **reference anchors A**：固定不变，用于评价 coverage
- **selected demonstrations D**：逐步增长，是实际选择的 episode 集合

整个 selection 过程中：
- `reference_anchor_count` 恒定 = N（总 episode 数）
- anchor universe 数量不随 candidate 变化

## V2 如何固定 dbar

`dbar_global` 和 `dbar_wrist` 在 `FixedAnchorSIC` 初始化时只计算一次：

```python
self.dbar_global, self.dbar_wrist, self.dbar_fallback_used = \
    compute_dbar_from_embeddings(self.phi_globals, self.phi_wrists)
```

之后所有 candidate scoring 都使用完全相同的 dbar 值。

预计算 kernel matrix：
```python
K_global[a, c] = exp(-||phi_global[a] - phi_global[c]|| / dbar_global)
K_wrist[a, c] = exp(-||phi_wrist[a] - phi_wrist[c]|| / dbar_wrist)
```

整个 greedy selection 过程直接查询 kernel matrix，不重复计算 pairwise distance。

## Sequential greedy 如何解决 top-9 redundancy

V1 的问题：对所有 candidate 打分 → 排序 → 一次加入 top 9。如果 A 和 B 相似，在旧 selected set 下都高分，但选择 A 后 B 的 marginal gain 应该下降。

V2 实现真正的 sequential greedy：

```python
while len(selected) < target_size:
    1. 根据当前 sigma 给所有 remaining candidate 计算 marginal gain
    2. 选择当前最大 marginal gain candidate
    3. 加入 selected
    4. 立即更新 sigma_global += tau * K_global[:, c]
    5. 立即更新 sigma_wrist += tau * K_wrist[:, c]
    6. 删除该 candidate
    7. 重新评价下一步 candidate
```

每次只正式决定一个 episode。`--n-add-per-round=9` 仅用于日志分组和 reporting。

## 是否使用了已有 embeddings

是。V2 直接复用已有的 embedding cache，没有重新提取。

`iterative_select_episodes_v2.py` 使用与 V1 完全相同的 `load_embeddings()` 函数读取 `.npy` 缓存文件。

## Unit Test 结果

10 项测试全部通过：

| Test | 说明 | 状态 |
|------|------|------|
| Test 1 | Fixed anchor universe | PASS |
| Test 2 | Fixed dbar | PASS |
| Test 3 | SIC monotonicity | PASS |
| Test 4 | Sequential redundancy penalty | PASS |
| Test 5 | Exactly target_size unique episodes | PASS |
| Test 6 | Reproducibility | PASS |
| Test 7 | Random B0 compatibility | PASS |
| Test 8 | Normalized SIC range | PASS |
| Test 9 | No NaN/Inf | PASS |
| Test 10 | Kernel matrix symmetry | PASS |

## V2 输出格式

### iterative_selection_result.json

新增字段：
- `selection_method`: `"fixed_anchor_sequential_sic_v2"`
- `reference_anchor_count`: 固定 reference anchor 数量
- `dbar_global`: 固定的 global 距离尺度
- `dbar_wrist`: 固定的 wrist 距离尺度
- `dbar_fallback_used`: 是否使用了 dbar fallback
- `normalized_sic`: SIC / (N * (1 + lambda_wrist))
- `selection_steps`: 每一步的详细诊断信息
  - `step`, `episode_index`, `marginal_gain`
  - `sic_before`, `sic_after`
  - `max_candidate_gain`, `median_candidate_gain`, `min_candidate_gain`
  - `time_seconds`
- `round_log`: 每 9 个 selection 的汇总
- `sigma_stats`: sigma 向量统计信息
- `total_runtime_seconds`: 总运行时间

### sic_curve.json

新增字段：
- `normalized_sic_scores`: 每轮的 normalized SIC
- `reference_anchor_count`: 固定值（不再变化）

## V2 Runtime

对于约 300 episodes 的 dataset：
- V1 每轮需要为每个 candidate 重新构建 AnchorSystem 并计算 dbar
- V2 预计算 kernel matrix 后，每步 candidate scoring 使用纯 numpy 向量化操作
- 预期速度提升显著（避免大量重复 Python loop）

## V1 完全可复现

V1 代码（`iterative_select_episodes.py`）未被修改，完全保留作为 baseline。

V2 的 `--b0-strategy random --seed 42` 产生的 B0 与 V1 的 `rng.choice()` 完全一致，确保公平 ablation。

## 未修改的内容

- Random selection：未修改
- Uniform selection：未修改
- MetaWorld eval image normalization：未修改
- 数据采集器：未修改
- SmolVLA 模型：未修改
- 现有训练好的 checkpoint：未修改
- Ours-v1 结果：未修改
- `extract_embeddings.py`：未修改

## WARNING: Training-time camera mismatch

`train_and_eval_v2.sh` 中的 training-time env eval camera 仍写死为 `corner`：

```bash
--env.camera_name="corner,gripperPOV"
```

因此 corner2 / corner3 的内置 16-episode eval **不能作为最终 benchmark**。

最终 benchmark 应继续使用 standalone `lerobot-eval`。

## 两个独立的研究问题

本项目涉及两个独立但相关的问题：

### A. Ours-v2 selection algorithm

通过固定 reference anchor universe 和预计算 kernel matrix，
实现更公平、更高效的 SIC-based episode selection。

### B. Dataset / embedding hypothesis validation

验证 SmolVLM 提取出的 episode embedding 是否真的构成一个
与 MetaWorld task state 有意义对应的表示空间。

**重要：** Ours-v2 fixed SIC 更高，不代表 policy success 一定更高。
必须先验证 embedding representation 本身是否与 task-state structure 相关。

只有当 embedding 确实编码了有意义的 task-state geometry 后，
继续优化 SIC episode selection 才有研究意义。

## Dataset Embedding Analysis

新增 `analyze_dataset_embedding.py` 用于验证 embedding 的 distinguishability 和 coverage。

### 运行示例

```bash
python personal/work2/see_dataset_after_eval/analyze_dataset_embedding.py \
    --dataset-root /data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner3 \
    --embeddings-dir /path/to/existing/embeddings \
    --random-subset /path/to/random_subset.json \
    --uniform-subset /path/to/uniform_subset.json \
    --ours-subset /path/to/ours_subset.json \
    --output-dir personal/work2/see_dataset_after_eval/analysis_results/corner3 \
    --n-bootstrap 1000 \
    --seed 42
```

### 输出文件

每个 dataset 输出：
- `analysis_summary.json` - 完整分析结果
- `embedding_metrics.csv` - embedding 统计指标
- `coverage_comparison.csv` - subset coverage 比较
- `analysis_report.md` - 人类可读报告（包含表格和假设评估）
- `figures/` - 可视化图表

### 核心假设

- **H1**: Embedding 是否具有可区分性？
- **H2**: Embedding distance 是否具有 task-state geometry？
- **H3**: Ours subset 是否显著优于 random coverage？

### Phase representation warning

当前 embedding 主要验证 initial/pre-grasp representation，
不能证明 transport/place phase representation 充分。

建议后续工作：
1. 提取不同 phase 的 embedding 并分别分析
2. 验证 embedding 是否编码 transport/place phase 信息
3. 分析 phase representation 与 task success 的关系