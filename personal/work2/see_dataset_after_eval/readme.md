# Ours / DynamicAnchor V2 Episode Selection

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