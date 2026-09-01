# Dynamic Anchor v2 - State-Action Manifold Coverage

## v1 vs v2 区别

| 特性 | v1 | v2 |
|------|----|----|
| **目标** | Visual embedding coverage maximization | State-action manifold coverage with redundancy control |
| **Embedding** | Global + Wrist visual embeddings | State-action joint embedding (visual + action trajectory) |
| **选择策略** | Batch top-k per round | Sequential greedy, one episode at a time with gain recomputation |
| **冗余控制** | 无显式冗余惩罚 | kNN-based redundancy penalty |
| **打分公式** | ΔSIC (marginal gain only) | marginal_gain / (1 + redundancy_penalty) |
| **Action 来源** | 无 | LeRobot demonstration dataset action trajectory |

## 核心思想

v2 不再只最大化视觉 embedding 的 coverage，而是选择能够覆盖 state-action manifold 且保持有效数据密度的 episode subset：

1. **保留 SIC diversity gain**，但改为 state+action 联合空间
2. **Sequential marginal gain greedy** 选择，每次选一个后重新计算所有候选的 gain
3. **kNN redundancy penalty** 自动惩罚与已选 episode 过于接近的候选
4. **无需人工调 alpha/beta/gamma 权重**，score = gain / (1 + penalty)
5. **Action embedding 基于 demonstration data**，不依赖训练好的 policy 内部信息

## 文件结构

```
our_v2/
├── core/
│   ├── embedding.py        # V2 embedding 构建 (global+wrist+action)
│   ├── action_embedding.py # Action trajectory embedding 提取 (基于 dataset)
│   ├── sic_v2.py           # V2 SIC 计算 (pairwise distance, marginal gain, redundancy)
│   └── anchors_v2.py       # Sequential greedy selection
├── experiments/
│   └── select_episodes_v2.py  # 命令行入口
├── config.py               # 默认配置
└── readme.md
```

## 运行方式

### 仅使用 visual embedding

```bash
python personal/work2/our_v2/experiments/select_episodes_v2.py \
    --dataset-root /path/to/dataset \
    --embedding-dir /path/to/embeddings \
    --output-dir /path/to/output \
    --num-selected 112 \
    --seed 42
```

### 使用 action embedding（默认推荐）

```bash
python personal/work2/our_v2/experiments/select_episodes_v2.py \
    --dataset-root /path/to/dataset \
    --embedding-dir /path/to/embeddings \
    --output-dir /path/to/output \
    --num-selected 112 \
    --seed 42 \
    --use-action-embedding
```

## 输出文件格式

输出 JSON 文件 (`dynamicanchor_v2_112_seed42.json`):

```json
{
    "selected_episode_indices": [0, 3, 7, 12, ...],
    "num_episodes": 112,
    "selection_method": "dynamic_anchor_v2",
    "parameters": {
        "target_size": 112,
        "bandwidth": 0.1234
    }
}
```

该格式与已有 train_and_eval_scripts 完全兼容。

## 与 v1 的主要算法变化

1. **Embedding 空间扩展**: v1 仅使用 global+wrist 视觉 embedding；v2 加入 action trajectory embedding（基于 demonstration dataset 的 action 统计特征），形成 state-action 联合空间。

2. **Sequential vs Batch 选择**: v1 每轮计算所有候选的 SIC 分数后取 top-k；v2 每次只选一个 score 最高的 episode，然后重新计算所有剩余候选的 marginal gain，确保每次选择都基于最新的 selected set。

3. **Redundancy 控制**: v1 无冗余惩罚；v2 使用 kNN 距离计算 candidate 与已选 episode 的局部重复程度，作为分母惩罚项自动降低冗余候选的 score。

4. **带宽估计**: v1 使用 anchor system 的 dbar 计算；v2 使用 mean kNN distance 自动估计 kernel bandwidth，减少人工参数。

5. **Action embedding 来源**: 基于 LeRobot demonstration dataset 的 action trajectory 统计特征（mean, std, velocity, range 等），不依赖训练好的 policy 内部 hidden/x_t/v_t 信息，确保 selection 只使用训练数据本身可获得的信息。