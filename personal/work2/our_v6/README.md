# our_v6: AdaptiveGrid + V5 Action Descriptors

## 设计理念

our_v6 结合了 v4 和 v5 的优势：
- **v4 的从0开始采集方式**：模拟真实机器人逐步采集数据的过程
- **v5 的 action descriptor**：使用预计算的 action descriptor（而非 v4 自己构建的 action embedding）

## 核心特性

### 与 v4 的区别
- 使用 v5 的 action descriptor 替代 v4 的 temporal+statistical action embedding
- 简化了 action embedding 的构建流程

### 与 v5 的区别
- 采用自适应网格的从0开始采集策略
- 严格的因果关系：只有 acquire 后才能获得 embedding
- 基于空间位置的 episode 选择（而非直接筛选候选集）

## 代码结构

```
our_v6/
├── __init__.py
├── config.py                          # 配置文件
├── core/
│   ├── __init__.py
│   ├── adaptive_grid.py               # 自适应网格（复用 v4）
│   ├── pool_adapter.py                # 数据集适配器（复用 v4）
│   ├── visual_embedding.py            # 视觉嵌入（复用 v4）
│   ├── action_embedding.py            # **核心修改**：使用 v5 action descriptor
│   ├── scoring.py                     # 评分模块（适配 v5 action）
│   └── planner.py                     # **核心修改**：V6Planner
└── experiments/
    └── select_episodes_v6.py          # 实验脚本
```

## 快速运行

### 单数据集选择
```bash
cd /data/zhonglinye/jun/lerobot
python our_v6/experiments/select_episodes_v6.py \
    --dataset_root /path/to/dataset \
    --embedding_dir /path/to/embeddings \
    --action_descriptor_dir /path/to/action_descriptors \
    --total_budget 112 \
    --seed 42 \
    --output_dir output_v6
```

### 多数据集联合训练（推荐）
```bash
# 使用默认配置（3个数据集，每个112 episodes）
bash duibi/train_and_eval_scripts/launch_ours_v6.sh 112 0 42

# 参数说明：
# $1: 每个数据集的 episode 数量（默认 112）
# $2: GPU ID（默认 0）
# $3: 随机种子（默认 42）
```

### 查看运行状态
```bash
# 附加到 tmux 会话
tmux attach -t our_v6_112x3_s42

# 查看日志
tail -f /data/zhonglinye/jun/lerobot/personal/work2/duibi/our_v6_112x3_seed42/logs/our_v6_112x3_seed42.log
```

## 配置参数

### config.py 主要参数
```python
TOTAL_BUDGET = 112              # 总采集预算
INITIAL_GRID_X = 7              # 初始网格 X 分辨率
INITIAL_GRID_Y = 4              # 初始网格 Y 分辨率
INITIAL_BUDGET = 28             # 第一阶段预算（7*4）
MAX_DEPTH = 3                   # 最大分裂深度
SPATIAL_WEIGHT = 1.0            # 空间需求权重
VISUAL_WEIGHT = 1.0             # 视觉分歧权重
ACTION_WEIGHT = 0.5             # 动作分歧权重
```

## 运行流程

1. **Episode 选择**：对每个数据集运行 V6 选择算法
   - Stage 1: 均匀覆盖采集（28 episodes）
   - Stage 2: 自适应采集（84 episodes）
   
2. **联合训练**：使用所有选中的 episodes 训练 SmolVLA 模型

3. **评估**：在三个数据集上分别评估模型性能

## 输出目录结构

```
our_v6_112x3_seed42/
├── logs/
│   ├── our_v6_112x3_seed42.log
│   ├── our_v6_112x3_seed42.time
│   └── our_v6_112x3_seed42.pid
├── selection/
│   ├── assembly-v3/
│   │   └── selected_episodes_v6.json
│   ├── button-press-v3/
│   │   └── selected_episodes_v6.json
│   └── drawer-open-v3/
│       └── selected_episodes_v6.json
├── combined_selected_episodes_v6.json
├── our_v6_112x3_seed42/          # 训练输出
│   └── checkpoints/
└── eval_results/
    └── our_v6_112x3_seed42_eval.json
```