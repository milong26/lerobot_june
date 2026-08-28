# MetaWorld 动态锚点迭代采集规划

## 项目概述

本项目实现基于状态信息覆盖（SIC）的动态锚点贪心采集规划方法，用于在有限采集预算下优化机器人示范数据采集方案。

核心思路：将锚点参考系从"预先固定"改为"随采集过程动态生长"，用SIC边际增益准则在"新构型探索"和"已有构型重复"之间做统一的贪心决策。

## 目录结构

```
personal/work2/our/
├── README.md                       # 本文件
├── core/
│   ├── anchors.py                  # AnchorSystem: 管理锚点集、计算d̄_global/d̄_wrist
│   ├── sic.py                      # SIC计算: K_global/K_wrist、τ(t)、支持度、饱和函数、SIC总分
│   ├── candidate_pool.py           # 112格点候选池: visited/unvisited/rep_count状态机
│   └── greedy_planner.py           # 通用贪心主循环: score_fn可插拔
├── variants/
│   ├── v1_dynamic_anchor_greedy.py # V1: 动态锚点贪心
│   ├── v2_two_phase_explore_exploit.py # V2: 两阶段探索-利用
│   ├── v3_explicit_tradeoff_greedy.py # V3: 显式权衡贪心
│   └── v4_probe_weighted_greedy.py # V4: 探针加权贪心
├── embeddings/
│   └── extract_embeddings.py       # 冻结VLM特征提取、global/wrist嵌入、PCA降维
├── experiments/
│   ├── check_pool_coverage.py      # 统计已有数据集覆盖的网格点数
│   ├── run_offline_simulation.py   # 离线模拟V1-V3，扫多个预算点
│   ├── run_online_collection.py    # 在线补采（当离线池覆盖不足时）
│   └── train_eval_selected_subset.py # 训练+评测选中子集
└── results/
    ├── logs/
    ├── tables/
    └── figures/
```

## 待核实项确认结果

1. **朝向/角度分量**: MetaWorld pick-place-v3任务的随机化只有x-y位置，没有角度分量。候选空间为14×8=112个网格点。

2. **VLM嵌入抽取**: 需要新实现，不能用欧氏距离替代。使用HuggingFaceTB/SmolVLM2-500M-Video-Instruct模型，global嵌入取前5帧平均，wrist嵌入取20%-70%区间帧平均，PCA降维到16或32维。

3. **tanzhen目录**: 不存在，V4自动退化为V1。

4. **claude.md冲突检查**: 本文档设计与claude.md不冲突，以本文档为准实现动态锚点系统。

## 复现步骤

### 第一步：检查数据集覆盖率

```bash
cd /data/zhonglinye/jun/lerobot

python personal/work2/our/experiments/check_pool_coverage.py \
    --datasets \
        personal/work2/dataset_view/random_112 \
        personal/work2/dataset_view/uniform_112 \
    --output personal/work2/our/results/tables/coverage_stats.json
```

### 第二步：提取VLM嵌入

```bash
python personal/work2/our/embeddings/extract_embeddings.py \
    --dataset-dir personal/work2/dataset_view/random_112 \
    --output-dir personal/work2/our/embeddings/cache/random \
    --n-components 32

python personal/work2/our/embeddings/extract_embeddings.py \
    --dataset-dir personal/work2/dataset_view/uniform_112 \
    --output-dir personal/work2/our/embeddings/cache/uniform \
    --n-components 32
```

### 第三步：离线模拟

```bash
python personal/work2/our/experiments/run_offline_simulation.py \
    --embeddings-dir personal/work2/our/embeddings/cache \
    --output-dir personal/work2/our/results/tables \
    --budgets 50 100 112 150 200 \
    --t-max 10 \
    --alpha 1.0 \
    --lambda-wrist 1.0 \
    --b-cov 50 \
    --gammas 0.0 0.1 0.5 1.0 2.0
```

### 第四步：训练与评测

```bash
python personal/work2/our/experiments/train_eval_selected_subset.py \
    --planning-result personal/work2/our/results/tables/best_planning_result.json \
    --dataset-dir personal/work2/dataset_view/random_112 \
    --output-dir personal/work2/our/eval_results \
    --gpu-id 0 \
    --n-steps 200
```

## 核心公式

### 次数软化函数
τ(t) = α · log((t+1)/t)

### 空间接近度核函数
K_global(a,c) = exp(-||φ_global(a) - φ_global(c)||_2 / d̄_global)
K_wrist(a,c) = exp(-||φ_wrist(a) - φ_wrist(c)||_2 / d̄_wrist)

### SIC总分
SIC(D) = Σ_a σ_global(a,D)/(1+σ_global(a,D)) + λ·Σ_a σ_wrist(a,D)/(1+σ_wrist(a,D))

其中：
σ_global(a,D) = Σ_(p,r)∈D Σ_(t=1)^Tp,r τ(t)K_global(a,p,r)
σ_wrist(a,D) = Σ_(p,r)∈D Σ_(t=1)^Tp,r τ(t)K_wrist(a,p,r)

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| α | 1.0 | 次数软化系数 |
| λ | 1.0 | wrist视角权重 |
| γ | 0.0 | V3探索奖励超参数 |
| t_max | 10 | 单配置最大采集次数 |
| B_cov | 50 | V2阶段一空间覆盖数 |
| n_components | 32 | PCA降维目标维度 |

## 输出规范

所有结果保存在 `personal/work2/our/` 目录下：
- 离线模拟结果: `results/tables/offline_sim_{variant}.csv`
- 覆盖率统计: `results/tables/coverage_stats.json`
- 评估结果: `eval_results/dynamicanchor_eval.json`
- 嵌入缓存: `embeddings/cache/`