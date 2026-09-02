# DemInf (Demonstration Information) - LeRobot/PyTorch Implementation

## 概述

本模块是 DemInf (Robot Data Curation with Mutual Information Estimators, RSS 2025) 核心算法在 LeRobot/PyTorch 框架下的 **算法级忠实复现 (algorithmically and numerically validated PyTorch/LeRobot reimplementation of the official DemInf estimator)**。

**不是**对原始 JAX/OpenX 代码的复制。所有实现仅依赖当前仓库已有环境（Python、PyTorch、NumPy、SciPy、LeRobot 数据读取接口），不引入 JAX、Flax、TensorFlow、RLDS、OpenX。除"JAX/OpenX→PyTorch/LeRobot 数据接口"这一工程适配外，DemInf 的 VAE 架构、VAE loss、训练超参数、KSG estimator、batch scoring 方式、repeat 方式、score clipping、score normalization、episode aggregation 和最终 ranking 与原始 DemInf 官方实现一致。

## DemInf 模块说明

### 数据输入

- **数据来源**：LeRobot 框架采集的 MetaWorld trajectory 数据
- **State 字段**：固定使用 `observation.environment_state`（39维）
- **Action 字段**：MetaWorld 原始 4维 action `(x, y, z, gripper)`
- **数据级别**：trajectory demonstration 级别的数据价值评估

### 评分流程

```
LeRobot MetaWorld trajectory
    → observation.environment_state (39维) + action (4维)
    → Normalization (state: Gaussian, action xyz: Gaussian, action gripper: Bounds)
    → State Beta-VAE (latent_dim=12) + Action Beta-VAE (latent_dim=6)
    → Latent encoding (posterior mean)
    → KSG Mutual Information Estimator (batch-local, ks=[5,6,7])
    → Per-timestep MI scores
    → NaN filtering → 1%/99% percentile clipping → global z-score normalization
    → Episode score (mean aggregation)
    → Top-K episode selection
```

### 模块限制

本模块 **不使用** 以下内容：
- ❌ 视觉 embedding（VLM、CLIP、image feature 等）
- ❌ Reward 信息
- ❌ Policy rollout / 策略执行结果
- ❌ 策略性能反馈
- ❌ 环境测试结果
- ❌ Ours 方法评分
- ❌ SIC 评分
- ❌ Embedding 评分

本模块 **仅依赖**：
- ✅ State Beta-VAE 编码的 latent
- ✅ Action Beta-VAE 编码的 latent
- ✅ KSG mutual information estimator
- ✅ LeRobot MetaWorld 数据（observation.environment_state + action）

### 独立运行

DemInf 模块必须独立运行，不依赖 `personal/work2` 中的 Ours 方法、embedding 方法、VLM 特征、视觉编码器、策略执行结果或 reward 信息。启动前会执行环境隔离检查，确保不存在 forbidden 模块依赖。

## 算法流程

DemInf 通过估计 state-action 之间的互信息 (Mutual Information) 来评估每条 demonstration 的信息量：

1. **读取完整 demonstration pool**：通过 LeRobotDataset 加载本地数据，基于 `dataset.hf_dataset` 的真实全局索引构造 episode mapping
2. **删除 terminal transition**：每个 episode 删除最后一个 timestep（官方 `_stepify` 行为：`x[:-1]`）
3. **提取 state/action**：默认使用 `observation.environment_state`（39维），action 为原始4维 `(x,y,z,gripper)`
4. **DemInf 归一化**：连续非 gripper 维 Gaussian normalization，gripper 维保持原尺度；action xyz Gaussian normalization，gripper bounds normalization
5. **训练 State VAE**：固定 50000 Adam optimizer steps（非 epochs），lr=1e-4，weight_decay=0，beta=0.05
6. **训练 Action VAE**：同上配置
7. **编码 latent**：使用 VAE posterior mean（不采样），得到每个 timestep 的 `z_s` 和 `z_a`
8. **构建 quality batches**：repeat=4，batch_size=1024，drop_remainder=True，每次用确定性 seed shuffle
9. **Batch-local KSG scoring**：在每个 1024 batch 内计算官方 KSG estimator
10. **后处理**：过滤 NaN → 全局 1st/99th percentile clipping → 全局 z-score normalization
11. **Episode 聚合**：对归属于同一 episode 的所有 normalized score 取算术平均
12. **排序选择**：按 deminf_score 降序排序，选出 top-K episode

### 数学公式

**VAE Loss**（先对 feature/latent 维求和，再对 batch 求 mean）：
```
recon_per_sample = sum_d (x_d - x_hat_d)^2          # 对 feature 维求和
recon_loss = mean_batch(recon_per_sample)            # 对 batch 求 mean

kl_per_sample = 0.5 * sum_j (-logvar_j - 1 + exp(logvar_j) + mu_j^2)  # 对 latent 维求和
kl_loss = mean_batch(kl_per_sample)                   # 对 batch 求 mean

total_loss = recon_loss + beta * kl_loss
```

**注意**：严禁使用 `F.mse_loss(..., reduction="mean")`（会同时对 batch 和 feature 平均），严禁对 latent 维直接 mean。

**KSG MI 估计**（官方 DemInf batch-local estimator）：

对于一个 batch 大小 B：
```
state_dist[i,j]  = ||z_s_i - z_s_j||_2              # 包含对角线 self distance = 0
action_dist[i,j] = ||z_a_i - z_a_j||_2              # 包含对角线 self distance = 0
joint_dist       = max(state_dist, action_dist)

sorted_joint = sort(joint_dist, dim=-1)              # 升序排序

# ks = [5, 6, 7] 是零基下标（Python/JAX 数组索引）
# 第0列是 self distance = 0，所以 [5,6,7] 对应第5/6/7个邻居
epsilon_k = sorted_joint[:, k]                       # 对每个 k 取阈值

# 严格 < 比较，self distance = 0 通常 < epsilon，自动计入 count
obs_count_k    = sum_j (state_dist[i,j]  < epsilon_k[i])   # 不要手动排除 self
action_count_k = sum_j (action_dist[i,j] < epsilon_k[i])   # 不要对 count +1

# 官方 DemInf ranking score（唯一用于 episode ranking 的公式）
score_i(k) = -(digamma(obs_count_k[i]) + digamma(action_count_k[i]))

# 最终 score：对所有 k 取平均
score_i = mean_k score_i(k)
```

**重要**：
- 对角线 self distance **不**设为 inf，保持为 0
- **不**给 epsilon 加 1e-10 偏移（官方是严格 `<`）
- **不**对 count 额外 +1
- **不**加 textbook KSG 中的 `psi(k) + psi(N)` 项
- **不使用** `psi(count + 1)`

**Quality Inference Pipeline**：
1. 对每个 repeat（默认4次），用 `seed + repeat_id` 确定性 shuffle 全部 transition
2. 按 `quality_batch_size=1024` 切 batch，`drop_remainder=True` 丢弃尾部不足 batch
3. 对每个 batch 调用 `deminf_ksg_batch_scores()` 得到 per-sample score
4. 同一 transition 在 4 次 repeat 中出现在不同 batch 上下文中，产生 4 个不同 score

**Score 后处理**（在 episode aggregation 之前执行）：
```
# 1. 过滤 NaN
scores = scores[isfinite(scores)]

# 2. 全局 percentile clipping
p1 = percentile(scores, 1)
p99 = percentile(scores, 99)
scores_clipped = clip(scores, p1, p99)

# 3. 全局 z-score normalization
scores_norm = (scores_clipped - mean(scores_clipped)) / std(scores_clipped)

# 4. Episode 聚合（算术平均）
episode_score[e] = mean(scores_norm for all timesteps belonging to episode e)
```

**Episode Score**：
```
S(tau_e) = mean_t ( normalized_score_{e,t} )
```
由于 quality_repeat=4，同一 episode 通常有多次 transition appearance，全部参与 mean。

## 文件结构

```
deminf/
├── __init__.py              # 对外暴露 API
├── config.py                # DemInfConfig dataclass（官方 state-based 配置）
├── dataset_adapter.py       # LeRobot 数据读取适配（全局 row index、归一化）
├── models.py                # Beta-VAE（MLP Encoder/Decoder，官方架构）
├── train_vae.py             # VAE 训练循环（固定 50000 steps）+ checkpoint
├── ksg.py                   # KSG MI 估计器（官方 batch-local 实现）
├── score_episodes.py        # Episode 评分流水线（quality inference）
├── select_subset.py         # Top-K 选择 + JSON 输出
├── run_deminf.py            # 主入口（完整 pipeline）
├── utils.py                 # 工具函数（seed, I/O, logging）
├── test_ksg.py              # KSG 单元测试
├── test_vae.py              # VAE 单元测试
├── test_quality_pipeline.py # Quality pipeline 单元测试
├── test_dataset_adapter.py  # Dataset adapter 单元测试
└── README.md                # 本文件
```

## 默认参数（官方 state-based 配置）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| state_latent_dim | 12 | State VAE latent 维度 |
| action_latent_dim | 6 | Action VAE latent 维度 |
| hidden_dims | [512, 512] | VAE hidden layers |
| vae_beta_state | 0.05 | State VAE KL 权重 |
| vae_beta_action | 0.05 | Action VAE KL 权重 |
| vae_lr | 1e-4 | VAE 学习率（Adam） |
| vae_steps | 50000 | VAE 固定 optimizer steps（非 epochs） |
| weight_decay | 0.0 | Adam weight decay（与官方 optax.adam 对齐） |
| vae_batch_size | 256 | VAE 训练 batch size |
| quality_batch_size | 1024 | Quality inference batch size |
| quality_repeat | 4 | KSG estimator repeat 次数 |
| quality_cache | True | 强制 effective_discard_fraction=0 |
| quality_discard_fraction | 0.5 | 请求的丢弃比例（cache=True 时实际为 0） |
| quality_drop_remainder | True | 丢弃不足 batch_size 的尾部 batch |
| ks | (5, 6, 7) | KSG k 值（零基下标） |
| score_clip_low | 1.0 | Score clipping 下百分位 |
| score_clip_high | 99.0 | Score clipping 上百分位 |
| state_source | observation.environment_state | State 来源（39维） |
| representation | state | 仅支持 state-based |

## State/Action 数据说明

**State**（默认 39 维 `observation.environment_state`）：
- 0:3 当前 hand xyz 位置
- 3 当前 gripper（**不做** Gaussian normalization）
- 4:18 当前 object 相关状态
- 18:36 上一帧 0:18 的复制
- 21 上一帧 gripper（**不做** Gaussian normalization）
- 36:39 goal xyz 位置

**注意**：不再拼接 4 维 `observation.state` 成 43 维，因为 `environment_state` 的前 4 维已包含相同 robot state，会重复加权。

**Action**（4 维 `(x, y, z, gripper)`）：
- 前三维 xyz 使用 Gaussian normalization
- 第 3 维 gripper：**始终**使用 bounds normalization 到 [-1,1]，基于 dataset 实际 min/max
  （即使原始值已经在 [-1,1] 范围内，仍然执行该变换；若 min=-1, max=1 则结果与原值相同）
  这与官方 `NormalizationType.BOUNDS` 完全一致，不再以"已在[-1,1]则跳过"
- MetaWorld expert action 属于增量/relative control，metadata 记录 `relative_action=true`

## 运行命令

### 正式运行

```bash
cd /data/zhonglinye/jun/lerobot

python personal/work2/deminf/run_deminf.py \
    --dataset-path /data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner3 \
    --output-dir /data/zhonglinye/jun/lerobot/personal/work2/duibi/deminf_112_seed42_corner3 \
    --target-episodes 112 \
    --seed 42 \
    --device cuda \
    --vae-steps 50000 \
    --vae-lr 1e-4 \
    --vae-batch-size 256 \
    --quality-batch-size 1024 \
    --quality-repeat 4 \
    --state-latent-dim 12 \
    --action-latent-dim 6 \
    --ks 5 6 7 \
    --state-source observation.environment_state \
    --quality-cache
```

### Smoke Test（快速验证）

```bash
python personal/work2/deminf/run_deminf.py \
    --dataset-path /data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner3 \
    --output-dir /tmp/deminf_smoke_test \
    --target-episodes 5 \
    --seed 42 \
    --device cuda \
    --vae-steps 50 \
    --vae-batch-size 64 \
    --quality-batch-size 128 \
    --max-timesteps 50
```

### 单元测试

```bash
cd /data/zhonglinye/jun/lerobot
python -m pytest personal/work2/deminf/test_ksg.py -v
python -m pytest personal/work2/deminf/test_vae.py -v
python -m pytest personal/work2/deminf/test_quality_pipeline.py -v
python -m pytest personal/work2/deminf/test_dataset_adapter.py -v
```

## 输出文件

运行完成后，output_dir 包含：

| 文件 | 说明 |
|------|------|
| `subsets/deminf_{K}_seed{seed}.json` | 与现有训练流程兼容的 subset JSON |
| `episode_scores.csv` | 所有 episode 的 DemInf score 排名 |
| `raw_timestep_scores.csv` | 每个 timestep 的 raw/clipped/normalized score |
| `score_rankings.csv` | 带 selected 标记的完整排名 |
| `config.json` | 运行配置 |
| `normalization_stats.npz` | State/Action 归一化统计量 |
| `latents.npz` | VAE 编码的 latent 缓存（带 manifest 验证） |
| `latents_manifest.json` | Latent cache fingerprint 元数据 |
| `checkpoints/state_vae_step50000.pt` | State VAE 最终 checkpoint |
| `checkpoints/action_vae_step50000.pt` | Action VAE 最终 checkpoint |
| `deminf_metadata.json` | 运行元数据（含 git commit） |
| `deminf.log` | 运行日志 |

### Subset JSON 格式

```json
{
  "selected_episode_indices": [0, 3, 5, ...],
  "num_episodes": 112,
  "selection_method": "deminf",
  "parameters": {
    "algorithm": "DemInf",
    "representation": "state_action",
    "state_key": "observation.environment_state",
    "action_key": "action",
    "state_dim": 39,
    "action_dim": 4,
    "state_latent_dim": 12,
    "action_latent_dim": 6,
    "score_type": "ksg_mutual_information",
    "episode_aggregation": "mean",
    "uses_policy_rollout": false,
    "uses_reward": false,
    "uses_visual_embedding": false,
    "official_deminf": true,
    "target_episodes": 112,
    "seed": 42,
    "vae_steps": 50000,
    "vae_lr": 1e-4,
    "vae_batch_size": 256,
    "quality_batch_size": 1024,
    "quality_repeat": 4,
    "requested_discard_fraction": 0.5,
    "effective_discard_fraction": 0.0,
    "quality_cache": true,
    "ks": [5, 6, 7],
    "state_source": "observation.environment_state",
    "vae_beta_state": 0.05,
    "vae_beta_action": 0.05,
    "hidden_dims": [512, 512],
    "relative_action": true,
    "dataset_path": "...",
    "score_clip_percentiles": [1.0, 99.0],
    "score_normalization": "global_zscore_after_clipping"
  }
}
```

## 接入策略训练

生成的 subset JSON 可直接用于当前训练流程，与 Random/SubZeroCore/ours_v3/ours_v4 完全兼容：

```bash
# 从 subset JSON 提取 episode indices
EPISODES=$(python -c "import json; data=json.load(open('subsets/deminf_112_seed42.json')); print('[' + ','.join(str(x) for x in data['selected_episode_indices']) + ']')")

# 用于 lerobot-train
lerobot-train \
    --dataset.episodes="$EPISODES" \
    ...  # 其他训练参数与已有实验完全相同
```

## 缓存机制

### Latent Cache

`latents.npz` 旁边保存 `latents_manifest.json`，包含 comprehensive fingerprint：
- dataset 路径 + info.json hash
- total episodes + total frames
- state_source + action_key
- normalization stats hash
- state/action input dims + latent dims + hidden dims
- VAE beta + lr + vae_steps
- state/action checkpoint SHA256
- 代码 git commit

只有所有关键字段一致才能 reuse，否则日志说明 cache mismatch 并重新 encode。

### Quality Score Cache

Quality score batch 本身还依赖 quality seed、batch size、repeat、discard fraction 和 ks，如需增加 score cache，必须单独具有 score fingerprint。

## 关键修复说明

本实现修复了以下早期版本中的问题：

| 问题 | 修复方式 |
|------|----------|
| 全局 N×N KNN pool | 改为 batch-local KSG（1024 batch 内计算） |
| VAE loss 使用 F.mse_loss reduction="mean" | 改为 feature 维 sum → batch 维 mean |
| KL 对 latent 维 mean | 改为 latent 维 sum → batch 维 mean |
| Parquet local index 误作 global index | 统一通过 LeRobotDataset hf_dataset 全局索引 |
| 全数据 torch.from_numpy(data).to(cuda) 后 DataLoader | 数据留 CPU，batch.to(device, non_blocking=True) |
| 43 维 state（拼接重复 robot state） | 默认只用 39 维 environment_state |
| Cache 无条件 reuse | 实现 comprehensive fingerprint 验证 |
| KSG epsilon 加 1e-10 偏移 | 移除，使用严格 `<` 比较 |
| Self distance 设为 inf | 保持为 0，自动计入 count |
| Count 额外 +1 | 移除，与官方一致 |
| VAE 训练使用 epochs + early stopping | 改为固定 50000 optimizer steps |
| PyTorch Linear 使用默认 Kaiming 初始化 | 改为 Xavier uniform，对齐官方 `nn.initializers.xavier_uniform` |
| Action gripper 已在[-1,1]则跳过 normalization | 始终执行 BOUNDS normalization，使用 dataset 实际 min/max |
| 50-step smoke checkpoint 被误用于 50000-step 正式实验 | 新增 `validate_vae_checkpoint()`，检查 global_step >= vae_steps + 全字段 fingerprint |
| Latent cache 实际上没有用于 quality scoring | 重构为 `score_latents()` 核心函数，cache 命中直接调用，不重新 encode |
| Cache fingerprint 使用空字符串代替 git commit | 新增 `get_git_commit()` 获取真实 git commit hash |
| NaN/Inf 过滤语义与官方不一致 | 只过滤 NaN（`~jnp.isnan`），Inf 视为异常并抛出明确 ValueError |
| `ksg_local_scores()` 被正式 pipeline 误调用 | 标记为 legacy/diagnostic，正式路径只使用 `deminf_ksg_batch_scores` |
| `test_ks_indices_are_zero_based_5_6_7` 使用错误的 `torch.diag` | 修正为直接比较 `sorted_joint[:, 0]` 与 zeros |
| Cache fingerprint 在 VAE 训练前构建，checkpoint hash 为空 | 重构执行顺序：先完成 VAE 训练→确定最终 checkpoint→计算 fingerprint→检查 cache |