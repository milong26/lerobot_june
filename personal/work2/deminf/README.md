# DemInf (Demonstration Information) - LeRobot/PyTorch Implementation

## 概述

本模块是 DemInf (Demonstration Information, Robot Data Curation with Mutual Information Estimators) 核心算法在当前 LeRobot/PyTorch pipeline 中的 **忠实复现 (faithful reimplementation)**。

**不是**对原始 JAX/OpenX 代码的复制。所有实现仅依赖当前仓库已有环境（Python、PyTorch、NumPy、SciPy、LeRobot 数据读取接口），不引入 JAX、Flax、TensorFlow、RLDS、OpenX。

## 算法流程

DemInf 通过估计 state-action 之间的互信息 (Mutual Information) 来评估每条 demonstration 的信息量：

1. **读取完整 demonstration pool**：每个 episode 的逐 timestep observation/state 和 action
2. **训练 State VAE**：得到每个 timestep 的低维 latent `z_s`
3. **训练 Action VAE**：得到每个 timestep 的低维 latent `z_a`
4. **KSG MI 估计**：在 latent space 中对每个 state-action pair 计算信息分数
5. **Episode 聚合**：将一个 episode 内所有有效 timestep 的 information contribution 取平均
6. **排序选择**：按 score 降序排序，选出 top-K episode

### 数学公式

**VAE Loss**:
```
L = L_recon + beta * L_KL
L_recon = MSE(recon, x)
L_KL = -0.5 * mean[1 + logvar - mu^2 - exp(logvar)]
```

**KSG MI 估计** (per DemInf paper):
```
d_s(i,j) = ||z_s_i - z_s_j||_2          # state latent 欧氏距离
d_a(i,j) = ||z_a_i - z_a_j||_2          # action latent 欧氏距离
d_joint(i,j) = max(d_s(i,j), d_a(i,j))  # 联合空间 max metric

epsilon_i^k = k-th nearest neighbor distance in joint space (excluding self)
n_s(i,k) = #{j != i | d_s(i,j) < epsilon_i^k}
n_a(i,k) = #{j != i | d_a(i,j) < epsilon_i^k}

# DemInf ranking mode (用于 episode 排序):
score_i(k) = -(psi(n_s(i,k) + 1) + psi(n_a(i,k) + 1))

# Full KSG MI mode:
i_hat_i(k) = psi(k) + psi(N) - psi(n_s(i,k) + 1) - psi(n_a(i,k) + 1)

# 最终 score: 对所有 k 取平均
score_i = mean_k score_i(k)
```

**Episode Score**:
```
S(tau_e) = 1/T_e * sum_{t=1}^{T_e} I_hat(z_s_{e,t}; z_a_{e,t})
```

## 文件结构

```
deminf/
├── __init__.py           # 对外暴露 API
├── config.py             # DemInfConfig dataclass
├── dataset_adapter.py    # LeRobot 数据读取适配
├── models.py             # Beta-VAE (MLP Encoder/Decoder)
├── train_vae.py          # VAE 训练循环 + checkpoint
├── ksg.py                # KSG MI 估计器 (核心算法)
├── score_episodes.py     # Episode 评分流水线
├── select_subset.py      # Top-K 选择 + JSON 输出
├── run_deminf.py         # 主入口 (唯一推荐直接运行的脚本)
├── utils.py              # 工具函数 (seed, I/O, logging)
├── test_ksg.py           # KSG 单元测试
├── test_vae.py           # VAE 冒烟测试
└── README.md             # 本文件
```

## 默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| state_latent_dim | 12 | State VAE latent 维度 |
| action_latent_dim | 6 | Action VAE latent 维度 |
| hidden_dims | [512, 512] | VAE hidden layers |
| vae_beta_state | 0.05 | State VAE KL 权重 |
| vae_beta_action | 0.05 | Action VAE KL 权重 |
| vae_lr | 1e-3 | VAE 学习率 |
| vae_epochs | 100 | VAE 训练轮数 |
| ks | (5, 6, 7) | KSG k 值 |
| batch_size | 256 | VAE 训练 batch size |
| score_batch_size | 1024 | KSG 评分 batch size |
| ksg_backend | chunked | 分块计算 (内存安全) |
| ksg_mode | deminf_rank | 用于排序的兼容模式 |
| representation | state | 仅支持 state-based |

## Relative Action 注意事项

DemInf 经验上更适合 **relative/delta action**。当前 MetaWorld 数据集的 action 为 `(dx, dy, dz, gripper)` 增量控制命令，已经是 relative action。代码会自动检测并在日志中记录 `relative_action=true`。

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
    --vae-epochs 100 \
    --batch-size 256 \
    --score-batch-size 1024 \
    --state-latent-dim 12 \
    --action-latent-dim 6 \
    --ks 5 6 7
```

### Smoke Test (快速验证)

```bash
python personal/work2/deminf/run_deminf.py \
    --dataset-path /data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner3 \
    --output-dir /tmp/deminf_smoke_test \
    --target-episodes 5 \
    --seed 42 \
    --device cuda \
    --vae-epochs 2 \
    --batch-size 64 \
    --max-timesteps 50
```

### 单元测试

```bash
cd /data/zhonglinye/jun/lerobot
python -m pytest personal/work2/deminf/test_ksg.py -v
python -m pytest personal/work2/deminf/test_vae.py -v
```

## 输出文件

运行完成后，output_dir 包含：

| 文件 | 说明 |
|------|------|
| `subsets/deminf_{K}_seed{seed}.json` | 与现有训练流程兼容的 subset JSON |
| `episode_scores.csv` | 所有 episode 的 DemInf score 排名 |
| `score_rankings.csv` | 带 selected 标记的完整排名 |
| `config.json` | 运行配置 |
| `normalization_stats.npz` | State/Action 归一化统计量 |
| `latents.npz` | VAE 编码的 latent 缓存 (可复用) |
| `checkpoints/state_vae.pt` | State VAE checkpoint |
| `checkpoints/action_vae.pt` | Action VAE checkpoint |
| `deminf_metadata.json` | 运行元数据 (含 git commit) |
| `deminf.log` | 运行日志 |

### Subset JSON 格式

```json
{
  "selected_episode_indices": [0, 3, 5, ...],
  "num_episodes": 112,
  "selection_method": "deminf",
  "parameters": {
    "target_episodes": 112,
    "seed": 42,
    "ks": [5, 6, 7],
    "state_latent_dim": 12,
    "action_latent_dim": 6,
    "vae_beta_state": 0.05,
    "vae_beta_action": 0.05,
    "vae_epochs": 100,
    "hidden_dims": [512, 512],
    "relative_action": true,
    "dataset_path": "...",
    "score_aggregation": "mean",
    "ksg_mode": "deminf_rank",
    "ksg_backend": "chunked",
    "representation": "state"
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

## 缓存文件含义

| 文件 | 内容 | 可复用场景 |
|------|------|-----------|
| `latents.npz` | z_state, z_action, episode_ids, timestep_ids | 修改 ks 时无需重新训练 VAE |
| `checkpoints/state_vae.pt` | State VAE 模型 + 优化器状态 | `--resume` 继续训练 |
| `checkpoints/action_vae.pt` | Action VAE 模型 + 优化器状态 | `--resume` 继续训练 |
| `normalization_stats.npz` | state_mean, state_std, action_mean, action_std | 确保评分使用同一套归一化 |

## 实验公平性

- DemInf 的 State/Action VAE **仅使用无监督表示学习**
- **不使用**环境 success label、策略 evaluation success rate、reward、最终 checkpoint 表现或人工质量标签
- 候选 full pool 可以全部参与无监督 VAE 训练和 MI scoring（这是 DemInf offline data curation 的定义）
- 最终策略模型使用和 Random、SubZeroCore、ours_v3/ours_v4 **完全相同的训练代码、训练步数和 evaluation set**，只改变选中的 episode indices