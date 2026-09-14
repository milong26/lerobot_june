# our_v6: Causal Configuration-Aware Adaptive Demonstration Acquisition

our_v6 保留 V5 的 configuration-region 选择思路，但把信息流改成严格因果形式：在 episode 被正式 acquire 之前，只允许读取 `episode_index` 与 `rand_vec`；视觉观测与 action 只能在 acquire 之后读取并参与后续 region priority。

## 核心流程

1. 使用全体 episode 的 `rand_vec` 元数据做 per-dimension min-max normalization。
2. 只基于 normalized `rand_vec` 做 KMeans，区域数为 `K=min(M,max(K_min,min(K_max,floor(rho*M))))`。
3. 确定性选择 `B0=max(1,floor(eta*K))` 个区域，并从每个区域选择离 centroid 最近的配置作为初始 acquisition。
4. acquire 后才提取该 episode 的 frozen SmolVLM visual feature 与 action sequence。
5. 后续 region priority 仅使用：configuration coverage gap、已 acquire visual heterogeneity、已 acquire action heterogeneity。
6. 在最高优先级 region 内，仅使用 configuration metadata 做 maximin target proposal；未 acquire visual/action 不参与 candidate selection。
7. 重复直到达到固定 budget。

## Visual variants

### `l2`（默认）
- frozen SmolVLM raw global branch
- frozen SmolVLM raw wrist branch
- branch-wise L2 normalization
- concatenate
- final L2 normalization

该模式完全不拟合 PCA。

### `online_pca`
- raw global/wrist feature 只在 episode acquire 后加入当前 acquired set
- PCA 只使用当前 acquired set 拟合
- 每次新增 episode 后重新拟合 branch-wise PCA
- 历史所有 acquired episodes 都重新投影到当前 PCA basis
- PCA 后 branch-wise L2，concatenate，再 final L2

因此 `online_pca` 不会使用未来 episode 的 hidden trajectory feature。不能复用旧 V1-V5 的 full-pool PCA embedding 作为该变体的输入。

## Action representation

保留 V5 的 action descriptor 信息内容：16-step temporal resampling、mean/std、delta mean/std、mean absolute delta、path length；但只在 acquire 后读取 action，并采用 group-wise normalization/weighting + final L2，避免 temporal block 维度主导距离。

## Ablations

支持：
- `full`
- `wo_action`：保持 regions、visual feedback、target proposal、causal access 不变，仅令 action region term 为 0。
- `wo_adaptive_priority`：保持相同 regions、target proposal、reveal rule，不使用 adaptive priority，而是在 non-exhausted regions 间 deterministic round-robin。

旧 V5 的 `w/o Region` 不再用于 causal V6。

## 直接运行：选择 + 合并 + 训练

从仓库根目录执行：

```bash
bash personal/work2/our_v6/run_select_train.sh 112 0 42 l2 full full
```

参数顺序：

```text
$1 episodes_per_dataset   默认 112
$2 gpu_id                 默认 0
$3 seed                   默认 42
$4 visual_variant         l2 | online_pca，默认 l2
$5 mode                   full | selection-only，默认 full
$6 ablation               full | wo_action | wo_adaptive_priority，默认 full
```

例如只运行 online PCA 选择：

```bash
bash personal/work2/our_v6/run_select_train.sh 112 0 42 online_pca selection-only full
```

运行 w/o Action 并训练：

```bash
bash personal/work2/our_v6/run_select_train.sh 112 0 42 l2 full wo_action
```

## tmux 兼容入口

原先的 launcher 路径仍可使用，但现在只是新 pipeline 的 tmux wrapper：

```bash
bash personal/work2/duibi/train_and_eval_scripts/launch_ours_v6.sh 112 0 42 l2 full full
```

查看运行：

```bash
tmux attach -t our_v6_l2_full_112x3_s42
```

## 三任务联合训练

默认处理：
- `coffee-button-v3_corner`
- `disassemble-v3_corner`
- `pick_place-v3_corner`

每个数据集先独立运行 causal V6 selection，然后通过现有 `merge_selected_episodes.py` 将三个 selected subsets 合成一个真正的 LeRobot merged dataset，再把该 merged dataset 交给 `lerobot-train`。不会再把三个数据集的 episode index 简单拼接后只指向第一个 dataset root。

## 主要文件

```text
personal/work2/our_v6/
├── config.py
├── core/
│   ├── planner.py             # causal V5-style region planner
│   ├── visual_embedding.py    # l2 / online_pca
│   └── action_embedding.py    # acquired-only action representation
├── experiments/
│   └── select_episodes_v6.py  # single-dataset selection CLI
└── run_select_train.sh        # end-to-end selection + merge + training
```

`adaptive_grid.py`、旧 scoring/pool helper 仍保留在目录中用于历史对照，但新的 `planner.py` 不再依赖旧 adaptive-grid 流程。
