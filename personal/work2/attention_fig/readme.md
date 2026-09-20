运行代码，检查注意力图

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=1

python personal/work2/attention_fig/plot_attention.py \
  --device cuda \
  --mode probe \
  --seed 10042 \
  --query-mode mean_suffix \
  --layers 0,3,7,11 \
  --average-heads \
  2>&1 | tee personal/work2/attention_fig/attention_probe_all.log

修改了代码以后执行新的
cd /data/zhonglinye/jun/lerobot
conda activate lb_server
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
nohup python personal/work2/attention_fig/plot_attention.py \
  --mode inference_trace \
  --device cuda \
  --seed 10042 \
  --noise-seed 10042 \
  --query-mode mean_suffix \
  --layers 3,7,11 \
  --trace-heatmap-steps first,mid,last \
  --sanity-check \
  --output-dir personal/work2/attention_fig/result_inference_trace \
  > personal/work2/attention_fig/inference_trace.log 2>&1 &

终于改完代码执行成功了，怎么看结果
默认代码只执行了
Models: ['random_corner_16k', 'uniform_corner_16k', 'ours_corner_16k']
这三个模型

之前已经执行过，在最后的sanity_check报错，所以
nohup python personal/work2/attention_fig/plot_attention.py \
  --mode inference_trace \
  --device cuda \
  --seed 10042 \
  --noise-seed 10042 \
  --query-mode mean_suffix \
  --layers 3,7,11 \
  --trace-heatmap-steps first,mid,last \
  --sanity-check \
  --load-from-cache \
  --output-dir personal/work2/attention_fig/result_inference_trace \
  > personal/work2/attention_fig/inference_trace_continue.log 2>&1 &


ok这边暂时不看了。

---

# 12k Corner 七模型分析

本章节针对 12k checkpoint 的 7 个 corner 模型（ours_v1、ours_v2、ours_v3、ours_v4、random、uniform、zero）进行 post-hoc attention 诊断分析。

**固定 10 个 seed**：10042,10043,10044,10045,10046,10047,10048,10049,10050,10051

**输出目录**：
- probe/rollout: `personal/work2/attention_fig/result_12k_corner`
- inference_trace: `personal/work2/attention_fig/result_inference_trace_12k_corner`
- 汇总分析: `personal/work2/attention_fig/analysis_12k_corner`

## 1. Probe 模式（initial attention）

```bash
cd /data/zhonglinye/jun/lerobot
conda activate lb_server
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python personal/work2/attention_fig/plot_attention.py \
  --device cuda \
  --model-set corner_12k \
  --mode probe \
  --seeds 10042,10043,10044,10045,10046,10047,10048,10049,10050,10051 \
  --query-mode mean_suffix \
  --layers 0,3,7,11 \
  --average-heads \
  2>&1 | tee personal/work2/attention_fig/probe_12k_corner.log
```

## 2. Rollout 模式（phase attention）

```bash
cd /data/zhonglinye/jun/lerobot
conda activate lb_server
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

nohup python personal/work2/attention_fig/plot_attention.py \
  --device cuda \
  --model-set corner_12k \
  --mode rollout \
  --seeds 10042,10043,10044,10045,10046,10047,10048,10049,10050,10051 \
  --query-mode mean_suffix \
  --layers 0,3,7,11 \
  --average-heads \
  --max-steps 500 \
  > personal/work2/attention_fig/rollout_12k_corner.log 2>&1 &
```

**注意**：rollout 不同模型在 post-grasp 和 pre-place 阶段的 observation 来自各自 policy trajectory，
因此这些 phase 结果属于行为诊断而不是严格 same-observation 因果比较。
严格 same-observation 比较以 inference_trace 的 initial 输入为准。

## 3. Inference Trace 模式（最严格公平比较）

每个 seed 的 7 模型必须从相同 initial observation 和相同 initial noise 开始。

```bash
cd /data/zhonglinye/jun/lerobot
conda activate lb_server
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

nohup python personal/work2/attention_fig/plot_attention.py \
  --device cuda \
  --model-set corner_12k \
  --mode inference_trace \
  --seeds 10042,10043,10044,10045,10046,10047,10048,10049,10050,10051 \
  --query-mode mean_suffix \
  --layers 3,7,11 \
  --average-heads \
  --trace-heatmap-steps first,mid,last \
  --sanity-check \
  > personal/work2/attention_fig/inference_trace_12k_corner.log 2>&1 &
```

如果中途中断，可从 cache 继续（逐 seed 读取对应 model_name/seed_/trace.pt）：

```bash
nohup python personal/work2/attention_fig/plot_attention.py \
  --device cuda \
  --model-set corner_12k \
  --mode inference_trace \
  --seeds 10042,10043,10044,10045,10046,10047,10048,10049,10050,10051 \
  --query-mode mean_suffix \
  --layers 3,7,11 \
  --average-heads \
  --trace-heatmap-steps first,mid,last \
  --sanity-check \
  --load-from-cache \
  > personal/work2/attention_fig/inference_trace_12k_corner_continue.log 2>&1 &
```

## 4. 汇总分析（离线统计，不运行模型）

```bash
cd /data/zhonglinye/jun/lerobot
conda activate lb_server

python personal/work2/attention_fig/aggregate_12k_corner.py \
  2>&1 | tee personal/work2/attention_fig/aggregate_12k_corner.log
```

汇总输出文件（位于 `personal/work2/attention_fig/analysis_12k_corner`）：
- `attention_aggregate.csv` — 跨 seed 的 attention mass 聚合（mean/std/count）
- `phase_reach_aggregate.csv` — 各 phase reached 比例
- `heatmap_statistics.csv` — attention 空间 entropy、最大值、中心坐标统计
- `inference_trace_aggregate.csv` — x_t_norm/v_t_norm/suffix_hidden_norm 聚合
- `pairwise_action_divergence_aggregate.csv` — 21 组模型对的 v_t_l2/v_t_cosine/x_t_l2/hidden_cosine 聚合
- `velocity_statistics.csv` — 跨 seed 的 v_t_norm 统计
- `model_performance_metadata.csv` — 7 模型名称/method/路径/pc_success/pc_grasp_success
- `analysis_12k_corner_summary.md` — 自动生成的分析报告

## 5. 分析原则

- **Layer 3/7/11 的 expert_cross** 作为主要可比较对象，优先分析 layer 11 深层视觉/action 相关 attention
- **Layer 0** 属于 joint_self，keyspace 与 expert_cross 不同，允许单独保存和观察，但**禁止**把 layer 0 绝对 attention mass 与 layer 3/7/11 放在同一个纵向数值排序中
- 重点观察 ours_v1->v2->v3->v4 以及 random/uniform/zero 在图像内部关注区域的稳定变化
- Global camera 重点观察 object/goal/robot/gripper/background 区域
- Wrist camera 重点观察 gripper/object/gripper-object 关系
- 报告必须明确区分 correlation 和 causality
- 如果不同 seed 结果不一致必须明确写不稳定
- 如果只有单个 seed 出现差异必须明确标记 single-seed artifact

## 6. 重要声明

**本分析仅用于解释性研究**：
- 仅用于解释 random、uniform、zero、ours_v1、ours_v2、ours_v3、ours_v4 之间的 policy attention 和 action trajectory 差异
- **不用于**修改 selection 策略
- **不用于**优化 success 率
- **不用于** our_v5 设计闭环
- eval_task_success 和 eval_grasp_success 仅作为 metadata 写入 model_performance_metadata.csv 和最终报告，**不参与**任何模型选择、参数调整或分析输入计算

# 40分钟快速 Surrogate Eval 验证

新版脚本不再只使用 7 个静态 12k 模型。它会在本机运行时扫描：

- `personal/work2/eval_model/**/.tasks.jsonl`
- 对应的非空 `eval_episode_results.json`

只纳入至少 100 个历史 eval episodes 的 checkpoint，并直接从 episode-level 文件恢复历史 success rate。旧 `summary.csv` 中 `n_episodes=0` 的错误条目不会作为历史成功率依据。

新版同时把 `lerobot-eval` 的 stdout/stderr 实时显示在终端，并每 10 秒输出一次 heartbeat，所以不会再出现“GPU 在跑但终端完全没输出”的情况。

## 两块 GPU 各跑一半

两个终端使用同一个 `--run-name`。脚本会自动把全部发现到的 checkpoint 均分成两个 shard。

终端 1：

```bash
python -u personal/work2/attention_fig/quick_surrogate_validate.py \
  --gpu-id 0 \
  --shard 0/2 \
  --run-name quick40 \
  2>&1 | tee personal/work2/attention_fig/quick40_gpu0.log
```

终端 2：

```bash
python -u personal/work2/attention_fig/quick_surrogate_validate.py \
  --gpu-id 1 \
  --shard 1/2 \
  --run-name quick40 \
  2>&1 | tee personal/work2/attention_fig/quick40_gpu1.log
```

默认参数：
- quick eval：5 episodes/checkpoint；
- historical full eval：至少 100 episodes 才进入模型池；
- 每个 checkpoint 最多 240 秒；
- 每个 shard 总预算 38 分钟；
- fixed start seed：1000；
- 相机根据模型路径自动识别 corner / corner2 / corner3。

每个 shard 输出：

`personal/work2/attention_fig/quick_surrogate_eval/quick40/shard_0_of_2/`

和

`personal/work2/attention_fig/quick_surrogate_eval/quick40/shard_1_of_2/`

当两个 shard 都完成且发现到的是同一个模型池时，后完成的进程会自动生成：

- `COMBINED_results.csv`
- `COMBINED_correlations.json`
- `COMBINED_CONCLUSION.md`

位置：

`personal/work2/attention_fig/quick_surrogate_eval/quick40/`

最终重点看 `COMBINED_CONCLUSION.md` 中 quick success 与历史 full success 的 Spearman 相关性。
