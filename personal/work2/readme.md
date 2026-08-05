# MetaWorld 单任务微调 · 示范数据配方研究

## 目标

研究"单任务微调时，什么样的示范数据分布能让成功率最高"。

## 快速开始

```bash
cd /data/zhonglinye/jun/lerobot

# 1. 生成固定评测集（一次性，所有配方共用）
python personal/work2/make_eval_set.py --n-states 200 --seed 42

# 2. 生成某个配方的数据集
python personal/work2/generate_dataset.py --task pick-place-v3 --strategy uniform --num-episodes 50 --output-dir personal/work2/generated/uniform_50 --repo-id test/uniform_50

# 3. 训练
lerobot-train --dataset.root=personal/work2/generated/uniform_50 --output_dir=outputs/exp_uniform_50 --steps=20000 --remove_features='["observation.environment_state"]'

# 4. 评测
python personal/work2/run_eval_with_states.py --eval-set personal/work2/fixed_eval_set.json --out-csv personal/work2/results/uniform_50.csv

# 5. 分析
python personal/work2/analyze_results.py --results-dir personal/work2/results/
```

## 目录结构

```
personal/work2/
  SPEC.md                        技术方案文档
  做完就打勾.md                   实施检查清单
  readme.md                      本文件
  mw_common/                     共享工具包
    __init__.py
    obs_utils.py                 39维观测解析
    state_injection.py           精确指定初始状态
    task_ranges.py               任务数值范围
  sampling_strategies.py         数据配方采样器
  generate_dataset.py            数据生成（重构版）
  collect_metaworld_dataset.py   原版数据采集（保留参考）
  make_eval_set.py               生成固定评测集
  run_eval_with_states.py        用固定评测集跑 eval
  run_experiment.py              编排: 生成 → 训练 → 评测
  analyze_results.py             统计分析
  view_obj_poses.py              查看数据集初始状态
  test_state_injection_smoke.py  状态注入 smoke test
  fixed_eval_set.json            固定评测集产物
  generated/                     各配方生成的数据集
  results/                       eval CSV + 汇总图表
  logs/                          实验日志
```

## 各脚本用途

| 脚本 | 用途 |
|---|---|
| `generate_dataset.py` | 用指定采样策略生成示范数据集 |
| `make_eval_set.py` | 生成固定评测集（所有配方共用） |
| `run_eval_with_states.py` | 用固定评测集评测策略，记录逐局结果 |
| `run_experiment.py` | 编排完整实验流程 |
| `analyze_results.py` | 统计分析各配方成功率 |
| `view_obj_poses.py` | 查看数据集的初始物体/目标位置 |
| `test_state_injection_smoke.py` | 验证状态注入机制是否正常工作 |

## 可用采样策略

| 策略 | 说明 |
|---|---|
| `uniform` | 均匀随机采样（基线） |
| `grid` | 规则网格采样 |
| `boundary` | 偏向边界采样 |
| `distance_stratified` | 按距离分层采样 |