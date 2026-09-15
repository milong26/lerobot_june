# RoboMME three-task training pipeline

This pipeline selects the same budget independently from all three local RoboMME datasets and then jointly trains one policy on the physically merged subset.

Datasets:
- `MoveCube_easy`
- `PatternLock_medium`
- `RouteStick_hard`

Default budget is 100 episodes per task, therefore the merged training dataset contains 300 episodes.

## One-command usage

```bash
cd /data/zhonglinye/jun/lerobot
bash personal/work2/robomme_pipeline/run_robomme_all.sh METHOD MODEL 100 GPU_ID 42 l2 full
```

`MODEL` accepts:
- `smovla` (compatibility spelling; resolved internally to the repository's `smolvla` policy)
- `smolvla`
- `minivla`

`METHOD` accepts:
- `random`
- `grid_uniform` or `grid-uniform`
- `fps`
- `deminf`
- `our_v6` or `ours` or `v6`

Examples:

```bash
# causal V6 + SmolVLA, 100 episodes from each task
bash personal/work2/robomme_pipeline/run_robomme_all.sh our_v6 smovla 100 0 42 l2 full

# random + MiniVLA
bash personal/work2/robomme_pipeline/run_robomme_all.sh random minivla 100 0 42 l2 full

# online acquired-only PCA variant of V6
bash personal/work2/robomme_pipeline/run_robomme_all.sh our_v6 minivla 100 0 42 online_pca full

# selection only, no merge/training
bash personal/work2/robomme_pipeline/run_robomme_all.sh grid-uniform smovla 100 0 42 l2 selection-only
```

The Python entry point exposes additional options:

```bash
python personal/work2/robomme_pipeline/run_robomme_all.py --help
```

Important options include `--train-steps`, `--batch-size`, `--dataset-base`, `--output-root`, `--minivla-init`, and `--force-merge`.

## Selection semantics

`random` samples episode ids with the requested seed.

`grid_uniform` uses only `episode_initial_states.json -> initial_configuration`. Numeric task-configuration leaves are flattened, constant dimensions are removed, remaining dimensions are normalized per dimension to `[0,1]`, and deterministic maximin coverage is used to spread selected configurations.

`fps` is a post-hoc full-pool visual baseline. It extracts frozen VLM features for all candidates, applies branch-wise L2 normalization, and runs greedy visual farthest-point sampling.

`deminf` reuses the repository DemInf implementation but uses `observation.state`, which is the state field in the local RoboMME datasets.

`our_v6` reuses the causal V6 algorithm through a RoboMME metadata adapter. Before acquisition it may use only the task `initial_configuration`; image/action content is revealed after an episode is acquired. `l2` and acquired-only `online_pca` visual variants are both supported.

## Model dispatch

SmolVLA uses `lerobot/smolvla_base`, frozen vision encoder, expert-only training, LR `1e-4`, default 12,000 steps and batch size 64.

MiniVLA uses `minivla_wrist`, explicit RoboMME image keys `observation.images.image` and `observation.images.wrist_image`, `extra_action_tokenizer` for the native 8-D RoboMME action, official optimizer defaults, default 50,000 steps and batch size 2. Its default initialization is `backbone_only` from `Stanford-ILIAD/minivla-libero90-prismatic`; the model must already be available in the Hugging Face cache, or use `--minivla-init none`.

Training-time environment evaluation is disabled (`env_eval_freq=0`) in this pipeline because one joint policy is trained across three different RoboMME tasks. Evaluation should be run task-by-task after training.

## Output

Runs are stored under:

```text
personal/work2/robomme_runs/robomme_<method>_<model>_<N>x3_seed<seed>/
```

The directory contains per-task selection JSON files, the merged selection manifest, a merged LeRobot dataset, model checkpoints, and `run_summary.json` after successful training.
