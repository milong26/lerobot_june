# MetaWorld three-task VLA pipeline

This directory provides the MetaWorld counterpart of `personal/work2/robomme_pipeline`.
One command performs the same sequence for all three MetaWorld datasets:

1. Select `N` episodes independently from each source dataset using one method.
2. Validate the three subsets.
3. Build one manifest containing `3N` selected demonstrations.
4. Physically merge the selected episodes into one LeRobot dataset.
5. Jointly train one selected VLA on the merged three-task dataset.

## Fixed datasets

- `coffee-button-v3_corner`
- `disassemble-v3_corner`
- `pick_place-v3_corner`

The default dataset base directory is `personal/work2/dataset_view`.

## Selection methods

- `our_v6`: causal V6 acquisition. Supports `l2` and `online_pca` visual variants and the existing `full`, `wo_action`, and `wo_adaptive_priority` ablations.
- `random`: seeded random episode sampling.
- `grid_uniform`: existing factorized `rand_vec` configuration-space coverage. Aliases `grid-uniform`, `gird_uniform`, and `gird-uniform` are accepted.
- `deminf`: existing state-based DemInf implementation using `observation.environment_state` for MetaWorld.
- `fps`: existing visual farthest-point sampling baseline. Missing frozen-VLM embeddings are generated/cached through `embedding_utils/ensure_embeddings.py` before FPS selection.

Each method selects episodes separately inside each task. Episode indices from different dataset roots are never directly treated as indices of one dataset; the selected subsets are physically merged first.

## Models

- `smovla` or `smolvla`: maps to the repository's SmolVLA implementation and `lerobot/smolvla_base`.
- `minivla`: uses `minivla_wrist`, primary key `observation.images.top`, wrist key `observation.images.wrist`, non-VQ `extra_action_tokenizer`, and the existing official optimizer defaults. By default the official MiniVLA checkpoint is used in `backbone_only` mode.
- `tinyvla_s`: TinyVLA-S.
- `tinyvla_b`: TinyVLA-B.
- `tinyvla` and `tinyvla-s` are aliases of `tinyvla_s`; `tinyvla-b` is an alias of `tinyvla_b`.

All training commands explicitly use `--env.use_self_mw=true`, matching the project's self-collected MetaWorld format. Training-time environment evaluation is disabled (`env_eval_freq=0`) because the merged training dataset contains three task instructions while a single `env.task` can name only one MetaWorld environment. Evaluate the three tasks after training with the existing evaluation task loop.

## Recommended command

```bash
bash personal/work2/metaworld_pipeline/run_metaworld_all.sh \
  our_v6 smovla 112 0 42 l2 full full
```

Arguments are:

```text
METHOD MODEL EPISODES_PER_TASK GPU_ID SEED VISUAL_VARIANT MODE OUR_V6_ABLATION
```

`OUR_V6_ABLATION` defaults to `full` and is ignored by non-`our_v6` methods.

For example:

```bash
# Our-V6 + SmolVLA, 112 demonstrations per task, 336 total
bash personal/work2/metaworld_pipeline/run_metaworld_all.sh \
  our_v6 smovla 112 0 42 l2 full full

# Our-V6 acquired-only online PCA
bash personal/work2/metaworld_pipeline/run_metaworld_all.sh \
  our_v6 smovla 112 0 42 online_pca full full

# Our-V6 w/o Action ablation
bash personal/work2/metaworld_pipeline/run_metaworld_all.sh \
  our_v6 smovla 112 0 42 l2 full wo_action

# Random + MiniVLA, 100 demonstrations per task
bash personal/work2/metaworld_pipeline/run_metaworld_all.sh \
  random minivla 100 0 42 l2 full

# Grid-Uniform + TinyVLA-S
bash personal/work2/metaworld_pipeline/run_metaworld_all.sh \
  grid_uniform tinyvla_s 112 0 42 l2 full

# The common typo gird_uniform is also accepted
bash personal/work2/metaworld_pipeline/run_metaworld_all.sh \
  gird_uniform smolvla 112 0 42 l2 full

# DemInf + SmolVLA
bash personal/work2/metaworld_pipeline/run_metaworld_all.sh \
  deminf smolvla 112 0 42 l2 full

# FPS selection only
bash personal/work2/metaworld_pipeline/run_metaworld_all.sh \
  fps smolvla 112 0 42 l2 selection-only
```

## Python entry point

The Python entry point exposes additional controls:

```bash
python personal/work2/metaworld_pipeline/run_metaworld_all.py \
  --method our_v6 \
  --model minivla \
  --episodes-per-task 112 \
  --gpu-id 0 \
  --seed 42 \
  --visual-variant online_pca \
  --our-v6-ablation full
```

Useful optional flags:

```text
--selection-only
--force-merge
--train-steps N
--batch-size N
--num-workers N
--dataset-base PATH
--output-root PATH
--minivla-init backbone_only|none
--minivla-pretrained MODEL_OR_PT_PATH
```

## Defaults

- SmolVLA: 12,000 steps, batch size 64, LR `1e-4`.
- MiniVLA: 50,000 steps, batch size 2, LR `2e-5`, constant scheduler.
- TinyVLA-S/B: 50,000 steps, batch size 4, peak LR `2e-4`, 0.5% warmup, cosine decay to `2.5e-6`.

All model defaults can be overridden by `--train-steps`, `--batch-size`, and `--num-workers` where applicable.

## Outputs

The default output root is:

```text
personal/work2/metaworld_runs/
```

Example:

```text
metaworld_our_v6_smolvla_112x3_seed42_l2_full/
├── selection/
│   ├── coffee-button-v3_corner/
│   ├── disassemble-v3_corner/
│   └── pick_place-v3_corner/
├── subsets/
├── merged_dataset/
├── train_smolvla/
└── run_summary.json
```

The merged dataset is generated with the existing `merge_selected_episodes.py`, which uses LeRobot's dataset split/merge helpers and recomputes statistics before training.
