# Our-V7: paper-faithful causal acquisition

`our_v7` is the ICRA-method-aligned successor to `our_v6`. It keeps the stable V6 frozen-VLM extraction, action descriptor, three-task merge, safe video split, and SmolVLA training pipeline, while changing the selection logic to match the current paper. The same launcher now supports both MetaWorld and RoboMME.

## What V7 changes from V6

1. Configuration normalization uses task reset-space bounds rather than raw whole-vector L2 normalization. MetaWorld uses `_random_reset_space.low/high`. RoboMME uses the task-relevant admissible support encoded in its reset metadata because the local JSON does not provide a separate analytic Bounds object.
2. KMeans initialization is deterministic: the first support point is nearest the global normalized-configuration centroid; later centers are chosen by farthest-point traversal; Lloyd updates then run from those centers.
3. Initial B0 regions are deterministic: first region centroid nearest the global centroid, followed by farthest-point traversal over region centroids.
4. Full region priority is exactly `g_k + minmax(u_visual) + minmax(u_action)`. The coverage gap `g_k = 1 - n_k/N_k` is not min-max normalized and the three terms are not weighted.
5. If a selected region has not yet produced a demonstration, its centroid is the target. Otherwise target selection is configuration-only maximin over unused configurations.
6. Archive matching is restricted to unused configurations in the selected region, with a global fallback only if that region is exhausted.

## Representations

V7 intentionally reuses the current V6 representation code:

- visual frame selection and frozen SmolVLM extraction are unchanged;
- action representation is unchanged and works with both MetaWorld 4-D and RoboMME 8-D actions;
- `l2`: branch-wise L2 -> concatenate -> final L2. This is the main ICRA representation;
- `online_pca`: acquired-only branch-wise PCA -> branch-wise L2 -> concatenate -> final L2. This is an experimental causal variant.

V7 adds `personal/work2/our_v7/shared_raw_embeddings/`. Raw VLM features are cached there only after an episode has been acquired. The causal gate is checked before cache lookup, so a cache created by another run does not expose an unacquired episode. L2 and online-PCA runs can therefore share raw extraction work without sharing future information.

Existing V6 raw features cannot be recovered from an already-finished V6 process because V6 only kept them in memory. V7 starts persistent caching from its first run onward.

## Benchmarks

### MetaWorld

Datasets:

- `coffee-button-v3_corner`
- `disassemble-v3_corner`
- `pick_place-v3_corner`

Configuration metadata comes from `episode_initial_states.json -> rand_vec`. MetaWorld's predefined `_random_reset_space.low/high` is used for per-dimension normalization.

### RoboMME

Datasets:

- `MoveCube_easy`
- `PatternLock_medium`
- `RouteStick_hard`

RoboMME does **not** use `rand_vec`. V7 reads `episode_initial_states.json -> episodes[*].initial_configuration` and flattens numeric leaves only from:

- `movable_objects`
- `randomized_targets`
- `articulations`
- `task_config`

The full low-level `scene_state` is explicitly excluded from configuration-space selection. Constant dimensions are removed by the existing RoboMME configuration adapter. The local dataset JSON exposes the admissible reset support but not a separate analytic Bounds object, so V7 normalizes each remaining dimension with the low/high values of that globally visible task-relevant support. This is causal because only reset metadata is used before acquisition.

RoboMME training keeps the existing working setup: native `observation.images.image` and `observation.images.wrist_image`, 8-D joint-angle action, `env.type=robomme`, SmolVLA LR `1e-4`, 12,000 steps, batch size 64. Training-time evaluation is disabled.

For either benchmark, each task is selected independently and the three selected subsets are physically merged before one joint SmolVLA is trained. Evaluation is run task-by-task later.

## Budget

The current ICRA experiment text specifies **100 demonstrations per task** for simulation. Therefore the paper-main setting is 100 per task, i.e. 300 demonstrations in each benchmark's joint training set. A 112-per-task run is still supported for compatibility/diagnostics but does not exactly match the current paper budget.

## Commands

Arguments:

```text
EPISODES_PER_DATASET GPU_ID SEED VISUAL_VARIANT MODE ABLATION BENCHMARK
```

`BENCHMARK` is `metaworld` or `robomme`. It defaults to `metaworld`, so the old six-argument V7 command is unchanged.

### Current paper main variant: MetaWorld, L2

```bash
nohup bash personal/work2/our_v7/run_select_train.sh \
  100 1 42 l2 full full metaworld \
  > our_v7_metaworld_l2_gpu1.log 2>&1 &
```

### Current paper main variant: RoboMME, L2

```bash
nohup bash personal/work2/our_v7/run_select_train.sh \
  100 2 42 l2 full full robomme \
  > our_v7_robomme_l2_gpu2.log 2>&1 &
```

### Causal online-PCA variants

```bash
nohup bash personal/work2/our_v7/run_select_train.sh \
  100 1 42 online_pca full full metaworld \
  > our_v7_metaworld_online_pca_gpu1.log 2>&1 &

nohup bash personal/work2/our_v7/run_select_train.sh \
  100 2 42 online_pca full full robomme \
  > our_v7_robomme_online_pca_gpu2.log 2>&1 &
```

### Selection only

```bash
bash personal/work2/our_v7/run_select_train.sh 100 1 42 l2 selection-only full metaworld
bash personal/work2/our_v7/run_select_train.sh 100 2 42 l2 selection-only full robomme
```

The original command remains valid and is interpreted as MetaWorld:

```bash
bash personal/work2/our_v7/run_select_train.sh 112 1 42 l2 full full
```

Outputs are benchmark/variant separated under `personal/work2/our_v7/outputs/`, while raw acquired VLM features are shared safely through `personal/work2/our_v7/shared_raw_embeddings/` using dataset-specific cache signatures.
