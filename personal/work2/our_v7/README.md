# Our-V7: paper-faithful causal acquisition

`our_v7` is the ICRA-method-aligned successor to `our_v6`. It keeps the stable V6 frozen-VLM extraction, action descriptor, three-task merge, safe video split, and SmolVLA training pipeline, while changing the selection logic to match the current paper.

## What V7 changes from V6

1. Configuration normalization uses MetaWorld's predefined `_random_reset_space.low/high` bounds. Archive sample min/max is not used.
2. KMeans initialization is deterministic: the first support point is nearest the global normalized-configuration centroid; later centers are chosen by farthest-point traversal; Lloyd updates then run from those centers.
3. Initial B0 regions are deterministic: first region centroid nearest the global centroid, followed by farthest-point traversal over region centroids.
4. Full region priority is exactly `g_k + minmax(u_visual) + minmax(u_action)`. The coverage gap `g_k = 1 - n_k/N_k` is not min-max normalized and the three terms are not weighted.
5. If a selected region has not yet produced a demonstration, its centroid is the target. Otherwise target selection is configuration-only maximin over unused configurations.
6. Archive matching is restricted to unused configurations in the selected region, with a global fallback only if that region is exhausted.

## Representations

V7 intentionally reuses the current V6 representation code:

- visual frame selection and frozen SmolVLM extraction are unchanged;
- action representation is unchanged;
- `l2`: branch-wise L2 -> concatenate -> final L2. This is the main ICRA representation;
- `online_pca`: acquired-only branch-wise PCA -> branch-wise L2 -> concatenate -> final L2. This is an experimental causal variant.

V7 adds `personal/work2/our_v7/shared_raw_embeddings/`. Raw VLM features are cached there only after an episode has been acquired. The causal gate is checked before cache lookup, so a cache created by another run does not expose an unacquired episode. L2 and online-PCA runs can therefore share raw extraction work without sharing future information.

Existing V6 raw features cannot be recovered from an already-finished V6 process because V6 only kept them in memory. V7 starts persistent caching from its first run onward.

## Three MetaWorld datasets

The end-to-end script independently selects from:

- `coffee-button-v3_corner`
- `disassemble-v3_corner`
- `pick_place-v3_corner`

With budget 112, the merged training set contains 336 demonstrations. Training-time environment evaluation is disabled in V7; evaluate each task separately after training.

## Commands

Main paper variant (L2) on GPU 1:

```bash
nohup bash personal/work2/our_v7/run_select_train.sh \
  112 1 42 l2 full full \
  > our_v7_l2_gpu1.log 2>&1 &
```

Causal online-PCA variant on GPU 2:

```bash
nohup bash personal/work2/our_v7/run_select_train.sh \
  112 2 42 online_pca full full \
  > our_v7_online_pca_gpu2.log 2>&1 &
```

Selection only:

```bash
bash personal/work2/our_v7/run_select_train.sh 112 1 42 l2 selection-only full
```

Arguments:

```text
EPISODES_PER_DATASET GPU_ID SEED VISUAL_VARIANT MODE ABLATION
```

Outputs are separated by visual variant under `personal/work2/our_v7/outputs/`, while raw acquired VLM features are shared through `personal/work2/our_v7/shared_raw_embeddings/`.
