# Distribution Analysis Module

## Purpose

This module analyzes demonstration distribution differences between **Candidate Pool**, **Random Selection**, **DemInf Selection**, and **Ours (Configuration-Aware) Selection** subsets under a **fixed selection budget**. It generates visualizations and statistical metrics for the paper section "Selection Behavior Analysis", proving that configuration-aware hierarchical acquisition changes the demonstration distribution in configuration space.

## Generated Outputs

| Output | Type | Description |
|--------|------|-------------|
| `configuration_distribution_comparison.pdf/png` | **Main Figure** | 4-panel configuration space comparison: (A) Candidate Pool, (B) Random, (C) DemInf, (D) Ours |
| `statistics.json` | Quantitative | Full distribution statistics for all three methods |
| `statistics.csv` | Quantitative | Statistics in CSV format for paper tables |
| `region_coverage_comparison.pdf/png` | Auxiliary | Bar chart showing episode count per region for each method |
| `region_coverage.csv` | Auxiliary | Region-level episode counts per method |
| `analysis.log` | Log | Execution log |

## Input Files

The module requires the following input files:

1. **Candidate Pool Metadata**: `episode_initial_states.json` containing episode indices with `rand_vec` fields
2. **Visual Embeddings**: Directory with `({episode_index}).npy` files containing `phi_global` and `phi_wrist`
3. **Action Descriptors**: Directory with subdirectories containing `({episode_index}).npy` files with `action_descriptor`
4. **Ours Selection Subset**: JSON file with `selected_episode_indices` or `selected_episode_ids` key
5. **Random Selection Subset**: JSON file with `selected_episode_indices` or `selected_episode_ids` key
6. **DemInf Selection Subset** (optional): JSON file with `selected_episode_indices` or `selected_episode_ids` key

## Usage

### Full Explicit Paths (recommended)

```bash
cd /data/zhonglinye/jun/lerobot && python personal/work2/distribution_analysis/analyze_distribution.py \
    --task Disassemble-v3 \
    --dataset-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset_view/disassemble-v3_corner \
    --candidate-pool /data/zhonglinye/jun/lerobot/personal/work2/dataset_view/disassemble-v3_corner/episode_initial_states.json \
    --visual-embedding-dir /data/zhonglinye/jun/lerobot/personal/work2/shared_embeddings/pick_place_disassemble-v3_corner/smolvlm2-500m_last-hidden-tokenmean_global-first5_wrist-20to70_temporal-mean_pca32_v1 \
    --action-descriptor-dir /data/zhonglinye/jun/lerobot/personal/work2/action_descriptors/pick_place_disassemble-v3_corner \
    --selection-ours /data/zhonglinye/jun/lerobot/personal/work2/differentvlm/experiments/tinyvla_s_disassemble-v3_corner/results/v5_selection_output/subsets/our_v5_112_seed42.json \
    --selection-random /data/zhonglinye/jun/lerobot/personal/work2/differentvlm/minivla/experiments/random_200_seed42_disassemblev3corner_minivla_random/subsets/random_200_seed42.json \
    --selection-deminf /data/zhonglinye/jun/lerobot/personal/work2/deminf_results/disassemble-v3_corner/subsets/deminf_112_seed42.json \
    --selection-budget 112 \
    --output-dir personal/work2/distribution_analysis/results/
```

### Supported Tasks

- `Disassemble-v3`
- `Pick-Place-v3`
- `Coffee-Button-v3`

## Statistics Metrics

| Metric | Description |
|--------|-------------|
| `selection_budget` | Fixed number of episodes for all methods |
| `coverage_ratio` | Fraction of configuration regions covered |
| `entropy` | Shannon entropy of region selection distribution |
| `nearest_neighbor_distance` | Average distance to nearest neighbor in configuration space |
| `pairwise_configuration_distance` | Mean pairwise distance between configurations |
| `visual_diversity` | Mean pairwise distance in visual embedding space |
| `action_diversity` | Mean pairwise distance in action descriptor space |
| `js_divergence` | Jensen-Shannon divergence from candidate pool distribution |
| `emd_distance` | Earth Mover Distance from candidate pool distribution |

## Key Design Decisions

1. **Fixed Budget**: All selection methods use the same episode count (`--selection-budget`, default 112)
2. **Single PCA Model**: PCA is fit once on the full candidate pool; all methods share the same 2D projection
3. **StandardScaler**: rand_vec features are standardized before PCA
4. **DemInf Optional**: If `--selection-deminf` is not provided, DemInf analysis is skipped with a warning
5. **Region Coverage as Auxiliary**: KMeans region划分 is retained for quantitative statistics but not used as the main visualization basis

## Dependencies

- numpy
- scipy (jensenshannon, wasserstein_distance)
- scikit-learn (StandardScaler, PCA, NearestNeighbors)
- matplotlib

## Notes

- Random seed is fixed at 42 for reproducibility
- All figure text is in English (no Chinese characters)
- Figures are saved in both PDF and PNG formats at 300 DPI
- Figure layout is designed for ICLR two-column paper format