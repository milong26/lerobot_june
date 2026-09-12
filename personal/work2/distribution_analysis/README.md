# Distribution Analysis Module

## Purpose

This module analyzes demonstration distribution differences between **Candidate Pool**, **Random Selection**, and **Ours (Configuration-Aware) Selection** subsets. It generates visualizations and statistical metrics for the paper section "Selection Behavior Analysis", proving that configuration-aware hierarchical acquisition changes the demonstration distribution.

## Generated Outputs

| Output | Description |
|--------|-------------|
| `configuration_space_comparison.pdf/png` | PCA 2D scatter plot of configuration space with three distributions |
| `region_coverage_comparison.pdf/png` | Bar chart showing which configuration regions are covered by each method |
| `region_coverage_bar.pdf/png` | Stacked bar chart of covered vs uncovered regions |
| `distribution_comparison_iclr.pdf/png` | ICLR-style 3-panel comparison figure (Candidate Pool / Random / Ours) |
| `statistics.json` | Full distribution statistics (coverage ratio, entropy, diversity metrics) |
| `statistics.csv` | Statistics in CSV format for paper tables |
| `region_coverage.csv` | Region coverage summary |
| `analysis.log` | Execution log |

## Input Files

The module requires the following input files:

1. **Candidate Pool Metadata**: `episode_initial_states.json` containing episode indices and `rand_vec` fields
2. **Visual Embeddings**: Directory with `({episode_index}).npy` files containing `phi_global` and `phi_wrist`
3. **Action Descriptors**: Directory with `({episode_index}).npy` files containing `action_descriptor`
4. **Ours Selection Subset**: JSON file with `selected_episode_indices` key
5. **Random Selection Subset**: JSON file with `selected_episode_indices` key

## Usage

### Basic Usage (auto-discovery)

```bash
python personal/work2/distribution_analysis/analyze_distribution.py \
    --task Disassemble-v3 \
    --output-dir personal/work2/distribution_analysis/results/
```

### Full Explicit Paths

```bash
python personal/work2/distribution_analysis/analyze_distribution.py \
    --task Disassemble-v3 \
    --dataset-dir /path/to/dataset_view/disassemble-v3_corner \
    --candidate-pool /path/to/dataset_view/disassemble-v3_corner/episode_initial_states.json \
    --visual-embedding-dir /path/to/shared_embeddings/disassemble-v3_corner/... \
    --action-descriptor-dir /path/to/action_descriptors/disassemble-v3_corner/... \
    --selection-ours /path/to/our_v5_112_seed42/subsets/our_v5_112_seed42.json \
    --selection-random /path/to/random_112_seed42/subsets/random_112_seed42.json \
    --output-dir personal/work2/distribution_analysis/results/
```

### Supported Tasks

- `Pick-Place-v3`
- `Disassemble-v3`
- `Coffee-Button-v3`

## Statistics Metrics

| Metric | Description |
|--------|-------------|
| `configuration_coverage_ratio` | Fraction of configuration regions covered by the subset |
| `region_entropy` | Shannon entropy of region selection distribution |
| `avg_nearest_neighbor_distance` | Average distance to nearest neighbor in configuration space |
| `mean_pairwise_configuration_distance` | Mean pairwise distance between configurations |
| `visual_diversity` | Mean pairwise distance in visual embedding space |
| `action_diversity` | Mean pairwise distance in action descriptor space |

## Dependencies

- numpy
- scikit-learn (PCA, KMeans, NearestNeighbors)
- matplotlib

## Notes

- Random seed is fixed at 42 for reproducibility
- All figure text is in English (no Chinese characters)
- Figures are saved in both PDF and PNG formats at 300 DPI
- Figure width is designed for ICLR two-column paper layout