# Dataset Embedding Analysis Report

## Summary

- Total episodes: 299
- Global dimension: 32
- Wrist dimension: 32
- Combined dimension: 64

## Embedding Quality

- Valid: True
- Exact duplicate groups (global): 0
- Exact duplicate groups (wrist): 0
- Near duplicates (global): 0
- Near duplicates (wrist): 0
- Zero norm (global): 0
- Zero norm (wrist): 0

## Table 1: Representation Quality

| Representation | Spearman XY | Permutation p | R2 x | R2 y | NN overlap@10 | Effective Rank |
|---------------|-------------|---------------|------|------|---------------|----------------|
| global | 0.0552 | 9.99e-04 | 0.0071 | -0.0109 | 0.0375 | 1.14 |
| wrist | 0.1340 | 9.99e-04 | -0.0006 | 0.0019 | 0.0542 | 12.12 |
| combined | 0.0511 | 9.99e-04 | 0.3467 | 0.3115 | 0.0481 | 8.74 |

## Table 2: Subset Coverage Comparison

| Method | Global Cover | Wrist Cover | Combined Cover | Combined Max Radius | Combined Redundancy | Fixed SIC | Fixed SIC BTR | Global BTR | Wrist BTR | Combined BTR |
|--------|-------------|-------------|----------------|---------------------|---------------------|-----------|---------------|------------|-----------|--------------|
| Random | 0.0115 | 0.0072 | 0.7866 | 1.1274 | 0.9375 | 0.8853 | 0.3230 | 0.7410 | 0.6060 | 0.9430 |
| Uniform | 0.0117 | 0.0073 | 0.8378 | 1.1953 | 0.9196 | 0.8841 | 0.0410 | 0.2400 | 0.1810 | 0.0000 |
| Ours | 0.0121 | 0.0072 | 0.9397 | 1.5671 | 0.9464 | 0.8191 | 0.0000 | 0.0050 | 0.7820 | 0.0000 |

## Ours vs Uniform Comparison

- Episode overlap count: 39
- Episode overlap ratio: 0.3482
- Conclusion: Insufficient evidence that Ours mainly reproduces Uniform initial-position coverage.

### Evidence Summary

- workspace_physical_unselected_mean_nearest_relative_delta: 1.5862
- workspace_physical_unselected_p95_relative_delta: 1.2635
- workspace_physical_unselected_max_radius_relative_delta: 2.7799
- global_coverage_unselected_mean_nearest_distance_relative_delta: 0.0271
- global_coverage_unselected_median_nearest_distance_relative_delta: -0.0142
- global_coverage_unselected_p90_nearest_distance_relative_delta: 0.0289
- global_coverage_unselected_p95_nearest_distance_relative_delta: 0.0332
- global_coverage_unselected_max_nearest_distance_relative_delta: 0.6039
- wrist_coverage_unselected_mean_nearest_distance_relative_delta: -0.0176
- wrist_coverage_unselected_median_nearest_distance_relative_delta: -0.0078
- wrist_coverage_unselected_p90_nearest_distance_relative_delta: -0.0282
- wrist_coverage_unselected_p95_nearest_distance_relative_delta: 0.0026
- wrist_coverage_unselected_max_nearest_distance_relative_delta: 0.0658
- combined_coverage_unselected_mean_nearest_distance_relative_delta: 0.1216
- combined_coverage_unselected_median_nearest_distance_relative_delta: 0.0740
- combined_coverage_unselected_p90_nearest_distance_relative_delta: 0.2625
- combined_coverage_unselected_p95_nearest_distance_relative_delta: 0.2841
- combined_coverage_unselected_max_nearest_distance_relative_delta: 0.3110
- global_redundancy_mean_nearest_relative_delta: 0.1465
- global_redundancy_median_nearest_relative_delta: 0.1008
- global_redundancy_p10_nearest_relative_delta: 0.5811
- global_redundancy_p50_nearest_relative_delta: 0.1008
- global_redundancy_p90_nearest_relative_delta: -0.0214
- global_redundancy_redundancy_fraction_relative_delta: -0.3889
- wrist_redundancy_mean_nearest_relative_delta: 0.0416
- wrist_redundancy_median_nearest_relative_delta: 0.0395
- wrist_redundancy_p10_nearest_relative_delta: 0.0318
- wrist_redundancy_p50_nearest_relative_delta: 0.0395
- wrist_redundancy_p90_nearest_relative_delta: 0.0261
- wrist_redundancy_redundancy_fraction_relative_delta: -0.0396
- combined_redundancy_mean_nearest_relative_delta: 0.0720
- combined_redundancy_median_nearest_relative_delta: 0.0943
- combined_redundancy_p10_nearest_relative_delta: 0.2111
- combined_redundancy_p50_nearest_relative_delta: 0.0943
- combined_redundancy_p90_nearest_relative_delta: -0.0314
- combined_redundancy_redundancy_fraction_relative_delta: 0.0291
- fixed_sic_delta: -0.0651
- fixed_sic_relative_delta: -0.0736
- note: No calibrated equivalence test was performed; inspect the multi-metric deltas above. Relative deltas are computed as (ours - uniform) / max(|uniform|, eps).

### Workspace Coverage Delta (Ours - Uniform)

| Metric | Delta |
|--------|-------|
| physical_unselected_mean_nearest | 0.0040 |
| physical_unselected_p95 | 0.0092 |
| physical_unselected_max_radius | 0.0239 |
| grid_7x4_ratio | -0.0714 |
| grid_14x8_ratio | -0.2679 |

### Global Coverage Delta (Ours - Uniform)

| Metric | Delta |
|--------|-------|
| unselected_mean_nearest_distance | 0.0003 |
| unselected_median_nearest_distance | -0.0002 |
| unselected_p90_nearest_distance | 0.0004 |
| unselected_p95_nearest_distance | 0.0005 |
| unselected_max_nearest_distance | 0.0126 |

### Wrist Coverage Delta (Ours - Uniform)

| Metric | Delta |
|--------|-------|
| unselected_mean_nearest_distance | -0.0001 |
| unselected_median_nearest_distance | -0.0001 |
| unselected_p90_nearest_distance | -0.0003 |
| unselected_p95_nearest_distance | 0.0000 |
| unselected_max_nearest_distance | 0.0008 |

### Combined Coverage Delta (Ours - Uniform)

| Metric | Delta |
|--------|-------|
| unselected_mean_nearest_distance | 0.1019 |
| unselected_median_nearest_distance | 0.0612 |
| unselected_p90_nearest_distance | 0.2770 |
| unselected_p95_nearest_distance | 0.3092 |
| unselected_max_nearest_distance | 0.3717 |

### Global Redundancy Delta (Ours - Uniform)

| Metric | Delta |
|--------|-------|
| mean_nearest | 0.0017 |
| median_nearest | 0.0012 |
| p10_nearest | 0.0041 |
| p50_nearest | 0.0012 |
| p90_nearest | -0.0003 |
| redundancy_fraction | -0.2500 |

### Wrist Redundancy Delta (Ours - Uniform)

| Metric | Delta |
|--------|-------|
| mean_nearest | 0.0003 |
| median_nearest | 0.0003 |
| p10_nearest | 0.0002 |
| p50_nearest | 0.0003 |
| p90_nearest | 0.0002 |
| redundancy_fraction | -0.0357 |

### Combined Redundancy Delta (Ours - Uniform)

| Metric | Delta |
|--------|-------|
| mean_nearest | 0.0561 |
| median_nearest | 0.0714 |
| p10_nearest | 0.1227 |
| p50_nearest | 0.0714 |
| p90_nearest | -0.0314 |
| redundancy_fraction | 0.0268 |

- Fixed SIC Delta (Ours - Uniform): -0.0651

## Hypothesis Evaluation

### H1: Embedding 是否具有可区分性？

- Status: **SUPPORTED**
- global: probe_x_significant=True, probe_y_significant=False, neighbor_significant=True, validity=True
- wrist: probe_x_significant=True, probe_y_significant=True, neighbor_significant=True, validity=True
- combined: probe_x_significant=True, probe_y_significant=True, neighbor_significant=True, validity=True
- Effective rank ratio: 0.0355

### H2: Embedding distance 是否具有 task-state geometry？

- Status: **SUPPORTED**
- global: spearman_significant=True, probe_x_significant=True, probe_y_significant=False, neighbor_significant=True
- wrist: spearman_significant=True, probe_x_significant=True, probe_y_significant=True, neighbor_significant=True
- combined: spearman_significant=True, probe_x_significant=True, probe_y_significant=True, neighbor_significant=True
- Significant Spearman tests: 3

### H3: Ours subset 是否显著优于 random coverage？

- Status: **NOT SUPPORTED**
- Strong evidence families (n_strong_families): 0
- Weak evidence families (n_weak_families): 0
- Random-range metrics (n_random): 0
- Poor metrics (n_poor): 7
- Evidence families detail:
  - coverage_quality: [{'representation': 'global', 'metric': 'mean_nearest', 'better_than_random_fraction': 0.005}, {'representation': 'global', 'metric': 'p95_nearest', 'better_than_random_fraction': 0.041}, {'representation': 'wrist', 'metric': 'mean_nearest', 'better_than_random_fraction': 0.782}, {'representation': 'wrist', 'metric': 'p95_nearest', 'better_than_random_fraction': 0.219}, {'representation': 'combined', 'metric': 'mean_nearest', 'better_than_random_fraction': 0.0}, {'representation': 'combined', 'metric': 'p95_nearest', 'better_than_random_fraction': 0.0}]
  - max_radius_tail: [{'representation': 'global', 'metric': 'max_radius', 'better_than_random_fraction': 0.0}, {'representation': 'wrist', 'metric': 'max_radius', 'better_than_random_fraction': 0.253}, {'representation': 'combined', 'metric': 'max_radius', 'better_than_random_fraction': 0.0}]
  - redundancy: [{'representation': 'global', 'metric': 'redundancy_fraction', 'better_than_random_fraction': 1.0}, {'representation': 'wrist', 'metric': 'redundancy_fraction', 'better_than_random_fraction': 0.945}, {'representation': 'combined', 'metric': 'redundancy_fraction', 'better_than_random_fraction': 0.392}]
  - fixed_sic: [{'representation': 'combined', 'metric': 'normalized_fixed_sic', 'better_than_random_fraction': 0.0}]

## Phase Representation Warning

当前 global embedding 主要来自 episode 前 5 帧。
当前 wrist embedding 主要来自 episode 20%-70% 区间平均 representation。
因此当前分析主要验证 initial / pre-grasp task-state representation。
它不能证明 transport / place / release phase representation 是充分的。
不要因为 embedding initial-state geometry 很强就得出完整 manipulation representation 已经有效的结论。
