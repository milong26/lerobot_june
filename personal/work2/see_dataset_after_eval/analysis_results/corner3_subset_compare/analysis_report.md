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
| global | 0.2546 | 9.99e-04 | 0.0360 | 0.0051 | 0.0819 | 4.81 |
| wrist | 0.1458 | 9.99e-04 | -0.0005 | 0.0014 | 0.0553 | 12.59 |
| combined | 0.2059 | 9.99e-04 | 0.3068 | 0.3383 | 0.0916 | 17.87 |

## Table 2: Subset Coverage Comparison

| Method | Global Cover | Wrist Cover | Combined Cover | Combined Max Radius | Combined Redundancy | Fixed SIC | Fixed SIC BTR | Global BTR | Wrist BTR | Combined BTR |
|--------|-------------|-------------|----------------|---------------------|---------------------|-----------|---------------|------------|-----------|--------------|
| Random | 0.0115 | 0.0073 | 1.0543 | 1.3920 | 0.9018 | 0.8832 | 0.9490 | 0.8010 | 0.8060 | 0.5250 |
| Uniform | 0.0118 | 0.0074 | 1.0958 | 1.6256 | 0.9018 | 0.8822 | 0.5920 | 0.0840 | 0.1380 | 0.0000 |
| Ours | 0.0128 | 0.0074 | 1.1204 | 1.6041 | 0.8661 | 0.8706 | 0.0000 | 0.0000 | 0.2650 | 0.0000 |

## Ours vs Uniform Comparison

- Episode overlap count: 35
- Episode overlap ratio: 0.3125
- Conclusion: Insufficient evidence that Ours mainly reproduces Uniform initial-position coverage.

### Evidence Summary

- workspace_physical_unselected_mean_nearest_relative_delta: 2.0280
- workspace_physical_unselected_p95_relative_delta: 1.1411
- workspace_physical_unselected_max_radius_relative_delta: 2.9646
- global_coverage_unselected_mean_nearest_distance_relative_delta: 0.0813
- global_coverage_unselected_median_nearest_distance_relative_delta: 0.0768
- global_coverage_unselected_p90_nearest_distance_relative_delta: 0.1507
- global_coverage_unselected_p95_nearest_distance_relative_delta: 0.1405
- global_coverage_unselected_max_nearest_distance_relative_delta: -0.2848
- wrist_coverage_unselected_mean_nearest_distance_relative_delta: -0.0037
- wrist_coverage_unselected_median_nearest_distance_relative_delta: 0.0053
- wrist_coverage_unselected_p90_nearest_distance_relative_delta: -0.0027
- wrist_coverage_unselected_p95_nearest_distance_relative_delta: -0.0104
- wrist_coverage_unselected_max_nearest_distance_relative_delta: -0.0526
- combined_coverage_unselected_mean_nearest_distance_relative_delta: 0.0224
- combined_coverage_unselected_median_nearest_distance_relative_delta: 0.0349
- combined_coverage_unselected_p90_nearest_distance_relative_delta: -0.0007
- combined_coverage_unselected_p95_nearest_distance_relative_delta: 0.0196
- combined_coverage_unselected_max_nearest_distance_relative_delta: -0.0132
- global_redundancy_mean_nearest_relative_delta: 0.0693
- global_redundancy_median_nearest_relative_delta: 0.0509
- global_redundancy_p10_nearest_relative_delta: 0.0932
- global_redundancy_p50_nearest_relative_delta: 0.0509
- global_redundancy_p90_nearest_relative_delta: 0.0701
- global_redundancy_redundancy_fraction_relative_delta: -0.0196
- wrist_redundancy_mean_nearest_relative_delta: 0.0691
- wrist_redundancy_median_nearest_relative_delta: 0.0616
- wrist_redundancy_p10_nearest_relative_delta: 0.0711
- wrist_redundancy_p50_nearest_relative_delta: 0.0616
- wrist_redundancy_p90_nearest_relative_delta: 0.0660
- wrist_redundancy_redundancy_fraction_relative_delta: -0.0882
- combined_redundancy_mean_nearest_relative_delta: 0.1081
- combined_redundancy_median_nearest_relative_delta: 0.1289
- combined_redundancy_p10_nearest_relative_delta: 0.2267
- combined_redundancy_p50_nearest_relative_delta: 0.1289
- combined_redundancy_p90_nearest_relative_delta: 0.0390
- combined_redundancy_redundancy_fraction_relative_delta: -0.0396
- fixed_sic_delta: -0.0116
- fixed_sic_relative_delta: -0.0132
- note: No calibrated equivalence test was performed; inspect the multi-metric deltas above. Relative deltas are computed as (ours - uniform) / max(|uniform|, eps).

### Workspace Coverage Delta (Ours - Uniform)

| Metric | Delta |
|--------|-------|
| physical_unselected_mean_nearest | 0.0051 |
| physical_unselected_p95 | 0.0083 |
| physical_unselected_max_radius | 0.0255 |
| grid_7x4_ratio | -0.0714 |
| grid_14x8_ratio | -0.2857 |

### Global Coverage Delta (Ours - Uniform)

| Metric | Delta |
|--------|-------|
| unselected_mean_nearest_distance | 0.0010 |
| unselected_median_nearest_distance | 0.0009 |
| unselected_p90_nearest_distance | 0.0021 |
| unselected_p95_nearest_distance | 0.0021 |
| unselected_max_nearest_distance | -0.0100 |

### Wrist Coverage Delta (Ours - Uniform)

| Metric | Delta |
|--------|-------|
| unselected_mean_nearest_distance | -0.0000 |
| unselected_median_nearest_distance | 0.0000 |
| unselected_p90_nearest_distance | -0.0000 |
| unselected_p95_nearest_distance | -0.0001 |
| unselected_max_nearest_distance | -0.0007 |

### Combined Coverage Delta (Ours - Uniform)

| Metric | Delta |
|--------|-------|
| unselected_mean_nearest_distance | 0.0246 |
| unselected_median_nearest_distance | 0.0383 |
| unselected_p90_nearest_distance | -0.0010 |
| unselected_p95_nearest_distance | 0.0277 |
| unselected_max_nearest_distance | -0.0215 |

### Global Redundancy Delta (Ours - Uniform)

| Metric | Delta |
|--------|-------|
| mean_nearest | 0.0008 |
| median_nearest | 0.0005 |
| p10_nearest | 0.0008 |
| p50_nearest | 0.0005 |
| p90_nearest | 0.0009 |
| redundancy_fraction | -0.0179 |

### Wrist Redundancy Delta (Ours - Uniform)

| Metric | Delta |
|--------|-------|
| mean_nearest | 0.0005 |
| median_nearest | 0.0004 |
| p10_nearest | 0.0004 |
| p50_nearest | 0.0004 |
| p90_nearest | 0.0006 |
| redundancy_fraction | -0.0804 |

### Combined Redundancy Delta (Ours - Uniform)

| Metric | Delta |
|--------|-------|
| mean_nearest | 0.1104 |
| median_nearest | 0.1307 |
| p10_nearest | 0.1657 |
| p50_nearest | 0.1307 |
| p90_nearest | 0.0510 |
| redundancy_fraction | -0.0357 |

- Fixed SIC Delta (Ours - Uniform): -0.0116

## Hypothesis Evaluation

### H1: Embedding 是否具有可区分性？

- Status: **SUPPORTED**
- global: probe_x_significant=True, probe_y_significant=True, neighbor_significant=True, validity=True
- wrist: probe_x_significant=True, probe_y_significant=True, neighbor_significant=True, validity=True
- combined: probe_x_significant=True, probe_y_significant=True, neighbor_significant=True, validity=True
- Effective rank ratio: 0.1504

### H2: Embedding distance 是否具有 task-state geometry？

- Status: **SUPPORTED**
- global: spearman_significant=True, probe_x_significant=True, probe_y_significant=True, neighbor_significant=True
- wrist: spearman_significant=True, probe_x_significant=True, probe_y_significant=True, neighbor_significant=True
- combined: spearman_significant=True, probe_x_significant=True, probe_y_significant=True, neighbor_significant=True
- Significant Spearman tests: 3

### H3: Ours subset 是否显著优于 random coverage？

- Status: **NOT SUPPORTED**
- Strong evidence families (n_strong_families): 1
- Weak evidence families (n_weak_families): 0
- Random-range metrics (n_random): 1
- Poor metrics (n_poor): 6
- Evidence families detail:
  - coverage_quality: [{'representation': 'global', 'metric': 'mean_nearest', 'better_than_random_fraction': 0.0}, {'representation': 'global', 'metric': 'p95_nearest', 'better_than_random_fraction': 0.037}, {'representation': 'wrist', 'metric': 'mean_nearest', 'better_than_random_fraction': 0.265}, {'representation': 'wrist', 'metric': 'p95_nearest', 'better_than_random_fraction': 0.431}, {'representation': 'combined', 'metric': 'mean_nearest', 'better_than_random_fraction': 0.0}, {'representation': 'combined', 'metric': 'p95_nearest', 'better_than_random_fraction': 0.0}]
  - max_radius_tail: [{'representation': 'global', 'metric': 'max_radius', 'better_than_random_fraction': 0.878}, {'representation': 'wrist', 'metric': 'max_radius', 'better_than_random_fraction': 0.649}, {'representation': 'combined', 'metric': 'max_radius', 'better_than_random_fraction': 0.028}]
  - redundancy: [{'representation': 'global', 'metric': 'redundancy_fraction', 'better_than_random_fraction': 0.352}, {'representation': 'wrist', 'metric': 'redundancy_fraction', 'better_than_random_fraction': 0.971}, {'representation': 'combined', 'metric': 'redundancy_fraction', 'better_than_random_fraction': 0.991}]
  - fixed_sic: [{'representation': 'combined', 'metric': 'normalized_fixed_sic', 'better_than_random_fraction': 0.0}]

## Phase Representation Warning

当前 global embedding 主要来自 episode 前 5 帧。
当前 wrist embedding 主要来自 episode 20%-70% 区间平均 representation。
因此当前分析主要验证 initial / pre-grasp task-state representation。
它不能证明 transport / place / release phase representation 是充分的。
不要因为 embedding initial-state geometry 很强就得出完整 manipulation representation 已经有效的结论。
