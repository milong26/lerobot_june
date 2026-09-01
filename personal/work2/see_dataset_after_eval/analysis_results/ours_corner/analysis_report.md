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

## Ours vs Uniform Comparison

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

- Status: **NOT EVALUATED**
- Strong evidence families (n_strong_families): 0
- Weak evidence families (n_weak_families): 0
- Random-range metrics (n_random): 0
- Poor metrics (n_poor): 0

## Phase Representation Warning

当前 global embedding 主要来自 episode 前 5 帧。
当前 wrist embedding 主要来自 episode 20%-70% 区间平均 representation。
因此当前分析主要验证 initial / pre-grasp task-state representation。
它不能证明 transport / place / release phase representation 是充分的。
不要因为 embedding initial-state geometry 很强就得出完整 manipulation representation 已经有效的结论。
