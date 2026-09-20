# Quick Surrogate Validation - shard 0/2

- Models with quick result: 24/52
- Verdict: **NOT_SUPPORTED**
- Runtime: 37.8 min

## Conclusion

当前 quick-eval 设置与历史 full-eval 排序一致性不足，不适合作为可靠 checkpoint filter。

## Correlation with historical full-eval success

| Signal | n | Pearson | Spearman |
|---|---:|---:|---:|
| quick_success | 24 | -0.671 | -0.673 |
| attention_l11_visual | 4 | 0.757 | 0.800 |
| v_t_norm_mean | 4 | 0.922 | 0.800 |
| trace_entropy_l11 | 4 | -0.913 | -0.800 |

## Screening retention

- Historical top-5 retained by quick-eval top-8: 0.0%

## Per-checkpoint results

| Model/checkpoint | Hist n | Full success | Quick success | Quick grasp | Task | Camera | Status |
|---|---:|---:|---:|---:|---|---|---|
| ours_v3_no_action_112_seed42_corner/dynamicanchor_v3_no_action_112_seed42@006000 | 200 | 36.5 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| random_42_corner/random_112_seed42@004000 | 200 | 36.0 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_v4_112_seed42_corner/dynamicgrid_v4_112_seed42@008000 | 200 | 35.0 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_v4_112_seed42_corner/dynamicgrid_v4_112_seed42@010000 | 200 | 35.0 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_v3_no_action_112_seed42_corner/dynamicanchor_v3_no_action_112_seed42@008000 | 200 | 34.5 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_v4_112_seed42_corner/dynamicgrid_v4_112_seed42@012000 | 200 | 34.5 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_v4_112_seed42_corner/dynamicgrid_v4_112_seed42@006000 | 200 | 33.5 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_v3_no_action_112_seed42_corner/dynamicanchor_v3_no_action_112_seed42@010000 | 200 | 31.5 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_v3_no_action_112_seed42_corner/dynamicanchor_v3_no_action_112_seed42@012000 | 200 | 31.5 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_v4_112_seed42_corner/dynamicgrid_v4_112_seed42@004000 | 200 | 30.5 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_v2_112_seed42_corner/dynamicanchor_v2_112_seed42@006000 | 200 | 30.0 | NA | NA | pick-place-v3 | corner,gripperPOV | timeout |
| ours_v3_no_action_112_seed42_corner/dynamicanchor_v3_no_action_112_seed42@004000 | 200 | 29.5 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_112_seed42_corner/dynamicanchor_112_seed42@010000 | 200 | 27.5 | 0.0 | 100.0 | pick-place-v3 | corner,gripperPOV | ok |
| ours_112_seed42_corner2/dynamicanchor_112_seed42@010000 | 200 | 27.5 | 0.0 | 60.0 | pick-place-v3 | corner2,gripperPOV | ok |
| ours_112_seed42_corner/dynamicanchor_112_seed42@004000 | 200 | 27.0 | 0.0 | 100.0 | pick-place-v3 | corner,gripperPOV | ok |
| ours_112_seed42_corner2/dynamicanchor_112_seed42@008000 | 200 | 27.0 | 0.0 | 60.0 | pick-place-v3 | corner2,gripperPOV | ok |
| ours_112_seed42_corner2/dynamicanchor_112_seed42@012000 | 200 | 26.5 | 0.0 | 60.0 | pick-place-v3 | corner2,gripperPOV | ok |
| ours_112_seed42_corner/dynamicanchor_112_seed42@008000 | 200 | 26.0 | 0.0 | 100.0 | pick-place-v3 | corner,gripperPOV | ok |
| ours_v2_112_seed42_corner/dynamicanchor_v2_112_seed42@008000 | 200 | 26.0 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_112_seed42_corner/dynamicanchor_112_seed42@002000 | 200 | 25.5 | 0.0 | 80.0 | pick-place-v3 | corner,gripperPOV | ok |
| ours_112_seed42_corner/dynamicanchor_112_seed42@006000 | 200 | 25.5 | 0.0 | 100.0 | pick-place-v3 | corner,gripperPOV | ok |
| ours_112_seed42_corner2/dynamicanchor_112_seed42@004000 | 200 | 25.5 | 20.0 | 100.0 | pick-place-v3 | corner2,gripperPOV | ok |
| ours_v2_112_seed42_corner/dynamicanchor_v2_112_seed42@012000 | 200 | 25.5 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_v2_112_seed42_corner/dynamicanchor_v2_112_seed42@010000 | 200 | 25.0 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_v2_112_seed42_corner/dynamicanchor_v2_112_seed42@016000 | 200 | 25.0 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_112_seed42_corner/dynamicanchor_112_seed42@012000 | 200 | 24.5 | 0.0 | 100.0 | pick-place-v3 | corner,gripperPOV | ok |
| ours_112_seed42_corner/dynamicanchor_112_seed42@016000 | 200 | 24.5 | 0.0 | 100.0 | pick-place-v3 | corner,gripperPOV | ok |
| ours_v2_112_seed42_corner/dynamicanchor_v2_112_seed42@004000 | 200 | 24.5 | 0.0 | 100.0 | pick-place-v3 | corner,gripperPOV | ok |
| ours_v3_no_action_112_seed42_corner/dynamicanchor_v3_no_action_112_seed42@002000 | 200 | 24.5 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_v4_112_seed42_corner/dynamicgrid_v4_112_seed42@002000 | 200 | 24.5 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_112_seed42_corner/dynamicanchor_112_seed42@014000 | 200 | 24.0 | 0.0 | 100.0 | pick-place-v3 | corner,gripperPOV | ok |
| ours_v2_112_seed42_corner/dynamicanchor_v2_112_seed42@014000 | 200 | 24.0 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| random_42_corner/random_112_seed42@002000 | 200 | 22.5 | NA | NA | pick-place-v3 | corner,gripperPOV | not_run_budget |
| ours_v2_112_seed42_corner3/dynamicanchor_v2_112_seed42@010000 | 200 | 21.5 | NA | NA | pick-place-v3 | corner3,gripperPOV | not_run_budget |
| ours_112_seed42_corner2/dynamicanchor_112_seed42@006000 | 200 | 21.0 | 0.0 | 60.0 | pick-place-v3 | corner2,gripperPOV | ok |
| ours_v2_112_seed42_corner/dynamicanchor_v2_112_seed42@002000 | 200 | 21.0 | 20.0 | 80.0 | pick-place-v3 | corner,gripperPOV | ok |
| ours_v2_112_seed42_corner3/dynamicanchor_v2_112_seed42@012000 | 200 | 20.0 | NA | NA | pick-place-v3 | corner3,gripperPOV | not_run_budget |
| ours_v2_112_seed42_corner3/dynamicanchor_v2_112_seed42@014000 | 200 | 20.0 | NA | NA | pick-place-v3 | corner3,gripperPOV | not_run_budget |
| ours_v2_112_seed42_corner3/dynamicanchor_v2_112_seed42@016000 | 200 | 20.0 | NA | NA | pick-place-v3 | corner3,gripperPOV | not_run_budget |
| ours_v2_112_seed42_corner3/dynamicanchor_v2_112_seed42@008000 | 200 | 19.5 | NA | NA | pick-place-v3 | corner3,gripperPOV | not_run_budget |
| ours_v2_112_seed42_corner3/dynamicanchor_v2_112_seed42@006000 | 200 | 18.5 | NA | NA | pick-place-v3 | corner3,gripperPOV | not_run_budget |
| ours_112_seed42_corner3/dynamicanchor_112_seed42@016000 | 200 | 16.5 | 40.0 | 100.0 | pick-place-v3 | corner3,gripperPOV | ok |
| ours_112_seed42_corner3/dynamicanchor_112_seed42@012000 | 200 | 16.0 | 60.0 | 100.0 | pick-place-v3 | corner3,gripperPOV | ok |
| ours_112_seed42_corner3/dynamicanchor_112_seed42@014000 | 200 | 16.0 | 60.0 | 100.0 | pick-place-v3 | corner3,gripperPOV | ok |
| ours_112_seed42_corner3/dynamicanchor_112_seed42@010000 | 200 | 14.0 | 40.0 | 100.0 | pick-place-v3 | corner3,gripperPOV | ok |
| ours_112_seed42_corner3/dynamicanchor_112_seed42@008000 | 200 | 13.5 | 40.0 | 100.0 | pick-place-v3 | corner3,gripperPOV | ok |
| ours_112_seed42_corner3/dynamicanchor_112_seed42@006000 | 200 | 12.5 | 40.0 | 100.0 | pick-place-v3 | corner3,gripperPOV | ok |
| ours_v2_112_seed42_corner3/dynamicanchor_v2_112_seed42@004000 | 200 | 12.0 | NA | NA | pick-place-v3 | corner3,gripperPOV | not_run_budget |
| ours_v2_112_seed42_corner3/dynamicanchor_v2_112_seed42@002000 | 200 | 11.0 | NA | NA | pick-place-v3 | corner3,gripperPOV | not_run_budget |
| ours_112_seed42_corner2/dynamicanchor_112_seed42@002000 | 200 | 10.5 | 0.0 | 40.0 | pick-place-v3 | corner2,gripperPOV | ok |
| ours_112_seed42_corner3/dynamicanchor_112_seed42@004000 | 200 | 10.5 | 40.0 | 100.0 | pick-place-v3 | corner3,gripperPOV | ok |
| ours_112_seed42_corner3/dynamicanchor_112_seed42@002000 | 200 | 6.5 | 20.0 | 40.0 | pick-place-v3 | corner3,gripperPOV | ok |

## Interpretation

- Spearman >= 0.60：可作为有用的 preliminary ranking signal。
- 0.30 <= Spearman < 0.60：只适合粗筛。
- Spearman < 0.30：当前 quick-eval 配置不适合作为 checkpoint filter。
- 即使结果为 SUPPORTED，仍然保留最终 full eval，只减少需要 full eval 的 checkpoint 数量。
- Attention / inference-trace 指标只对已有对应分析的少数 checkpoint 附加展示，不作为单独成功率预测器。
