# Quick Surrogate Validation - shard 1/2

- Models with quick result: 25/52
- Verdict: **NOT_SUPPORTED**
- Runtime: 37.8 min

## Conclusion

当前 quick-eval 设置与历史 full-eval 排序一致性不足，不适合作为可靠 checkpoint filter。

## Correlation with historical full-eval success

| Signal | n | Pearson | Spearman |
|---|---:|---:|---:|
| quick_success | 25 | -0.225 | -0.398 |
| attention_l11_visual | 3 | 0.511 | 0.866 |
| v_t_norm_mean | 3 | 0.726 | 0.866 |
| trace_entropy_l11 | 3 | -0.192 | 0.000 |

## Screening retention

- Historical top-5 retained by quick-eval top-8: 0.0%

## Per-checkpoint results

| Model/checkpoint | Hist n | Full success | Quick success | Quick grasp | Task | Camera | Status |
|---|---:|---:|---:|---:|---|---|---|
| random_42_corner/random_112_seed42@008000 | 200 | 32.0 | NA | NA | pick-place-v3 | corner,gripperPOV | checkpoint_missing |
| random_42_corner/random_112_seed42@012000 | 200 | 32.0 | NA | NA | pick-place-v3 | corner,gripperPOV | checkpoint_missing |
| subzerocore_112_seed42_corner/subzerocore_112_seed42@006000 | 200 | 32.0 | 40.0 | 100.0 | pick-place-v3 | corner,gripperPOV | ok |
| subzerocore_112_seed42_corner/subzerocore_112_seed42@012000 | 200 | 32.0 | 20.0 | 100.0 | pick-place-v3 | corner,gripperPOV | ok |
| random_42_corner/random_112_seed42@016000 | 200 | 31.5 | NA | NA | pick-place-v3 | corner,gripperPOV | checkpoint_missing |
| subzerocore_112_seed42_corner/subzerocore_112_seed42@010000 | 200 | 30.5 | 40.0 | 100.0 | pick-place-v3 | corner,gripperPOV | ok |
| random_42_corner/random_112_seed42@014000 | 200 | 30.0 | NA | NA | pick-place-v3 | corner,gripperPOV | checkpoint_missing |
| subzerocore_112_seed42_corner/subzerocore_112_seed42@008000 | 200 | 30.0 | 20.0 | 100.0 | pick-place-v3 | corner,gripperPOV | ok |
| random_42_corner/random_112_seed42@010000 | 200 | 29.5 | NA | NA | pick-place-v3 | corner,gripperPOV | checkpoint_missing |
| random_42_corner2/random_112_seed42@014000 | 200 | 29.0 | NA | NA | pick-place-v3 | corner2,gripperPOV | checkpoint_missing |
| random_42_corner2/random_112_seed42@016000 | 200 | 28.5 | NA | NA | pick-place-v3 | corner2,gripperPOV | checkpoint_missing |
| random_42_corner/random_112_seed42@006000 | 200 | 28.0 | NA | NA | pick-place-v3 | corner,gripperPOV | checkpoint_missing |
| uniform_42_corner/uniform_112_seed42@012000 | 200 | 28.0 | 40.0 | 100.0 | pick-place-v3 | corner,gripperPOV | ok |
| uniform_42_corner/uniform_112_seed42@014000 | 200 | 28.0 | 40.0 | 100.0 | pick-place-v3 | corner,gripperPOV | ok |
| random_42_corner2/random_112_seed42@012000 | 200 | 27.5 | NA | NA | pick-place-v3 | corner2,gripperPOV | checkpoint_missing |
| uniform_42_corner/uniform_112_seed42@010000 | 200 | 27.0 | 40.0 | 100.0 | pick-place-v3 | corner,gripperPOV | ok |
| uniform_42_corner/uniform_112_seed42@016000 | 200 | 27.0 | 20.0 | 100.0 | pick-place-v3 | corner,gripperPOV | ok |
| subzerocore_112_seed42_corner/subzerocore_112_seed42@004000 | 200 | 26.0 | 40.0 | 80.0 | pick-place-v3 | corner,gripperPOV | ok |
| random_42_corner3/random_112_seed42@014000 | 200 | 25.5 | 40.0 | 100.0 | pick-place-v3 | corner3,gripperPOV | ok |
| random_42_corner3/random_112_seed42@016000 | 200 | 25.5 | 40.0 | 100.0 | pick-place-v3 | corner3,gripperPOV | ok |
| uniform_42_corner/uniform_112_seed42@004000 | 200 | 25.5 | 40.0 | 80.0 | pick-place-v3 | corner,gripperPOV | ok |
| uniform_42_corner/uniform_112_seed42@008000 | 200 | 25.5 | 40.0 | 80.0 | pick-place-v3 | corner,gripperPOV | ok |
| random_42_corner2/random_112_seed42@010000 | 200 | 25.0 | NA | NA | pick-place-v3 | corner2,gripperPOV | checkpoint_missing |
| random_42_corner2/random_112_seed42@008000 | 200 | 24.5 | NA | NA | pick-place-v3 | corner2,gripperPOV | checkpoint_missing |
| random_42_corner3/random_112_seed42@006000 | 200 | 24.5 | 60.0 | 100.0 | pick-place-v3 | corner3,gripperPOV | ok |
| random_42_corner3/random_112_seed42@008000 | 200 | 24.5 | 60.0 | 100.0 | pick-place-v3 | corner3,gripperPOV | ok |
| random_42_corner3/random_112_seed42@012000 | 200 | 24.5 | 40.0 | 100.0 | pick-place-v3 | corner3,gripperPOV | ok |
| uniform_42_corner/uniform_112_seed42@006000 | 200 | 24.5 | 40.0 | 80.0 | pick-place-v3 | corner,gripperPOV | ok |
| random_42_corner3/random_112_seed42@010000 | 200 | 24.0 | 60.0 | 100.0 | pick-place-v3 | corner3,gripperPOV | ok |
| uniform_42_corner3/uniform_112_seed42@014000 | 200 | 24.0 | NA | NA | pick-place-v3 | corner3,gripperPOV | not_run_budget |
| uniform_42_corner/uniform_112_seed42@002000 | 200 | 23.0 | 40.0 | 60.0 | pick-place-v3 | corner,gripperPOV | ok |
| uniform_42_corner3/uniform_112_seed42@010000 | 200 | 23.0 | NA | NA | pick-place-v3 | corner3,gripperPOV | not_run_budget |
| uniform_42_corner3/uniform_112_seed42@016000 | 200 | 23.0 | NA | NA | pick-place-v3 | corner3,gripperPOV | not_run_budget |
| random_42_corner3/random_112_seed42@004000 | 200 | 22.5 | 40.0 | 100.0 | pick-place-v3 | corner3,gripperPOV | ok |
| uniform_42_corner2/uniform_112_seed42@010000 | 200 | 22.5 | NA | NA | pick-place-v3 | corner2,gripperPOV | checkpoint_missing |
| uniform_42_corner3/uniform_112_seed42@006000 | 200 | 22.5 | 60.0 | 100.0 | pick-place-v3 | corner3,gripperPOV | ok |
| random_42_corner2/random_112_seed42@006000 | 200 | 21.5 | NA | NA | pick-place-v3 | corner2,gripperPOV | checkpoint_missing |
| uniform_42_corner3/uniform_112_seed42@012000 | 200 | 21.5 | NA | NA | pick-place-v3 | corner3,gripperPOV | not_run_budget |
| uniform_42_corner3/uniform_112_seed42@008000 | 200 | 21.0 | NA | NA | pick-place-v3 | corner3,gripperPOV | timeout |
| uniform_42_corner2/uniform_112_seed42@014000 | 200 | 20.5 | NA | NA | pick-place-v3 | corner2,gripperPOV | checkpoint_missing |
| uniform_42_corner2/uniform_112_seed42@006000 | 200 | 19.5 | NA | NA | pick-place-v3 | corner2,gripperPOV | checkpoint_missing |
| uniform_42_corner2/uniform_112_seed42@012000 | 200 | 19.5 | NA | NA | pick-place-v3 | corner2,gripperPOV | checkpoint_missing |
| uniform_42_corner2/uniform_112_seed42@016000 | 200 | 19.5 | NA | NA | pick-place-v3 | corner2,gripperPOV | checkpoint_missing |
| uniform_42_corner2/uniform_112_seed42@004000 | 200 | 17.5 | NA | NA | pick-place-v3 | corner2,gripperPOV | checkpoint_missing |
| uniform_42_corner2/uniform_112_seed42@008000 | 200 | 17.5 | NA | NA | pick-place-v3 | corner2,gripperPOV | checkpoint_missing |
| random_42_corner2/random_112_seed42@002000 | 200 | 16.5 | NA | NA | pick-place-v3 | corner2,gripperPOV | checkpoint_missing |
| random_42_corner3/random_112_seed42@002000 | 200 | 16.5 | 40.0 | 80.0 | pick-place-v3 | corner3,gripperPOV | ok |
| uniform_42_corner3/uniform_112_seed42@004000 | 200 | 16.0 | 40.0 | 60.0 | pick-place-v3 | corner3,gripperPOV | ok |
| subzerocore_112_seed42_corner/subzerocore_112_seed42@002000 | 200 | 14.5 | 60.0 | 80.0 | pick-place-v3 | corner,gripperPOV | ok |
| uniform_42_corner3/uniform_112_seed42@002000 | 200 | 13.0 | 20.0 | 40.0 | pick-place-v3 | corner3,gripperPOV | ok |
| random_42_corner2/random_112_seed42@004000 | 200 | 11.5 | NA | NA | pick-place-v3 | corner2,gripperPOV | checkpoint_missing |
| uniform_42_corner2/uniform_112_seed42@002000 | 200 | 7.0 | NA | NA | pick-place-v3 | corner2,gripperPOV | checkpoint_missing |

## Interpretation

- Spearman >= 0.60：可作为有用的 preliminary ranking signal。
- 0.30 <= Spearman < 0.60：只适合粗筛。
- Spearman < 0.30：当前 quick-eval 配置不适合作为 checkpoint filter。
- 即使结果为 SUPPORTED，仍然保留最终 full eval，只减少需要 full eval 的 checkpoint 数量。
- Attention / inference-trace 指标只对已有对应分析的少数 checkpoint 附加展示，不作为单独成功率预测器。
