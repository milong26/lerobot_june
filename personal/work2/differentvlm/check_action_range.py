#!/usr/bin/env python3
"""
检查模型输出的动作范围
"""
import sys
sys.path.insert(0, "/data/zhonglinye/jun/lerobot/src")

import torch
import numpy as np
from safetensors.torch import load_file

# 加载归一化统计
norm_path = "/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/experiments/tinyvla_s_disassemble-v3_corner/checkpoints/tinyvla_tinyvla_s_disassemble-v3_corner/checkpoints/012000/pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors"
norm_stats = load_file(norm_path)

action_mean = norm_stats["action.mean"]
action_std = norm_stats["action.std"]
action_min = norm_stats["action.min"]
action_max = norm_stats["action.max"]

print("="*60)
print("动作空间分析")
print("="*60)
print(f"\n数据集动作统计:")
print(f"  mean:  {action_mean.tolist()}")
print(f"  std:   {action_std.tolist()}")
print(f"  min:   {action_min.tolist()}")
print(f"  max:   {action_max.tolist()}")
print(f"  range: {(action_max - action_min).tolist()}")

print(f"\nMetaWorld环境动作范围: [-1, 1]")
print(f"\n数据集中超出[-1,1]的维度:")
for i in range(4):
    if action_min[i] < -1 or action_max[i] > 1:
        print(f"  维度{i}: [{action_min[i]:.3f}, {action_max[i]:.3f}] - 超出范围!")
    else:
        print(f"  维度{i}: [{action_min[i]:.3f}, {action_max[i]:.3f}] - 正常")

print(f"\n结论:")
print(f"  模型输出的是原始动作空间（未归一化到[-1,1]）")
print(f"  评估环境期望[-1,1]范围")
print(f"  模型输出会被clip，导致动作幅度不足")

# 计算如果模型输出数据集范围内的动作，clip后的影响
print(f"\nClip影响分析:")
for i in range(4):
    min_val = float(action_min[i])
    max_val = float(action_max[i])
    clipped_min = max(min_val, -1.0)
    clipped_max = min(max_val, 1.0)
    original_range = max_val - min_val
    clipped_range = clipped_max - clipped_min
    loss_pct = (1 - clipped_range / original_range) * 100
    print(f"  维度{i}: 原始范围[{min_val:.3f}, {max_val:.3f}] (range={original_range:.3f})")
    print(f"         Clip后[{clipped_min:.3f}, {clipped_max:.3f}] (range={clipped_range:.3f})")
    print(f"         信息损失: {loss_pct:.1f}%")