#!/usr/bin/env python3
"""
检查合并数据集的 action 值分布，查找异常值。
使用 LeRobot 的方式读取数据集。
"""

import sys
sys.path.insert(0, '/data/zhonglinye/jun/lerobot/src')

import numpy as np
import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

# 当前使用的数据集路径
dataset_path = Path("/data/zhonglinye/jun/lerobot/personal/work2/duibi/our_v5_multi_84_seed42/merged_dataset")

print("=" * 60)
print("检查合并数据集的 action 值分布")
print("=" * 60)

# 加载数据集元数据
meta = LeRobotDatasetMetadata(str(dataset_path))
print(f"总帧数: {meta.total_frames}")
print(f"Episode 数: {meta.total_episodes}")
print(f"FPS: {meta.fps}")

# 加载数据集
dataset = LeRobotDataset(
    repo_id="lerobot/metaworld_pick_place",
    root=str(dataset_path),
)

print(f"\n数据集加载完成，共 {len(dataset)} 帧")

# 收集所有 action
all_actions = []
for i in range(len(dataset)):
    item = dataset[i]
    if 'action' in item:
        all_actions.append(item['action'].numpy())

if len(all_actions) > 0:
    actions = np.concatenate(all_actions, axis=0)
    print(f"\nAction shape: {actions.shape}")
    print(f"Action dtype: {actions.dtype}")
    
    # 检查每个维度的统计信息
    for i in range(actions.shape[1]):
        dim_values = actions[:, i]
        print(f"\nAction dim {i}:")
        print(f"  mean: {dim_values.mean():.6f}")
        print(f"  std: {dim_values.std():.6f}")
        print(f"  min: {dim_values.min():.6f}")
        print(f"  max: {dim_values.max():.6f}")
        print(f"  q01: {np.percentile(dim_values, 1):.6f}")
        print(f"  q99: {np.percentile(dim_values, 99):.6f}")
        
        # 检查是否有 NaN 或 Inf
        nan_count = np.isnan(dim_values).sum()
        inf_count = np.isinf(dim_values).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  ⚠️ NaN count: {nan_count}, Inf count: {inf_count}")
        
        # 检查是否有异常大的值
        large_values = np.abs(dim_values) > 100
        if large_values.any():
            print(f"  ⚠️ 异常大值 count (|x| > 100): {large_values.sum()}")
            print(f"  异常值示例: {dim_values[large_values][:10]}")
else:
    print("没有找到 action 数据！")

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)