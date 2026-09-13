#!/usr/bin/env python3
"""
验证修复后的数据集（merged_dataset_fixed2）的 index 列和 action_is_pad。
"""

import sys
sys.path.insert(0, '/data/zhonglinye/jun/lerobot/src')

import torch
import pandas as pd
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

# 修复后的数据集路径
fixed_dataset_path = Path("/data/zhonglinye/jun/lerobot/personal/work2/duibi/our_v5_multi_84_seed42/merged_dataset_fixed2")

print("=" * 60)
print("验证修复后的数据集（merged_dataset_fixed2）")
print("=" * 60)

# 1. 检查 index 列
print(f"\n=== 1. 检查 index 列 ===")
data_file = fixed_dataset_path / "data" / "chunk-000" / "file-000.parquet"
data_df = pd.read_parquet(data_file)

print(f"Data 前10行:")
print(data_df[['index', 'frame_index', 'episode_index']].head(10).to_string())

print(f"\nindex 范围: {data_df['index'].min()} - {data_df['index'].max()}")
print(f"frame_index 范围: {data_df['frame_index'].min()} - {data_df['frame_index'].max()}")

if (data_df['index'] == data_df['frame_index']).all():
    print(f"✓ index 等于 frame_index")
else:
    print(f"✗ index 不等于 frame_index")

# 2. 检查 action_is_pad
print(f"\n=== 2. 检查 action_is_pad ===")
fixed_meta = LeRobotDatasetMetadata(str(fixed_dataset_path))
print(f"fps: {fixed_meta.fps}")
print(f"total_episodes: {fixed_meta.total_episodes}")
print(f"total_frames: {fixed_meta.total_frames}")

# 加载数据集
delta_timestamps = {
    'action': [i / fixed_meta.fps for i in range(50)],
    'observation.state': [0.0],
}

fixed_dataset = LeRobotDataset(
    repo_id="lerobot/metaworld_pick_place",
    root=str(fixed_dataset_path),
    delta_timestamps=delta_timestamps,
)

# 检查 episodes 元数据
episodes_file = fixed_dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
episodes_df = pd.read_parquet(episodes_file)

# 统计 action_is_pad 情况
action_is_pad_stats = {
    'all_true': 0,
    'all_false': 0,
    'partial': 0,
    'total': 0,
}

abnormal_episodes = []

# 检查所有 episode 的第一个 frame
for ep_idx in range(len(episodes_df)):
    ep = episodes_df.iloc[ep_idx]
    ep_start = ep['dataset_from_index']
    
    item = fixed_dataset[ep_start]
    if 'action_is_pad' in item:
        is_pad = item['action_is_pad']
        all_true = is_pad.all().item()
        all_false = (~is_pad).all().item()
        
        if all_true:
            action_is_pad_stats['all_true'] += 1
            abnormal_episodes.append(ep_idx)
        elif all_false:
            action_is_pad_stats['all_false'] += 1
        else:
            action_is_pad_stats['partial'] += 1
        
        action_is_pad_stats['total'] += 1

print(f"\n=== action_is_pad 统计 ===")
print(f"  总检查数: {action_is_pad_stats['total']}")
print(f"  全部为 True (异常): {action_is_pad_stats['all_true']}")
print(f"  全部为 False (正常): {action_is_pad_stats['all_false']}")
print(f"  部分为 True (正常): {action_is_pad_stats['partial']}")

if abnormal_episodes:
    print(f"\n❌ 仍有 {len(abnormal_episodes)} 个异常 episode")
    print(f"  Episode 索引: {abnormal_episodes[:10]}...")
else:
    print(f"\n✅ 所有 episode 的 action_is_pad 都正常!")

# 额外验证：检查几个 episode 的不同位置
print(f"\n=== 额外验证：检查不同位置的 action_is_pad ===")
for test_ep_idx in [0, 1, 42, 83]:
    ep = episodes_df.iloc[test_ep_idx]
    ep_start = ep['dataset_from_index']
    ep_length = ep['length']
    
    # 第一个 frame
    item_first = fixed_dataset[ep_start]
    if 'action_is_pad' in item_first:
        is_pad_first = item_first['action_is_pad']
        true_count = is_pad_first.sum().item()
        total_count = len(is_pad_first)
        print(f"  Episode {test_ep_idx}, Frame {ep_start} (第一个): {true_count}/{total_count} 为 True")
    
    # 中间 frame
    mid_offset = ep_length // 2
    item_mid = fixed_dataset[ep_start + mid_offset]
    if 'action_is_pad' in item_mid:
        is_pad_mid = item_mid['action_is_pad']
        true_count = is_pad_mid.sum().item()
        total_count = len(is_pad_mid)
        print(f"  Episode {test_ep_idx}, Frame {ep_start + mid_offset} (中间): {true_count}/{total_count} 为 True")
    
    # 最后一个 frame
    item_last = fixed_dataset[ep_start + ep_length - 1]
    if 'action_is_pad' in item_last:
        is_pad_last = item_last['action_is_pad']
        true_count = is_pad_last.sum().item()
        total_count = len(is_pad_last)
        print(f"  Episode {test_ep_idx}, Frame {ep_start + ep_length - 1} (最后一个): {true_count}/{total_count} 为 True")

print("\n" + "=" * 60)
print("验证完成")
print("=" * 60)