#!/usr/bin/env python3
"""
验证修复后的数据集 action_is_pad 是否正确。
"""

import sys
sys.path.insert(0, '/data/zhonglinye/jun/lerobot/src')

import torch
import pandas as pd
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

# 修复后的数据集路径
fixed_dataset_path = Path("/data/zhonglinye/jun/lerobot/personal/work2/duibi/our_v5_multi_84_seed42/merged_dataset_fixed")

print("=" * 60)
print("验证修复后的数据集 action_is_pad")
print("=" * 60)

# 加载数据集元数据
fixed_meta = LeRobotDatasetMetadata(str(fixed_dataset_path))
print(f"fps: {fixed_meta.fps}")
print(f"total_episodes: {fixed_meta.total_episodes}")
print(f"total_frames: {fixed_meta.total_frames}")

# 检查 episodes 元数据
fixed_episodes_file = fixed_dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
fixed_episodes_df = pd.read_parquet(fixed_episodes_file)
print(f"\nEpisodes 元数据 (前5行):")
print(fixed_episodes_df[['episode_index', 'length', 'dataset_from_index', 'dataset_to_index']].head(5).to_string())

# 测试修复后数据集的 action_is_pad
print(f"\n=== 测试修复后数据集的 action_is_pad ===")
delta_timestamps = {
    'action': [i / fixed_meta.fps for i in range(50)],
    'observation.state': [0.0],
}

fixed_dataset = LeRobotDataset(
    repo_id="lerobot/metaworld_pick_place",
    root=str(fixed_dataset_path),
    delta_timestamps=delta_timestamps,
)

# 统计 action_is_pad 情况
action_is_pad_stats = {
    'all_true': 0,
    'all_false': 0,
    'partial': 0,
    'total': 0,
}

# 检查所有 episode 的第一个 frame
abnormal_episodes = []
for ep_idx in range(len(fixed_episodes_df)):
    ep = fixed_episodes_df.iloc[ep_idx]
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

# 额外验证：检查几个 episode 的中间和最后一个 frame
print(f"\n=== 额外验证：检查 episode 0 的不同位置 ===")
ep0 = fixed_episodes_df.iloc[0]
ep0_start = ep0['dataset_from_index']
ep0_length = ep0['length']

# 第一个 frame
item_first = fixed_dataset[ep0_start]
if 'action_is_pad' in item_first:
    is_pad_first = item_first['action_is_pad']
    print(f"  Frame {ep0_start} (第一个): action_is_pad 全True={is_pad_first.all().item()}, 全False={(~is_pad_first).all().item()}")

# 中间 frame
item_mid = fixed_dataset[ep0_start + ep0_length // 2]
if 'action_is_pad' in item_mid:
    is_pad_mid = item_mid['action_is_pad']
    print(f"  Frame {ep0_start + ep0_length // 2} (中间): action_is_pad 全True={is_pad_mid.all().item()}, 全False={(~is_pad_mid).all().item()}")

# 最后一个 frame
item_last = fixed_dataset[ep0_start + ep0_length - 1]
if 'action_is_pad' in item_last:
    is_pad_last = item_last['action_is_pad']
    print(f"  Frame {ep0_start + ep0_length - 1} (最后一个): action_is_pad 全True={is_pad_last.all().item()}, 全False={(~is_pad_last).all().item()}")

print("\n" + "=" * 60)
print("验证完成")
print("=" * 60)