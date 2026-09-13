#!/usr/bin/env python3
"""
调试 dataset_reader 的 _get_query_indices 方法。
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
print("调试 _get_query_indices 方法")
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

# 获取 reader
reader = fixed_dataset._ensure_reader()

# 手动调用 _get_query_indices 来调试
print(f"\n=== 手动调试 _get_query_indices ===")

# Episode 0
ep0_idx = 0
ep0 = reader._meta.episodes[ep0_idx]
print(f"Episode 0 元数据:")
print(f"  episode_index: {ep0['episode_index']}")
print(f"  dataset_from_index: {ep0['dataset_from_index']}")
print(f"  dataset_to_index: {ep0['dataset_to_index']}")
print(f"  length: {ep0['length']}")

# 调用 _get_query_indices
abs_idx = ep0['dataset_from_index']  # 0
ep_idx = ep0_idx  # 0
print(f"\n调用 _get_query_indices(abs_idx={abs_idx}, ep_idx={ep_idx})")

query_indices, padding = reader._get_query_indices(abs_idx, ep_idx)

print(f"  query_indices['action'] (前10个): {query_indices['action'][:10]}")
print(f"  padding['action_is_pad'] (前10个): {padding['action_is_pad'][:10]}")
print(f"  padding['action_is_pad'] 全True: {padding['action_is_pad'].all().item()}")
print(f"  padding['action_is_pad'] 全False: {(~padding['action_is_pad']).all().item()}")

# 检查 delta_indices
print(f"\nreader.delta_indices: {reader.delta_indices}")

# 手动计算
ep_start = ep0['dataset_from_index']
ep_end = ep0['dataset_to_index']
delta_idx = reader.delta_indices.get('action', list(range(50)))

print(f"\n手动计算:")
print(f"  ep_start: {ep_start}")
print(f"  ep_end: {ep_end}")
print(f"  delta_idx (前10个): {delta_idx[:10]}")

for delta in delta_idx[:10]:
    is_pad = (abs_idx + delta < ep_start) or (abs_idx + delta >= ep_end)
    print(f"    delta={delta}: abs_idx+delta={abs_idx+delta}, is_pad={is_pad}")

print("\n" + "=" * 60)
print("调试完成")
print("=" * 60)