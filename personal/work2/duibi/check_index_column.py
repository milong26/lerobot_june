#!/usr/bin/env python3
"""
检查合并后数据集中 index 列的值。
"""

import sys
sys.path.insert(0, '/data/zhonglinye/jun/lerobot/src')

import pandas as pd
from pathlib import Path

# 修复后的数据集路径
fixed_dataset_path = Path("/data/zhonglinye/jun/lerobot/personal/work2/duibi/our_v5_multi_84_seed42/merged_dataset_fixed")

print("=" * 60)
print("检查合并后数据集中 index 列的值")
print("=" * 60)

# 检查 data parquet
data_file = fixed_dataset_path / "data" / "chunk-000" / "file-000.parquet"
data_df = pd.read_parquet(data_file)

print(f"\nData 列名: {list(data_df.columns)}")
print(f"\nData 前10行:")
print(data_df[['index', 'frame_index', 'episode_index']].head(10).to_string())

print(f"\nData 最后10行:")
print(data_df[['index', 'frame_index', 'episode_index']].tail(10).to_string())

# 检查 index 和 frame_index 的关系
print(f"\nindex 范围: {data_df['index'].min()} - {data_df['index'].max()}")
print(f"frame_index 范围: {data_df['frame_index'].min()} - {data_df['frame_index'].max()}")
print(f"episode_index 范围: {data_df['episode_index'].min()} - {data_df['episode_index'].max()}")

# 检查 index 是否等于 frame_index
if (data_df['index'] == data_df['frame_index']).all():
    print(f"\n✓ index 等于 frame_index")
else:
    print(f"\n✗ index 不等于 frame_index")
    diff = (data_df['index'] != data_df['frame_index']).sum()
    print(f"  有 {diff} 行不同")

# 检查 episodes 元数据
episodes_file = fixed_dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
episodes_df = pd.read_parquet(episodes_file)

print(f"\nEpisodes 元数据 (前10行):")
print(episodes_df[['episode_index', 'length', 'dataset_from_index', 'dataset_to_index']].head(10).to_string())

# 检查 dataset_reader 如何使用 index
print(f"\n=== 分析 dataset_reader 逻辑 ===")
print("dataset_reader.py 第316行:")
print("  abs_idx = item['index'].item()")
print("  ep_idx = item['episode_index'].item()")
print()
print("然后调用 _get_query_indices(abs_idx, ep_idx)")
print("在 _get_query_indices 中:")
print("  ep = self._meta.episodes[ep_idx]")
print("  ep_start = ep['dataset_from_index']")
print("  ep_end = ep['dataset_to_index']")
print("  padding = [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]")
print()

# 模拟 Episode 0 的第一个 frame
ep0 = episodes_df.iloc[0]
ep0_start = ep0['dataset_from_index']
ep0_end = ep0['dataset_to_index']

# 获取 Episode 0 的第一个 frame 的数据
ep0_first_frame = data_df[data_df['episode_index'] == 0].iloc[0]
abs_idx = ep0_first_frame['index']
ep_idx = ep0_first_frame['episode_index']

print(f"Episode 0 第一个 frame:")
print(f"  index (abs_idx): {abs_idx}")
print(f"  episode_index (ep_idx): {ep_idx}")
print(f"  ep_start: {ep0_start}")
print(f"  ep_end: {ep0_end}")

# 计算 padding
delta_idx = list(range(50))
padding = [(abs_idx + delta < ep0_start) or (abs_idx + delta >= ep0_end) for delta in delta_idx]
print(f"  padding (前10个): {padding[:10]}")
print(f"  padding 全True: {all(padding)}")
print(f"  padding 全False: {all(not p for p in padding)}")

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)