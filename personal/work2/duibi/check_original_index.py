#!/usr/bin/env python3
"""
检查原始数据集中 index 列的值。
"""

import sys
sys.path.insert(0, '/data/zhonglinye/jun/lerobot/src')

import pandas as pd
from pathlib import Path

# 原始数据集路径
original_dataset_path = Path("/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/coffee-button-v3_corner")

print("=" * 60)
print("检查原始数据集中 index 列的值")
print("=" * 60)

# 检查 data parquet
data_file = original_dataset_path / "data" / "chunk-000" / "file-000.parquet"
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

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)