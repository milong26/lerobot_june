#!/usr/bin/env python3
"""
检查合并后的数据集中 action_is_pad 的生成情况。
使用 lerobot 的 LeRobotDataset 来加载数据并检查 action_is_pad。
"""

import sys
sys.path.insert(0, '/data/zhonglinye/jun/lerobot/src')

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

# 数据集路径
merged_dataset_path = Path("/data/zhonglinye/jun/lerobot/personal/work2/duibi/our_v5_multi_84_seed42/merged_dataset")

print("=" * 60)
print("检查合并后的数据集")
print("=" * 60)

# 1. 检查 info.json
info_file = merged_dataset_path / "meta" / "info.json"
if info_file.exists():
    with open(info_file, 'r') as f:
        info = json.load(f)
    print(f"\n=== info.json 内容 ===")
    print(f"fps: {info.get('fps', 'N/A')}")
    print(f"total_episodes: {info.get('total_episodes', 'N/A')}")
    print(f"total_frames: {info.get('total_frames', 'N/A')}")
    print(f"features: {list(info.get('features', {}).keys())}")
    print(f"features 详情:")
    for key, val in info.get('features', {}).items():
        print(f"  {key}: {val}")
else:
    print(f"ERROR: info.json 不存在于 {info_file}")
    sys.exit(1)

# 2. 检查 episodes 元数据
episodes_file = merged_dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
if episodes_file.exists():
    episodes_df = pd.read_parquet(episodes_file)
    print(f"\n=== Episodes 元数据 (前10行) ===")
    print(f"总 episode 数: {len(episodes_df)}")
    print(episodes_df[['episode_index', 'length', 'dataset_from_index', 'dataset_to_index']].head(10).to_string())
    print(f"\n=== Episodes 元数据 (最后10行) ===")
    print(episodes_df[['episode_index', 'length', 'dataset_from_index', 'dataset_to_index']].tail(10).to_string())
    
    # 检查 dataset_from_index 和 dataset_to_index 是否连续
    print(f"\n=== 检查 episode 边界连续性 ===")
    for i in range(len(episodes_df)):
        ep = episodes_df.iloc[i]
        ep_start = ep['dataset_from_index']
        ep_end = ep['dataset_to_index']
        ep_length = ep['length']
        expected_length = ep_end - ep_start + 1
        if ep_length != expected_length:
            print(f"  Episode {ep['episode_index']}: length={ep_length}, expected={expected_length}, from={ep_start}, to={ep_end}")
else:
    print(f"ERROR: episodes parquet 不存在于 {episodes_file}")

# 3. 使用 LeRobotDataset 加载并检查 action_is_pad
print(f"\n=== 使用 LeRobotDataset 加载数据 ===")
print("配置 delta_timestamps 以模拟训练时的配置...")

# 训练时使用的 action_delta_indices (从日志中可以看到 chunk_size 相关配置)
# 需要查看训练配置中的 action_delta_indices
# 通常 smolvla 使用 chunk_size 来配置 action 窗口

# 先不配置 delta_timestamps，检查原始数据
try:
    ds_meta = LeRobotDatasetMetadata(str(merged_dataset_path))
    print(f"数据集元数据加载成功")
    print(f"  fps: {ds_meta.fps}")
    print(f"  total_episodes: {ds_meta.total_episodes}")
    print(f"  total_frames: {ds_meta.total_frames}")
    print(f"  features: {list(ds_meta.features.keys())}")
    print(f"  video_keys: {ds_meta.video_keys}")
    
    # 检查 action 特征
    if 'action' in ds_meta.features:
        print(f"  action 特征: {ds_meta.features['action']}")
    
except Exception as e:
    print(f"ERROR: 加载数据集元数据失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 检查原始数据中的 action 数据
print(f"\n=== 检查原始数据文件 ===")
data_file = merged_dataset_path / "data" / "chunk-000" / "file-000.parquet"
if data_file.exists():
    data_df = pd.read_parquet(data_file)
    print(f"总 frame 数: {len(data_df)}")
    print(f"列名: {list(data_df.columns)}")
    
    if 'action' in data_df.columns:
        actions = data_df['action'].values
        print(f"action 形状: {actions.shape}")
        print(f"action 样例 (前5行): {actions[:5]}")
        
        # 检查是否有 NaN 或异常值
        has_nan = np.any(np.isnan(actions)) if actions.dtype == np.float64 else False
        print(f"action 包含 NaN: {has_nan}")
    
    if 'episode_index' in data_df.columns:
        # 检查每个 episode 的 frame 数
        ep_counts = data_df['episode_index'].value_counts().sort_index()
        print(f"\n每个 episode 的 frame 数:")
        print(ep_counts.to_string())
        
        # 检查 episode_index 是否连续
        ep_indices = sorted(data_df['episode_index'].unique())
        print(f"\nEpisode 索引范围: {min(ep_indices)} - {max(ep_indices)}")
        print(f"Episode 索引是否连续: {ep_indices == list(range(min(ep_indices), max(ep_indices) + 1))}")
else:
    print(f"ERROR: data parquet 不存在于 {data_file}")

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)