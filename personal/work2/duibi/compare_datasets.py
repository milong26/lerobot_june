#!/usr/bin/env python3
"""
对比原始数据集和合并后的数据集，找出 action_is_pad 异常的原因。
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

print("=" * 60)
print("对比原始数据集和合并后的数据集")
print("=" * 60)

# 原始数据集路径
original_dataset_path = Path("/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/coffee-button-v3_corner")
# 合并后的数据集路径
merged_dataset_path = Path("/data/zhonglinye/jun/lerobot/personal/work2/duibi/our_v5_multi_84_seed42/merged_dataset")

# 1. 检查原始数据集
print("\n=== 1. 检查原始数据集 (coffee-button-v3_corner) ===")
try:
    original_meta = LeRobotDatasetMetadata(str(original_dataset_path))
    print(f"fps: {original_meta.fps}")
    print(f"total_episodes: {original_meta.total_episodes}")
    print(f"total_frames: {original_meta.total_frames}")
    
    # 检查 episodes 元数据
    original_episodes_file = original_dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if original_episodes_file.exists():
        original_episodes_df = pd.read_parquet(original_episodes_file)
        print(f"\nEpisodes 元数据 (前5行):")
        print(original_episodes_df[['episode_index', 'length', 'dataset_from_index', 'dataset_to_index']].head(5).to_string())
        
        # 测试原始数据集的 action_is_pad
        print(f"\n=== 测试原始数据集的 action_is_pad ===")
        delta_timestamps = {
            'action': [i / original_meta.fps for i in range(50)],
            'observation.state': [0.0],
        }
        
        original_dataset = LeRobotDataset(
            repo_id="lerobot/coffee-button-v3_corner",
            root=str(original_dataset_path),
            delta_timestamps=delta_timestamps,
        )
        
        # 检查前5个 episode 的第一个 frame
        for ep_idx in range(5):
            ep = original_episodes_df.iloc[ep_idx]
            ep_start = ep['dataset_from_index']
            item = original_dataset[ep_start]
            if 'action_is_pad' in item:
                is_pad = item['action_is_pad']
                all_true = is_pad.all().item()
                all_false = (~is_pad).all().item()
                print(f"  Episode {ep_idx} (frame {ep_start}): action_is_pad 全部为 {('True (异常!)' if all_true else 'False (正常)' if all_false else '部分 (正常)')}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# 2. 检查合并后的数据集
print(f"\n=== 2. 检查合并后的数据集 ===")
try:
    merged_meta = LeRobotDatasetMetadata(str(merged_dataset_path))
    print(f"fps: {merged_meta.fps}")
    print(f"total_episodes: {merged_meta.total_episodes}")
    print(f"total_frames: {merged_meta.total_frames}")
    
    # 检查 episodes 元数据
    merged_episodes_file = merged_dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if merged_episodes_file.exists():
        merged_episodes_df = pd.read_parquet(merged_episodes_file)
        print(f"\nEpisodes 元数据 (前5行):")
        print(merged_episodes_df[['episode_index', 'length', 'dataset_from_index', 'dataset_to_index']].head(5).to_string())
        
        # 测试合并后数据集的 action_is_pad
        print(f"\n=== 测试合并后数据集的 action_is_pad ===")
        delta_timestamps = {
            'action': [i / merged_meta.fps for i in range(50)],
            'observation.state': [0.0],
        }
        
        merged_dataset = LeRobotDataset(
            repo_id="lerobot/metaworld_pick_place",
            root=str(merged_dataset_path),
            delta_timestamps=delta_timestamps,
        )
        
        # 检查前5个 episode 的第一个 frame
        for ep_idx in range(5):
            ep = merged_episodes_df.iloc[ep_idx]
            ep_start = ep['dataset_from_index']
            item = merged_dataset[ep_start]
            if 'action_is_pad' in item:
                is_pad = item['action_is_pad']
                all_true = is_pad.all().item()
                all_false = (~is_pad).all().item()
                print(f"  Episode {ep_idx} (frame {ep_start}): action_is_pad 全部为 {('True (异常!)' if all_true else 'False (正常)' if all_false else '部分 (正常)')}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# 3. 对比关键差异
print(f"\n=== 3. 对比关键差异 ===")

# 检查 info.json 中的差异
original_info_file = original_dataset_path / "meta" / "info.json"
merged_info_file = merged_dataset_path / "meta" / "info.json"

if original_info_file.exists() and merged_info_file.exists():
    with open(original_info_file, 'r') as f:
        original_info = json.load(f)
    with open(merged_info_file, 'r') as f:
        merged_info = json.load(f)
    
    print(f"\n原始数据集 info.json:")
    print(f"  fps: {original_info.get('fps')}")
    print(f"  total_episodes: {original_info.get('total_episodes')}")
    print(f"  total_frames: {original_info.get('total_frames')}")
    print(f"  features: {list(original_info.get('features', {}).keys())}")
    
    print(f"\n合并后数据集 info.json:")
    print(f"  fps: {merged_info.get('fps')}")
    print(f"  total_episodes: {merged_info.get('total_episodes')}")
    print(f"  total_frames: {merged_info.get('total_frames')}")
    print(f"  features: {list(merged_info.get('features', {}).keys())}")

# 4. 检查 dataset_reader.py 中的 _get_query_indices 逻辑
print(f"\n=== 4. 分析 _get_query_indices 逻辑 ===")
print("根据 dataset_reader.py 第227-231行:")
print("  padding = {")
print("      f'{key}_is_pad': torch.BoolTensor(")
print("          [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]")
print("      )")
print("      for key, delta_idx in self.delta_indices.items()")
print("  }")
print()
print("这意味着 action_is_pad 为 True 当:")
print("  abs_idx + delta < ep_start  或  abs_idx + delta >= ep_end")
print()
print("对于 episode 的第一个 frame (abs_idx = ep_start):")
print("  abs_idx + delta = ep_start + delta")
print("  条件变为: ep_start + delta < ep_start (不可能) 或 ep_start + delta >= ep_end")
print("  即: delta >= ep_end - ep_start = length")
print()
print("如果所有 action_is_pad 都为 True，说明:")
print("  对于所有 delta in [0, 1, ..., 49]，都有 delta >= length")
print("  这意味着 length <= 0，或者 ep_end - ep_start 计算有误")

# 5. 检查合并脚本中的 dataset_to_index 计算
print(f"\n=== 5. 检查合并脚本中的 dataset_to_index 计算 ===")
print("查看 merge_selected_episodes.py 第163行:")
print("  'dataset_to_index': current_frame_idx + num_frames - 1,")
print()
print("这意味着 dataset_to_index 是最后一个 frame 的索引（包含）")
print("而 _get_query_indices 中使用的是:")
print("  ep_end = ep['dataset_to_index']")
print("  条件: abs_idx + delta >= ep_end")
print()
print("如果 ep_end 是最后一个 frame 的索引，那么:")
print("  对于 episode 的最后一个 frame (abs_idx = ep_end):")
print("    abs_idx + 0 = ep_end >= ep_end → True (padding)")
print("  这是错误的！最后一个 frame 不应该是 padding")
print()
print("问题可能在于: dataset_to_index 应该是最后一个 frame 的索引 + 1（exclusive）")
print("但合并脚本中设置的是最后一个 frame 的索引（inclusive）")

# 6. 验证假设
print(f"\n=== 6. 验证假设 ===")
merged_episodes_df = pd.read_parquet(merged_episodes_file)
for ep_idx in range(5):
    ep = merged_episodes_df.iloc[ep_idx]
    ep_start = ep['dataset_from_index']
    ep_end = ep['dataset_to_index']
    length = ep['length']
    
    # 检查 ep_end 是否是最后一个 frame 的索引
    expected_end_inclusive = ep_start + length - 1
    expected_end_exclusive = ep_start + length
    
    print(f"Episode {ep_idx}:")
    print(f"  dataset_from_index: {ep_start}")
    print(f"  dataset_to_index: {ep_end}")
    print(f"  length: {length}")
    print(f"  如果 to_index 是 inclusive: 应该是 {expected_end_inclusive}, 实际是 {ep_end}, {'✓' if ep_end == expected_end_inclusive else '✗'}")
    print(f"  如果 to_index 是 exclusive: 应该是 {expected_end_exclusive}, 实际是 {ep_end}, {'✓' if ep_end == expected_end_exclusive else '✗'}")

print("\n" + "=" * 60)
print("分析完成")
print("=" * 60)