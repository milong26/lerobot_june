#!/usr/bin/env python3
"""
检查合并后的数据集中 action_is_pad 的生成情况。
模拟训练时的数据加载，使用 delta_timestamps 来检查 action_is_pad。
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

# merged_dataset_path = Path("/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place-v3_corner")

print("=" * 60)
print("检查 action_is_pad 生成情况（模拟训练时配置）")
print("=" * 60)

# 加载数据集元数据
ds_meta = LeRobotDatasetMetadata(str(merged_dataset_path))
print(f"数据集 fps: {ds_meta.fps}")
print(f"总 episode 数: {ds_meta.total_episodes}")
print(f"总 frame 数: {ds_meta.total_frames}")

# 查看训练配置中的 chunk_size
# 从日志中可以看到 smolvla 使用 chunk_size 来配置 action 窗口
# 需要查看 policy 配置
print("\n=== 检查训练配置中的 action_delta_indices ===")

# 从日志文件中提取配置
log_file = Path("/data/zhonglinye/jun/lerobot/personal/work2/duibi/our_v5_multi_84_seed42/logs/our_v5_multi_84_seed42.log")
if log_file.exists():
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # 查找 chunk_size 或 action_delta_indices
    import re
    chunk_size_match = re.search(r'chunk_size.*?(\d+)', log_content)
    if chunk_size_match:
        print(f"找到 chunk_size: {chunk_size_match.group(1)}")
    
    # 查找 action_delta_indices
    action_delta_match = re.search(r'action_delta_indices.*?\[([^\]]+)\]', log_content)
    if action_delta_match:
        print(f"找到 action_delta_indices: [{action_delta_match.group(1)}]")
    else:
        print("未找到 action_delta_indices，使用默认配置")

# 根据 smolvla 的常见配置，chunk_size 通常是 50 或 64
# 假设使用 chunk_size=50，action_delta_indices 可能是 [0, 1, 2, ..., 49]
# 但更常见的是使用 [0, 1, 2, ..., chunk_size-1]

# 让我先尝试不配置 delta_timestamps，看看原始数据
print("\n=== 测试1: 不使用 delta_timestamps（原始数据） ===")
try:
    dataset_no_delta = LeRobotDataset(
        repo_id="lerobot/metaworld_pick_place",
        root=str(merged_dataset_path),
        delta_timestamps=None,
    )
    print(f"数据集加载成功，总 frame 数: {len(dataset_no_delta)}")
    
    # 获取第一个 item
    item0 = dataset_no_delta[0]
    print(f"Item 0 的 keys: {list(item0.keys())}")
    if 'action' in item0:
        print(f"  action shape: {item0['action'].shape}")
    if 'action_is_pad' in item0:
        print(f"  action_is_pad: {item0['action_is_pad']}")
    else:
        print("  action_is_pad: 不存在（这是正常的，因为没有配置 delta_timestamps）")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# 现在模拟训练时的配置
# 从日志中查找实际的配置
print("\n=== 测试2: 使用 delta_timestamps 模拟训练配置 ===")

# 需要找到训练时使用的 action_delta_indices
# 让我查看 policy 配置文件
policy_config_path = Path("/data/zhonglinye/hfdata/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205/config.json")
if policy_config_path.exists():
    with open(policy_config_path, 'r') as f:
        policy_config = json.load(f)
    print(f"Policy 配置加载成功")
    print(f"  chunk_size: {policy_config.get('chunk_size', 'N/A')}")
    print(f"  n_obs_steps: {policy_config.get('n_obs_steps', 'N/A')}")
    print(f"  n_action_steps: {policy_config.get('n_action_steps', 'N/A')}")
    
    chunk_size = policy_config.get('chunk_size', 50)
    
    # 构建 delta_timestamps
    # action_delta_indices 通常是 [0, 1, 2, ..., chunk_size-1]
    action_delta_indices = list(range(chunk_size))
    
    # observation_delta_indices 通常是 [0] 或 [-1, 0]
    observation_delta_indices = [0]
    
    delta_timestamps = {
        'action': [i / ds_meta.fps for i in action_delta_indices],
        'observation.state': [i / ds_meta.fps for i in observation_delta_indices],
    }
    
    print(f"\n使用 delta_timestamps:")
    print(f"  action: {len(delta_timestamps['action'])} 个时间戳，范围 [0, {delta_timestamps['action'][-1]:.3f}]")
    print(f"  observation.state: {delta_timestamps['observation.state']}")
    
    try:
        dataset_with_delta = LeRobotDataset(
            repo_id="lerobot/metaworld_pick_place",
            root=str(merged_dataset_path),
            delta_timestamps=delta_timestamps,
        )
        print(f"\n数据集加载成功，总 frame 数: {len(dataset_with_delta)}")
        
        # 检查多个 item 的 action_is_pad
        print(f"\n=== 检查 action_is_pad 生成情况 ===")
        
        # 检查每个 episode 的第一个和最后一个 frame
        episodes_meta = pd.read_parquet(merged_dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        
        action_is_pad_stats = {
            'all_true': 0,
            'all_false': 0,
            'partial': 0,
            'total': 0,
        }
        
        # 检查前 10 个 episode
        for ep_idx in range(min(10, len(episodes_meta))):
            ep = episodes_meta.iloc[ep_idx]
            ep_start = ep['dataset_from_index']
            ep_length = ep['length']
            
            # 检查 episode 的第一个 frame
            item_first = dataset_with_delta[ep_start]
            if 'action_is_pad' in item_first:
                is_pad = item_first['action_is_pad']
                all_true = is_pad.all().item()
                all_false = (~is_pad).all().item()
                
                if all_true:
                    action_is_pad_stats['all_true'] += 1
                    print(f"  Episode {ep_idx} (frame {ep_start}): action_is_pad 全部为 True (异常!)")
                elif all_false:
                    action_is_pad_stats['all_false'] += 1
                    print(f"  Episode {ep_idx} (frame {ep_start}): action_is_pad 全部为 False (正常)")
                else:
                    action_is_pad_stats['partial'] += 1
                    print(f"  Episode {ep_idx} (frame {ep_start}): action_is_pad 部分为 True (正常，padding)")
            
            action_is_pad_stats['total'] += 1
        
        print(f"\n=== action_is_pad 统计 ===")
        print(f"  总检查数: {action_is_pad_stats['total']}")
        print(f"  全部为 True (异常): {action_is_pad_stats['all_true']}")
        print(f"  全部为 False (正常): {action_is_pad_stats['all_false']}")
        print(f"  部分为 True (正常): {action_is_pad_stats['partial']}")
        
        # 特别检查那些可能异常的 episode
        print(f"\n=== 详细检查所有 episode 的第一个 frame ===")
        abnormal_episodes = []
        for ep_idx in range(len(episodes_meta)):
            ep = episodes_meta.iloc[ep_idx]
            ep_start = ep['dataset_from_index']
            
            item = dataset_with_delta[ep_start]
            if 'action_is_pad' in item:
                is_pad = item['action_is_pad']
                if is_pad.all().item():
                    abnormal_episodes.append(ep_idx)
        
        if abnormal_episodes:
            print(f"发现 {len(abnormal_episodes)} 个异常 episode (action_is_pad 全部为 True):")
            print(f"  Episode 索引: {abnormal_episodes}")
            
            # 检查这些 episode 的特征
            for ep_idx in abnormal_episodes[:5]:  # 只显示前5个
                ep = episodes_meta.iloc[ep_idx]
                print(f"\n  Episode {ep_idx} 详情:")
                print(f"    length: {ep['length']}")
                print(f"    dataset_from_index: {ep['dataset_from_index']}")
                print(f"    dataset_to_index: {ep['dataset_to_index']}")
        else:
            print("所有 episode 的 action_is_pad 都正常!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)