#!/usr/bin/env python3
"""
检查 preprocessor 是否正确归一化了 action。
验证训练时 action 归一化是否正常工作。
"""

import sys
sys.path.insert(0, '/data/zhonglinye/jun/lerobot/src')

import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.smolvla import make_smolvla_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

# 当前使用的数据集路径
dataset_path = Path("/data/zhonglinye/jun/lerobot/personal/work2/duibi/our_v5_multi_84_seed42/merged_dataset")

print("=" * 60)
print("检查 preprocessor 的 action 归一化")
print("=" * 60)

# 加载数据集元数据
meta = LeRobotDatasetMetadata(str(dataset_path))
print(f"\nDataset stats (action):")
print(f"  mean: {meta.stats['action'].get('mean')}")
print(f"  std: {meta.stats['action'].get('std')}")

# 创建 smolvla 配置
config = SmolVLAConfig(
    n_obs_steps=1,
    chunk_size=50,
    n_action_steps=50,
    device="cpu",
)

# 使用 make_smolvla_pre_post_processors 创建 preprocessor（这是训练时实际使用的方式）
print(f"\n正在使用 make_smolvla_pre_post_processors 创建 preprocessor...")
print(f"  dataset_stats: {list(meta.stats.keys())}")

preprocessor, postprocessor = make_smolvla_pre_post_processors(
    config=config,
    dataset_stats=meta.stats,
)

print(f"\nPreprocessor steps:")
for i, step in enumerate(preprocessor.steps):
    step_type = type(step).__name__
    print(f"  {i}: {step_type}")
    
    # 检查 normalizer_processor 的 stats
    if step_type == "NormalizerProcessorStep":
        print(f"      stats keys: {list(step.stats.keys()) if step.stats else 'None'}")
        if step.stats and 'action' in step.stats:
            print(f"      action stats:")
            print(f"        mean: {step.stats['action'].get('mean')}")
            print(f"        std: {step.stats['action'].get('std')}")

# 加载数据集
delta_timestamps = {
    'action': [i / meta.fps for i in range(50)],
    'observation.state': [0.0],
}

dataset = LeRobotDataset(
    repo_id="lerobot/metaworld_pick_place",
    root=str(dataset_path),
    delta_timestamps=delta_timestamps,
)

# 获取一个 sample
item = dataset[0]

# 添加 batch 维度
batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in item.items()}

print(f"\n=== 预处理前的 action ===")
print(f"  shape: {batch['action'].shape}")
print(f"  mean: {batch['action'].mean().item():.4f}")
print(f"  std: {batch['action'].std().item():.4f}")
print(f"  min: {batch['action'].min().item():.4f}")
print(f"  max: {batch['action'].max().item():.4f}")

# 应用 preprocessor
processed_batch = preprocessor(batch)

print(f"\n=== 预处理后的 action ===")
if 'action' in processed_batch:
    action = processed_batch['action']
    print(f"  shape: {action.shape}")
    print(f"  mean: {action.mean().item():.4f}")
    print(f"  std: {action.std().item():.4f}")
    print(f"  min: {action.min().item():.4f}")
    print(f"  max: {action.max().item():.4f}")
    
    # 检查归一化是否正确
    if action.std().item() < 0.1 or action.std().item() > 10:
        print(f"\n  ⚠️ 警告：归一化后的 std 异常 ({action.std().item():.4f})")
    if action.mean().item() > 5 or action.mean().item() < -5:
        print(f"\n  ⚠️ 警告：归一化后的 mean 异常 ({action.mean().item():.4f})")
else:
    print("  action 不在 processed_batch 中！")
    print(f"  keys: {processed_batch.keys()}")

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)