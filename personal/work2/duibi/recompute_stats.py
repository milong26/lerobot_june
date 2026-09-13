#!/usr/bin/env python3
"""
重新计算合并后数据集的 stats.json。
"""

import sys
sys.path.insert(0, '/data/zhonglinye/jun/lerobot/src')

import json
import numpy as np
import pandas as pd
from pathlib import Path

# 合并后的数据集路径
dataset_path = Path("/data/zhonglinye/jun/lerobot/personal/work2/duibi/our_v5_multi_84_seed42/merged_dataset_fixed2")

print("=" * 60)
print("重新计算合并后数据集的 stats.json")
print("=" * 60)

# 加载 data parquet
data_file = dataset_path / "data" / "chunk-000" / "file-000.parquet"
data_df = pd.read_parquet(data_file)

print(f"总帧数: {len(data_df)}")
print(f"列名: {list(data_df.columns)}")

# 计算每个特征的统计信息
stats = {}

# 需要计算统计信息的特征
features_to_compute = ['action', 'observation.state', 'observation.environment_state', 'next.reward', 'next.success']

for feature in features_to_compute:
    if feature not in data_df.columns:
        print(f"跳过 {feature}（不存在）")
        continue
    
    values = np.stack(data_df[feature].values)
    
    # 检查是否是布尔类型
    is_bool = values.dtype == bool
    
    # 计算统计信息
    feature_stats = {
        'min': values.min(axis=0).tolist(),
        'max': values.max(axis=0).tolist(),
        'mean': values.mean(axis=0).tolist(),
        'std': values.std(axis=0).tolist(),
        'count': [len(values)],
    }
    
    # 布尔类型不计算分位数
    if not is_bool:
        feature_stats.update({
            'q01': np.percentile(values, 1, axis=0).tolist(),
            'q10': np.percentile(values, 10, axis=0).tolist(),
            'q50': np.percentile(values, 50, axis=0).tolist(),
            'q90': np.percentile(values, 90, axis=0).tolist(),
            'q99': np.percentile(values, 99, axis=0).tolist(),
        })
    
    stats[feature] = feature_stats
    print(f"\n{feature}:")
    print(f"  mean: {feature_stats['mean']}")
    print(f"  std: {feature_stats['std']}")
    print(f"  min: {feature_stats['min']}")
    print(f"  max: {feature_stats['max']}")

# 对于图像，保持原来的统计信息（因为视频是 symlink 的）
old_stats_file = dataset_path / "meta" / "stats.json"
with open(old_stats_file, 'r') as f:
    old_stats = json.load(f)

# 保留图像的统计信息
for key in old_stats:
    if 'images' in key or 'video' in key:
        stats[key] = old_stats[key]

# 保存新的 stats.json
new_stats_file = dataset_path / "meta" / "stats.json"
with open(new_stats_file, 'w') as f:
    json.dump(stats, f, indent=4)

print(f"\n✓ 新的 stats.json 已保存到: {new_stats_file}")
print("\n" + "=" * 60)
print("完成")
print("=" * 60)