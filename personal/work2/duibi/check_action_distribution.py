#!/usr/bin/env python3
"""
检查合并数据集的 action 值分布，查找异常值。
"""

import sys
sys.path.insert(0, '/data/zhonglinye/jun/lerobot/src')

import numpy as np
import pandas as pd
from pathlib import Path

# 当前使用的数据集路径
dataset_path = Path("/data/zhonglinye/jun/lerobot/personal/work2/duibi/our_v5_multi_84_seed42/merged_dataset")

print("=" * 60)
print("检查合并数据集的 action 值分布")
print("=" * 60)

# 加载 data.parquet
data_file = dataset_path / "data" / "data.parquet"
data_df = pd.read_parquet(data_file)

print(f"总帧数: {len(data_df)}")
print(f"列名: {list(data_df.columns)}")

# 检查 action 列
if 'action' in data_df.columns:
    actions = np.stack(data_df['action'].values)
    print(f"\nAction shape: {actions.shape}")
    print(f"Action dtype: {actions.dtype}")
    
    # 检查每个维度的统计信息
    for i in range(actions.shape[1]):
        dim_values = actions[:, i]
        print(f"\nAction dim {i}:")
        print(f"  mean: {dim_values.mean():.6f}")
        print(f"  std: {dim_values.std():.6f}")
        print(f"  min: {dim_values.min():.6f}")
        print(f"  max: {dim_values.max():.6f}")
        print(f"  q01: {np.percentile(dim_values, 1):.6f}")
        print(f"  q99: {np.percentile(dim_values, 99):.6f}")
        
        # 检查是否有 NaN 或 Inf
        nan_count = np.isnan(dim_values).sum()
        inf_count = np.isinf(dim_values).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  ⚠️ NaN count: {nan_count}, Inf count: {inf_count}")
        
        # 检查是否有异常大的值
        large_values = np.abs(dim_values) > 100
        if large_values.any():
            print(f"  ⚠️ 异常大值 count (|x| > 100): {large_values.sum()}")
            print(f"  异常值示例: {dim_values[large_values][:10]}")
else:
    print("action 列不存在！")

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)