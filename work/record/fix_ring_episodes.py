#!/usr/bin/env python3
"""修复 ep10/ring 数据集的 episode metadata schema 不一致问题"""

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
import shutil

episodes_dir = Path('/home/qwe/.cache/huggingface/lerobot/ep10/ring/meta/episodes')

# 备份
backup_dir = episodes_dir.parent / 'episodes_backup'
if not backup_dir.exists():
    print(f"备份原始文件到 {backup_dir}")
    shutil.copytree(episodes_dir, backup_dir)
else:
    print(f"从备份恢复原始文件...")
    shutil.rmtree(episodes_dir)
    shutil.copytree(backup_dir, episodes_dir)

parquet_files = sorted(episodes_dir.glob('**/*.parquet'))

# 需要修复的字段
int_fields = [
    'stats/frame_index/min', 'stats/frame_index/max',
    'stats/episode_index/min', 'stats/episode_index/max',
    'stats/index/min', 'stats/index/max',
    'stats/task_index/min', 'stats/task_index/max',
]

image_fields = [
    'stats/observation.images.wrist/min', 'stats/observation.images.wrist/max',
    'stats/observation.images.wrist/mean', 'stats/observation.images.wrist/std',
    'stats/observation.images.wrist/q01', 'stats/observation.images.wrist/q10',
    'stats/observation.images.wrist/q50', 'stats/observation.images.wrist/q90',
    'stats/observation.images.wrist/q99',
    'stats/observation.images.top/min', 'stats/observation.images.top/max',
    'stats/observation.images.top/mean', 'stats/observation.images.top/std',
    'stats/observation.images.top/q01', 'stats/observation.images.top/q10',
    'stats/observation.images.top/q50', 'stats/observation.images.top/q90',
    'stats/observation.images.top/q99',
]

def flatten_nested_list(val):
    """将嵌套列表展平为 1D 列表"""
    if val is None:
        return None
    if isinstance(val, (list, np.ndarray)):
        arr = np.array(val)
        return arr.flatten().tolist()
    return val

def to_int(val):
    """转换为整数"""
    if val is None:
        return None
    if isinstance(val, (np.integer, np.floating)):
        return int(val)
    if isinstance(val, (list, np.ndarray)):
        return int(np.array(val).flatten()[0])
    return int(val)

for f in parquet_files:
    print(f"\n修复文件: {f}")
    table = pq.read_table(str(f))
    
    new_columns = {}
    fixed_count = 0
    
    for col_name in int_fields:
        if col_name in table.column_names:
            col = table.column(col_name)
            # 将 list<int64> 或 list<double> 转换为 int64
            new_values = [to_int(val.as_py()) for val in col]
            new_columns[col_name] = pa.array(new_values, type=pa.int64())
            fixed_count += 1
            print(f"  修复 {col_name}: list -> int64")
    
    for col_name in image_fields:
        if col_name in table.column_names:
            col = table.column(col_name)
            # 将 list<list<list<double>>> 转换为 list<double>
            new_values = [flatten_nested_list(val.as_py()) for val in col]
            new_columns[col_name] = pa.array(new_values, type=pa.list_(pa.float64()))
            fixed_count += 1
            print(f"  修复 {col_name}: 3D list -> 1D list")
    
    # 构建新 table
    new_table = table
    for col_name, new_col in new_columns.items():
        idx = new_table.column_names.index(col_name)
        new_table = new_table.set_column(idx, col_name, new_col)
    
    print(f"  共修复 {fixed_count} 个字段")
    
    # 写入修复后的文件
    pq.write_table(new_table, str(f))
    print(f"  已写入修复后的文件")

print("\n" + "="*60)
print("修复完成!")
print("="*60)

# 验证修复后的 schema
print("\n验证修复后的 schema:")
for f in parquet_files:
    print(f"\n文件: {f}")
    pf = pq.ParquetFile(str(f))
    schema = pf.schema_arrow
    for field in schema:
        if 'observation.images' in field.name or 'frame_index' in field.name or 'episode_index' in field.name or 'index' in field.name or 'task_index' in field.name:
            print(f'  {field.name}: {field.type}')