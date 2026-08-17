#!/usr/bin/env python3
"""
修复 ring 数据集中 file-002.parquet 的 schema 不一致问题。

问题：
1. 图像 stats 字段被保存为 3D 嵌套列表，应该是 1D 列表
2. 整数字段 (frame_index, episode_index, index, task_index) 的 min/max 被保存为 double，应该是 int64
"""

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from pathlib import Path
import shutil

EPISODES_DIR = Path('/home/qwe/.cache/huggingface/lerobot/ep10/ring/meta/episodes')

# 需要修复的字段
IMAGE_STATS_FIELDS = [
    'stats/observation.images.wrist/min',
    'stats/observation.images.wrist/max',
    'stats/observation.images.wrist/mean',
    'stats/observation.images.wrist/std',
    'stats/observation.images.wrist/count',
    'stats/observation.images.wrist/q01',
    'stats/observation.images.wrist/q10',
    'stats/observation.images.wrist/q50',
    'stats/observation.images.wrist/q90',
    'stats/observation.images.wrist/q99',
    'stats/observation.images.top/min',
    'stats/observation.images.top/max',
    'stats/observation.images.top/mean',
    'stats/observation.images.top/std',
    'stats/observation.images.top/count',
    'stats/observation.images.top/q01',
    'stats/observation.images.top/q10',
    'stats/observation.images.top/q50',
    'stats/observation.images.top/q90',
    'stats/observation.images.top/q99',
]

INTEGER_FIELDS = [
    'stats/frame_index/min',
    'stats/frame_index/max',
    'stats/episode_index/min',
    'stats/episode_index/max',
    'stats/index/min',
    'stats/index/max',
    'stats/task_index/min',
    'stats/task_index/max',
]


def flatten_nested_list(value):
    """将嵌套的多层列表展平为 1D 列表。"""
    if isinstance(value, (list, np.ndarray)):
        result = []
        for item in value:
            result.extend(flatten_nested_list(item))
        return result
    else:
        return [value]


def fix_parquet_file(file_path: Path, output_path: Path):
    """修复单个 parquet 文件。"""
    print(f"\n修复文件: {file_path}")
    
    # 读取原始数据
    table = pq.read_table(str(file_path))
    
    print(f"  原始行数: {table.num_rows}")
    print(f"  原始列数: {table.num_columns}")
    
    # 构建新的列数据
    new_columns = {}
    fixed_count = 0
    
    for col_name in table.column_names:
        col_data = table.column(col_name)
        
        if col_name in IMAGE_STATS_FIELDS:
            # 检查是否需要展平
            sample = col_data[0].as_py()
            if isinstance(sample, list):
                # 检查嵌套深度
                depth = 0
                temp = sample
                while isinstance(temp, list) and len(temp) > 0:
                    depth += 1
                    temp = temp[0]
                
                if depth > 1:
                    print(f"  修复 {col_name}: 嵌套深度 {depth} -> 1")
                    # 展平每个值
                    flattened = []
                    for i in range(len(col_data)):
                        val = col_data[i].as_py()
                        flat_val = flatten_nested_list(val)
                        flattened.append(flat_val)
                    
                    # 创建正确的 list<double> 类型
                    new_columns[col_name] = pa.array(flattened, type=pa.list_(pa.float64()))
                    fixed_count += 1
                else:
                    # 已经是 1D，保持原样
                    new_columns[col_name] = col_data
            else:
                new_columns[col_name] = col_data
        
        elif col_name in INTEGER_FIELDS:
            # 检查是否是 float64 类型
            sample = col_data[0].as_py()
            if isinstance(sample, list):
                if len(sample) > 0 and isinstance(sample[0], float):
                    print(f"  修复 {col_name}: float64 -> int64")
                    # 转换为 int64
                    int_values = []
                    for i in range(len(col_data)):
                        val = col_data[i].as_py()
                        int_val = [int(x) for x in val]
                        int_values.append(int_val)
                    
                    new_columns[col_name] = pa.array(int_values, type=pa.list_(pa.int64()))
                    fixed_count += 1
                else:
                    new_columns[col_name] = col_data
            else:
                new_columns[col_name] = col_data
        else:
            # 其他字段保持原样
            new_columns[col_name] = col_data
    
    print(f"  修复了 {fixed_count} 个字段")
    
    # 构建新的 table
    new_arrays = []
    new_fields = []
    
    for col_name in table.column_names:
        arr = new_columns[col_name]
        new_arrays.append(arr)
        new_fields.append(pa.field(col_name, arr.type))
    
    new_schema = pa.schema(new_fields)
    new_table = pa.Table.from_arrays(new_arrays, schema=new_schema)
    
    # 写入修复后的数据
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, str(output_path))
    print(f"  写入修复后的文件: {output_path}")
    
    return fixed_count


def main():
    print("="*60)
    print("修复 rectangular 数据集的 episode metadata")
    print("="*60)
    
    # 从备份恢复原始文件
    backup_dir = EPISODES_DIR.with_name(EPISODES_DIR.name + "_backup")
    if backup_dir.exists():
        print(f"\n从备份恢复原始文件...")
        if EPISODES_DIR.exists():
            shutil.rmtree(EPISODES_DIR)
        shutil.copytree(backup_dir, EPISODES_DIR)
        print(f"  已恢复")
    else:
        print(f"\n备份不存在，直接修复当前文件")
        # 创建备份
        if EPISODES_DIR.exists():
            backup_dir = EPISODES_DIR.with_name(EPISODES_DIR.name + "_backup2")
            if not backup_dir.exists():
                shutil.copytree(EPISODES_DIR, backup_dir)
    
    # 修复 file-002.parquet
    file_002 = EPISODES_DIR / "chunk-000" / "file-002.parquet"
    if file_002.exists():
        fix_parquet_file(file_002, file_002)
    else:
        print(f"文件不存在: {file_002}")
        return
    
    print("\n" + "="*60)
    print("修复完成!")
    print("="*60)
    
    # 验证修复后的 schema
    print("\n验证修复后的 schema:")
    for f in sorted(EPISODES_DIR.glob('**/*.parquet')):
        print(f"\n文件: {f}")
        pf = pq.ParquetFile(str(f))
        schema = pf.schema_arrow
        # 只打印图像 stats 和整数字段
        for field in schema:
            if field.name in IMAGE_STATS_FIELDS or field.name in INTEGER_FIELDS:
                print(f"  {field.name}: {field.type}")


if __name__ == "__main__":
    main()