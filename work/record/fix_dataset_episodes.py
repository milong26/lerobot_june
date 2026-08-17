#!/usr/bin/env python3
"""通用修复 LeRobot 数据集 episode metadata schema 不一致问题"""

import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
import shutil

def fix_dataset(repo_id, root=None, dry_run=False):
    from lerobot.utils.constants import HF_LEROBOT_HOME
    
    root_path = Path(root) if root else HF_LEROBOT_HOME / repo_id
    episodes_dir = root_path / 'meta' / 'episodes'
    
    if not episodes_dir.exists():
        print(f"错误: 找不到 episodes 目录 {episodes_dir}")
        return
    
    # 备份
    backup_dir = episodes_dir.parent / 'episodes_backup'
    if not dry_run:
        if not backup_dir.exists():
            print(f"备份原始文件到 {backup_dir}")
            shutil.copytree(episodes_dir, backup_dir)
        else:
            print(f"从备份恢复原始文件...")
            shutil.rmtree(episodes_dir)
            shutil.copytree(backup_dir, episodes_dir)
    
    parquet_files = sorted(episodes_dir.glob('**/*.parquet'))
    print(f"找到 {len(parquet_files)} 个 episode parquet 文件")
    
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
        'stats/observation.images.side/min', 'stats/observation.images.side/max',
        'stats/observation.images.side/mean', 'stats/observation.images.side/std',
        'stats/observation.images.side/q01', 'stats/observation.images.side/q10',
        'stats/observation.images.side/q50', 'stats/observation.images.side/q90',
        'stats/observation.images.side/q99',
        'stats/observation.images.side_depth/min', 'stats/observation.images.side_depth/max',
        'stats/observation.images.side_depth/mean', 'stats/observation.images.side_depth/std',
        'stats/observation.images.side_depth/q01', 'stats/observation.images.side_depth/q10',
        'stats/observation.images.side_depth/q50', 'stats/observation.images.side_depth/q90',
        'stats/observation.images.side_depth/q99',
    ]
    
    def flatten_nested_list(val):
        if val is None:
            return None
        if isinstance(val, (list, np.ndarray)):
            arr = np.array(val)
            return arr.flatten().tolist()
        return val
    
    def to_int(val):
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
                new_values = [to_int(val.as_py()) for val in col]
                new_columns[col_name] = pa.array(new_values, type=pa.int64())
                fixed_count += 1
                print(f"  修复 {col_name}: list -> int64")
        
        for col_name in image_fields:
            if col_name in table.column_names:
                col = table.column(col_name)
                new_values = [flatten_nested_list(val.as_py()) for val in col]
                new_columns[col_name] = pa.array(new_values, type=pa.list_(pa.float64()))
                fixed_count += 1
                print(f"  修复 {col_name}: 3D list -> 1D list")
        
        if dry_run:
            print(f"  (dry run) 将修复 {fixed_count} 个字段")
            continue
        
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
    if dry_run:
        print("Dry Run 完成! 没有实际修改文件")
    else:
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

def main():
    parser = argparse.ArgumentParser(description="修复 LeRobot 数据集 episode metadata schema 不一致问题")
    parser.add_argument("--repo_id", type=str, required=True, help="数据集 repository ID")
    parser.add_argument("--root", type=str, default=None, help="数据集根目录")
    parser.add_argument("--dry_run", action="store_true", help="只检查不修复")
    args = parser.parse_args()
    
    fix_dataset(args.repo_id, args.root, args.dry_run)

if __name__ == "__main__":
    main()