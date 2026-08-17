#!/usr/bin/env python3
"""综合检查和修复 LeRobot 数据集的工具。

这个脚本会：
1. 检查所有 episode 的统计字段类型是否一致
2. 检查 episode metadata 是否正确引用了 parquet 文件
3. 自动修复发现的问题
4. 打印详细的检查和修复报告

使用方法:
    # 只检查不修复
    python check_and_fix_dataset.py --repo_id ep10/charger_old --verbose
    
    # 检查并修复到新目录
    python check_and_fix_dataset.py --repo_id ep10/charger_old \
        --new_repo_id ep10/charger_fixed \
        --new_root /home/qwe/.cache/huggingface/lerobot/ep10/charger_fixed \
        --fix
    
    # 检查并原地修复（会创建备份）
    python check_and_fix_dataset.py --repo_id ep10/charger_old --fix --in_place
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 添加 lerobot 到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata, DEFAULT_EPISODES_PATH
from lerobot.datasets.io_utils import load_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_DATA_PATH, EPISODES_DIR
from lerobot.utils.constants import HF_LEROBOT_HOME


def load_episode_parquet_data(meta: LeRobotDatasetMetadata, episode_idx: int) -> dict:
    """从 parquet 文件加载单个 episode 的完整数据（包含 stats）。"""
    ep_meta = meta.episodes[episode_idx]
    chunk_idx = ep_meta["meta/episodes/chunk_index"]
    file_idx = ep_meta["meta/episodes/file_index"]

    parquet_path = meta.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    df = pd.read_parquet(parquet_path)

    episode_row = df[df["episode_index"] == episode_idx].iloc[0]
    return episode_row.to_dict()


def check_stats_consistency(repo_id: str, root: str = None, verbose: bool = False) -> dict:
    """检查数据集所有 episode 的统计字段类型一致性。
    
    Returns:
        dict: 检查结果，包含 inconsistent_fields 和 episode_type_map
    """
    print(f"\n{'='*60}")
    print("检查 1: 统计字段类型一致性")
    print(f"{'='*60}")
    
    try:
        meta = LeRobotDatasetMetadata(repo_id, root=root)
    except Exception as e:
        print(f"错误: 无法加载数据集 {repo_id}")
        print(f"详细信息: {e}")
        return {"valid": False, "error": str(e)}
    
    if meta.episodes is None or len(meta.episodes) == 0:
        print("警告: 没有找到 episode 数据")
        return {"valid": False, "error": "No episodes found"}
    
    print(f"数据集: {repo_id}")
    print(f"总 episode 数: {meta.info.total_episodes}")
    print(f"总帧数: {meta.info.total_frames}")
    print()
    
    # 收集所有 episode 的 stats 字段类型
    all_stats_fields = set()
    episode_field_types = {}  # {ep_idx: {field_name: type_str}}
    episode_parquet_data = {}
    
    print("正在加载所有 episode 的 parquet 数据...")
    for idx in range(len(meta.episodes)):
        try:
            episode_data = load_episode_parquet_data(meta, idx)
            episode_parquet_data[idx] = episode_data
            
            field_types = {}
            for key in episode_data.keys():
                if key.startswith("stats/"):
                    all_stats_fields.add(key)
                    value = episode_data[key]
                    if isinstance(value, np.ndarray):
                        field_types[key] = f"ndarray(shape={value.shape}, dtype={value.dtype})"
                    else:
                        field_types[key] = type(value).__name__
            
            episode_field_types[idx] = field_types
            
            if verbose:
                print(f"  Episode {idx}: 加载成功，{len(field_types)} 个 stats 字段")
        except Exception as e:
            print(f"  Episode {idx}: 加载失败 - {e}")
            episode_field_types[idx] = {}
    
    # 检查类型一致性
    field_type_map = {}  # {field_name: {type_str: [ep_indices]}}
    for ep_idx, field_types in episode_field_types.items():
        for field_name, type_str in field_types.items():
            if field_name not in field_type_map:
                field_type_map[field_name] = {}
            if type_str not in field_type_map[field_name]:
                field_type_map[field_name][type_str] = []
            field_type_map[field_name][type_str].append(ep_idx)
    
    # 找出不一致的字段
    inconsistent_fields = {}
    for field_name, types in field_type_map.items():
        if len(types) > 1:
            inconsistent_fields[field_name] = types
    
    # 打印结果
    print(f"\n发现 {len(all_stats_fields)} 个统计字段")
    
    if inconsistent_fields:
        print(f"\n✗ 发现 {len(inconsistent_fields)} 个字段存在类型不一致:")
        for field_name, types in sorted(inconsistent_fields.items()):
            print(f"\n  {field_name}:")
            for type_str, indices in sorted(types.items()):
                ep_range = f"{min(indices)}-{max(indices)}" if len(indices) > 1 else str(indices[0])
                print(f"    - {type_str}: episodes {ep_range} ({len(indices)} episodes)")
        
        return {
            "valid": False,
            "inconsistent_fields": inconsistent_fields,
            "field_type_map": field_type_map,
            "episode_field_types": episode_field_types,
            "episode_parquet_data": episode_parquet_data,
        }
    else:
        print("\n✓ 所有统计字段类型一致")
        return {
            "valid": True,
            "inconsistent_fields": {},
            "field_type_map": field_type_map,
            "episode_field_types": episode_field_types,
            "episode_parquet_data": episode_parquet_data,
        }


def check_episode_metadata(dataset: LeRobotDataset) -> dict:
    """检查 episode metadata 是否正确引用了 parquet 文件。
    
    Returns:
        dict: 检查结果
    """
    print(f"\n{'='*60}")
    print("检查 2: Episode Metadata 文件引用")
    print(f"{'='*60}")
    
    root = dataset.root
    episodes_meta = dataset.meta.episodes
    
    issues = {
        "total_episodes": len(episodes_meta),
        "missing_episode_files": [],
        "missing_data_files": [],
        "is_valid": True,
    }
    
    print(f"检查 {issues['total_episodes']} 个 episode 的文件引用...")
    
    for ep_idx in range(issues['total_episodes']):
        ep_meta = episodes_meta[ep_idx]
        
        episode_chunk = ep_meta.get("meta/episodes/chunk_index")
        episode_file = ep_meta.get("meta/episodes/file_index")
        data_chunk = ep_meta.get("data/chunk_index")
        data_file = ep_meta.get("data/file_index")
        
        if episode_chunk is not None and episode_file is not None:
            episode_path = root / DEFAULT_EPISODES_PATH.format(
                chunk_index=episode_chunk, file_index=episode_file
            )
            if not episode_path.exists():
                issues["missing_episode_files"].append({
                    "episode_index": ep_idx,
                    "expected_path": str(episode_path),
                    "chunk_index": episode_chunk,
                    "file_index": episode_file,
                })
                issues["is_valid"] = False
                print(f"  ✗ Episode {ep_idx}: 缺少 episode metadata 文件 {episode_path}")
        
        if data_chunk is not None and data_file is not None:
            data_path = root / DEFAULT_DATA_PATH.format(
                chunk_index=data_chunk, file_index=data_file
            )
            if not data_path.exists():
                issues["missing_data_files"].append({
                    "episode_index": ep_idx,
                    "expected_path": str(data_path),
                    "chunk_index": data_chunk,
                    "file_index": data_file,
                })
                issues["is_valid"] = False
                print(f"  ✗ Episode {ep_idx}: 缺少 data 文件 {data_path}")
    
    if issues["is_valid"]:
        print("  ✓ 所有 episode metadata 文件引用正确")
    
    return issues


def fix_stats_types(
    source_meta: LeRobotDatasetMetadata,
    output_dir: Path,
    verbose: bool = False,
) -> dict:
    """修复统计字段类型不一致的问题。
    
    策略：
    1. 对于整数类型字段（episode_index, frame_index, index, task_index）的 min/max，统一为 int64
    2. 对于其他字段，统一为 float64
    3. 保持原始数据集的文件结构，按照原始 chunk/file 索引分组写入
    
    Returns:
        dict: 修复结果
    """
    print(f"\n{'='*60}")
    print("修复: 统计字段类型")
    print(f"{'='*60}")
    
    if source_meta.episodes is None or len(source_meta.episodes) == 0:
        print("错误: 没有 episode 数据可修复")
        return {"success": False, "error": "No episodes found"}
    
    # 加载所有 episode 数据，并记录原始的 chunk/file 索引
    episode_data_list = []
    for idx in range(len(source_meta.episodes)):
        try:
            ep_data = load_episode_parquet_data(source_meta, idx)
            # 记录原始的 meta/episodes 索引
            orig_chunk = ep_data.get("meta/episodes/chunk_index")
            orig_file = ep_data.get("meta/episodes/file_index")
            # 如果是数组，提取第一个值
            if isinstance(orig_chunk, np.ndarray):
                orig_chunk = int(orig_chunk[0])
            if isinstance(orig_file, np.ndarray):
                orig_file = int(orig_file[0])
            
            episode_data_list.append({
                "idx": idx,
                "data": ep_data,
                "chunk_idx": orig_chunk,
                "file_idx": orig_file,
            })
        except Exception as e:
            print(f"  ✗ Episode {idx}: 加载失败 - {e}")
            return {"success": False, "error": f"Failed to load episode {idx}: {e}"}
    
    print(f"  加载了 {len(episode_data_list)} 个 episode 的数据")
    
    # 确定每个字段应该使用的目标类型
    # 规则：整数索引字段的 min/max 使用 int64，其他使用 float64
    integer_fields = {"episode_index", "frame_index", "index", "task_index"}
    
    def get_target_dtype(field_name: str, stat_name: str) -> np.dtype:
        """确定字段的目标数据类型。"""
        base_field = field_name.replace("stats/", "").split("/")[0]
        
        if base_field in integer_fields and stat_name in ["min", "max", "count"]:
            return np.int64
        elif stat_name == "count":
            return np.int64
        else:
            return np.float64
    
    # 修复每个 episode 的数据
    fixed_count = 0
    for ep_info in episode_data_list:
        ep_data = ep_info["data"]
        modified = False
        for key in list(ep_data.keys()):
            if not key.startswith("stats/"):
                continue
            
            value = ep_data[key]
            if not isinstance(value, np.ndarray):
                continue
            
            stat_name = key.split("/")[-1]
            target_dtype = get_target_dtype(key, stat_name)
            
            if value.dtype != target_dtype:
                if verbose:
                    print(f"  Episode {ep_info['idx']}: {key} {value.dtype} -> {target_dtype}")
                ep_data[key] = value.astype(target_dtype)
                modified = True
        
        if modified:
            fixed_count += 1
    
    print(f"  修复了 {fixed_count}/{len(episode_data_list)} 个 episode 的类型问题")
    
    # 写入修复后的数据
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制原始数据集
    print(f"  复制数据集到 {output_dir}...")
    if output_dir.exists():
        backup_path = output_dir.with_name(output_dir.name + "_backup")
        print(f"  备份已存在的目录到 {backup_path}")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.move(output_dir, backup_path)
    
    shutil.copytree(source_meta.root, output_dir)
    
    # 重写 episode parquet 文件，保持原始的文件结构
    episodes_dir = output_dir / EPISODES_DIR
    if episodes_dir.exists():
        shutil.rmtree(episodes_dir)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    
    # 按照 chunk/file 索引分组
    file_groups = {}  # {(chunk_idx, file_idx): [episode_records]}
    for ep_info in episode_data_list:
        key = (ep_info["chunk_idx"], ep_info["file_idx"])
        if key not in file_groups:
            file_groups[key] = []
        file_groups[key].append(ep_info["data"])
    
    # 写入每个文件
    total_written = 0
    for (chunk_idx, file_idx), records in sorted(file_groups.items()):
        chunk_dir = episodes_dir / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        file_path = chunk_dir / f"file-{file_idx:03d}.parquet"
        
        df = pd.DataFrame(records)
        df.to_parquet(file_path, index=False)
        
        print(f"  写入 {len(records)} 个 episode 到 {file_path}")
        total_written += len(records)
    
    print(f"  总共写入 {total_written} 个 episode 到 {len(file_groups)} 个文件")
    
    return {
        "success": True,
        "fixed_episodes": fixed_count,
        "total_episodes": len(episode_data_list),
        "output_dir": output_dir,
    }


def print_summary(stats_check: dict, metadata_check: dict, fix_result: dict = None):
    """打印检查和修复的总结报告。"""
    print(f"\n{'='*60}")
    print("总结报告")
    print(f"{'='*60}")
    
    # 统计字段检查
    print("\n1. 统计字段类型一致性:")
    if stats_check.get("valid"):
        print("   ✓ 通过 - 所有字段类型一致")
    else:
        inconsistent = stats_check.get("inconsistent_fields", {})
        print(f"   ✗ 失败 - {len(inconsistent)} 个字段类型不一致")
        for field_name, types in sorted(inconsistent.items()):
            type_list = ", ".join([f"{t} ({len(indices)} episodes)" for t, indices in types.items()])
            print(f"     - {field_name}: {type_list}")
    
    # Metadata 检查
    print("\n2. Episode Metadata 文件引用:")
    if metadata_check.get("is_valid"):
        print("   ✓ 通过 - 所有文件引用正确")
    else:
        print(f"   ✗ 失败")
        if metadata_check.get("missing_episode_files"):
            print(f"     - {len(metadata_check['missing_episode_files'])} 个 episode 缺少 metadata 文件")
        if metadata_check.get("missing_data_files"):
            print(f"     - {len(metadata_check['missing_data_files'])} 个 episode 缺少 data 文件")
    
    # 修复结果
    if fix_result:
        print("\n3. 修复结果:")
        if fix_result.get("success"):
            print(f"   ✓ 成功修复 {fix_result['fixed_episodes']}/{fix_result['total_episodes']} 个 episode")
            print(f"   输出目录: {fix_result['output_dir']}")
        else:
            print(f"   ✗ 修复失败: {fix_result.get('error', 'Unknown error')}")
    
    # 总体结论
    print(f"\n{'='*60}")
    all_valid = stats_check.get("valid") and metadata_check.get("is_valid")
    if all_valid:
        print("✓ 数据集没有问题")
    elif fix_result and fix_result.get("success"):
        print("✓ 问题已修复，请使用新数据集")
    else:
        print("✗ 数据集存在问题，需要修复")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="综合检查和修复 LeRobot 数据集"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="数据集仓库 ID，例如: ep10/charger"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="数据集根目录（默认: $HF_LEROBOT_HOME/repo_id）"
    )
    parser.add_argument(
        "--new_repo_id",
        type=str,
        default=None,
        help="修复后的数据集仓库 ID"
    )
    parser.add_argument(
        "--new_root",
        type=str,
        default=None,
        help="修复后的数据集根目录"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="自动修复发现的问题"
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="原地修复（会创建备份）"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细信息"
    )
    
    args = parser.parse_args()
    
    # 确定路径
    root_path = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
    
    print(f"数据集: {args.repo_id}")
    print(f"路径: {root_path}")
    
    # 检查 1: 统计字段类型一致性
    stats_check = check_stats_consistency(args.repo_id, args.root, args.verbose)
    
    # 检查 2: Episode Metadata 文件引用
    try:
        dataset = LeRobotDataset(args.repo_id, root=args.root)
        metadata_check = check_episode_metadata(dataset)
    except Exception as e:
        print(f"\n错误: 无法加载完整数据集 - {e}")
        metadata_check = {"is_valid": False, "error": str(e)}
    
    # 修复
    fix_result = None
    if args.fix:
        if not stats_check.get("valid") or not metadata_check.get("is_valid"):
            # 确定输出目录
            if args.in_place:
                output_dir = root_path
            else:
                output_repo_id = args.new_repo_id or f"{args.repo_id}_fixed"
                output_dir = Path(args.new_root) if args.new_root else HF_LEROBOT_HOME / output_repo_id
            
            fix_result = fix_stats_types(
                dataset.meta if hasattr(dataset, 'meta') else LeRobotDatasetMetadata(args.repo_id, root=args.root),
                output_dir,
                args.verbose,
            )
            
            if fix_result["success"]:
                print(f"\n修复完成！")
                if args.in_place:
                    print(f"原地修复，备份在: {root_path.with_name(root_path.name + '_backup')}")
                else:
                    print(f"新数据集: {output_repo_id}")
                    print(f"位置: {output_dir}")
        else:
            print("\n数据集没有问题，无需修复")
    
    # 打印总结
    print_summary(stats_check, metadata_check, fix_result)


if __name__ == "__main__":
    main()