#!/usr/bin/env python3
"""检查 LeRobot 数据集所有 episode 的统计字段完整性。

这个脚本会：
1. 加载指定数据集的所有 episode metadata
2. 检查每个 episode 的统计字段是否完整
3. 报告缺失或不一致的字段
4. 帮助诊断 delete_episodes 操作中的 schema 不匹配问题

使用方法:
    python check_dataset_episodes.py --repo_id ep10/charger
    python check_dataset_episodes.py --repo_id ep10/charger --verbose
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 添加 lerobot 到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata, DEFAULT_EPISODES_PATH
from lerobot.datasets.io_utils import load_episodes


def load_episode_parquet_data(meta: LeRobotDatasetMetadata, episode_idx: int) -> dict:
    """从 parquet 文件加载单个 episode 的完整数据（包含 stats）。"""
    ep_meta = meta.episodes[episode_idx]
    chunk_idx = ep_meta["meta/episodes/chunk_index"]
    file_idx = ep_meta["meta/episodes/file_index"]

    parquet_path = meta.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    df = pd.read_parquet(parquet_path)

    episode_row = df[df["episode_index"] == episode_idx].iloc[0]
    return episode_row.to_dict()


def check_episode_stats(repo_id: str, verbose: bool = False) -> bool:
    """检查数据集所有 episode 的统计字段。

    Args:
        repo_id: 数据集仓库 ID
        verbose: 是否输出详细信息

    Returns:
        bool: 如果所有 episode 都一致返回 True，否则返回 False
    """
    print(f"加载数据集: {repo_id}")
    
    try:
        meta = LeRobotDatasetMetadata(repo_id)
    except Exception as e:
        print(f"错误: 无法加载数据集 {repo_id}")
        print(f"详细信息: {e}")
        return False

    print(f"总 episode 数: {meta.info.total_episodes}")
    print(f"总帧数: {meta.info.total_frames}")
    print(f"FPS: {meta.info.fps}")
    print(f"Features: {list(meta.features.keys())}")
    print(f"视频 keys: {meta.video_keys}")
    print()

    if meta.episodes is None or len(meta.episodes) == 0:
        print("警告: 没有找到 episode 数据")
        return False

    # 收集所有可能的 stats 字段
    all_stats_fields = set()
    episode_stats_fields = {}  # 记录每个 episode 的 stats 字段
    episode_parquet_data = {}  # 缓存每个 episode 的 parquet 数据

    print("正在加载所有 episode 的 parquet 数据...")
    for idx in range(len(meta.episodes)):
        try:
            episode_data = load_episode_parquet_data(meta, idx)
            episode_parquet_data[idx] = episode_data
            
            stats_fields = set()
            for key in episode_data.keys():
                if key.startswith("stats/"):
                    stats_fields.add(key)
                    all_stats_fields.add(key)
            
            episode_stats_fields[idx] = stats_fields
            
            if verbose:
                print(f"  Episode {idx}: 加载成功，{len(stats_fields)} 个 stats 字段")
        except Exception as e:
            print(f"  Episode {idx}: 加载失败 - {e}")
            episode_stats_fields[idx] = set()

    print(f"\n发现 {len(all_stats_fields)} 个统计字段:")
    for field in sorted(all_stats_fields):
        print(f"  - {field}")
    print()

    # 检查每个 episode 的字段完整性
    all_consistent = True
    missing_fields_report = []

    for idx in range(len(meta.episodes)):
        if idx not in episode_parquet_data:
            all_consistent = False
            missing_fields_report.append((idx, all_stats_fields))
            print(f"Episode {idx}: ✗ 无法加载数据")
            continue
            
        episode_fields = episode_stats_fields[idx]
        missing_fields = all_stats_fields - episode_fields
        
        if missing_fields:
            all_consistent = False
            missing_fields_report.append((idx, missing_fields))
            
            if verbose:
                print(f"Episode {idx}: 缺失 {len(missing_fields)} 个字段")
                for field in sorted(missing_fields):
                    print(f"    ✗ {field}")
        elif verbose:
            print(f"Episode {idx}: ✓ 完整 ({len(episode_fields)} 个字段)")

    # 打印汇总报告
    print("=" * 60)
    print("检查报告")
    print("=" * 60)
    
    if all_consistent:
        print("✓ 所有 episode 的统计字段完整且一致")
    else:
        print(f"✗ 发现 {len(missing_fields_report)} 个 episode 存在字段缺失:")
        for ep_idx, missing in missing_fields_report:
            print(f"\n  Episode {ep_idx}:")
            for field in sorted(missing):
                print(f"    ✗ {field}")
        
        print("\n" + "=" * 60)
        print("建议:")
        print("1. 这些缺失的字段会导致 delete_episodes 操作失败")
        print("2. 需要确保所有 episode 都有相同的统计字段结构")
        print("3. 可以考虑重新录制缺失统计信息的 episode")
        print("4. 或者修改代码在写入时自动填充缺失字段")

    # 检查字段值的一致性
    print("\n" + "=" * 60)
    print("字段值类型检查")
    print("=" * 60)
    
    field_types = {}
    for idx in episode_parquet_data:
        episode = episode_parquet_data[idx]
        for key in all_stats_fields:
            if key in episode:
                value = episode[key]
                if isinstance(value, np.ndarray):
                    type_str = f"ndarray(shape={value.shape}, dtype={value.dtype})"
                elif isinstance(value, list):
                    type_str = f"list(len={len(value)})"
                else:
                    type_str = type(value).__name__
                
                if key not in field_types:
                    field_types[key] = {}
                if type_str not in field_types[key]:
                    field_types[key][type_str] = []
                field_types[key][type_str].append(idx)

    for field in sorted(field_types.keys()):
        types = field_types[field]
        if len(types) > 1:
            print(f"\n⚠ {field}: 存在多种类型")
            for type_str, indices in types.items():
                print(f"    {type_str}: episodes {indices[:5]}{'...' if len(indices) > 5 else ''}")
        elif verbose:
            type_str = list(types.keys())[0]
            print(f"✓ {field}: {type_str}")

    return all_consistent


def main():
    parser = argparse.ArgumentParser(
        description="检查 LeRobot 数据集所有 episode 的统计字段完整性"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="数据集仓库 ID，例如: ep10/charger"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细信息"
    )
    
    args = parser.parse_args()
    
    success = check_episode_stats(args.repo_id, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()