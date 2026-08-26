"""
覆盖空隙计算模块

对112个格点中的每一个，计算它到训练集中最近样本的欧氏距离。
复用analyze_results.py中的compute_nearest_dist逻辑，扩展为批量计算。
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def distance_to_nearest_anchor(point: np.ndarray, anchor_set: List[np.ndarray]) -> float:
    """
    计算点到最近锚点的欧氏距离

    参数:
        point: 目标点坐标，shape (3,) 或 (2,)
        anchor_set: 锚点集合，list of ndarray

    返回:
        到最近锚点的距离
    """
    if not anchor_set:
        return float("inf")

    anchor_array = np.array(anchor_set)
    dists = np.linalg.norm(anchor_array - point, axis=1)
    return float(dists.min())


def compute_coverage_gap(
    grid_positions: List[Dict],
    anchor_positions: List[Dict],
    use_xy_only: bool = True,
) -> Dict[int, float]:
    """
    计算所有格点的覆盖空隙

    参数:
        grid_positions: 格点位置列表，每项包含 {"grid_id": int, "obj_pos": [x, y, z]}
        anchor_positions: 训练样本位置列表，格式同上
        use_xy_only: 是否只使用x-y平面距离（默认True，因为z轴变化不大）

    返回:
        {grid_id: gap_value} 字典
    """
    # 提取坐标
    if use_xy_only:
        grid_coords = [np.array(p["obj_pos"][:2]) for p in grid_positions]
        anchor_coords = [np.array(p["obj_pos"][:2]) for p in anchor_positions]
    else:
        grid_coords = [np.array(p["obj_pos"]) for p in grid_positions]
        anchor_coords = [np.array(p["obj_pos"]) for p in anchor_positions]

    # 计算每个格点到最近训练样本的距离
    gaps = {}
    for i, grid_coord in enumerate(grid_coords):
        grid_id = grid_positions[i]["grid_id"]
        gaps[grid_id] = distance_to_nearest_anchor(grid_coord, anchor_coords)

    return gaps


def load_grid_positions_from_subset(subset_file: str, dataset_metadata: str) -> List[Dict]:
    """
    从subset文件加载格点位置

    参数:
        subset_file: subset JSON文件路径
        dataset_metadata: episode_initial_states.json路径

    返回:
        list of {"grid_id": int, "obj_pos": [x, y, z]}
    """
    with open(subset_file) as f:
        subset_data = json.load(f)

    with open(dataset_metadata) as f:
        metadata = json.load(f)

    episodes = metadata["episodes"]

    # subset文件格式: {"method": "random", "num_episodes": 112, "seed": 42, "selected_episode_indices": [...]}
    if isinstance(subset_data, list):
        episode_indices = subset_data
    elif isinstance(subset_data, dict):
        episode_indices = subset_data.get("selected_episode_indices", 
                           subset_data.get("episodes", 
                           subset_data.get("included_episodes", [])))
    else:
        raise ValueError(f"未知的subset文件格式: {subset_file}")

    grid_positions = []
    for idx in episode_indices:
        episode = episodes[idx]
        grid_positions.append({
            "grid_id": idx,
            "obj_pos": episode["obj_init_pos"],
        })

    return grid_positions


def load_anchor_positions_from_subset(subset_file: str, dataset_metadata: str) -> List[Dict]:
    """
    从subset文件加载训练样本位置（锚点）

    参数与返回值格式同load_grid_positions_from_subset
    """
    return load_grid_positions_from_subset(subset_file, dataset_metadata)


def main():
    """测试覆盖空隙计算"""
    import argparse

    parser = argparse.ArgumentParser(description="覆盖空隙计算")
    parser.add_argument("--subset-file", type=str, required=True)
    parser.add_argument("--dataset-metadata", type=str, required=True)
    parser.add_argument("--output", type=str, default="coverage_gaps.json")
    args = parser.parse_args()

    # 加载格点和锚点
    grid_positions = load_grid_positions_from_subset(args.subset_file, args.dataset_metadata)
    anchor_positions = load_anchor_positions_from_subset(args.subset_file, args.dataset_metadata)

    print(f"加载了 {len(grid_positions)} 个格点")
    print(f"加载了 {len(anchor_positions)} 个锚点")

    # 计算覆盖空隙
    gaps = compute_coverage_gap(grid_positions, anchor_positions)

    # 输出统计
    gap_values = list(gaps.values())
    print(f"\n覆盖空隙统计:")
    print(f"  最小值: {min(gap_values):.4f}")
    print(f"  最大值: {max(gap_values):.4f}")
    print(f"  平均值: {np.mean(gap_values):.4f}")
    print(f"  中位数: {np.median(gap_values):.4f}")

    # 保存结果
    with open(args.output, "w") as f:
        json.dump(gaps, f, indent=2)
    print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()