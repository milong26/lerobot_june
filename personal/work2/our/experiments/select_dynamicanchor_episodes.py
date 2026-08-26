#!/usr/bin/env python
"""
根据离线模拟结果选择episodes并输出为子集JSON文件

从run_offline_simulation.py输出的CSV中读取规划结果，
选择对应的episodes并保存为train_and_eval.sh可读取的格式。
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


OBJ_LOW = np.array([-0.1, 0.6, 0.02])
OBJ_HIGH = np.array([0.1, 0.7, 0.02])
GRID_SHAPE = (14, 8)


def pos_to_grid_coord(obj_pos: np.ndarray) -> Tuple[int, int]:
    """将物体位置转换为网格坐标"""
    x, y = obj_pos[0], obj_pos[1]
    
    x_range = OBJ_HIGH[0] - OBJ_LOW[0]
    y_range = OBJ_HIGH[1] - OBJ_LOW[1]
    
    i = int((x - OBJ_LOW[0]) / x_range * GRID_SHAPE[0])
    j = int((y - OBJ_LOW[1]) / y_range * GRID_SHAPE[1])
    
    i = max(0, min(i, GRID_SHAPE[0] - 1))
    j = max(0, min(j, GRID_SHAPE[1] - 1))
    
    return (i, j)


def select_episodes(
    planning_result: Dict[Tuple[int, int], int],
    dataset_dir: Path
) -> List[int]:
    """
    根据规划结果选中对应的episode索引
    
    参数:
        planning_result: Dict[grid_coord, repeat_count]
        dataset_dir: 数据集目录
    
    返回:
        选中的episode索引列表
    """
    meta_file = dataset_dir / "episode_initial_states.json"
    if not meta_file.exists():
        print(f"警告: 元数据文件不存在: {meta_file}")
        return []
    
    with open(meta_file) as f:
        meta = json.load(f)
        episodes = meta.get("episodes", [])
    
    grid_to_episodes: Dict[Tuple[int, int], List[int]] = {}
    
    for ep_idx, ep in enumerate(episodes):
        obj_pos = ep.get("obj_init_pos")
        if obj_pos is None:
            continue
        
        coord = pos_to_grid_coord(np.array(obj_pos))
        if coord not in grid_to_episodes:
            grid_to_episodes[coord] = []
        grid_to_episodes[coord].append(ep_idx)
    
    selected_episodes = []
    
    for grid_coord, repeat_count in planning_result.items():
        available = grid_to_episodes.get(grid_coord, [])
        selected = available[:repeat_count]
        selected_episodes.extend(selected)
    
    selected_episodes = sorted(list(set(selected_episodes)))
    
    print(f"选中 {len(selected_episodes)} 个episodes")
    print(f"  episode索引: {selected_episodes[:20]}{'...' if len(selected_episodes) > 20 else ''}")
    
    return selected_episodes


def main():
    parser = argparse.ArgumentParser(description="根据规划结果选择episodes")
    parser.add_argument("--planning-result", type=str, required=True,
                       help="规划结果JSON文件路径")
    parser.add_argument("--dataset-dir", type=str, required=True,
                       help="数据集目录")
    parser.add_argument("--output", type=str, required=True,
                       help="输出子集JSON文件路径")
    args = parser.parse_args()
    
    print(f"加载规划结果: {args.planning_result}")
    with open(args.planning_result) as f:
        planning_result = json.load(f)
    print(f"规划结果包含 {len(planning_result)} 个网格配置")
    
    planning_result = {
        tuple(map(int, k.strip("()").split(","))): v
        for k, v in planning_result.items()
    }
    
    dataset_dir = Path(args.dataset_dir)
    print(f"数据集目录: {dataset_dir}")
    
    print("\n开始选择 episodes...")
    selected_episodes = select_episodes(planning_result, dataset_dir)
    
    if not selected_episodes:
        print("没有选中任何episodes，退出")
        return
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "selected_episode_indices": selected_episodes,
        "num_episodes": len(selected_episodes),
        "planning_result": {
            f"({k[0]},{k[1]})": v
            for k, v in planning_result.items()
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"子集文件已保存到: {output_path}")


if __name__ == "__main__":
    main()