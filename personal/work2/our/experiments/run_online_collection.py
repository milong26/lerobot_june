"""
在线补采脚本

仅当离线池覆盖不足时才用，调用 collect_metaworld_dataset.py 补采缺失的网格点。
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


def grid_coord_to_pos(grid_coord: Tuple[int, int]) -> np.ndarray:
    """将网格坐标转换为物体位置（网格中心）"""
    i, j = grid_coord
    
    x_range = OBJ_HIGH[0] - OBJ_LOW[0]
    y_range = OBJ_HIGH[1] - OBJ_LOW[1]
    
    x = OBJ_LOW[0] + (i + 0.5) / GRID_SHAPE[0] * x_range
    y = OBJ_LOW[1] + (j + 0.5) / GRID_SHAPE[1] * y_range
    z = OBJ_LOW[2]
    
    return np.array([x, y, z])


def find_uncovered_grids(
    dataset_dirs: List[Path],
    target_coverage: float = 1.0
) -> List[Tuple[int, int]]:
    """
    找到未被覆盖的网格点
    
    参数:
        dataset_dirs: 已有数据集目录列表
        target_coverage: 目标覆盖率
    
    返回:
        未覆盖的网格坐标列表
    """
    covered_grids = set()
    
    for dataset_dir in dataset_dirs:
        meta_file = dataset_dir / "episode_initial_states.json"
        if not meta_file.exists():
            continue
        
        with open(meta_file) as f:
            meta = json.load(f)
            episodes = meta.get("episodes", [])
        
        for ep in episodes:
            obj_pos = ep.get("obj_init_pos")
            if obj_pos is None:
                continue
            
            coord = pos_to_grid_coord(np.array(obj_pos))
            covered_grids.add(coord)
    
    all_grids = set(
        (i, j) for i in range(GRID_SHAPE[0]) for j in range(GRID_SHAPE[1])
    )
    
    uncovered = all_grids - covered_grids
    
    n_total = len(all_grids)
    n_covered = len(covered_grids)
    coverage = n_covered / n_total
    
    print(f"已覆盖: {n_covered}/{n_total} ({coverage:.1%})")
    print(f"未覆盖: {len(uncovered)} 个网格点")
    
    return list(uncovered)


def collect_for_grid(
    grid_coord: Tuple[int, int],
    output_dir: Path,
    seed_start: int = 0,
    task: str = "pick-place-v3"
):
    """
    为指定网格点采集数据
    
    参数:
        grid_coord: 网格坐标
        output_dir: 输出目录
        seed_start: 起始seed
        task: 任务名称
    """
    from collect_dataset.collect_metaworld_dataset import (
        create_metaworld_env,
        run_episode,
        LeRobotDataset
    )
    
    target_pos = grid_coord_to_pos(grid_coord)
    print(f"\n采集网格 {grid_coord}，目标位置: {target_pos}")
    
    seed = seed_start + grid_coord[0] * GRID_SHAPE[1] + grid_coord[1]
    
    env_top = create_metaworld_env(task, seed=seed, camera_name="corner2")
    env_wrist = create_metaworld_env(task, seed=seed, camera_name="gripperPOV")
    
    ep_info = run_episode(env_top, env_wrist, image_size=480)
    
    print(f"  采集完成: success={ep_info['success']}")
    
    return ep_info


def main():
    parser = argparse.ArgumentParser(description="在线补采缺失网格点")
    parser.add_argument("--datasets", type=str, nargs="+",
                       help="已有数据集目录列表")
    parser.add_argument("--output-dir", type=str,
                       default="personal/work2/our/collected",
                       help="补采数据输出目录")
    parser.add_argument("--target-coverage", type=float, default=1.0,
                       help="目标覆盖率")
    parser.add_argument("--seed-start", type=int, default=10000,
                       help="起始seed")
    args = parser.parse_args()
    
    dataset_dirs = [Path(d) for d in args.datasets]
    
    uncovered = find_uncovered_grids(dataset_dirs, args.target_coverage)
    
    if not uncovered:
        print("所有网格点已覆盖，无需补采")
        return
    
    print(f"\n开始补采 {len(uncovered)} 个网格点...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for grid_coord in uncovered:
        collect_for_grid(grid_coord, output_dir, args.seed_start)


if __name__ == "__main__":
    main()