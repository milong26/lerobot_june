"""
检查已有数据集的网格覆盖率

统计 random_42 和 uniform_42 各自覆盖了 112 个网格点中的多少个、
每个点的重复次数分布。
"""

"""
TODO:
需要修改代码：警告: 元数据文件不存在: personal/work2/dataset_view/random_112/episode_initial_states.json
元数据是这样的结构：
 personal/work2/dataset_view/pickplacev3文件夹是lerobot格式数据集的root
 random_112和uniform112筛选出来的数据集在personal/work2/duibi/uniform_42/subsets/uniform_112_seed42.json和personal/work2/duibi/random_42/subsets/random_112_seed42.json
 所以需要修改代码。
 在修改代码之前，“各自覆盖了 112 个网格点中的多少个”是什么意思？
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


OBJ_LOW = np.array([-0.1, 0.6, 0.02])
OBJ_HIGH = np.array([0.1, 0.7, 0.02])
GRID_SHAPE = (14, 8)


def pos_to_grid_coord(obj_pos: np.ndarray) -> Tuple[int, int]:
    """
    将物体位置转换为网格坐标
    
    参数:
        obj_pos: 物体初始位置 [x, y, z]
    
    返回:
        (grid_i, grid_j)
    """
    x, y = obj_pos[0], obj_pos[1]
    
    x_range = OBJ_HIGH[0] - OBJ_LOW[0]
    y_range = OBJ_HIGH[1] - OBJ_LOW[1]
    
    i = int((x - OBJ_LOW[0]) / x_range * GRID_SHAPE[0])
    j = int((y - OBJ_LOW[1]) / y_range * GRID_SHAPE[1])
    
    i = max(0, min(i, GRID_SHAPE[0] - 1))
    j = max(0, min(j, GRID_SHAPE[1] - 1))
    
    return (i, j)


def load_episode_metadata(subset_json: Path, dataset_root: Path) -> List[Dict]:
    """
    加载子集JSON中指定的episode的初始状态元数据
    
    参数:
        subset_json: 子集JSON文件路径（如random_112_seed42.json）
        dataset_root: LeRobot数据集根目录（如personal/work2/dataset_view/pickplacev3）
    
    返回:
        List[Dict] - 每个episode的初始状态信息
    """
    if not subset_json.exists():
        print(f"  警告: 子集JSON文件不存在: {subset_json}")
        return []
    
    with open(subset_json) as f:
        subset_data = json.load(f)
        episode_indices = subset_data.get("episodes", [])
    
    meta_file = dataset_root / "episode_initial_states.json"
    if not meta_file.exists():
        print(f"  警告: 元数据文件不存在: {meta_file}")
        return []
    
    with open(meta_file) as f:
        all_metadata = json.load(f)
        all_episodes = all_metadata.get("episodes", [])
    
    selected_episodes = [all_episodes[i] for i in episode_indices if i < len(all_episodes)]
    
    return selected_episodes


def compute_coverage(dataset_configs: List[Dict]) -> Dict:
    """
    计算多个数据集的合并覆盖率
    
    参数:
        dataset_configs: List[Dict] - 每个元素包含 {"name": str, "subset_json": Path, "dataset_root": Path}
    
    返回:
        覆盖率统计信息
    """
    grid_counts: Dict[Tuple[int, int], int] = {}
    total_episodes = 0
    
    for config in dataset_configs:
        name = config["name"]
        subset_json = config["subset_json"]
        dataset_root = config["dataset_root"]
        
        print(f"\n处理数据集: {name}")
        
        episodes = load_episode_metadata(subset_json, dataset_root)
        print(f"  找到 {len(episodes)} 个episodes")
        
        for ep in episodes:
            obj_pos = ep.get("obj_init_pos")
            if obj_pos is None:
                continue
            
            obj_pos = np.array(obj_pos)
            coord = pos_to_grid_coord(obj_pos)
            
            grid_counts[coord] = grid_counts.get(coord, 0) + 1
            total_episodes += 1

def main():
    parser = argparse.ArgumentParser(description="检查数据集网格覆盖率")
    parser.add_argument("--dataset-root", type=str,
                       default="personal/work2/dataset_view/pickplacev3",
                       help="LeRobot数据集根目录")
    parser.add_argument("--subset-files", type=str, nargs="+",
                       default=[
                           "personal/work2/duibi/random_42/subsets/random_112_seed42.json",
                           "personal/work2/duibi/uniform_42/subsets/uniform_112_seed42.json"
                       ],
                       help="子集JSON文件列表")
    parser.add_argument("--output", type=str,
                       default="personal/work2/our/results/tables/coverage_stats.json",
                       help="输出文件路径")
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    
    dataset_configs = []
    for subset_file in args.subset_files:
        name = Path(subset_file).stem
        dataset_configs.append({
            "name": name,
            "subset_json": Path(subset_file),
            "dataset_root": dataset_root
        })
    
    print("=" * 60)
    print("数据集网格覆盖率统计")
    print("=" * 60)
    print(f"网格形状: {GRID_SHAPE}")
    print(f"总网格数: {GRID_SHAPE[0] * GRID_SHAPE[1]}")
    print(f"工作空间: OBJ_LOW={OBJ_LOW}, OBJ_HIGH={OBJ_HIGH}")
    
    result = compute_coverage(dataset_configs)
    
    print("\n" + "=" * 60)
    print("覆盖率统计结果")
    print("=" * 60)
    print(f"总网格数: {result['n_total_cells']}")
    print(f"已覆盖: {result['n_covered']}")
    print(f"未覆盖: {result['n_uncovered']}")
    print(f"覆盖率: {result['coverage_rate']:.1f}%")
    print(f"总episodes: {result['total_episodes']}")
    print(f"平均重复次数: {result['avg_count']:.2f}")
    print(f"最大重复次数: {result['max_count']}")
    print(f"最小重复次数: {result['min_count']}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n结果已保存到: {output_path}")
    
    if result['coverage_rate'] >= 90:
        print("\n覆盖率 >= 90%，可以完全离线模拟，不需要新采集。")
    else:
        print(f"\n覆盖率 {result['coverage_rate']:.1f}% < 90%，可能需要在线补采。")


if __name__ == "__main__":
    import argparse
    main()