#!/usr/bin/env python
"""
Select episodes with true uniform distribution in the full state space (rand_vec).

Key improvements over select_uniform_episodes.py:
1. Uses rand_vec (full state space) instead of just obj_init_pos
2. Uses theoretical space bounds from env_state_structure (reset_space_low/high)
3. Better handling of empty cells - selects nearest in rand_vec space
4. Supports any number of objects/tasks by using the full randomization vector

Usage:
    python select_new_uniform.py --num-episodes 100 --seed 42 \
        --dataset-root /path/to/dataset --output-dir /path/to/output
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_episode_metadata(json_path: str) -> tuple[list[int], np.ndarray, dict]:
    """Load episode indices, rand_vecs, and env_state_structure from JSON file."""
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"找不到 JSON 文件: {json_file}")

    with open(json_file, "r") as f:
        metadata = json.load(f)

    episodes = metadata["episodes"]
    indices = [ep["episode_index"] for ep in episodes]
    
    has_rand_vec = [ep.get("rand_vec") is not None for ep in episodes]
    if all(has_rand_vec):
        rand_vecs = np.array([ep["rand_vec"] for ep in episodes])
    else:
        valid_indices = [i for i, ep in enumerate(episodes) if ep.get("rand_vec") is not None]
        rand_vecs = np.array([episodes[i]["rand_vec"] for i in valid_indices])
        indices = [indices[i] for i in valid_indices]
        print(f"警告: {len(has_rand_vec) - len(valid_indices)} 个 episode 缺少 rand_vec，已跳过")

    env_state_structure = metadata.get("env_state_structure", {})

    return indices, rand_vecs, env_state_structure


def get_theoretical_bounds(env_state_structure: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Get theoretical bounds from env_state_structure.
    
    Returns (low, high) arrays representing the full randomization space bounds.
    Falls back to None if structure is incomplete.
    """
    if not env_state_structure:
        print("未找到 env_state_structure")
        return None, None
    
    reset_low = env_state_structure.get("reset_space_low")
    reset_high = env_state_structure.get("reset_space_high")
    
    if reset_low and reset_high:
        low = np.array(reset_low)
        high = np.array(reset_high)
        
        print(f"使用理论空间范围 (来自 env_state_structure):")
        print(f"  维度: {len(low)}")
        print(f"  low: {low[:5].tolist()}... (前5维)")
        print(f"  high: {high[:5].tolist()}... (前5维)")
        
        return low, high
    
    print("env_state_structure 不完整，缺少 reset_space_low/high")
    return None, None


def project_to_2d_for_visualization(rand_vecs: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """
    Project high-dimensional rand_vec to 2D for grid-based selection.
    
    Uses the first two varying dimensions (where low != high) as x and y.
    This works for any task regardless of number of objects.
    """
    varying_mask = low != high
    varying_dims = np.where(varying_mask)[0]
    
    if len(varying_dims) >= 2:
        dim_x = varying_dims[0]
        dim_y = varying_dims[1]
        return rand_vecs[:, [dim_x, dim_y]], dim_x, dim_y
    elif len(varying_dims) == 1:
        dim_x = varying_dims[0]
        positions_2d = np.zeros((len(rand_vecs), 2))
        positions_2d[:, 0] = rand_vecs[:, dim_x]
        positions_2d[:, 1] = 0
        return positions_2d, dim_x, None
    else:
        positions_2d = np.zeros((len(rand_vecs), 2))
        return positions_2d, None, None


def select_new_uniform_episodes(num_episodes, seed, dataset_root, output_dir=None):
    """Select episodes with true uniform distribution in full rand_vec space."""
    rng = np.random.RandomState(seed)
    
    json_path = Path(dataset_root) / "episode_initial_states.json"
    print(f"加载 episode 数据: {json_path}")
    indices, rand_vecs, env_state_structure = load_episode_metadata(str(json_path))

    print(f"Total episodes: {len(rand_vecs)}")
    print(f"rand_vec dimension: {rand_vecs.shape[1]}")
    print(f"rand_vec range (per dimension):")
    for i in range(min(5, rand_vecs.shape[1])):
        print(f"  dim {i}: [{rand_vecs[:, i].min():.3f}, {rand_vecs[:, i].max():.3f}]")

    low, high = get_theoretical_bounds(env_state_structure)
    
    if low is None or high is None:
        print("警告: 无法获取理论空间范围，使用实际数据范围")
        low = rand_vecs.min(axis=0)
        high = rand_vecs.max(axis=0)
        use_theoretical = False
    else:
        use_theoretical = True
        
        data_coverage = []
        for i in range(len(low)):
            if high[i] > low[i]:
                coverage = (rand_vecs[:, i].max() - rand_vecs[:, i].min()) / (high[i] - low[i]) * 100
                data_coverage.append(coverage)
        
        if data_coverage:
            avg_coverage = np.mean(data_coverage)
            print(f"\n平均数据覆盖率: {avg_coverage:.1f}%")
            print(f"覆盖率 < 50% 的维度数: {sum(1 for c in data_coverage if c < 50)}/{len(data_coverage)}")

    positions_2d, dim_x, dim_y = project_to_2d_for_visualization(rand_vecs, low, high)
    
    if dim_x is not None:
        print(f"\n使用 rand_vec 维度 {dim_x} 和 {dim_y} 进行 2D 网格划分")
        x_min, x_max = low[dim_x], high[dim_x]
        y_min, y_max = low[dim_y] if dim_y is not None else 0, high[dim_y] if dim_y is not None else 0
    else:
        print("\n警告: 没有变化的维度，使用实际数据范围")
        x_min, x_max = positions_2d[:, 0].min(), positions_2d[:, 0].max()
        y_min, y_max = positions_2d[:, 1].min(), positions_2d[:, 1].max()

    total_episodes = len(rand_vecs)
    range_x = x_max - x_min
    range_y = y_max - y_min
    aspect_ratio = range_x / range_y if range_y > 0 else 1.0

    n_x = int(np.floor(np.sqrt(num_episodes * aspect_ratio)))
    n_y = int(np.ceil(num_episodes / n_x))
    
    best_nx, best_ny = n_x, n_y
    best_diff = abs(n_x * n_y - num_episodes)
    
    for nx in range(max(1, n_x - 2), n_x + 3):
        ny = int(np.ceil(num_episodes / nx))
        diff = abs(nx * ny - num_episodes)
        if diff < best_diff or (diff == best_diff and abs(nx / ny - aspect_ratio) < abs(best_nx / best_ny - aspect_ratio)):
            best_nx, best_ny = nx, ny
            best_diff = diff
    
    n_x, n_y = best_nx, best_ny
    
    print(f"Grid size: {n_x} x {n_y} = {n_x * n_y} cells (target: {num_episodes})")

    x_bins = np.linspace(x_min, x_max, n_x + 1)
    y_bins = np.linspace(y_min, y_max, n_y + 1)
    
    cell_centers = []
    for i in range(n_x):
        for j in range(n_y):
            cx = (x_bins[i] + x_bins[i + 1]) / 2
            cy = (y_bins[j] + y_bins[j + 1]) / 2
            cell_centers.append((i, j, cx, cy))
    
    rng.shuffle(cell_centers)
    
    selected_set = set()
    used_episodes = set()
    empty_cells = []
    
    for i, j, cx, cy in cell_centers:
        if len(selected_set) >= num_episodes:
            break
        
        x_mask = (positions_2d[:, 0] >= x_bins[i]) & (positions_2d[:, 0] < x_bins[i + 1])
        y_mask = (positions_2d[:, 1] >= y_bins[j]) & (positions_2d[:, 1] < y_bins[j + 1])
        cell_mask = x_mask & y_mask
        cell_indices = np.where(cell_mask)[0]
        
        available_indices = [idx for idx in cell_indices if idx not in used_episodes]
        
        if len(available_indices) > 0:
            best_dist = float("inf")
            best_idx = None
            for idx in available_indices:
                dx = positions_2d[idx, 0] - cx
                dy = positions_2d[idx, 1] - cy
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_idx = int(idx)
            
            if best_idx is not None:
                selected_set.add(best_idx)
                used_episodes.add(best_idx)
        else:
            empty_cells.append((i, j, cx, cy))
    
    if empty_cells:
        print(f"\n有 {len(empty_cells)} 个空单元格（该区域没有数据）")
        print("尝试从整个空间中选择最近的 episode...")
        
        for i, j, cx, cy in empty_cells:
            if len(selected_set) >= num_episodes:
                break
            
            best_dist = float("inf")
            best_idx = None
            
            for idx in range(total_episodes):
                if idx in used_episodes:
                    continue
                
                dx = positions_2d[idx, 0] - cx
                dy = positions_2d[idx, 1] - cy
                dist = dx * dx + dy * dy
                
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            
            if best_idx is not None:
                selected_set.add(best_idx)
                used_episodes.add(best_idx)
                print(f"  空单元格 ({i},{j}) 选择最近 episode (dist={best_dist:.4f})")
    
    if len(selected_set) < num_episodes:
        remaining = [ep for ep in range(total_episodes) if ep not in used_episodes]
        extra_needed = num_episodes - len(selected_set)
        if len(remaining) >= extra_needed:
            extra = rng.choice(remaining, extra_needed, replace=False)
            selected_set.update(extra.tolist())
        else:
            selected_set.update(remaining)
    
    selected = sorted(list(selected_set))[:num_episodes]
    
    selected_rand_vecs = rand_vecs[selected]
    print(f"\nSelected {len(selected)} episodes")
    print(f"Selected rand_vec range (first 5 dims):")
    for i in range(min(5, selected_rand_vecs.shape[1])):
        print(f"  dim {i}: [{selected_rand_vecs[:, i].min():.3f}, {selected_rand_vecs[:, i].max():.3f}]")

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_file = output_path / f"new_uniform_{num_episodes}_seed{seed}.json"
        subset_data = {
            "method": "new_uniform_full_state_space",
            "num_episodes": num_episodes,
            "seed": seed,
            "rand_vec_dimension": rand_vecs.shape[1],
            "selected_episode_indices": [indices[i] for i in selected],
            "theoretical_bounds": {
                "low": low.tolist() if use_theoretical else None,
                "high": high.tolist() if use_theoretical else None
            },
            "grid_dimensions": {
                "dim_x": int(dim_x) if dim_x is not None else None,
                "dim_y": int(dim_y) if dim_y is not None else None,
                "x_range": [float(x_min), float(x_max)],
                "y_range": [float(y_min), float(y_max)]
            }
        }
        with open(subset_file, "w") as f:
            json.dump(subset_data, f, indent=2)
        print(f"Saved {num_episodes} new uniform episodes (seed={seed}) to {subset_file}")

    return [indices[i] for i in selected]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-root", type=str, required=True, help="数据集根目录，JSON 文件位于此目录下")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    select_new_uniform_episodes(
        num_episodes=args.num_episodes,
        seed=args.seed,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )