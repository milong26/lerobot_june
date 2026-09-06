#!/usr/bin/env python
"""
Select episodes with true uniform distribution in the full state space (rand_vec).

Key improvements over select_uniform_episodes.py:
1. Uses rand_vec (full state space) instead of just obj_init_pos
2. Uses theoretical space bounds from env_state_structure (reset_space_low/high)
3. True D-dimensional grid-based uniform selection (not just 2D projection)
4. Adaptive grid allocation across all varying dimensions
5. Supports any number of objects/tasks by using the full randomization vector

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


def normalize_to_unit(rand_vecs: np.ndarray, low: np.ndarray, high: np.ndarray, varying_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize rand_vec to [0, 1] for all varying dimensions.
    Non-varying dimensions are set to 0.
    
    Returns normalized positions and the scale factors used.
    """
    n_dims = rand_vecs.shape[1]
    normalized = np.zeros_like(rand_vecs)
    scales = np.ones(n_dims)
    
    for i in range(n_dims):
        if varying_mask[i]:
            range_val = high[i] - low[i]
            if range_val > 1e-10:
                normalized[:, i] = (rand_vecs[:, i] - low[i]) / range_val
                scales[i] = range_val
            else:
                normalized[:, i] = 0.5
        else:
            normalized[:, i] = 0.0
    
    return normalized, scales


def allocate_grid_dimensions(num_episodes: int, n_varying: int, target_cells_ratio: float = 1.5) -> list[int]:
    """
    Allocate grid cells per dimension to achieve approximately num_episodes total cells.
    
    Strategy: 
    - Target total cells = num_episodes * target_cells_ratio (oversample to handle empty cells)
    - Distribute cells across dimensions based on their importance (equal for now)
    - Ensure each dimension gets at least 2 cells
    
    Returns list of grid sizes per varying dimension.
    """
    target_cells = int(num_episodes * target_cells_ratio)
    
    if n_varying == 1:
        return [target_cells]
    
    # Start with equal allocation
    base_per_dim = max(2, int(np.ceil(target_cells ** (1.0 / n_varying))))
    
    # Adjust to get closer to target
    grid_sizes = [base_per_dim] * n_varying
    
    # Fine-tune: try to get product close to target_cells
    current_product = np.prod(grid_sizes)
    
    # If too many cells, reduce some dimensions
    while current_product > target_cells * 1.5 and max(grid_sizes) > 2:
        max_idx = np.argmax(grid_sizes)
        grid_sizes[max_idx] -= 1
        current_product = np.prod(grid_sizes)
    
    # If too few cells, increase some dimensions
    while current_product < target_cells * 0.8:
        min_idx = np.argmin(grid_sizes)
        grid_sizes[min_idx] += 1
        current_product = np.prod(grid_sizes)
    
    return grid_sizes


def generate_d_grid_centers(grid_sizes: list[int]) -> list[tuple]:
    """
    Generate all cell centers for a D-dimensional grid.
    Each cell center is represented as (cell_indices..., center_coords...).
    """
    n_dims = len(grid_sizes)
    
    # Generate all cell index combinations
    from itertools import product
    all_indices = list(product(*[range(s) for s in grid_sizes]))
    
    centers = []
    for indices in all_indices:
        # Cell center in normalized space: (idx + 0.5) / size
        center_coords = tuple((idx + 0.5) / size for idx, size in zip(indices, grid_sizes))
        centers.append((indices, center_coords))
    
    return centers


def find_nearest_episode(normalized_positions: np.ndarray, center: tuple, varying_dims: list[int], 
                         used_episodes: set, total_episodes: int) -> tuple[int, float]:
    """
    Find the nearest unused episode to a D-dimensional cell center.
    Returns (episode_index, distance).
    """
    best_dist = float("inf")
    best_idx = None
    
    center_arr = np.array(center)
    
    for idx in range(total_episodes):
        if idx in used_episodes:
            continue
        
        # Calculate squared distance in normalized D-space
        diff = normalized_positions[idx, varying_dims] - center_arr
        dist = np.dot(diff, diff)
        
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    
    return best_idx, best_dist


def select_new_uniform_episodes(num_episodes, seed, dataset_root, output_dir=None):
    """Select episodes with true uniform distribution in full D-dimensional rand_vec space."""
    rng = np.random.RandomState(seed)
    
    json_path = Path(dataset_root) / "episode_initial_states.json"
    print(f"加载 episode 数据: {json_path}")
    indices, rand_vecs, env_state_structure = load_episode_metadata(str(json_path))

    n_episodes = len(rand_vecs)
    n_dims = rand_vecs.shape[1]
    
    print(f"Total episodes: {n_episodes}")
    print(f"rand_vec dimension: {n_dims}")
    print(f"rand_vec range (per dimension):")
    for i in range(min(5, n_dims)):
        print(f"  dim {i}: [{rand_vecs[:, i].min():.3f}, {rand_vecs[:, i].max():.3f}]")
    if n_dims > 5:
        print(f"  ... ({n_dims - 5} more dimensions)")

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

    # Identify varying dimensions
    varying_mask = low != high
    varying_dims = np.where(varying_mask)[0].tolist()
    n_varying = len(varying_dims)
    
    print(f"\n变化维度数: {n_varying} / {n_dims}")
    print(f"变化维度索引: {varying_dims}")
    
    if n_varying == 0:
        print("错误: 没有变化的维度，无法进行均匀选择")
        # Fallback to random selection
        selected = rng.choice(n_episodes, min(num_episodes, n_episodes), replace=False).tolist()
        selected_indices = [indices[i] for i in selected]
        print(f"随机选择 {len(selected_indices)} episodes")
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            subset_file = output_path / f"state_uniform_{num_episodes}_seed{seed}.json"
            subset_data = {
                "method": "random_fallback_no_varying_dims",
                "num_episodes": num_episodes,
                "seed": seed,
                "selected_episode_indices": selected_indices,
            }
            with open(subset_file, "w") as f:
                json.dump(subset_data, f, indent=2)
        return selected_indices

    # Normalize to [0, 1] for varying dimensions
    normalized_positions, scales = normalize_to_unit(rand_vecs, low, high, varying_mask)
    
    print(f"\n归一化后，在 {n_varying}D 空间中进行网格划分")

    # Allocate grid dimensions
    grid_sizes = allocate_grid_dimensions(num_episodes, n_varying)
    total_cells = int(np.prod(grid_sizes))
    
    print(f"网格配置: {grid_sizes}")
    print(f"总网格数: {total_cells} (目标: {num_episodes})")

    # Generate all cell centers in normalized space
    cell_centers = generate_d_grid_centers(grid_sizes)
    
    # Shuffle for random selection order
    rng.shuffle(cell_centers)
    
    # Select episodes
    selected_set = set()
    used_episodes = set()
    empty_cells = 0
    
    print(f"\n开始选择 episodes...")
    
    for cell_indices, center_coords in cell_centers:
        if len(selected_set) >= num_episodes:
            break
        
        # Find episodes in this cell
        cell_mask = np.ones(n_episodes, dtype=bool)
        for dim_idx, grid_size in enumerate(grid_sizes):
            dim_global = varying_dims[dim_idx]
            cell_idx = cell_indices[dim_idx]
            
            lower_bound = cell_idx / grid_size
            upper_bound = (cell_idx + 1) / grid_size
            
            cell_mask &= (normalized_positions[:, dim_global] >= lower_bound) & \
                         (normalized_positions[:, dim_global] < upper_bound)
        
        cell_indices_list = np.where(cell_mask)[0]
        available_indices = [idx for idx in cell_indices_list if idx not in used_episodes]
        
        if len(available_indices) > 0:
            # Find nearest to cell center
            best_dist = float("inf")
            best_idx = None
            center_arr = np.array(center_coords)
            
            for idx in available_indices:
                diff = normalized_positions[idx, varying_dims] - center_arr
                dist = np.dot(diff, diff)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            
            if best_idx is not None:
                selected_set.add(best_idx)
                used_episodes.add(best_idx)
        else:
            empty_cells += 1
    
    print(f"网格选择完成: {len(selected_set)} episodes, {empty_cells} 空单元格")
    
    # Handle empty cells: find nearest from entire space
    if empty_cells > 0 and len(selected_set) < num_episodes:
        print(f"\n处理 {empty_cells} 个空单元格...")
        
        remaining_centers = []
        for cell_indices, center_coords in cell_centers:
            cell_key = tuple(cell_indices)
            # Check if this cell was empty (simplified: just continue from where we left off)
            if len(selected_set) >= num_episodes:
                break
            
            # Re-check if cell has available episodes
            cell_mask = np.ones(n_episodes, dtype=bool)
            for dim_idx, grid_size in enumerate(grid_sizes):
                dim_global = varying_dims[dim_idx]
                c_idx = cell_indices[dim_idx]
                lower_bound = c_idx / grid_size
                upper_bound = (c_idx + 1) / grid_size
                cell_mask &= (normalized_positions[:, dim_global] >= lower_bound) & \
                             (normalized_positions[:, dim_global] < upper_bound)
            
            available = [idx for idx in np.where(cell_mask)[0] if idx not in used_episodes]
            if len(available) == 0:
                # Find nearest from entire space
                best_idx, best_dist = find_nearest_episode(
                    normalized_positions, center_coords, varying_dims, 
                    used_episodes, n_episodes
                )
                if best_idx is not None:
                    selected_set.add(best_idx)
                    used_episodes.add(best_idx)
                    print(f"  空单元格选择最近 episode (dist={best_dist:.4f})")
    
    # Fill remaining if needed
    if len(selected_set) < num_episodes:
        remaining = [ep for ep in range(n_episodes) if ep not in used_episodes]
        extra_needed = num_episodes - len(selected_set)
        if len(remaining) >= extra_needed:
            extra = rng.choice(remaining, extra_needed, replace=False)
            selected_set.update(extra.tolist())
            print(f"随机补充 {extra_needed} episodes")
        else:
            selected_set.update(remaining)
            print(f"随机补充 {len(remaining)} episodes (不足)")
    
    selected = sorted(list(selected_set))[:num_episodes]
    
    selected_rand_vecs = rand_vecs[selected]
    print(f"\n最终选择 {len(selected)} episodes")
    print(f"Selected rand_vec range (first 5 dims):")
    for i in range(min(5, n_dims)):
        print(f"  dim {i}: [{selected_rand_vecs[:, i].min():.3f}, {selected_rand_vecs[:, i].max():.3f}]")
    
    # Calculate coverage metrics
    if n_varying <= 6:
        print(f"\nD维空间覆盖度评估:")
        for dim_idx, dim_global in enumerate(varying_dims):
            if high[dim_global] > low[dim_global]:
                full_range = high[dim_global] - low[dim_global]
                sel_min = selected_rand_vecs[:, dim_global].min()
                sel_max = selected_rand_vecs[:, dim_global].max()
                coverage = (sel_max - sel_min) / full_range * 100
                print(f"  dim {dim_global}: [{sel_min:.3f}, {sel_max:.3f}] (覆盖率: {coverage:.1f}%)")

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_file = output_path / f"state_uniform_{num_episodes}_seed{seed}.json"
        subset_data = {
            "method": "true_d_dimensional_uniform",
            "num_episodes": num_episodes,
            "seed": seed,
            "rand_vec_dimension": n_dims,
            "n_varying_dimensions": n_varying,
            "varying_dimensions": varying_dims,
            "selected_episode_indices": [indices[i] for i in selected],
            "grid_sizes_per_varying_dim": grid_sizes,
            "total_grid_cells": total_cells,
            "theoretical_bounds": {
                "low": low.tolist() if use_theoretical else None,
                "high": high.tolist() if use_theoretical else None
            },
            "normalization_scales": scales.tolist()
        }
        with open(subset_file, "w") as f:
            json.dump(subset_data, f, indent=2)
        print(f"Saved {num_episodes} true D-dimensional uniform episodes (seed={seed}) to {subset_file}")

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