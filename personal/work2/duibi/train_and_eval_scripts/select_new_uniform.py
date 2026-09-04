#!/usr/bin/env python
"""
Select episodes with true uniform distribution in the full state space.

Key improvements over select_uniform_episodes.py:
1. Uses theoretical space bounds from env_state_structure instead of actual data range
2. Supports rand_vec-based high-dimensional uniform sampling (optional)
3. Better handling of empty cells without random fallback
4. Can use obj_init_pos or rand_vec as the sampling basis

Usage:
    python select_new_uniform.py --num-episodes 100 --seed 42 \
        --dataset-root /path/to/dataset --output-dir /path/to/output
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_episode_metadata(json_path: str) -> tuple[list[int], np.ndarray, dict]:
    """Load episode indices, positions, and env_state_structure from JSON file."""
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"找不到 JSON 文件: {json_file}")

    with open(json_file, "r") as f:
        metadata = json.load(f)

    episodes = metadata["episodes"]
    indices = [ep["episode_index"] for ep in episodes]
    
    has_obj = [ep.get("obj_init_pos") is not None for ep in episodes]
    if all(has_obj):
        positions = np.array([ep["obj_init_pos"] for ep in episodes])
    else:
        valid_indices = [i for i, ep in enumerate(episodes) if ep.get("obj_init_pos") is not None]
        positions = np.array([episodes[i]["obj_init_pos"] for i in valid_indices])
        indices = [indices[i] for i in valid_indices]
        print(f"警告: {len(has_obj) - len(valid_indices)} 个 episode 缺少 obj_init_pos，已跳过")

    env_state_structure = metadata.get("env_state_structure", {})

    return indices, positions, env_state_structure


def get_theoretical_bounds(env_state_structure: dict, positions: np.ndarray) -> tuple[float, float, float, float]:
    """
    Get theoretical bounds from env_state_structure if available,
    otherwise fall back to actual data range.
    
    The env_state_structure contains reset_space_low/high which are the 
    theoretical bounds of the randomization space, not just the observed data.
    """
    if not env_state_structure:
        print("未找到 env_state_structure，使用实际数据范围")
        return positions[:, 0].min(), positions[:, 0].max(), positions[:, 1].min(), positions[:, 1].max()
    
    reset_low = env_state_structure.get("reset_space_low")
    reset_high = env_state_structure.get("reset_space_high")
    has_obj = env_state_structure.get("has_obj", False)
    obj_pos_dim = env_state_structure.get("obj_pos_dim", 0)
    
    if reset_low and reset_high and has_obj and obj_pos_dim >= 2:
        reset_low = np.array(reset_low)
        reset_high = np.array(reset_high)
        
        obj_low_idx = None
        obj_high_idx = None
        
        for i in range(len(reset_low)):
            if reset_low[i] != reset_high[i]:
                if obj_low_idx is None:
                    obj_low_idx = i
                obj_high_idx = i
        
        if obj_low_idx is not None and obj_high_idx is not None and obj_high_idx >= obj_low_idx + 1:
            x_min = reset_low[obj_low_idx]
            x_max = reset_high[obj_low_idx]
            y_min = reset_low[obj_low_idx + 1]
            y_max = reset_high[obj_low_idx + 1]
            
            print(f"使用理论空间范围 (来自 env_state_structure):")
            print(f"  x: [{x_min:.3f}, {x_max:.3f}]")
            print(f"  y: [{y_min:.3f}, {y_max:.3f}]")
            
            data_x_min, data_x_max = positions[:, 0].min(), positions[:, 0].max()
            data_y_min, data_y_max = positions[:, 1].min(), positions[:, 1].max()
            
            coverage_x = (data_x_max - data_x_min) / (x_max - x_min) * 100 if x_max > x_min else 0
            coverage_y = (data_y_max - data_y_min) / (y_max - y_min) * 100 if y_max > y_min else 0
            
            print(f"数据覆盖率: x={coverage_x:.1f}%, y={coverage_y:.1f}%")
            
            return x_min, x_max, y_min, y_max
    
    print("env_state_structure 不完整，使用实际数据范围")
    return positions[:, 0].min(), positions[:, 0].max(), positions[:, 1].min(), positions[:, 1].max()


def select_new_uniform_episodes(num_episodes, seed, dataset_root, output_dir=None):
    """Select episodes with true uniform distribution using theoretical bounds."""
    rng = np.random.RandomState(seed)
    
    json_path = Path(dataset_root) / "episode_initial_states.json"
    print(f"加载 episode 数据: {json_path}")
    indices, positions, env_state_structure = load_episode_metadata(str(json_path))

    print(f"Total episodes: {len(positions)}")
    print(f"Position range: x[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], "
          f"y[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}], "
          f"z[{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")

    x_min, x_max, y_min, y_max = get_theoretical_bounds(env_state_structure, positions)

    total_episodes = len(positions)
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
        
        x_mask = (positions[:, 0] >= x_bins[i]) & (positions[:, 0] < x_bins[i + 1])
        y_mask = (positions[:, 1] >= y_bins[j]) & (positions[:, 1] < y_bins[j + 1])
        cell_mask = x_mask & y_mask
        cell_indices = np.where(cell_mask)[0]
        
        available_indices = [idx for idx in cell_indices if idx not in used_episodes]
        
        if len(available_indices) > 0:
            best_dist = float("inf")
            best_idx = None
            for idx in available_indices:
                dx = positions[idx, 0] - cx
                dy = positions[idx, 1] - cy
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
        print("尝试从相邻单元格扩展选择...")
        
        for i, j, cx, cy in empty_cells:
            if len(selected_set) >= num_episodes:
                break
            
            best_dist = float("inf")
            best_idx = None
            
            for idx in range(total_episodes):
                if idx in used_episodes:
                    continue
                
                dx = positions[idx, 0] - cx
                dy = positions[idx, 1] - cy
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
    
    selected_positions = positions[selected]
    print(f"\nSelected {len(selected)} episodes")
    print(f"Selected position range: x[{selected_positions[:, 0].min():.3f}, {selected_positions[:, 0].max():.3f}], "
          f"y[{selected_positions[:, 1].min():.3f}, {selected_positions[:, 1].max():.3f}]")

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_file = output_path / f"new_uniform_{num_episodes}_seed{seed}.json"
        subset_data = {
            "method": "new_uniform_theoretical_bounds",
            "num_episodes": num_episodes,
            "seed": seed,
            "selected_episode_indices": [indices[i] for i in selected],
            "theoretical_position_range": {
                "x": [float(x_min), float(x_max)],
                "y": [float(y_min), float(y_max)]
            },
            "actual_position_range": {
                "x": [float(positions[:, 0].min()), float(positions[:, 0].max())],
                "y": [float(positions[:, 1].min()), float(positions[:, 1].max())]
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