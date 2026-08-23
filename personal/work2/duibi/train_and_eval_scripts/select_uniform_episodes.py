#!/usr/bin/env python
"""
Select episodes with uniform distribution in workspace based on obj_init_pos from JSON file.
Uses a grid-based approach: divides workspace into cells, picks one episode per cell closest to center.
Usage:
    python select_uniform_episodes.py --num-episodes 100 --seed 42 --dataset-root /path/to/dataset --output-dir /path/to/output
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_episode_positions(json_path: str) -> tuple[list[int], np.ndarray]:
    """Load episode indices and initial object positions from JSON file."""
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"找不到 JSON 文件: {json_file}")

    with open(json_file, "r") as f:
        metadata = json.load(f)

    episodes = metadata["episodes"]
    indices = [ep["episode_index"] for ep in episodes]
    positions = np.array([ep["obj_init_pos"] for ep in episodes])

    return indices, positions


def select_uniform_episodes(num_episodes, seed, dataset_root, output_dir=None):
    """Select episodes with uniform distribution in workspace."""
    rng = np.random.RandomState(seed)
    
    json_path = Path(dataset_root) / "episode_initial_states.json"
    print(f"加载 episode 数据: {json_path}")
    indices, positions = load_episode_positions(str(json_path))

    print(f"Total episodes: {len(positions)}")
    print(f"Position range: x[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], "
          f"y[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}], "
          f"z[{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")

    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

    total_episodes = len(positions)
    range_x = x_max - x_min
    range_y = y_max - y_min
    aspect_ratio = range_x / range_y if range_y > 0 else 1.0

    # Calculate grid dimensions proportional to aspect ratio
    n_x = int(np.floor(np.sqrt(num_episodes * aspect_ratio)))
    n_y = int(np.ceil(num_episodes / n_x))
    
    # Ensure we have exactly enough cells (prefer n_x * n_y == num_episodes)
    # Try to find a better combination
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

    # Create grid cell centers
    x_bins = np.linspace(x_min, x_max, n_x + 1)
    y_bins = np.linspace(y_min, y_max, n_y + 1)
    
    cell_centers = []
    for i in range(n_x):
        for j in range(n_y):
            cx = (x_bins[i] + x_bins[i + 1]) / 2
            cy = (y_bins[j] + y_bins[j + 1]) / 2
            cell_centers.append((i, j, cx, cy))
    
    # Shuffle cell order for randomness
    rng.shuffle(cell_centers)
    
    # For each cell, find the episode closest to the cell center
    selected_set = set()
    used_episodes = set()
    
    for i, j, cx, cy in cell_centers:
        if len(selected_set) >= num_episodes:
            break
        
        # Find episodes in this cell
        x_mask = (positions[:, 0] >= x_bins[i]) & (positions[:, 0] < x_bins[i + 1])
        y_mask = (positions[:, 1] >= y_bins[j]) & (positions[:, 1] < y_bins[j + 1])
        cell_mask = x_mask & y_mask
        cell_indices = np.where(cell_mask)[0]
        
        # Filter out already used episodes
        available_indices = [idx for idx in cell_indices if idx not in used_episodes]
        
        if len(available_indices) > 0:
            # Pick the episode closest to the cell center
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
    
    # If we still need more episodes, randomly select from remaining
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
    print(f"Selected {len(selected)} episodes")
    print(f"Selected position range: x[{selected_positions[:, 0].min():.3f}, {selected_positions[:, 0].max():.3f}], "
          f"y[{selected_positions[:, 1].min():.3f}, {selected_positions[:, 1].max():.3f}]")

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_file = output_path / f"uniform_{num_episodes}_seed{seed}.json"
        subset_data = {
            "method": "uniform_workspace",
            "num_episodes": num_episodes,
            "seed": seed,
            "selected_episode_indices": [indices[i] for i in selected],
            "position_range": {
                "x": [float(x_min), float(x_max)],
                "y": [float(y_min), float(y_max)]
            }
        }
        with open(subset_file, "w") as f:
            json.dump(subset_data, f, indent=2)
        print(f"Saved {num_episodes} uniform episodes (seed={seed}) to {subset_file}")

    return [indices[i] for i in selected]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-root", type=str, required=True, help="数据集根目录，JSON 文件位于此目录下")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    select_uniform_episodes(
        num_episodes=args.num_episodes,
        seed=args.seed,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )