#!/usr/bin/env python
"""
Select episodes with uniform distribution in workspace based on environment_state.
Usage:
    python select_uniform_episodes.py --num-episodes 100 --seed 42 --dataset-root /path/to/dataset --output-dir /path/to/output
"""
import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets import LeRobotDataset


def extract_initial_positions(dataset):
    """Extract initial object positions from the first frame of each episode."""
    num_episodes = dataset.num_episodes
    print(f"Total episodes: {num_episodes}")
    print("Extracting initial object positions from first frame of each episode...")

    positions = []
    batch_size = 50

    for start in range(0, num_episodes, batch_size):
        end = min(start + batch_size, num_episodes)
        for ep_idx in range(start, end):
            from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
            frame = dataset[int(from_idx)]
            env_state = frame["observation.environment_state"].numpy()
            pos = [env_state[4], env_state[5], env_state[6]]
            positions.append(pos)

        print(f"  Processed {end}/{num_episodes} episodes...")

    return np.array(positions)


def select_uniform_episodes(num_episodes, seed, dataset_root, output_dir=None):
    """Select episodes with uniform distribution in workspace."""
    rng = np.random.RandomState(seed)
    
    print("Loading dataset...")
    dataset = LeRobotDataset(
        repo_id="lerobot/metaworld_pick_place",
        root=dataset_root
    )

    print("Extracting initial object positions...")
    positions = extract_initial_positions(dataset)
    print(f"Position range: x[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], "
          f"y[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}], "
          f"z[{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")

    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

    total_episodes = len(positions)

    range_x = x_max - x_min
    range_y = y_max - y_min
    range_z = z_max - z_min

    # Detect if z is constant (degenerate to 2D grid on x-y plane)
    z_is_constant = range_z < 1e-6

    if z_is_constant:
        print("Z is constant, using 2D grid on x-y plane...")
        # Calculate 2D grid size proportional to x-y ranges
        aspect_ratio = range_x / range_y if range_y > 0 else 1.0
        n_x = int(round(np.sqrt(num_episodes * aspect_ratio)))
        n_y = int(round(num_episodes / n_x))

        # Ensure n_x * n_y >= num_episodes
        while n_x * n_y < num_episodes:
            n_y += 1

        n_z = 1
        print(f"2D Grid size: {n_x} x {n_y} = {n_x * n_y} cells")
    else:
        # 3D grid
        total_range = range_x + range_y + range_z
        n_x = max(1, int(round(num_episodes ** (1/3) * range_x / total_range * 3)))
        n_y = max(1, int(round(num_episodes ** (1/3) * range_y / total_range * 3)))
        n_z = max(1, num_episodes // (n_x * n_y))

        if n_x * n_y * n_z < num_episodes:
            n_z += 1

        print(f"3D Grid size: {n_x} x {n_y} x {n_z} = {n_x * n_y * n_z} cells")

    x_bins = np.linspace(x_min, x_max, n_x + 1)
    y_bins = np.linspace(y_min, y_max, n_y + 1)
    z_bins = np.linspace(z_min, z_max, n_z + 1) if n_z > 1 else [z_min, z_max + 1e-6]

    # Select episodes: for each cell, pick the episode closest to the cell center
    selected_set = set()
    empty_cells = []

    for i in range(n_x):
        for j in range(n_y):
            if len(selected_set) >= num_episodes:
                break

            cx = (x_bins[i] + x_bins[i + 1]) / 2
            cy = (y_bins[j] + y_bins[j + 1]) / 2

            # Find episodes in this cell
            x_mask = (positions[:, 0] >= x_bins[i]) & (positions[:, 0] < x_bins[i + 1])
            y_mask = (positions[:, 1] >= y_bins[j]) & (positions[:, 1] < y_bins[j + 1])
            cell_mask = x_mask & y_mask
            cell_indices = np.where(cell_mask)[0]

            if len(cell_indices) > 0:
                # Pick the episode closest to the cell center
                best_dist = float("inf")
                best_idx = None
                for idx in cell_indices:
                    dx = positions[idx, 0] - cx
                    dy = positions[idx, 1] - cy
                    dist = dx * dx + dy * dy
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = int(idx)
                selected_set.add(best_idx)
            else:
                empty_cells.append((i, j, cx, cy))

    # Fill empty cells with nearest unselected episode
    if empty_cells and len(selected_set) < num_episodes:
        unselected = [ep for ep in range(total_episodes) if ep not in selected_set]
        for i, j, cx, cy in empty_cells:
            if len(selected_set) >= num_episodes:
                break
            best_dist = float("inf")
            best_idx = None
            for ep_idx in unselected:
                dx = positions[ep_idx, 0] - cx
                dy = positions[ep_idx, 1] - cy
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_idx = ep_idx
            if best_idx is not None:
                selected_set.add(best_idx)
                unselected.remove(best_idx)

    # Final fallback: random fill if still not enough
    if len(selected_set) < num_episodes:
        remaining = [ep for ep in range(total_episodes) if ep not in selected_set]
        extra = rng.choice(remaining, num_episodes - len(selected_set), replace=False)
        selected_set.update(extra.tolist())

    selected = sorted(list(selected_set))[:num_episodes]

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_file = output_path / f"uniform_{num_episodes}_seed{seed}.json"
        subset_data = {
            "method": "uniform_workspace",
            "num_episodes": num_episodes,
            "seed": seed,
            "selected_episode_indices": selected,
            "position_range": {
                "x": [float(x_min), float(x_max)],
                "y": [float(y_min), float(y_max)],
                "z": [float(z_min), float(z_max)]
            }
        }
        with open(subset_file, "w") as f:
            json.dump(subset_data, f, indent=2)
        print(f"Saved {num_episodes} uniform episodes (seed={seed}) to {subset_file}")

    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    select_uniform_episodes(
        num_episodes=args.num_episodes,
        seed=args.seed,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )