#!/usr/bin/env python
"""
Randomly select N episodes from the dataset and save the indices.
Reads obj_init_pos from episode_initial_states.json for fast analysis.
Usage:
    python select_random_episodes.py --num-episodes 100 --seed 42 --dataset-root /path/to/dataset --output-dir /path/to/output
"""
import argparse
import json
import random
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


def select_random_episodes(num_episodes, seed, dataset_root, total_episodes=None, output_dir=None):
    """Randomly select episodes and save indices."""
    json_path = Path(dataset_root) / "episode_initial_states.json"
    print(f"加载 episode 数据: {json_path}")
    indices, all_positions = load_episode_positions(str(json_path))

    if total_episodes is None:
        total_episodes = len(indices)
    else:
        total_episodes = min(total_episodes, len(indices))

    print(f"Total episodes: {total_episodes}")

    rng = random.Random(seed)
    selected = sorted(rng.sample(range(total_episodes), num_episodes))

    selected_positions = all_positions[selected]

    print(f"Position range: x[{all_positions[:, 0].min():.3f}, {all_positions[:, 0].max():.3f}], "
          f"y[{all_positions[:, 1].min():.3f}, {all_positions[:, 1].max():.3f}], "
          f"z[{all_positions[:, 2].min():.3f}, {all_positions[:, 2].max():.3f}]")

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_file = output_path / f"random_{num_episodes}_seed{seed}.json"
        subset_data = {
            "method": "random",
            "num_episodes": num_episodes,
            "seed": seed,
            "selected_episode_indices": [indices[i] for i in selected],
            "position_range": {
                "x": [float(all_positions[:, 0].min()), float(all_positions[:, 0].max())],
                "y": [float(all_positions[:, 1].min()), float(all_positions[:, 1].max())],
                "z": [float(all_positions[:, 2].min()), float(all_positions[:, 2].max())]
            },
            "selected_position_stats": {
                "x_mean": float(np.mean(selected_positions[:, 0])),
                "x_std": float(np.std(selected_positions[:, 0])),
                "y_mean": float(np.mean(selected_positions[:, 1])),
                "y_std": float(np.std(selected_positions[:, 1])),
                "z_mean": float(np.mean(selected_positions[:, 2])),
                "z_std": float(np.std(selected_positions[:, 2]))
            }
        }
        with open(subset_file, "w") as f:
            json.dump(subset_data, f, indent=2)
        print(f"Saved {num_episodes} random episodes (seed={seed}) to {subset_file}")

    return [indices[i] for i in selected]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-root", type=str, required=True, help="数据集根目录，JSON 文件位于此目录下")
    parser.add_argument("--total-episodes", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    select_random_episodes(
        num_episodes=args.num_episodes,
        seed=args.seed,
        dataset_root=args.dataset_root,
        total_episodes=args.total_episodes,
        output_dir=args.output_dir,
    )