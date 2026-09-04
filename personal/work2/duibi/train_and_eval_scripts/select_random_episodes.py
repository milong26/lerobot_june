#!/usr/bin/env python
"""
Randomly select N episodes from the LeRobot dataset and save the indices.
Directly reads episodes from LeRobotDataset without requiring episode_initial_states.json.
Usage:
    python select_random_episodes.py --num-episodes 100 --seed 42 --dataset-root /path/to/dataset --repo-id work2/dataset_name --output-dir /path/to/output
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np


def load_episodes_from_dataset(dataset_root: str) -> list[int]:
    """Load episode indices from LeRobotDataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    print(f"加载 LeRobotDataset from {dataset_root}")
    dataset = LeRobotDataset(repo_id="1/2", root=dataset_root)
    
    total_episodes = dataset.num_episodes
    print(f"数据集总 episode 数: {total_episodes}")
    
    # Get all episode indices
    indices = list(range(total_episodes))
    
    return indices


def select_random_episodes(num_episodes, seed, dataset_root, total_episodes=None, output_dir=None):
    """Randomly select episodes and save indices."""
    indices = load_episodes_from_dataset(dataset_root)

    if total_episodes is None:
        total_episodes = len(indices)
    else:
        total_episodes = min(total_episodes, len(indices))

    print(f"可用 episode 数: {total_episodes}")
    print(f"需要选择: {num_episodes} episodes")

    rng = random.Random(seed)
    selected = sorted(rng.sample(range(total_episodes), num_episodes))

    print(f"已选择 episodes: {selected[:10]}{'...' if len(selected) > 10 else ''}")

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_file = output_path / f"random_{num_episodes}_seed{seed}.json"
        subset_data = {
            "method": "random",
            "num_episodes": num_episodes,
            "seed": seed,
            "selected_episode_indices": [indices[i] for i in selected],
        }
        with open(subset_file, "w") as f:
            json.dump(subset_data, f, indent=2)
        print(f"Saved {num_episodes} random episodes (seed={seed}) to {subset_file}")

    return [indices[i] for i in selected]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-root", type=str, required=True, help="LeRobot 数据集根目录")
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