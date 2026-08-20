#!/usr/bin/env python
"""
Randomly select N episodes from the dataset and save the indices.
Usage:
    python select_random_episodes.py --num-episodes 100 --seed 42 --dataset-root /path/to/dataset --output-dir /path/to/output
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from lerobot.datasets import LeRobotDataset


def extract_initial_positions(dataset, episode_indices):
    """Extract initial object positions from the first frame of specified episodes."""
    positions = []

    for ep_idx in episode_indices:
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        frame = dataset[int(from_idx)]
        env_state = frame["observation.environment_state"].numpy()
        pos = [env_state[4], env_state[5], env_state[6]]
        positions.append(pos)

    return np.array(positions)


def select_random_episodes(num_episodes, seed, dataset_root, total_episodes=None, output_dir=None):
    """Randomly select episodes and save indices."""
    print("Loading dataset...")
    dataset = LeRobotDataset(
        repo_id="lerobot/metaworld_pick_place",
        root=dataset_root
    )

    if total_episodes is None:
        total_episodes = dataset.num_episodes

    print(f"Total episodes: {total_episodes}")

    rng = random.Random(seed)
    selected = sorted(rng.sample(range(total_episodes), num_episodes))

    print("Extracting initial object positions for analysis...")
    all_positions = extract_initial_positions(dataset, range(total_episodes))
    selected_positions = extract_initial_positions(dataset, selected)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_file = output_path / f"random_{num_episodes}_seed{seed}.json"
        subset_data = {
            "method": "random",
            "num_episodes": num_episodes,
            "seed": seed,
            "selected_episode_indices": selected,
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

    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
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