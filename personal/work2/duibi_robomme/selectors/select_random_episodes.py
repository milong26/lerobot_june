#!/usr/bin/env python
"""
Random episode selector for Robomme pipeline.
Reproduces the behavior of personal/work2/duibi/train_and_eval_scripts/select_random_episodes.py
but with a clean interface that works with LeRobotDataset directly.

Usage:
    python select_random_episodes.py \
        --dataset-root /path/to/dataset \
        --num-episodes 28 \
        --seed 42 \
        --output-dir /path/to/output
"""
import argparse
import json
import random
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def select_random_episodes(num_episodes, seed, dataset_root, output_dir=None):
    """Randomly select episodes and save indices."""
    print(f"Loading LeRobotDataset from {dataset_root}")
    dataset = LeRobotDataset(repo_id="1/2", root=dataset_root)

    total_episodes = dataset.num_episodes
    print(f"Total episodes: {total_episodes}")
    print(f"Selecting {num_episodes} episodes with seed={seed}")

    if num_episodes > total_episodes:
        raise ValueError(
            f"num_episodes ({num_episodes}) > total_episodes ({total_episodes})"
        )

    rng = random.Random(seed)
    selected = sorted(rng.sample(range(total_episodes), num_episodes))

    print(f"Selected episodes: {selected[:10]}{'...' if len(selected) > 10 else ''}")

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_file = output_path / f"random_{num_episodes}_seed{seed}.json"
        subset_data = {
            "method": "random",
            "num_episodes": num_episodes,
            "seed": seed,
            "selected_episode_indices": selected,
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
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    select_random_episodes(
        num_episodes=args.num_episodes,
        seed=args.seed,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )