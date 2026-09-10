#!/usr/bin/env python
"""
Select episodes from LeRobot dataset using different strategies.

Usage:
    python select_episodes.py --method random --num-episodes 100 --seed 42 \
        --dataset-root /path/to/dataset --output-dir /path/to/output
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np


def load_episode_count(dataset_root: str) -> int:
    """Load episode count from LeRobotDataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading LeRobotDataset from {dataset_root}")
    dataset = LeRobotDataset(repo_id="1/2", root=dataset_root)
    total_episodes = dataset.num_episodes
    print(f"Total episodes: {total_episodes}")
    return total_episodes


def select_random(total_episodes: int, num_episodes: int, seed: int) -> list[int]:
    """Randomly select episodes."""
    rng = random.Random(seed)
    selected = sorted(rng.sample(range(total_episodes), num_episodes))
    print(f"Random selection: {num_episodes} episodes from {total_episodes}")
    print(f"Selected: {selected[:10]}{'...' if len(selected) > 10 else ''}")
    return selected


def select_grid_uniform(total_episodes: int, num_episodes: int, seed: int) -> list[int]:
    """Select episodes with uniform spacing (grid-based)."""
    if num_episodes >= total_episodes:
        selected = list(range(total_episodes))
    else:
        step = total_episodes / num_episodes
        selected = sorted([int(i * step + step / 2) for i in range(num_episodes)])
        selected = [min(ep, total_episodes - 1) for ep in selected]
    print(f"Grid uniform selection: {num_episodes} episodes from {total_episodes}")
    print(f"Selected: {selected[:10]}{'...' if len(selected) > 10 else ''}")
    return selected


def select_first_n(total_episodes: int, num_episodes: int, seed: int = 0) -> list[int]:
    """Select first N episodes."""
    selected = list(range(min(num_episodes, total_episodes)))
    print(f"First N selection: {num_episodes} episodes from {total_episodes}")
    print(f"Selected: {selected[:10]}{'...' if len(selected) > 10 else ''}")
    return selected


def select_last_n(total_episodes: int, num_episodes: int, seed: int = 0) -> list[int]:
    """Select last N episodes."""
    start = max(0, total_episodes - num_episodes)
    selected = list(range(start, total_episodes))
    print(f"Last N selection: {num_episodes} episodes from {total_episodes}")
    print(f"Selected: {selected[:10]}{'...' if len(selected) > 10 else ''}")
    return selected


METHODS = {
    "random": select_random,
    "grid_uniform": select_grid_uniform,
    "first_n": select_first_n,
    "last_n": select_last_n,
}


def main():
    parser = argparse.ArgumentParser(description="Select episodes from LeRobot dataset")
    parser.add_argument("--method", type=str, default="random", choices=METHODS.keys())
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    total_episodes = load_episode_count(args.dataset_root)
    num_episodes = min(args.num_episodes, total_episodes)

    select_fn = METHODS[args.method]
    selected = select_fn(total_episodes, num_episodes, args.seed)

    # Save to output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "selected_episodes.json"
    with open(output_file, "w") as f:
        json.dump({
            "method": args.method,
            "seed": args.seed,
            "total_episodes": total_episodes,
            "selected_episodes": selected,
            "num_selected": len(selected),
        }, f, indent=2)

    print(f"\nSaved {len(selected)} episode indices to {output_file}")
    return str(output_file)


if __name__ == "__main__":
    main()