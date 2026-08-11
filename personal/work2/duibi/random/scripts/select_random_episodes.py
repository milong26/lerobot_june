#!/usr/bin/env python
"""
Randomly select N episodes from the dataset and save the indices.
Usage:
    python select_random_episodes.py --num-episodes 100 --seed 42 --output-dir personal/work2/duibi/random/subsets
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np


def select_random_episodes(num_episodes: int, seed: int, total_episodes: int = 500, output_dir: str = None):
    """Randomly select episodes and save indices."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    # Randomly select without replacement
    selected = sorted(rng.sample(range(total_episodes), num_episodes))
    
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
    parser.add_argument("--total-episodes", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    
    select_random_episodes(
        num_episodes=args.num_episodes,
        seed=args.seed,
        total_episodes=args.total_episodes,
        output_dir=args.output_dir,
    )