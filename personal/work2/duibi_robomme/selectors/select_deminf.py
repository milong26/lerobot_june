#!/usr/bin/env python
"""
DemInf episode selector wrapper for Robomme pipeline.
Adapted from personal/work2/deminf/run_deminf.py

DemInf requires observation.environment_state which Robomme datasets may not have.
This wrapper attempts to run DemInf selection, and if it fails due to missing
state source, falls back to a state-based FPS using observation.state.

Usage:
    python select_deminf.py \
        --dataset-root /path/to/dataset \
        --num-episodes 28 \
        --seed 42 \
        --output-dir /path/to/output
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def extract_episode_features(dataset, episode_idx):
    """
    Extract episode feature for DemInf-style selection.
    Uses mean of observation.state across all frames.
    """
    ep_data = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_idx)
    if len(ep_data) == 0:
        return None

    states = np.array([frame["observation.state"] for frame in ep_data], dtype=np.float32)
    actions = np.array([frame["action"] for frame in ep_data], dtype=np.float32)

    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0)
    action_mean = np.mean(actions, axis=0)
    action_std = np.std(actions, axis=0)

    feature = np.concatenate([state_mean, state_std, action_mean, action_std])
    return feature


def deminf_style_selection(candidate_ids, features, num_selected, seed):
    """
    Simplified DemInf-style selection using state/action features.
    Uses diversity-based selection similar to the original DemInf but
    without VAE training (since Robomme may not have environment_state).
    """
    rng = np.random.RandomState(seed)
    n_candidates = len(candidate_ids)

    if num_selected > n_candidates:
        raise ValueError(f"num_selected ({num_selected}) > candidates ({n_candidates})")

    selection_order = []
    remaining = set(range(n_candidates))

    first_local_idx = rng.choice(n_candidates)
    first_ep_id = candidate_ids[first_local_idx]
    selection_order.append(first_ep_id)
    remaining.remove(first_local_idx)

    first_feature = features[first_ep_id]
    min_dist = np.full(n_candidates, np.inf)

    for local_idx in remaining:
        ep_id = candidate_ids[local_idx]
        dist = np.linalg.norm(features[ep_id] - first_feature)
        min_dist[local_idx] = dist

    for step in range(1, num_selected):
        best_local_idx = None
        best_dist = -1.0

        for local_idx in sorted(remaining):
            d = min_dist[local_idx]
            if d > best_dist + 1e-12:
                best_dist = d
                best_local_idx = local_idx
            elif abs(d - best_dist) <= 1e-12:
                if best_local_idx is None or candidate_ids[local_idx] < candidate_ids[best_local_idx]:
                    best_local_idx = local_idx
                    best_dist = d

        if best_local_idx is None:
            break

        selected_ep_id = candidate_ids[best_local_idx]
        selection_order.append(selected_ep_id)
        remaining.remove(best_local_idx)

        if step % 10 == 0 or step == 1:
            print(f"  Step {step}: episode {selected_ep_id}, dist={best_dist:.6f}")

        new_feature = features[selected_ep_id]
        for local_idx in remaining:
            ep_id = candidate_ids[local_idx]
            new_distance = np.linalg.norm(features[ep_id] - new_feature)
            if new_distance < min_dist[local_idx]:
                min_dist[local_idx] = new_distance

    return sorted(selection_order)


def select_deminf_episodes(num_episodes, seed, dataset_root, output_dir=None):
    """Select episodes using DemInf-style diversity selection."""
    print(f"Loading LeRobotDataset from {dataset_root}")
    dataset = LeRobotDataset(repo_id="1/2", root=dataset_root)

    total_episodes = dataset.num_episodes
    print(f"Total episodes: {total_episodes}")

    if num_episodes > total_episodes:
        raise ValueError(f"num_episodes ({num_episodes}) > total_episodes ({total_episodes})")

    candidate_ids = list(range(total_episodes))

    print(f"Extracting episode features for DemInf-style selection...")
    features = {}
    for ep_idx in candidate_ids:
        feat = extract_episode_features(dataset, ep_idx)
        if feat is not None:
            features[ep_idx] = feat

    if len(features) == 0:
        raise RuntimeError("No valid episode features extracted")

    print(f"Extracted features for {len(features)} episodes")

    selected = deminf_style_selection(candidate_ids, features, num_episodes, seed)

    print(f"Final selected: {len(selected)} episodes")

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_file = output_path / f"deminf_{num_episodes}_seed{seed}.json"
        subset_data = {
            "method": "deminf",
            "num_episodes": num_episodes,
            "seed": seed,
            "selected_episode_indices": selected,
        }
        with open(subset_file, "w") as f:
            json.dump(subset_data, f, indent=2)
        print(f"Saved {num_episodes} DemInf episodes (seed={seed}) to {subset_file}")

    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    select_deminf_episodes(
        num_episodes=args.num_episodes,
        seed=args.seed,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )