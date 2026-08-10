"""
Step 1: Build candidate pool from 500 episodes.
Extract object position, goal position, and metadata.
Save to work2/ours/pool/
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

POOL_DIR = Path(__file__).parent / "pool"
DATASET_META = Path(__file__).parent.parent / "dataset" / "episode_initial_states.json"


def build_pool(meta_path=None, output_dir=None):
    if meta_path is None:
        meta_path = DATASET_META
    if output_dir is None:
        output_dir = POOL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_path) as f:
        meta = json.load(f)

    episodes = meta["episodes"]
    n = len(episodes)
    print(f"Loading {n} episodes from {meta_path}")

    obj_positions = []
    goal_positions = []
    episode_indices = []
    success_flags = []
    num_frames = []

    for ep in episodes:
        idx = ep["episode_index"]
        obj_pos = ep["obj_init_pos"]
        goal = ep["goal_pose"]
        obj_positions.append(obj_pos)
        goal_positions.append(goal)
        episode_indices.append(idx)
        success_flags.append(ep["success"])
        num_frames.append(ep["num_frames"])

    obj_positions = np.array(obj_positions, dtype=np.float32)
    goal_positions = np.array(goal_positions, dtype=np.float32)
    episode_indices = np.array(episode_indices, dtype=np.int32)
    success_flags = np.array(success_flags, dtype=bool)
    num_frames = np.array(num_frames, dtype=np.int32)

    # Save CSV
    df = pd.DataFrame({
        "episode_index": episode_indices,
        "obj_x": obj_positions[:, 0],
        "obj_y": obj_positions[:, 1],
        "obj_z": obj_positions[:, 2],
        "goal_x": goal_positions[:, 0],
        "goal_y": goal_positions[:, 1],
        "goal_z": goal_positions[:, 2],
        "success": success_flags,
        "num_frames": num_frames,
    })
    csv_path = output_dir / "episode_metadata.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved episode metadata to {csv_path}")

    # Save numpy arrays
    np.save(output_dir / "obj_positions.npy", obj_positions)
    np.save(output_dir / "goal_positions.npy", goal_positions)
    np.save(output_dir / "episode_indices.npy", episode_indices)

    # Save workspace statistics
    workspace_stats = {
        "num_episodes": int(n),
        "obj_position": {
            "x_min": float(obj_positions[:, 0].min()),
            "x_max": float(obj_positions[:, 0].max()),
            "y_min": float(obj_positions[:, 1].min()),
            "y_max": float(obj_positions[:, 1].max()),
            "z_min": float(obj_positions[:, 2].min()),
            "z_max": float(obj_positions[:, 2].max()),
            "x_mean": float(obj_positions[:, 0].mean()),
            "y_mean": float(obj_positions[:, 1].mean()),
            "z_mean": float(obj_positions[:, 2].mean()),
            "x_std": float(obj_positions[:, 0].std()),
            "y_std": float(obj_positions[:, 1].std()),
            "z_std": float(obj_positions[:, 2].std()),
        },
        "goal_position": {
            "x_min": float(goal_positions[:, 0].min()),
            "x_max": float(goal_positions[:, 0].max()),
            "y_min": float(goal_positions[:, 1].min()),
            "y_max": float(goal_positions[:, 1].max()),
            "z_min": float(goal_positions[:, 2].min()),
            "z_max": float(goal_positions[:, 2].max()),
        },
        "success_rate": float(success_flags.mean()),
        "mean_frames": float(num_frames.mean()),
    }

    stats_path = output_dir / "workspace_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(workspace_stats, f, indent=2)
    print(f"Saved workspace statistics to {stats_path}")

    # Print summary
    print("\n=== Pool Summary ===")
    print(f"Episodes: {n}")
    print(f"Object position range: x=[{workspace_stats['obj_position']['x_min']:.4f}, {workspace_stats['obj_position']['x_max']:.4f}], "
          f"y=[{workspace_stats['obj_position']['y_min']:.4f}, {workspace_stats['obj_position']['y_max']:.4f}], "
          f"z=[{workspace_stats['obj_position']['z_min']:.4f}, {workspace_stats['obj_position']['z_max']:.4f}]")
    print(f"Goal position range: x=[{workspace_stats['goal_position']['x_min']:.4f}, {workspace_stats['goal_position']['x_max']:.4f}], "
          f"y=[{workspace_stats['goal_position']['y_min']:.4f}, {workspace_stats['goal_position']['y_max']:.4f}], "
          f"z=[{workspace_stats['goal_position']['z_min']:.4f}, {workspace_stats['goal_position']['z_max']:.4f}]")
    print(f"Success rate: {workspace_stats['success_rate']:.2%}")
    print(f"Mean frames: {workspace_stats['mean_frames']:.1f}")

    return obj_positions, goal_positions, episode_indices, workspace_stats


if __name__ == "__main__":
    build_pool()