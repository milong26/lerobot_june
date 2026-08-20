#!/usr/bin/env python
"""
Visualize the spatial distribution of selected episodes from uniform sampling.

Reads selected episode indices from uniform_100_seed42.json,
extracts initial object positions (xyz) from each episode,
and plots a 2D or 3D scatter plot saved as see_uniform_100.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, no popup window
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lerobot.datasets import LeRobotDataset

DATASET_ROOT = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pickplacev3"
SUBSET_FILE = "/data/zhonglinye/jun/lerobot/personal/work2/duibi/uniform_42/subsets/uniform_112_seed42.json"
OUTPUT_IMAGE = str(Path(__file__).parent / "see_uniform_112.png")


def main():
    # Load subset JSON
    print(f"Loading subset file: {SUBSET_FILE}")
    with open(SUBSET_FILE, "r") as f:
        subset_data = json.load(f)

    selected_indices = subset_data["selected_episode_indices"]
    print(f"Selected episodes: {len(selected_indices)}")

    # Load dataset
    print(f"Loading dataset from: {DATASET_ROOT}")
    dataset = LeRobotDataset(
        repo_id="work2/metaworld_pick_place",
        root=DATASET_ROOT
    )
    print(f"Total episodes in dataset: {dataset.num_episodes}")

    # Extract initial object positions for selected episodes
    print("Extracting initial object positions...")
    positions = []

    for ep_idx in selected_indices:
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        frame = dataset[int(from_idx)]
        env_state = frame["observation.environment_state"].numpy()
        x, y, z = env_state[4], env_state[5], env_state[6]
        positions.append([x, y, z])

    positions = np.array(positions)
    x_vals = positions[:, 0]
    y_vals = positions[:, 1]
    z_vals = positions[:, 2]

    print(f"Position range: x[{x_vals.min():.3f}, {x_vals.max():.3f}], "
          f"y[{y_vals.min():.3f}, {y_vals.max():.3f}], "
          f"z[{z_vals.min():.3f}, {z_vals.max():.3f}]")

    # Check if z values are all the same (within tolerance)
    z_range = z_vals.max() - z_vals.min()
    z_is_constant = z_range < 1e-6

    if z_is_constant:
        print("Z values are constant, plotting 2D scatter...")
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(x_vals, y_vals, c=range(len(x_vals)), cmap="viridis", s=50, alpha=0.8, edgecolors="black", linewidth=0.5)
        ax.set_xlabel("X (object position)", fontsize=12)
        ax.set_ylabel("Y (object position)", fontsize=12)
        ax.set_title(f"Uniform Sampling: 100 Episodes in Workspace (Z={z_vals[0]:.3f})", fontsize=14)
        ax.set_aspect("equal")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Episode Index", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        print("Z values vary, plotting 3D scatter...")
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(x_vals, y_vals, z_vals, c=range(len(x_vals)), cmap="viridis", s=50, alpha=0.8, edgecolors="black", linewidth=0.5)
        ax.set_xlabel("X (object position)", fontsize=12)
        ax.set_ylabel("Y (object position)", fontsize=12)
        ax.set_zlabel("Z (object position)", fontsize=12)
        ax.set_title("Uniform Sampling: 100 Episodes in 3D Workspace", fontsize=14)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label("Episode Index", fontsize=11)

    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {OUTPUT_IMAGE}")
    plt.close()


if __name__ == "__main__":
    main()