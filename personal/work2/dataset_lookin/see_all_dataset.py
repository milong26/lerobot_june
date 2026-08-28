#!/usr/bin/env python
"""
Visualize ALL episodes in the dataset to see the original distribution.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JSON_PATH = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pickplacev3/episode_initial_states.json"
OUTPUT_IMAGE = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_lookin/see_all_episodes.png"


def main():
    print(f"Loading JSON: {JSON_PATH}")
    with open(JSON_PATH, "r") as f:
        metadata = json.load(f)

    episodes = metadata["episodes"]
    positions = np.array([ep["obj_init_pos"] for ep in episodes])

    x_vals = positions[:, 0]
    y_vals = positions[:, 1]
    z_vals = positions[:, 2]

    print(f"Total episodes: {len(positions)}")
    print(f"Position range: x[{x_vals.min():.3f}, {x_vals.max():.3f}], "
          f"y[{y_vals.min():.3f}, {y_vals.max():.3f}], "
          f"z[{z_vals.min():.3f}, {z_vals.max():.3f}]")

    z_range = z_vals.max() - z_vals.min()
    z_is_constant = z_range < 1e-6

    if z_is_constant:
        print("Z is constant, plotting 2D...")
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Left: all episodes
        ax = axes[0]
        scatter = ax.scatter(x_vals, y_vals, c=range(len(x_vals)), cmap="viridis", s=20, alpha=0.6)
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_title(f"All {len(positions)} Episodes", fontsize=14)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="Episode Index")

        # Right: 2D histogram
        ax = axes[1]
        hist = ax.hist2d(x_vals, y_vals, bins=30, cmap="YlOrRd")
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_title("Density Heatmap", fontsize=14)
        ax.set_aspect("equal")
        plt.colorbar(hist[3], ax=ax, label="Count")

        plt.tight_layout()
    else:
        print("Z varies, plotting 3D...")
        fig = plt.figure(figsize=(15, 6))

        ax1 = fig.add_subplot(121, projection="3d")
        ax1.scatter(x_vals, y_vals, z_vals, c=range(len(x_vals)), cmap="viridis", s=20, alpha=0.6)
        ax1.set_title(f"All {len(positions)} Episodes")

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap="viridis", s=20, alpha=0.6)
        ax2.set_title("Colored by Z")

    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
    print(f"Saved to: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()