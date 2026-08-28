#!/usr/bin/env python
"""
Check the uniform selection results and compare with all episodes.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JSON_PATH = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pickplacev3/episode_initial_states.json"
SUBSET_FILE = "/data/zhonglinye/jun/lerobot/personal/work2/duibi/uniform_42/subsets/uniform_112_seed42.json"
OUTPUT_IMAGE = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_lookin/check_uniform_selection.png"


def main():
    with open(JSON_PATH, "r") as f:
        metadata = json.load(f)

    episodes = metadata["episodes"]
    all_positions = np.array([ep["obj_init_pos"] for ep in episodes])
    all_indices = [ep["episode_index"] for ep in episodes]

    with open(SUBSET_FILE, "r") as f:
        subset_data = json.load(f)

    selected_indices = subset_data["selected_episode_indices"]

    # Map episode_index to position
    idx_to_pos = {ep["episode_index"]: ep["obj_init_pos"] for ep in episodes}
    selected_positions = np.array([idx_to_pos[idx] for idx in selected_indices])

    x_all = all_positions[:, 0]
    y_all = all_positions[:, 1]
    x_sel = selected_positions[:, 0]
    y_sel = selected_positions[:, 1]

    print(f"All episodes: {len(all_positions)}")
    print(f"Selected episodes: {len(selected_positions)}")
    print(f"Selected indices: {selected_indices[:20]}...")

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Left: all episodes
    ax = axes[0]
    ax.scatter(x_all, y_all, s=15, alpha=0.5, color="gray", label="All")
    ax.set_title(f"All {len(all_positions)} Episodes", fontsize=14)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Middle: selected episodes
    ax = axes[1]
    ax.scatter(x_all, y_all, s=10, alpha=0.3, color="gray", label="All")
    ax.scatter(x_sel, y_sel, s=50, alpha=0.8, color="red", edgecolors="black", linewidth=0.5, label="Selected")
    ax.set_title(f"Selected {len(selected_positions)} Episodes", fontsize=14)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right: 2D histogram of selected
    ax = axes[2]
    hist = ax.hist2d(x_sel, y_sel, bins=10, cmap="YlOrRd", range=[[-0.1, 0.1], [0.6, 0.7]])
    ax.set_title("Selected Episodes Density", fontsize=14)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_aspect("equal")
    plt.colorbar(hist[3], ax=ax, label="Count")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
    print(f"Saved to: {OUTPUT_IMAGE}")

    # Print grid info
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    range_x = x_max - x_min
    range_y = y_max - y_min
    aspect_ratio = range_x / range_y
    n_x = int(round(np.sqrt(112 * aspect_ratio)))
    n_y = int(round(112 / n_x))
    while n_x * n_y < 112:
        n_y += 1
    print(f"\nGrid: n_x={n_x}, n_y={n_y}, total cells={n_x*n_y}")
    print(f"Aspect ratio: {aspect_ratio:.2f}")


if __name__ == "__main__":
    main()