#!/usr/bin/env python
"""
Step 2: Construct B0 - select 1 episode per cell (closest to cell center in physical space).
Output: B0 episode_index list.
"""
import json
import pickle
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("Step 2: Construct B0 (1 episode per cell)")
    print("=" * 60)

    with open(RESULTS_DIR / "step0_all_data.pkl", "rb") as f:
        data = pickle.load(f)

    with open(RESULTS_DIR / "step1_grid_info.json", "r") as f:
        grid_info = json.load(f)

    obj_positions = data["obj_positions"]
    episode_indices = data["episode_indices"]
    cell_centers = grid_info["cell_centers"]

    b0_episodes = []
    empty_cells = []

    for cell in cell_centers:
        cx, cy = cell["x_center"], cell["y_center"]
        phys_2d = obj_positions[:, :2]
        dists = np.sqrt((phys_2d[:, 0] - cx) ** 2 + (phys_2d[:, 1] - cy) ** 2)
        best_idx = np.argmin(dists)
        best_ep = episode_indices[best_idx]
        b0_episodes.append(int(best_ep))

    b0_unique = sorted(set(b0_episodes))
    print(f"B0 size (with duplicates): {len(b0_episodes)}")
    print(f"B0 unique episodes: {len(b0_unique)}")
    print(f"Empty cells: {len(empty_cells)}")

    b0_info = {
        "b0_episodes": b0_episodes,
        "b0_unique_episodes": b0_unique,
        "empty_cells": empty_cells,
    }

    out_path = RESULTS_DIR / "step2_b0_episodes.json"
    with open(out_path, "w") as f:
        json.dump(b0_info, f, indent=2)
    print(f"\nSaved B0 episodes to {out_path}")
    print(f"B0 episode indices: {b0_unique}")

    print("\nDone.")


if __name__ == "__main__":
    main()