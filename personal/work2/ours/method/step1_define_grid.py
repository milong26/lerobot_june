#!/usr/bin/env python
"""
Step 1: Define 5x5 physical grid based on episode initial positions (obj_x, obj_y).
Generate 25 cell center coordinates and save grid info.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_GRID_X = 5
N_GRID_Y = 5


def main():
    print("=" * 60)
    print("Step 1: Define 5x5 physical grid")
    print("=" * 60)

    with open(RESULTS_DIR / "step0_all_data.pkl", "rb") as f:
        data = pickle.load(f)

    obj_positions = data["obj_positions"]
    episode_indices = data["episode_indices"]

    xs = obj_positions[:, 0]
    ys = obj_positions[:, 1]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    print(f"X range: {x_min:.4f} ~ {x_max:.4f}")
    print(f"Y range: {y_min:.4f} ~ {y_max:.4f}")

    x_edges = np.linspace(x_min, x_max, N_GRID_X + 1)
    y_edges = np.linspace(y_min, y_max, N_GRID_Y + 1)

    cell_centers = []
    for i in range(N_GRID_X):
        for j in range(N_GRID_Y):
            cx = (x_edges[i] + x_edges[i + 1]) / 2
            cy = (y_edges[j] + y_edges[j + 1]) / 2
            cell_centers.append({
                "cell_id": i * N_GRID_Y + j,
                "grid_i": i,
                "grid_j": j,
                "x_center": float(cx),
                "y_center": float(cy),
                "x_min": float(x_edges[i]),
                "x_max": float(x_edges[i + 1]),
                "y_min": float(y_edges[j]),
                "y_max": float(y_edges[j + 1]),
            })

    print(f"\nGenerated {len(cell_centers)} cell centers")

    grid_info = {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "n_grid_x": N_GRID_X,
        "n_grid_y": N_GRID_Y,
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
        "cell_centers": cell_centers,
    }

    out_path = RESULTS_DIR / "step1_grid_info.json"
    with open(out_path, "w") as f:
        json.dump(grid_info, f, indent=2)
    print(f"Saved grid info to {out_path}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, c="blue", s=10, alpha=0.5, label="Episodes")
    for cell in cell_centers:
        rect = plt.Rectangle(
            (cell["x_min"], cell["y_min"]),
            cell["x_max"] - cell["x_min"],
            cell["y_max"] - cell["y_min"],
            fill=False, edgecolor="red", linewidth=0.8, alpha=0.6,
        )
        ax.add_patch(rect)
        ax.plot(cell["x_center"], cell["y_center"], "r+", markersize=6)
    ax.set_xlabel("Object X")
    ax.set_ylabel("Object Y")
    ax.set_title("5x5 Grid over Episode Initial Positions")
    ax.legend()
    ax.set_aspect("equal")
    fig_path = RESULTS_DIR / "step1_grid_visualization.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved grid visualization to {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()