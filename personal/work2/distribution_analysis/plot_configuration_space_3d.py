#!/usr/bin/env python
"""
3D Configuration Space Visualization for Paper

Generates a 2x2 subplot figure showing 3D scatter + occupied volume + 2D occupancy inset
for four demonstration selection methods: Grid-Uniform, Random, DemInf, Ours.

This visualization validates whether configuration-aware selection changes the
demonstration coverage structure in configuration space.

Usage:
    python plot_configuration_space_3d.py \
        --candidate-pool /path/to/episode_initial_states.json \
        --selection-grid /path/to/grid_uniform_subset.json \
        --selection-random /path/to/random_subset.json \
        --selection-deminf /path/to/deminf_subset.json \
        --selection-ours /path/to/ours_subset.json \
        --budget 112 \
        --output-dir personal/work2/distribution_analysis/results/
"""

import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

WORK2_ROOT = Path(__file__).resolve().parent.parent
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

from our_v5.select_our_v5 import load_rand_vecs

METHOD_COLORS = {
    "grid_uniform": "#1F77B4",
    "random": "#E74C3C",
    "deminf": "#FF8C00",
    "ours": "#2ECC71",
}

METHOD_LABELS = {
    "grid_uniform": "Grid-Uniform",
    "random": "Random",
    "deminf": "DemInf",
    "ours": "Ours",
}

METHOD_SUBPLOT_LETTERS = {
    "grid_uniform": "(a)",
    "random": "(b)",
    "deminf": "(c)",
    "ours": "(d)",
}

VOXEL_BINS = 8
DPI = 300
FIGSIZE_2X2 = (14.0, 12.0)


def setup_logging(output_dir: Path) -> logging.Logger:
    log_file = output_dir / "config_3d_vis.log"
    logger = logging.getLogger("config_3d_vis")
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_subset_indices(subset_path: Path, logger: logging.Logger) -> List[int]:
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_path}")
    with open(subset_path, "r") as f:
        data = json.load(f)
    for key in ["selected_episode_indices", "selected_episode_ids"]:
        if key in data:
            indices = data[key]
            logger.info(f"Loaded {len(indices)} indices from {subset_path.name} (key={key})")
            return [int(x) for x in indices]
    raise ValueError(f"No episode index list found in {subset_path}. "
                     f"Expected keys: selected_episode_indices or selected_episode_ids")


def load_candidate_indices(pool_path: Path, logger: logging.Logger) -> List[int]:
    if not pool_path.exists():
        raise FileNotFoundError(f"Candidate pool metadata not found: {pool_path}")
    with open(pool_path, "r") as f:
        metadata = json.load(f)
    episodes = metadata.get("episodes", [])
    indices = [int(ep["episode_index"]) for ep in episodes if "episode_index" in ep]
    logger.info(f"Loaded {len(indices)} candidate pool episodes from {pool_path.name}")
    return indices


def load_candidate_rand_vecs(pool_path: Path, logger: logging.Logger) -> Dict[int, np.ndarray]:
    with open(pool_path, "r") as f:
        metadata = json.load(f)
    rand_vecs = {}
    episodes = metadata.get("episodes", [])
    for ep_info in episodes:
        ep_idx = ep_info.get("episode_index")
        if ep_idx is None:
            continue
        rv = ep_info.get("rand_vec")
        if rv is not None:
            rand_vecs[int(ep_idx)] = np.array(rv, dtype=np.float32)
    logger.info(f"Loaded {len(rand_vecs)} rand_vecs from candidate pool")
    return rand_vecs


def select_budget_indices(indices: List[int], budget: int, seed: int = SEED) -> List[int]:
    if len(indices) <= budget:
        return list(indices)
    rng = np.random.RandomState(seed)
    selected = rng.choice(indices, size=budget, replace=False).tolist()
    return [int(x) for x in sorted(selected)]


def compute_occupied_voxels(coords_3d: np.ndarray, bins: int = VOXEL_BINS) -> np.ndarray:
    normalized = np.zeros_like(coords_3d)
    for d in range(3):
        col = coords_3d[:, d]
        c_min, c_max = col.min(), col.max()
        if c_max - c_min < 1e-10:
            normalized[:, d] = 0.5
        else:
            normalized[:, d] = (col - c_min) / (c_max - c_min)
    voxel_grid = np.zeros((bins, bins, bins), dtype=np.int32)
    voxel_indices = np.floor(normalized * bins).astype(np.int32)
    voxel_indices = np.clip(voxel_indices, 0, bins - 1)
    for vi in voxel_indices:
        voxel_grid[vi[0], vi[1], vi[2]] = 1
    return voxel_grid


def select_discriminative_dimensions(
    candidate_vecs: np.ndarray,
    method_coords: Dict[str, np.ndarray],
    logger: logging.Logger,
) -> List[int]:
    n_dims = candidate_vecs.shape[1]
    if n_dims < 3:
        top3 = list(range(n_dims))
        logger.info(f"Feature dimension: {n_dims}")
        logger.info(f"Selected dimensions: {top3} (fallback: too few dimensions)")
        return top3

    def _compute_emd_3d(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
        all_coords = np.vstack([coords_a, coords_b])
        mins = all_coords.min(axis=0)
        maxs = all_coords.max(axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-10] = 1.0
        norm_a = (coords_a - mins) / ranges
        norm_b = (coords_b - mins) / ranges
        bins_1d = 10
        hist_a, _ = np.histogramdd(norm_a, bins=bins_1d, range=[(0, 1)] * 3)
        hist_b, _ = np.histogramdd(norm_b, bins=bins_1d, range=[(0, 1)] * 3)
        p = hist_a.flatten() + 1e-10
        q = hist_b.flatten() + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        js_dist = np.sqrt(0.5 * np.sum(p * np.log(p / q)) + 0.5 * np.sum(q * np.log(q / p)))
        return js_dist

    best_score = -1.0
    best_combo = None
    all_combos = list(combinations(range(n_dims), 3))

    logger.info(f"Evaluating {len(all_combos)} dimension combinations for discriminative power...")

    for combo in all_combos:
        combo = list(combo)
        ours_coords = method_coords.get("ours")[:, combo]
        score = 0.0
        for method in ["grid_uniform", "random", "deminf"]:
            if method in method_coords:
                other_coords = method_coords[method][:, combo]
                score += _compute_emd_3d(ours_coords, other_coords)

        if score > best_score:
            best_score = score
            best_combo = combo

    combo_sorted = sorted(best_combo)
    logger.info(f"Feature dimension: {n_dims}")
    logger.info(f"Selected dimensions: {combo_sorted}")
    logger.info(f"Discrimination score: {best_score:.4f}")
    return combo_sorted


def draw_occupied_voxels(ax, voxel_grid: np.ndarray, color: str, alpha: float = 0.08):
    occupied = np.argwhere(voxel_grid > 0)
    if len(occupied) == 0:
        return
    faces_list = []
    for (x, y, z) in occupied:
        v = np.array([
            [[x, y, z], [x+1, y, z], [x+1, y+1, z], [x, y+1, z]],
            [[x, y, z+1], [x+1, y, z+1], [x+1, y+1, z+1], [x, y+1, z+1]],
            [[x, y, z], [x, y+1, z], [x, y+1, z+1], [x, y, z+1]],
            [[x+1, y, z], [x+1, y+1, z], [x+1, y+1, z+1], [x+1, y, z+1]],
            [[x, y, z], [x, y, z+1], [x+1, y, z+1], [x+1, y, z]],
            [[x, y+1, z], [x, y+1, z+1], [x+1, y+1, z+1], [x+1, y+1, z]],
        ])
        faces_list.append(v)
    collection = Poly3DCollection(
        np.vstack(faces_list),
        alpha=alpha,
        facecolor=color,
        edgecolor=None,
        linewidths=0,
    )
    ax.add_collection3d(collection)


dim_labels = []


def main():
    global dim_labels

    parser = argparse.ArgumentParser(description="3D Configuration Space Visualization")
    parser.add_argument("--candidate-pool", type=str, required=True,
                        help="Path to episode_initial_states.json")
    parser.add_argument("--selection-grid", type=str, default=None,
                        help="Path to Grid-Uniform selection subset JSON")
    parser.add_argument("--selection-random", type=str, required=True,
                        help="Path to Random selection subset JSON")
    parser.add_argument("--selection-deminf", type=str, required=True,
                        help="Path to DemInf selection subset JSON")
    parser.add_argument("--selection-ours", type=str, required=True,
                        help="Path to Ours selection subset JSON")
    parser.add_argument("--budget", type=int, default=112,
                        help="Selection budget (default: 112)")
    parser.add_argument("--output-dir", type=str,
                        default="personal/work2/distribution_analysis/results",
                        help="Output directory for results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    print("\n================================")
    print("3D Configuration Visualization")
    print("================================")

    candidate_pool_path = Path(args.candidate_pool)
    selection_paths = {
        "grid_uniform": Path(args.selection_grid) if args.selection_grid else None,
        "random": Path(args.selection_random),
        "deminf": Path(args.selection_deminf),
        "ours": Path(args.selection_ours),
    }
    budget = args.budget

    candidate_indices = load_candidate_indices(candidate_pool_path, logger)
    candidate_rand_vecs = load_candidate_rand_vecs(candidate_pool_path, logger)

    valid_candidate_indices = sorted(candidate_rand_vecs.keys())
    candidate_vecs = np.array([candidate_rand_vecs[ep] for ep in valid_candidate_indices])

    selection_indices = {}
    budgets = {}
    method_coords_all = {}

    for method, path in selection_paths.items():
        if path is None or not path.exists():
            if method == "grid_uniform":
                logger.warning(f"Grid-Uniform selection file not found, generating from candidate pool")
                selection_indices[method] = select_budget_indices(valid_candidate_indices, budget, seed=SEED)
            else:
                logger.error(f"{method} selection file not found: {path}")
                sys.exit(1)
        else:
            raw_indices = load_subset_indices(path, logger)
            valid_raw = [ep for ep in raw_indices if ep in candidate_rand_vecs]
            if method == "random":
                selection_indices[method] = select_budget_indices(valid_candidate_indices, budget, seed=SEED)
            else:
                selection_indices[method] = select_budget_indices(valid_raw, budget, seed=SEED)
        budgets[method] = len(selection_indices[method])
        vecs = np.array([candidate_rand_vecs[ep] for ep in selection_indices[method]])
        method_coords_all[method] = vecs

    print(f"Feature dimension: {candidate_vecs.shape[1]}")
    print(f"Grid budget: {budgets['grid_uniform']}")
    print(f"Random budget: {budgets['random']}")
    print(f"DemInf budget: {budgets['deminf']}")
    print(f"Ours budget: {budgets['ours']}")

    top3_dims = select_discriminative_dimensions(candidate_vecs, method_coords_all, logger)
    dim_labels = top3_dims
    print(f"Selected dimensions: Dim {top3_dims[0]}, Dim {top3_dims[1]}, Dim {top3_dims[2]}")

    method_coords_3d = {}
    for method, vecs in method_coords_all.items():
        method_coords_3d[method] = vecs[:, top3_dims]

    candidate_coords_3d = candidate_vecs[:, top3_dims]

    x_min = candidate_coords_3d[:, 0].min()
    x_max = candidate_coords_3d[:, 0].max()
    y_min = candidate_coords_3d[:, 1].min()
    y_max = candidate_coords_3d[:, 1].max()
    z_min = candidate_coords_3d[:, 2].min()
    z_max = candidate_coords_3d[:, 2].max()

    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05
    margin_z = (z_max - z_min) * 0.05
    shared_ranges = [
        (x_min - margin_x, x_max + margin_x),
        (y_min - margin_y, y_max + margin_y),
        (z_min - margin_z, z_max + margin_z),
    ]

    candidate_voxel_grid = compute_occupied_voxels(candidate_coords_3d, bins=VOXEL_BINS)
    total_candidate_voxels = int(np.sum(candidate_voxel_grid > 0))

    fig = plt.figure(figsize=FIGSIZE_2X2)

    method_order = ["grid_uniform", "random", "deminf", "ours"]
    method_voxel_results = {}

    for idx, method in enumerate(method_order):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        ax.set_facecolor("white")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("gray")
        ax.yaxis.pane.set_edgecolor("gray")
        ax.zaxis.pane.set_edgecolor("gray")
        ax.grid(True, alpha=0.15)

        color = METHOD_COLORS[method]
        selected = method_coords_3d[method]

        method_voxel_grid = compute_occupied_voxels(selected, bins=VOXEL_BINS)
        method_voxel_results[method] = method_voxel_grid

        draw_occupied_voxels(ax, method_voxel_grid, color=color, alpha=0.08)

        ax.scatter(
            candidate_coords_3d[:, 0],
            candidate_coords_3d[:, 1],
            candidate_coords_3d[:, 2],
            c="#CCCCCC",
            s=4,
            alpha=0.2,
            marker="o",
            zorder=1,
        )

        ax.scatter(
            selected[:, 0],
            selected[:, 1],
            selected[:, 2],
            c=color,
            s=35,
            alpha=0.9,
            marker="o",
            zorder=3,
            edgecolors="white",
            linewidths=0.4,
        )

        x_min_r, x_max_r = shared_ranges[0]
        y_min_r, y_max_r = shared_ranges[1]
        z_min_r, z_max_r = shared_ranges[2]
        ax.set_xlim(x_min_r, x_max_r)
        ax.set_ylim(y_min_r, y_max_r)
        ax.set_zlim(z_min_r, z_max_r)

        ax.set_box_aspect([1, 1, 1])

        ax.view_init(elev=25, azim=45)

        ax.set_xlabel(f"Dim {dim_labels[0]}", fontsize=9, labelpad=5)
        ax.set_ylabel(f"Dim {dim_labels[1]}", fontsize=9, labelpad=5)
        ax.set_zlabel(f"Dim {dim_labels[2]}", fontsize=9, labelpad=5)

        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.tick_params(axis="z", which="major", labelsize=7)

        ax.set_title(f"{METHOD_SUBPLOT_LETTERS[method]} {METHOD_LABELS[method]} (Budget={budget})",
                     fontsize=11, fontweight="bold", pad=8)

        inset_ax = ax.inset_axes([0.55, 0.55, 0.40, 0.40])
        density_xy = method_voxel_grid.sum(axis=2)
        binary_map = (density_xy > 0).astype(np.float32)
        im = inset_ax.imshow(
            binary_map.T,
            origin="lower",
            cmap="Greys",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_title("XY Occupancy", fontsize=6, pad=2)

    fig.tight_layout(pad=2.0)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333",
                   markersize=8, alpha=0.9, label="Selected Demonstrations"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#AAAAAA",
                   markersize=8, alpha=0.3, label="Occupied Voxel"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, 0.01))

    out_path = output_dir / "configuration_space_3d_comparison.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    print(f"\nSaved: {out_path}")

    stats = {}
    for method in method_order:
        occupied_count = int(np.sum(method_voxel_results[method] > 0))
        coverage_ratio = occupied_count / total_candidate_voxels if total_candidate_voxels > 0 else 0.0

        coords = method_coords_3d[method]
        stats[method] = {
            "num_selected": len(selection_indices[method]),
            "occupied_voxel_count": occupied_count,
            "voxel_coverage_ratio": round(coverage_ratio, 6),
            "x_range": [float(coords[:, 0].min()), float(coords[:, 0].max())],
            "y_range": [float(coords[:, 1].min()), float(coords[:, 1].max())],
            "z_range": [float(coords[:, 2].min()), float(coords[:, 2].max())],
        }

    stats_json = {
        "visualization_dimensions": [int(d) for d in top3_dims],
        "voxel_bins": VOXEL_BINS,
        "total_candidate_voxel_count": total_candidate_voxels,
        "candidate_pool_size": len(valid_candidate_indices),
        "methods": stats,
    }

    stats_path = output_dir / "configuration_space_3d_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats_json, f, indent=2)
    logger.info(f"Saved: {stats_path}")
    print(f"Saved: {stats_path}")

    print(f"\nGrid occupied voxel: {stats['grid_uniform']['occupied_voxel_count']}")
    print(f"Random occupied voxel: {stats['random']['occupied_voxel_count']}")
    print(f"DemInf occupied voxel: {stats['deminf']['occupied_voxel_count']}")
    print(f"Ours occupied voxel: {stats['ours']['occupied_voxel_count']}")


if __name__ == "__main__":
    main()