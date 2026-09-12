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
from typing import Dict, List, Tuple
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
STD_THRESHOLD_RATIO = 0.01


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


def select_grid_uniform_spatial(
    candidate_indices: List[int],
    candidate_rand_vecs: Dict[int, np.ndarray],
    budget: int,
    seed: int = SEED,
) -> List[int]:
    main_dims = [0, 1, 3, 4]
    coords = np.array([candidate_rand_vecs[ep][main_dims] for ep in candidate_indices])

    n_per_dim = int(np.ceil(budget ** 0.25))
    n_per_dim = max(n_per_dim, 3)

    normalized_coords = np.zeros_like(coords)
    for d in range(4):
        col = coords[:, d]
        c_min, c_max = col.min(), col.max()
        if c_max - c_min < 1e-10:
            normalized_coords[:, d] = 0.5
        else:
            normalized_coords[:, d] = (col - c_min) / (c_max - c_min)

    cell_size = 1.0 / n_per_dim
    cell_centers = []
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            for k in range(n_per_dim):
                for l in range(n_per_dim):
                    cell_centers.append((
                        (i + 0.5) * cell_size,
                        (j + 0.5) * cell_size,
                        (k + 0.5) * cell_size,
                        (l + 0.5) * cell_size,
                    ))

    rng = np.random.RandomState(seed)
    rng.shuffle(cell_centers)

    selected_indices = []
    used = set()

    for center in cell_centers:
        if len(selected_indices) >= budget:
            break
        dists = np.sum((normalized_coords - np.array(center)) ** 2, axis=1)
        sorted_indices = np.argsort(dists)
        for rank in sorted_indices:
            if rank not in used:
                selected_indices.append(candidate_indices[rank])
                used.add(rank)
                break

    if len(selected_indices) < budget:
        remaining = [i for i in range(len(candidate_indices)) if i not in used]
        if remaining:
            extra = rng.choice(remaining, size=min(budget - len(selected_indices), len(remaining)), replace=False)
            selected_indices.extend([candidate_indices[i] for i in extra])

    return [int(x) for x in sorted(selected_indices)]


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


def analyze_dimension_variance(
    candidate_vecs: np.ndarray,
    logger: logging.Logger,
) -> Tuple[Dict, Dict]:
    n_dims = candidate_vecs.shape[1]
    dim_stats = {}
    variance_analysis = {}

    logger.info("=" * 60)
    logger.info("Dimension Variance Analysis")
    logger.info("=" * 60)

    for d in range(n_dims):
        col = candidate_vecs[:, d]
        mean_val = float(np.mean(col))
        std_val = float(np.std(col))
        min_val = float(np.min(col))
        max_val = float(np.max(col))
        value_range = max_val - min_val

        dim_stats[d] = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "range": value_range,
        }

        range_span = max_val - min_val
        threshold = range_span * STD_THRESHOLD_RATIO if range_span > 0 else 1e-10
        is_valid = std_val > threshold

        variance_analysis[d] = {
            "is_valid": is_valid,
            "std": std_val,
            "threshold": threshold,
        }

        logger.info(
            f"Dim {d}: mean={mean_val:.6f}, std={std_val:.6f}, "
            f"min={min_val:.6f}, max={max_val:.6f}, "
            f"range={value_range:.6f}, valid={is_valid}"
        )

    logger.info("-" * 60)
    logger.info("Object 1 (dims 0-2): Position [x1, y1, z1]")
    logger.info("Object 2 (dims 3-5): Position [x2, y2, z2]")
    logger.info("-" * 60)

    obj1_valid = [d for d in range(3) if variance_analysis.get(d, {}).get("is_valid", False)]
    obj2_valid = [d for d in range(3, 6) if variance_analysis.get(d, {}).get("is_valid", False)]

    logger.info(f"Object 1 valid dimensions: {obj1_valid} ({len(obj1_valid)}/3)")
    logger.info(f"Object 2 valid dimensions: {obj2_valid} ({len(obj2_valid)}/3)")

    return dim_stats, variance_analysis


def select_plotting_dimensions(
    variance_analysis: Dict,
    logger: logging.Logger,
) -> Tuple[List[int], List[int], str]:
    obj1_dims = [0, 1, 2]
    obj2_dims = [3, 4, 5]

    obj1_valid = [d for d in obj1_dims if variance_analysis.get(d, {}).get("is_valid", False)]
    obj2_valid = [d for d in obj2_dims if variance_analysis.get(d, {}).get("is_valid", False)]

    obj1_invalid = [d for d in obj1_dims if d not in obj1_valid]
    obj2_invalid = [d for d in obj2_dims if d not in obj2_valid]

    if len(obj1_valid) == 3:
        main_dims = list(obj1_dims)
        inset_dims = [d for d in obj2_dims if d in obj2_valid][:2]
        if len(inset_dims) < 2:
            inset_dims = [3, 4]
        reason = "Object 1 XYZ all valid, using Object 1 XYZ for main plot"
        logger.info(f"Plotting mode: Object 1 XYZ (reason: {reason})")
        return main_dims, inset_dims, reason

    fallback_dims = []
    reason_parts = []

    for d in obj1_dims:
        if variance_analysis.get(d, {}).get("is_valid", False):
            fallback_dims.append(d)
        else:
            reason_parts.append(f"Dim {d} std~0")

    for d in obj2_dims:
        if len(fallback_dims) >= 3:
            break
        if variance_analysis.get(d, {}).get("is_valid", False) and d not in fallback_dims:
            fallback_dims.append(d)

    if len(fallback_dims) < 3:
        all_valid = [d for d in range(6) if variance_analysis.get(d, {}).get("is_valid", False)]
        fallback_dims = all_valid[:3]
        reason_parts.append("fallback to first 3 valid dims across all objects")

    fallback_dims = fallback_dims[:3]
    reason = "; ".join(reason_parts) if reason_parts else "auto-selected 3 valid dimensions"

    logger.info(f"Plotting mode: Fallback dimensions {fallback_dims} (reason: {reason})")

    inset_dims = [d for d in obj2_dims if variance_analysis.get(d, {}).get("is_valid", False)][:2]
    if len(inset_dims) < 2:
        inset_dims = [3, 4]

    return fallback_dims, inset_dims, reason


def get_axis_labels(main_dims: List[int]) -> List[str]:
    dim_to_label = {
        0: "Object 1 X",
        1: "Object 1 Y",
        2: "Object 1 Z",
        3: "Object 2 X",
        4: "Object 2 Y",
        5: "Object 2 Z",
    }
    return [dim_to_label.get(d, f"Dim {d}") for d in main_dims]


def draw_occupied_voxels(ax, voxel_grid: np.ndarray, color: str, alpha: float = 0.08):
    pass


def main():
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

    candidate_rand_vecs = load_candidate_rand_vecs(candidate_pool_path, logger)
    valid_candidate_indices = sorted(candidate_rand_vecs.keys())
    candidate_vecs = np.array([candidate_rand_vecs[ep] for ep in valid_candidate_indices])

    dim_stats, variance_analysis = analyze_dimension_variance(candidate_vecs, logger)

    main_dims, inset_dims, plot_reason = select_plotting_dimensions(variance_analysis, logger)
    axis_labels = get_axis_labels(main_dims)

    print(f"Main plot dimensions: {main_dims}")
    print(f"Axis labels: {axis_labels}")
    print(f"Inset dimensions: {inset_dims}")

    selection_indices = {}
    budgets = {}
    method_coords_main = {}
    method_coords_inset = {}

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
            if method == "grid_uniform":
                selection_indices[method] = select_grid_uniform_spatial(
                    valid_candidate_indices, candidate_rand_vecs, budget, seed=SEED
                )
            elif method == "random":
                selection_indices[method] = select_budget_indices(valid_candidate_indices, budget, seed=SEED)
            else:
                selection_indices[method] = select_budget_indices(valid_raw, budget, seed=SEED)

        budgets[method] = len(selection_indices[method])
        vecs = np.array([candidate_rand_vecs[ep] for ep in selection_indices[method]])
        method_coords_main[method] = vecs[:, main_dims]
        method_coords_inset[method] = vecs[:, inset_dims]

    print(f"\nGrid budget: {budgets['grid_uniform']}")
    print(f"Random budget: {budgets['random']}")
    print(f"DemInf budget: {budgets['deminf']}")
    print(f"Ours budget: {budgets['ours']}")

    all_main_coords = np.vstack([method_coords_main[m] for m in method_coords_main])
    x_min, x_max = all_main_coords[:, 0].min(), all_main_coords[:, 0].max()
    y_min, y_max = all_main_coords[:, 1].min(), all_main_coords[:, 1].max()
    z_min, z_max = all_main_coords[:, 2].min(), all_main_coords[:, 2].max()

    margin_x = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
    margin_y = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
    margin_z = (z_max - z_min) * 0.05 if z_max > z_min else 0.1
    shared_ranges = [
        (x_min - margin_x, x_max + margin_x),
        (y_min - margin_y, y_max + margin_y),
        (z_min - margin_z, z_max + margin_z),
    ]

    all_inset_coords = np.vstack([method_coords_inset[m] for m in method_coords_inset])
    inset_x_min, inset_x_max = all_inset_coords[:, 0].min(), all_inset_coords[:, 0].max()
    inset_y_min, inset_y_max = all_inset_coords[:, 1].min(), all_inset_coords[:, 1].max()

    fig = plt.figure(figsize=FIGSIZE_2X2)

    method_order = ["grid_uniform", "random", "deminf", "ours"]
    method_voxel_results = {}

    max_voxels = VOXEL_BINS ** 3

    z_label_override = "Rotation"

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
        selected = method_coords_main[method]

        method_voxel_grid = compute_occupied_voxels(selected, bins=VOXEL_BINS)
        method_voxel_results[method] = method_voxel_grid

        draw_occupied_voxels(ax, method_voxel_grid, color=color, alpha=0.08)

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

        ax.set_xlabel(axis_labels[0], fontsize=9, labelpad=5)
        ax.set_ylabel(axis_labels[1], fontsize=9, labelpad=5)
        ax.set_zlabel(z_label_override, fontsize=9, labelpad=5)

        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.tick_params(axis="z", which="major", labelsize=7)

        title = f"{METHOD_SUBPLOT_LETTERS[method]} {METHOD_LABELS[method]}"
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

        inset_ax = ax.inset_axes([0.55, 0.55, 0.40, 0.40])
        inset_selected = method_coords_inset[method]

        inset_ax.scatter(
            inset_selected[:, 0],
            inset_selected[:, 1],
            c=color,
            s=10,
            alpha=0.8,
            marker="o",
            zorder=3,
            edgecolors="none",
        )
        inset_ax.set_xlim(inset_x_min, inset_x_max)
        inset_ax.set_ylim(inset_y_min, inset_y_max)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_title("Object 2 XY", fontsize=6, pad=2)

    fig.tight_layout(pad=2.0)

    out_path = output_dir / "configuration_space_3d_comparison.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    print(f"\nSaved: {out_path}")

    stats = {}
    for method in method_order:
        occupied_count = int(np.sum(method_voxel_results[method] > 0))
        coverage_ratio = occupied_count / max_voxels

        coords = method_coords_main[method]
        stats[method] = {
            "num_selected": len(selection_indices[method]),
            "occupied_voxel_count": occupied_count,
            "voxel_coverage_ratio": round(coverage_ratio, 6),
            "x_range": [float(coords[:, 0].min()), float(coords[:, 0].max())],
            "y_range": [float(coords[:, 1].min()), float(coords[:, 1].max())],
            "z_range": [float(coords[:, 2].min()), float(coords[:, 2].max())],
        }

    stats_json = {
        "main_plot_dimensions": [int(d) for d in main_dims],
        "main_plot_axis_labels": axis_labels,
        "inset_dimensions": [int(d) for d in inset_dims],
        "plotting_reason": plot_reason,
        "voxel_bins": VOXEL_BINS,
        "max_voxel_count": max_voxels,
        "dimension_variance": {
            str(k): v for k, v in dim_stats.items()
        },
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