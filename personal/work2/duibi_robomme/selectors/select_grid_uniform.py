#!/usr/bin/env python
"""
Grid-uniform episode selector for Robomme pipeline.

Selects episodes by uniformly covering the initial-configuration feature space.
Replaces the old rand_vec-based approach with configuration features extracted
from episode_initial_states.json.

Algorithm:
  1. Load configuration features for all episodes
  2. Detect active dimensions (meaningful variation)
  3. Normalize active dimensions to [0,1]
  4. Partition configuration space into grid regions
  5. Allocate balanced quotas to each valid joint region
  6. Select episodes closest to region centers

Usage:
    python select_grid_uniform.py \
        --dataset-root /path/to/dataset \
        --task-name MoveCube_easy \
        --num-episodes 28 \
        --seed 42 \
        --output-dir /path/to/output
"""
import argparse
import json
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow direct execution (python script.py) to find sibling modules
_selectors_dir = Path(__file__).resolve().parent
if str(_selectors_dir) not in sys.path:
    sys.path.insert(0, str(_selectors_dir))

from configuration_features import (
    load_configuration_features,
    TASK_FEATURE_BUILDERS,
    TASK_FEATURE_DIMS,
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Configuration feature grouping per task
# ---------------------------------------------------------------------------

def get_configuration_groups(
    task_name: str,
    features: Dict[int, np.ndarray],
) -> Tuple[List[str], List[List[int]]]:
    """
    Split configuration features into logical groups based on task structure.

    Returns (group_names, group_dim_lists) where each group_dim_list contains
    the indices within the feature vector belonging to that group.
    """
    dim = TASK_FEATURE_DIMS[task_name]
    all_dims = list(range(dim))

    if task_name in ("MoveCube_easy", "MoveCube"):
        # cube pos(3)+quat(4)=7, cube_2 pos(3)+quat(4)=7,
        # goal_site pos(3)+quat(4)=7, goal_site_2 pos(3)+quat(4)=7,
        # pegs qpos=2, task_config=5
        groups = [
            ("cube_pose", list(range(0, 7))),
            ("cube2_pose", list(range(7, 14))),
            ("goal_pose", list(range(14, 21))),
            ("goal2_pose", list(range(21, 28))),
            ("pegs_qpos", list(range(28, 30))),
            ("task_config", list(range(30, 35))),
        ]
    elif task_name in ("PatternLock_medium", "PatternLock"):
        # 4 buttons * 7 = 28, task_config = 2
        groups = []
        for i in range(4):
            groups.append((f"button_{i}_pose", list(range(i * 7, (i + 1) * 7))))
        groups.append(("task_config", list(range(28, 30))))
    elif task_name in ("RouteStick_hard", "RouteStick"):
        # 4 waypoints * 7 = 28, task_config = 3
        groups = []
        for i in range(4):
            groups.append((f"waypoint_{i}_pose", list(range(i * 7, (i + 1) * 7))))
        groups.append(("task_config", list(range(28, 31))))
    else:
        # Fallback: single group with all dimensions
        groups = [("all", all_dims)]

    group_names = [g[0] for g in groups]
    group_dim_lists = [g[1] for g in groups]
    return group_names, group_dim_lists


# ---------------------------------------------------------------------------
# Active dimension detection and normalization
# ---------------------------------------------------------------------------

def find_active_dimensions(positions: np.ndarray, eps: float = 1e-4) -> List[int]:
    """Find dimensions with meaningful variation."""
    low = positions.min(axis=0)
    high = positions.max(axis=0)
    ranges = high - low
    return [i for i, r in enumerate(ranges) if r > eps]


def normalize_space(positions: np.ndarray, active_dims: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize positions to [0,1] for active dimensions."""
    if len(active_dims) == 0:
        return positions[:, :0], np.array([]), np.array([])
    low = positions[:, active_dims].min(axis=0)
    high = positions[:, active_dims].max(axis=0)
    ranges = high - low
    normalized = (positions[:, active_dims] - low) / np.where(ranges > 1e-10, ranges, 1.0)
    return normalized, low, high


# ---------------------------------------------------------------------------
# Region allocation and center generation
# ---------------------------------------------------------------------------

def allocate_region_counts(num_episodes: int, group_active_dims: List[List[int]]) -> List[int]:
    """Allocate K region counts for each group."""
    n_groups = len(group_active_dims)
    n_active_groups = sum(1 for d in group_active_dims if len(d) > 0)

    if n_active_groups == 0:
        return [1] * n_groups

    base = max(1, int(np.round(num_episodes ** (1.0 / n_active_groups))))
    return [base if len(d) > 0 else 1 for d in group_active_dims]


def generate_uniform_region_centers(k: int, dim: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate K uniform region centers in [0,1]^dim via farthest-point sampling."""
    if k == 1:
        return np.full((1, dim), 0.5)

    if dim == 1:
        return np.linspace(0.5 / k, 1.0 - 0.5 / k, k).reshape(-1, 1)

    points_per_dim = max(2, int(np.ceil(k ** (1.0 / dim))) + 2)
    grids = [np.linspace(0, 1, points_per_dim) for _ in range(dim)]
    candidates = np.array(list(product(*grids)), dtype=np.float64)

    space_center = np.full(dim, 0.5)
    dists = np.sum((candidates - space_center) ** 2, axis=1)
    first_idx = np.argmin(dists)

    centers = [candidates[first_idx]]
    min_dists = np.full(len(candidates), np.inf)

    for _ in range(1, k):
        last_center = centers[-1]
        new_dists = np.sum((candidates - last_center) ** 2, axis=1)
        min_dists = np.minimum(min_dists, new_dists)

        min_dists_copy = min_dists.copy()
        for c in centers:
            exact_match = np.all(candidates == c, axis=1)
            min_dists_copy[exact_match] = -1.0

        best_idx = np.argmax(min_dists_copy)
        centers.append(candidates[best_idx])

    return np.array(centers, dtype=np.float64)


def assign_regions(positions_norm: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each position to its nearest region center."""
    diff = positions_norm[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)
    return np.argmin(dists, axis=1)


# ---------------------------------------------------------------------------
# Joint region construction and balanced quota allocation
# ---------------------------------------------------------------------------

def build_valid_joint_regions(region_ids_list: List[np.ndarray], n_episodes: int) -> Dict[Tuple, List[int]]:
    """Build dict of valid joint regions."""
    n_groups = len(region_ids_list)
    joint_regions: Dict[Tuple, List[int]] = {}

    for i in range(n_episodes):
        key = tuple(int(region_ids_list[g][i]) for g in range(n_groups))
        if key not in joint_regions:
            joint_regions[key] = []
        joint_regions[key].append(i)

    return joint_regions


def allocate_balanced_quotas(num_episodes: int, valid_regions: Dict[Tuple, List[int]], seed: int) -> Dict[Tuple, int]:
    """Allocate episode quotas to valid joint regions."""
    v = len(valid_regions)
    if v == 0:
        return {}

    base = num_episodes // v
    remainder = num_episodes % v

    sorted_keys = sorted(valid_regions.keys())

    quotas = {}
    for i, key in enumerate(sorted_keys):
        extra = 1 if i < remainder else 0
        quotas[key] = base + extra

    max_iterations = 10
    for _ in range(max_iterations):
        shortfall_total = 0
        surplus_regions = []

        for key in sorted_keys:
            available = len(valid_regions[key])
            if quotas[key] > available:
                shortfall_total += quotas[key] - available
                quotas[key] = available
            else:
                surplus_regions.append(key)

        if shortfall_total == 0 or not surplus_regions:
            break

        per_region = shortfall_total // len(surplus_regions)
        extra = shortfall_total % len(surplus_regions)

        for i, key in enumerate(surplus_regions):
            additional = per_region + (1 if i < extra else 0)
            max_can_add = len(valid_regions[key]) - quotas[key]
            quotas[key] += min(additional, max_can_add)

    return quotas


def select_from_joint_regions(
    valid_regions: Dict[Tuple, List[int]],
    quotas: Dict[Tuple, int],
    group_positions_norm: List[np.ndarray],
    group_centers: List[np.ndarray],
    group_weights: Optional[List[float]] = None,
) -> List[int]:
    """Select episodes from each joint region (closest to center)."""
    if group_weights is None:
        group_weights = [1.0] * len(group_centers)

    selected = []

    for key in sorted(quotas.keys()):
        quota = quotas[key]
        episode_indices = valid_regions[key]

        if quota <= 0:
            continue

        distances = []
        for idx in episode_indices:
            joint_dist = 0.0
            for g, center in enumerate(group_centers):
                region_id = key[g]
                d_g = np.sum((group_positions_norm[g][idx] - center[region_id]) ** 2)
                joint_dist += group_weights[g] * d_g
            distances.append((joint_dist, idx))

        distances.sort(key=lambda x: (x[0], x[1]))

        n_select = min(quota, len(distances))
        for j in range(n_select):
            selected.append(distances[j][1])

    return selected


# ---------------------------------------------------------------------------
# Main selection function
# ---------------------------------------------------------------------------

def select_grid_uniform_episodes(
    num_episodes: int,
    seed: int,
    dataset_root: str,
    task_name: str,
    output_dir: Optional[str] = None,
) -> List[int]:
    """
    Select episodes with factorized configuration-feature joint-region uniform distribution.

    Args:
        num_episodes: target number of episodes to select (K)
        seed: random seed for reproducibility
        dataset_root: path to LeRobotDataset root
        task_name: one of MoveCube_easy, PatternLock_medium, RouteStick_hard
        output_dir: directory to save selection results

    Returns:
        sorted list of selected episode indices
    """
    rng = np.random.RandomState(seed)

    print(f"Loading configuration features for task={task_name}")
    config_features = load_configuration_features(dataset_root, task_name)

    if len(config_features) == 0:
        raise ValueError("No configuration features found")

    episode_indices = sorted(config_features.keys())
    n_total = len(episode_indices)
    print(f"Total episodes with configuration features: {n_total}")

    if num_episodes > n_total:
        raise ValueError(f"num_episodes ({num_episodes}) > available episodes ({n_total})")

    # Stack features in episode order
    feat_array = np.stack([config_features[ep] for ep in episode_indices], axis=0)

    # Get configuration groups
    group_names, group_dim_lists = get_configuration_groups(task_name, config_features)
    n_groups = len(group_names)
    print(f"Configuration groups: {group_names}")

    # Extract per-group feature arrays
    group_arrays = []
    for g, dim_list in enumerate(group_dim_lists):
        group_arrays.append(feat_array[:, dim_list])

    # Find active dimensions per group
    group_active_dims = []
    for g, arr in enumerate(group_arrays):
        active = find_active_dimensions(arr)
        group_active_dims.append(active)
        print(f"  Group {g} ({group_names[g]}): {len(active)} active dims out of {arr.shape[1]}")

    if all(len(d) == 0 for d in group_active_dims):
        raise ValueError("No active dimensions in any group")

    # Normalize active dimensions
    group_norm = []
    group_low = []
    group_high = []
    for g, arr in enumerate(group_arrays):
        norm, low, high = normalize_space(arr, group_active_dims[g])
        group_norm.append(norm)
        group_low.append(low)
        group_high.append(high)

    # Allocate region counts
    group_ks = allocate_region_counts(num_episodes, [len(d) for d in group_active_dims])
    theoretical_regions = 1
    for k in group_ks:
        theoretical_regions *= k
    print(f"Region allocation: K={group_ks}, theoretical joint regions={theoretical_regions}")

    # Generate region centers
    group_centers = []
    for g, k in enumerate(group_ks):
        dim = len(group_active_dims[g])
        if dim == 0:
            group_centers.append(np.array([]).reshape(0, 0))
        else:
            centers = generate_uniform_region_centers(k, dim, rng)
            group_centers.append(centers)

    # Assign regions
    group_region_ids = []
    for g in range(n_groups):
        dim = len(group_active_dims[g])
        if dim == 0:
            group_region_ids.append(np.zeros(n_total, dtype=np.int64))
        else:
            region_ids = assign_regions(group_norm[g], group_centers[g])
            group_region_ids.append(region_ids)

    # Build valid joint regions
    valid_joint_regions = build_valid_joint_regions(group_region_ids, n_total)
    n_valid_joint = len(valid_joint_regions)
    print(f"Valid joint regions: {n_valid_joint} / {theoretical_regions}")

    if n_valid_joint == 0:
        raise ValueError("No valid joint regions")

    # Allocate balanced quotas
    quotas = allocate_balanced_quotas(num_episodes, valid_joint_regions, seed)

    # Select episodes
    selected_local = select_from_joint_regions(
        valid_joint_regions, quotas, group_norm, group_centers
    )

    # Map local indices back to global episode indices
    selected_global = [episode_indices[i] for i in selected_local]
    selected_global = sorted(selected_global)[:num_episodes]

    # Verify we got exactly K distinct episodes
    assert len(selected_global) == num_episodes, (
        f"Selected {len(selected_global)} episodes, expected {num_episodes}"
    )
    assert len(set(selected_global)) == num_episodes, "Duplicate episodes selected"

    print(f"Final selected: {len(selected_global)} episodes")

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_file = output_path / f"grid_uniform_{num_episodes}_seed{seed}.json"

        subset_data = {
            "method": "grid_uniform",
            "num_episodes": num_episodes,
            "seed": seed,
            "task_name": task_name,
            "n_total_episodes": n_total,
            "n_valid_episodes": n_total,
            "group_names": group_names,
            "group_ks": group_ks,
            "n_valid_joint_regions": n_valid_joint,
            "selected_episode_indices": selected_global,
        }

        with open(subset_file, "w") as f:
            json.dump(subset_data, f, indent=2, cls=NumpyEncoder)
        print(f"Saved {num_episodes} grid_uniform episodes (seed={seed}) to {subset_file}")

        # Save region assignments for inspection
        region_file = output_path / f"grid_uniform_{num_episodes}_seed{seed}_regions.json"
        region_data = {
            "group_names": group_names,
            "group_ks": group_ks,
            "group_active_dims": group_active_dims,
            "episode_to_region": {
                str(episode_indices[i]): {
                    g: int(group_region_ids[g][i]) for g in range(n_groups)
                }
                for i in range(n_total)
            },
        }
        with open(region_file, "w") as f:
            json.dump(region_data, f, indent=2, cls=NumpyEncoder)

    return selected_global


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--task-name", type=str, required=True,
                       choices=["MoveCube_easy", "PatternLock_medium", "RouteStick_hard"])
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    select_grid_uniform_episodes(
        num_episodes=args.num_episodes,
        seed=args.seed,
        dataset_root=args.dataset_root,
        task_name=args.task_name,
        output_dir=args.output_dir,
    )