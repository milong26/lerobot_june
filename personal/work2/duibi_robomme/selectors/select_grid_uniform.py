#!/usr/bin/env python
"""
Grid-uniform episode selector for Robomme pipeline.
Reproduces the behavior of personal/work2/duibi/train_and_eval_scripts/select_grid_uniform.py
but with a clean interface.

Usage:
    python select_grid_uniform.py \
        --dataset-root /path/to/dataset \
        --num-episodes 28 \
        --seed 42 \
        --output-dir /path/to/output
"""
import argparse
import json
from pathlib import Path
from itertools import product

import numpy as np


def load_episode_metadata(json_path: str):
    """Load episode indices and raw metadata from JSON file."""
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"Episode metadata not found: {json_file}")

    with open(json_file, "r") as f:
        metadata = json.load(f)

    episodes = metadata["episodes"]
    indices = [ep["episode_index"] for ep in episodes]
    return indices, episodes


def extract_rand_vec_groups(episodes, group_size=3):
    """Extract rand_vec from episodes and split into groups."""
    valid_indices = []
    all_rand_vecs = []
    missing = 0

    for i, ep in enumerate(episodes):
        rand_vec = ep.get("rand_vec")
        if rand_vec is None:
            missing += 1
            continue
        valid_indices.append(i)
        all_rand_vecs.append(np.array(rand_vec, dtype=np.float64))

    if missing > 0:
        print(f"Skipping {missing} episodes without rand_vec")

    if not all_rand_vecs:
        return [], []

    dim = len(all_rand_vecs[0])
    n_groups = dim // group_size

    group_arrays = []
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group_arr = np.stack([rv[start:end] for rv in all_rand_vecs], axis=0)
        group_arrays.append(group_arr)

    return valid_indices, group_arrays


def find_active_dimensions(positions, eps=1e-4):
    """Find dimensions with meaningful variation."""
    low = positions.min(axis=0)
    high = positions.max(axis=0)
    ranges = high - low
    return [i for i, r in enumerate(ranges) if r > eps]


def normalize_space(positions, active_dims):
    """Normalize positions to [0,1] for active dimensions."""
    low = positions[:, active_dims].min(axis=0)
    high = positions[:, active_dims].max(axis=0)
    ranges = high - low
    normalized = (positions[:, active_dims] - low) / np.where(ranges > 1e-10, ranges, 1.0)
    return normalized, low, high


def allocate_region_counts(num_episodes, group_active_dims):
    """Allocate K region counts for each group."""
    n_groups = len(group_active_dims)
    n_active_groups = sum(1 for d in group_active_dims if d > 0)

    if n_active_groups == 0:
        return [1] * n_groups

    base = max(1, int(np.round(num_episodes ** (1.0 / n_active_groups))))
    return [base if d > 0 else 1 for d in group_active_dims]


def generate_uniform_region_centers(k, dim, rng):
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


def assign_regions(positions_norm, centers):
    """Assign each position to its nearest region center."""
    diff = positions_norm[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)
    return np.argmin(dists, axis=1)


def build_valid_joint_regions(region_ids_list, n_episodes):
    """Build dict of valid joint regions."""
    n_groups = len(region_ids_list)
    joint_regions = {}

    for i in range(n_episodes):
        key = tuple(int(region_ids_list[g][i]) for g in range(n_groups))
        if key not in joint_regions:
            joint_regions[key] = []
        joint_regions[key].append(i)

    return joint_regions


def allocate_balanced_quotas(num_episodes, valid_regions, seed):
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


def select_from_joint_regions(valid_regions, quotas, group_positions_norm, group_centers, group_weights=None):
    """Select episodes from each joint region."""
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


def select_grid_uniform_episodes(num_episodes, seed, dataset_root, output_dir=None, group_size=3):
    """Select episodes with factorized rand_vec-group joint-region uniform distribution."""
    rng = np.random.RandomState(seed)

    json_path = Path(dataset_root) / "episode_initial_states.json"
    print(f"Loading episode data: {json_path}")
    indices, episodes = load_episode_metadata(str(json_path))
    n_total = len(episodes)
    print(f"Total episodes in dataset: {n_total}")

    valid_local_indices, group_arrays = extract_rand_vec_groups(episodes, group_size=group_size)
    n_valid = len(valid_local_indices)
    n_groups = len(group_arrays)

    if n_valid == 0:
        raise ValueError("No episodes with rand_vec found")

    print(f"Valid episodes: {n_valid}")
    print(f"rand_vec groups: {n_groups} (each {group_size} dims)")

    group_active_dims = []
    for g, arr in enumerate(group_arrays):
        active = find_active_dimensions(arr)
        group_active_dims.append(active)
        print(f"Group {g} active dims: {active}")

    if all(len(d) == 0 for d in group_active_dims):
        raise ValueError("No active dimensions in any group")

    group_norm = []
    group_low = []
    group_high = []
    for g, arr in enumerate(group_arrays):
        norm, low, high = normalize_space(arr, group_active_dims[g])
        group_norm.append(norm)
        group_low.append(low)
        group_high.append(high)

    group_ks = allocate_region_counts(num_episodes, [len(d) for d in group_active_dims])
    theoretical_regions = 1
    for k in group_ks:
        theoretical_regions *= k
    print(f"Region allocation: K={group_ks}, theoretical joint regions={theoretical_regions}")

    group_centers = []
    for g, k in enumerate(group_ks):
        dim = len(group_active_dims[g])
        centers = generate_uniform_region_centers(k, dim, rng)
        group_centers.append(centers)

    group_region_ids = []
    for g in range(n_groups):
        region_ids = assign_regions(group_norm[g], group_centers[g])
        group_region_ids.append(region_ids)

    valid_joint_regions = build_valid_joint_regions(group_region_ids, n_valid)
    n_valid_joint = len(valid_joint_regions)
    print(f"Valid joint regions: {n_valid_joint} / {theoretical_regions}")

    if n_valid_joint == 0:
        raise ValueError("No valid joint regions")

    quotas = allocate_balanced_quotas(num_episodes, valid_joint_regions, seed)

    selected_local = select_from_joint_regions(
        valid_joint_regions, quotas, group_norm, group_centers
    )

    selected_global = [indices[valid_local_indices[i]] for i in selected_local]
    selected_global = sorted(selected_global)[:num_episodes]

    print(f"Final selected: {len(selected_global)} episodes")

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_file = output_path / f"grid_uniform_{num_episodes}_seed{seed}.json"

        subset_data = {
            "method": "grid_uniform",
            "num_episodes": num_episodes,
            "seed": seed,
            "group_size": group_size,
            "n_groups": n_groups,
            "n_total_episodes": n_total,
            "n_valid_episodes": n_valid,
            "selected_episode_indices": selected_global,
        }

        with open(subset_file, "w") as f:
            json.dump(subset_data, f, indent=2)
        print(f"Saved {num_episodes} grid_uniform episodes (seed={seed}) to {subset_file}")

    return selected_global


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--group-size", type=int, default=3)
    args = parser.parse_args()

    select_grid_uniform_episodes(
        num_episodes=args.num_episodes,
        seed=args.seed,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        group_size=args.group_size,
    )