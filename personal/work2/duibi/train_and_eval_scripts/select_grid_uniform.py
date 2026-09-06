#!/usr/bin/env python
"""
Select episodes with factorized rand_vec-group joint-region uniform distribution.

Algorithm overview:
1. Extract rand_vec from episode_initial_states.json
2. Split rand_vec into groups of 3 dimensions each (e.g., group_A=rand_vec[:3], group_B=rand_vec[3:6])
3. Find active dimensions for each group (range > eps)
4. Normalize each group to [0,1]^D independently
5. Generate K_A and K_B uniform region centers via farthest-point sampling
6. Assign each episode to (group_A_region, group_B_region) joint region
7. Build valid joint regions (only those with at least one real episode)
8. Allocate balanced quotas across valid joint regions
9. Select episodes closest to joint region centers

Usage:
    python select_grid_uniform.py --num-episodes 112 --seed 42 \
        --dataset-root /path/to/dataset --output-dir /path/to/output
"""
import argparse
import json
from pathlib import Path
from itertools import product

import numpy as np


# ─── Data Loading ────────────────────────────────────────────────────

def load_episode_metadata(json_path: str) -> tuple[list[int], list[dict]]:
    """Load episode indices and raw metadata from JSON file."""
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"找不到 JSON 文件: {json_file}")

    with open(json_file, "r") as f:
        metadata = json.load(f)

    episodes = metadata["episodes"]
    indices = [ep["episode_index"] for ep in episodes]
    return indices, episodes


# ─── Rand Vec Group Extraction ───────────────────────────────────────

def extract_rand_vec_groups(episodes: list[dict], group_size: int = 3) -> tuple[list[int], list[np.ndarray]]:
    """
    Extract rand_vec from episodes and split into groups of `group_size` dimensions.
    
    Returns:
        valid_indices: indices of episodes with valid rand_vec
        group_arrays: list of (N_valid, group_size) arrays, one per group
    """
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
        print(f"跳过 {missing} 个缺少 rand_vec 的 episode")

    if not all_rand_vecs:
        return [], []

    # Determine number of groups from first episode
    dim = len(all_rand_vecs[0])
    n_groups = dim // group_size

    # Split each rand_vec into groups
    group_arrays = []
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group_arr = np.stack([rv[start:end] for rv in all_rand_vecs], axis=0)
        group_arrays.append(group_arr)

    return valid_indices, group_arrays


# ─── Active Dimension Detection ──────────────────────────────────────

def find_active_dimensions(positions: np.ndarray, eps: float = 1e-4) -> list[int]:
    """
    Find dimensions with meaningful variation.
    
    Uses range > eps criterion to avoid treating near-fixed dimensions
    (e.g., z=0.025~0.02501) as active degrees of freedom.
    """
    low = positions.min(axis=0)
    high = positions.max(axis=0)
    ranges = high - low
    active = [i for i, r in enumerate(ranges) if r > eps]
    return active


# ─── Normalization ───────────────────────────────────────────────────

def normalize_space(positions: np.ndarray, active_dims: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize positions to [0,1] for active dimensions only.
    
    Returns:
        normalized: (N, D_active) normalized positions
        low: (D_active,) observed minimums
        high: (D_active,) observed maximums
    """
    low = positions[:, active_dims].min(axis=0)
    high = positions[:, active_dims].max(axis=0)
    ranges = high - low

    normalized = (positions[:, active_dims] - low) / np.where(ranges > 1e-10, ranges, 1.0)
    return normalized, low, high


# ─── Region Count Allocation ─────────────────────────────────────────

def allocate_region_counts(num_episodes: int, group_active_dims: list[int]) -> list[int]:
    """
    Allocate K region counts for each group.
    
    Default: K_i ≈ num_episodes^(1/n_groups) for each group with active dimensions.
    If a group has no active dimensions, K=1 for that group.
    """
    n_groups = len(group_active_dims)
    n_active_groups = sum(1 for d in group_active_dims if d > 0)

    if n_active_groups == 0:
        return [1] * n_groups

    base = max(1, int(np.round(num_episodes ** (1.0 / n_active_groups))))
    ks = []
    for d in group_active_dims:
        if d > 0:
            ks.append(base)
        else:
            ks.append(1)

    return ks


# ─── Farthest-Point Region Center Generation ─────────────────────────

def generate_uniform_region_centers(k: int, dim: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Generate K uniform region centers in [0,1]^dim via deterministic farthest-point sampling.
    
    1. Build a dense candidate lattice in [0,1]^dim
    2. Select first center deterministically (closest to space center)
    3. Iteratively select the candidate farthest from all selected centers
    """
    if k == 1:
        return np.full((1, dim), 0.5)

    if dim == 1:
        return np.linspace(0.5 / k, 1.0 - 0.5 / k, k).reshape(-1, 1)

    # Build candidate lattice
    points_per_dim = max(2, int(np.ceil(k ** (1.0 / dim))) + 2)
    grids = [np.linspace(0, 1, points_per_dim) for _ in range(dim)]
    candidates = np.array(list(product(*grids)), dtype=np.float64)

    # Select first center: closest to space center (0.5, 0.5, ...)
    space_center = np.full(dim, 0.5)
    dists = np.sum((candidates - space_center) ** 2, axis=1)
    first_idx = np.argmin(dists)

    centers = [candidates[first_idx]]
    min_dists = np.full(len(candidates), np.inf)

    for _ in range(1, k):
        # Update minimum distances to nearest selected center
        last_center = centers[-1]
        new_dists = np.sum((candidates - last_center) ** 2, axis=1)
        min_dists = np.minimum(min_dists, new_dists)

        # Select candidate with maximum minimum distance
        min_dists_copy = min_dists.copy()
        for c in centers:
            # Mark already-selected candidates
            exact_match = np.all(candidates == c, axis=1)
            min_dists_copy[exact_match] = -1.0

        best_idx = np.argmax(min_dists_copy)
        centers.append(candidates[best_idx])

    return np.array(centers, dtype=np.float64)


# ─── Region Assignment ───────────────────────────────────────────────

def assign_regions(positions_norm: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Assign each position to its nearest region center (Voronoi-style).
    
    Returns:
        region_ids: (N,) array of region indices
    """
    diff = positions_norm[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)
    return np.argmin(dists, axis=1)


# ─── Valid Joint Region Building ─────────────────────────────────────

def build_valid_joint_regions(
    region_ids_list: list[np.ndarray],
    n_episodes: int
) -> dict[tuple[int, ...], list[int]]:
    """
    Build dict of valid joint regions: (region_A, region_B, ...) -> [episode_indices].
    
    Only regions with at least one real episode are included.
    """
    n_groups = len(region_ids_list)
    joint_regions: dict[tuple[int, ...], list[int]] = {}

    for i in range(n_episodes):
        key = tuple(int(region_ids_list[g][i]) for g in range(n_groups))
        if key not in joint_regions:
            joint_regions[key] = []
        joint_regions[key].append(i)

    return joint_regions


# ─── Balanced Quota Allocation ───────────────────────────────────────

def allocate_balanced_quotas(
    num_episodes: int,
    valid_regions: dict[tuple[int, ...], list[int]],
    seed: int
) -> dict[tuple[int, ...], int]:
    """
    Allocate episode quotas to valid joint regions.
    
    1. base = N // V, remainder = N % V
    2. Each region gets base quota
    3. Remainder distributed evenly across regions (deterministic, sorted order)
    4. If a region has fewer episodes than quota, redistribute the shortfall
    """
    v = len(valid_regions)
    if v == 0:
        return {}

    base = num_episodes // v
    remainder = num_episodes % v

    # Sort region keys for deterministic ordering
    sorted_keys = sorted(valid_regions.keys())

    # Initial allocation
    quotas = {}
    for i, key in enumerate(sorted_keys):
        extra = 1 if i < remainder else 0
        quotas[key] = base + extra

    # Handle regions with insufficient episodes
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

        # Redistribute shortfall to regions with surplus capacity
        per_region = shortfall_total // len(surplus_regions)
        extra = shortfall_total % len(surplus_regions)

        for i, key in enumerate(surplus_regions):
            additional = per_region + (1 if i < extra else 0)
            max_can_add = len(valid_regions[key]) - quotas[key]
            quotas[key] += min(additional, max_can_add)

    return quotas


# ─── Episode Selection from Joint Regions ────────────────────────────

def select_from_joint_regions(
    valid_regions: dict[tuple[int, ...], list[int]],
    quotas: dict[tuple[int, ...], int],
    group_positions_norm: list[np.ndarray],
    group_centers: list[np.ndarray],
    group_weights: list[float] | None = None
) -> list[int]:
    """
    Select episodes from each joint region, prioritizing those closest to region center.
    
    For episode e in region (R0, R1, ...):
        d(e) = sum_i w_i * ||group_norm_i(e) - group_center_i[Ri]||^2
    """
    if group_weights is None:
        group_weights = [1.0] * len(group_centers)

    selected = []

    for key in sorted(quotas.keys()):
        quota = quotas[key]
        episode_indices = valid_regions[key]

        if quota <= 0:
            continue

        # Compute joint center distances
        distances = []
        for idx in episode_indices:
            joint_dist = 0.0
            for g, center in enumerate(group_centers):
                region_id = key[g]
                d_g = np.sum((group_positions_norm[g][idx] - center[region_id]) ** 2)
                joint_dist += group_weights[g] * d_g
            distances.append((joint_dist, idx))

        # Sort by distance (closest first)
        distances.sort(key=lambda x: (x[0], x[1]))

        # Select top quota episodes
        n_select = min(quota, len(distances))
        for j in range(n_select):
            selected.append(distances[j][1])

    return selected


# ─── Main Selection Pipeline ─────────────────────────────────────────

def select_grid_uniform_episodes(num_episodes, seed, dataset_root, output_dir=None, group_size: int = 3):
    """
    Select episodes with factorized rand_vec-group joint-region uniform distribution.
    """
    rng = np.random.RandomState(seed)

    # 1. Load data
    json_path = Path(dataset_root) / "episode_initial_states.json"
    print(f"加载 episode 数据: {json_path}")
    indices, episodes = load_episode_metadata(str(json_path))
    n_total = len(episodes)
    print(f"Total episodes in dataset: {n_total}")

    # 2. Extract rand_vec groups
    valid_local_indices, group_arrays = extract_rand_vec_groups(episodes, group_size=group_size)
    n_valid = len(valid_local_indices)
    n_groups = len(group_arrays)

    if n_valid == 0:
        print("错误: 没有包含 rand_vec 的 episode")
        return []

    print(f"有效 episode: {n_valid}")
    print(f"rand_vec 分组数: {n_groups} (每组 {group_size} 维)")
    for g, arr in enumerate(group_arrays):
        print(f"  Group {g} shape: {arr.shape}")

    # 3. Find active dimensions for each group
    group_active_dims = []
    for g, arr in enumerate(group_arrays):
        active = find_active_dimensions(arr)
        group_active_dims.append(active)
        print(f"Group {g} active dimensions: {active} (共 {len(active)} 维)")

    if all(len(d) == 0 for d in group_active_dims):
        print("错误: 所有 group 都没有有效变化维度")
        return []

    # 4. Normalize each group
    group_norm = []
    group_low = []
    group_high = []
    for g, arr in enumerate(group_arrays):
        norm, low, high = normalize_space(arr, group_active_dims[g])
        group_norm.append(norm)
        group_low.append(low)
        group_high.append(high)
        print(f"Group {g} 归一化范围: low={low.tolist()}, high={high.tolist()}")

    # 5. Allocate region counts
    group_ks = allocate_region_counts(num_episodes, [len(d) for d in group_active_dims])
    theoretical_regions = 1
    for k in group_ks:
        theoretical_regions *= k
    print(f"\n区域分配: K={[k for k in group_ks]}, 理论联合区域数={theoretical_regions}")

    # 6. Generate region centers
    group_centers = []
    for g, k in enumerate(group_ks):
        dim = len(group_active_dims[g])
        centers = generate_uniform_region_centers(k, dim, rng)
        group_centers.append(centers)
        print(f"Group {g} centers shape: {centers.shape}")

    # 7. Assign regions
    group_region_ids = []
    for g in range(n_groups):
        region_ids = assign_regions(group_norm[g], group_centers[g])
        group_region_ids.append(region_ids)

    # 8. Build valid joint regions
    valid_joint_regions = build_valid_joint_regions(group_region_ids, n_valid)
    n_valid_joint = len(valid_joint_regions)

    print(f"\n实际有效联合区域数: {n_valid_joint} / {theoretical_regions}")

    if n_valid_joint == 0:
        print("错误: 没有有效的联合区域")
        return []

    # Print region distribution
    print("联合区域分布:")
    for key in sorted(valid_joint_regions.keys()):
        print(f"  {key}: {len(valid_joint_regions[key])} episodes")

    # 9. Allocate balanced quotas
    quotas = allocate_balanced_quotas(num_episodes, valid_joint_regions, seed)

    print(f"\n配额分配:")
    total_quota = sum(quotas.values())
    for key in sorted(quotas.keys()):
        print(f"  {key}: quota={quotas[key]}, available={len(valid_joint_regions[key])}")
    print(f"  总配额: {total_quota}")

    # 10. Select episodes
    selected_local = select_from_joint_regions(
        valid_joint_regions, quotas,
        group_norm, group_centers
    )

    # Map local indices back to global episode indices
    selected_global = [indices[valid_local_indices[i]] for i in selected_local]
    selected_global = sorted(selected_global)[:num_episodes]

    print(f"\n最终选择 {len(selected_global)} episodes")

    # Print selected region distribution
    selected_region_counts = {}
    for i in selected_local:
        key = tuple(int(group_region_ids[g][i]) for g in range(n_groups))
        selected_region_counts[key] = selected_region_counts.get(key, 0) + 1

    print("选中 episode 的联合区域分布:")
    for key in sorted(selected_region_counts.keys()):
        print(f"  {key}: {selected_region_counts[key]} episodes")

    # 11. Save results
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_file = output_path / f"grid_uniform_{num_episodes}_seed{seed}.json"

        # Build metadata for JSON output
        valid_region_counts = {
            str(k): len(v) for k, v in sorted(valid_joint_regions.items())
        }
        selected_region_counts_str = {
            str(k): v for k, v in sorted(selected_region_counts.items())
        }

        subset_data = {
            "method": "factorized_rand_vec_group_uniform",
            "num_episodes": num_episodes,
            "seed": seed,
            "group_size": group_size,
            "n_groups": n_groups,
            "n_total_episodes": n_total,
            "n_valid_episodes": n_valid,
            "group_active_dims": {str(g): group_active_dims[g] for g in range(n_groups)},
            "group_observed_low": {str(g): group_low[g].tolist() for g in range(n_groups)},
            "group_observed_high": {str(g): group_high[g].tolist() for g in range(n_groups)},
            "group_k": {str(g): group_ks[g] for g in range(n_groups)},
            "num_possible_joint_regions": theoretical_regions,
            "num_valid_joint_regions": n_valid_joint,
            "group_centers": {str(g): group_centers[g].tolist() for g in range(n_groups)},
            "valid_joint_region_counts": valid_region_counts,
            "selected_joint_region_counts": selected_region_counts_str,
            "selected_episode_indices": selected_global,
        }

        with open(subset_file, "w") as f:
            json.dump(subset_data, f, indent=2)
        print(f"Saved {num_episodes} factorized rand_vec-group uniform episodes (seed={seed}) to {subset_file}")

    return selected_global


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-root", type=str, required=True, help="数据集根目录，JSON 文件位于此目录下")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--group-size", type=int, default=3, help="rand_vec 每组维度数 (默认3)")
    args = parser.parse_args()

    select_grid_uniform_episodes(
        num_episodes=args.num_episodes,
        seed=args.seed,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        group_size=args.group_size,
    )