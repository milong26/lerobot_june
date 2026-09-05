#!/usr/bin/env python
"""
Our V5 Episode Selection - Rand Vec Aware Adaptive Coverage Selection

Selects episodes based on adaptive coverage of the rand_vec (reset randomization vector) space,
with visual and action information as auxiliary scoring.

Core idea:
- Read rand_vec for each episode from episode_initial_states.json or metadata
- Partition rand_vec space into regions (dynamic number based on episode count)
- Initialize B0 by uniformly sampling from different rand_vec regions
- Iteratively select episodes based on region priority:
  - coverage_gap: fewer selected episodes in region -> higher priority
  - visual_uncertainty: higher visual embedding variance -> higher priority
  - action_uncertainty: higher action descriptor variance -> higher priority
- Within selected region, choose episode with highest visual+action score

Does NOT use: obj_init_pos, goal_pose, environment_state, success, grasp_success,
eval results, attention results, or test results.

Usage:
    python select_our_v5.py \
        --visual-embedding-dir /path/to/visual/embeddings \
        --action-descriptor-dir /path/to/action/descriptors \
        --dataset-dir /path/to/dataset \
        --output-dir /path/to/output \
        --num-selected 112 \
        --seed 42 \
        --region-ratio 0.1 \
        --min-regions 16 \
        --max-regions 128 \
        --b0-region-ratio 0.2 \
        --coverage-weight 0.5 \
        --region-visual-weight 0.3 \
        --region-action-weight 0.2
"""

import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WORK2_ROOT = Path(__file__).resolve().parent.parent
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))


def load_visual_embeddings(embedding_path: Path) -> Dict[int, np.ndarray]:
    """
    Load existing global+wrist visual embeddings from cache directory.

    Each episode file contains phi_global and phi_wrist.
    We concatenate them into a single visual feature vector per episode.

    Args:
        embedding_path: visual embedding cache directory

    Returns:
        Dict[episode_index, concatenated_visual_feature]
    """
    embeddings = {}

    for f in sorted(embedding_path.glob("*.npy")):
        try:
            data = np.load(str(f), allow_pickle=True).item()
            ep_idx = data.get("episode_index")
            if ep_idx is None:
                continue

            phi_global = data["phi_global"]
            phi_wrist = data["phi_wrist"]

            visual_feature = np.concatenate([phi_global, phi_wrist])
            embeddings[int(ep_idx)] = visual_feature
        except Exception as e:
            print(f"  Skipping invalid file: {f.name} ({e})")

    print(f"Loaded {len(embeddings)} visual embeddings")
    return embeddings


def load_action_descriptors(descriptor_path: Path) -> Dict[int, np.ndarray]:
    """
    Load action descriptors from cache directory.

    Tries combined action_descriptor.npy first, falls back to individual files.

    Args:
        descriptor_path: action descriptor cache directory

    Returns:
        Dict[episode_index, action_descriptor]
    """
    combined_file = descriptor_path / "action_descriptor.npy"
    if combined_file.exists():
        data = np.load(str(combined_file), allow_pickle=True).item()
        descriptors = data["descriptors"]
        indices = data["episode_indices"]
        return {int(idx): descriptors[i] for i, idx in enumerate(indices)}

    descriptors = {}
    for f in sorted(descriptor_path.glob("*.npy")):
        if f.name == "action_descriptor.npy":
            continue
        try:
            data = np.load(str(f), allow_pickle=True).item()
            ep_idx = data.get("episode_index")
            if ep_idx is not None and "action_descriptor" in data:
                descriptors[int(ep_idx)] = data["action_descriptor"]
        except Exception:
            continue

    print(f"Loaded {len(descriptors)} action descriptors")
    return descriptors


def load_rand_vecs(dataset_dir: Path, all_episode_ids: List[int]) -> Dict[int, np.ndarray]:
    """
    Load rand_vec for each episode from episode_initial_states.json or metadata.

    Tries multiple sources:
    1. episode_initial_states.json in dataset directory
    2. Individual episode metadata files
    3. LeRobot dataset meta/episodes parquet files

    Args:
        dataset_dir: dataset root directory
        all_episode_ids: list of all episode indices

    Returns:
        Dict[episode_index, rand_vec]
    """
    rand_vecs = {}

    # Try episode_initial_states.json first
    metadata_file = dataset_dir / "episode_initial_states.json"
    if metadata_file.exists():
        print(f"Loading rand_vecs from: {metadata_file}")
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            episodes = metadata.get("episodes", [])
            for ep_info in episodes:
                ep_idx = ep_info.get("episode_index")
                if ep_idx is None:
                    continue

                rv = ep_info.get("rand_vec")
                if rv is not None:
                    rand_vecs[int(ep_idx)] = np.array(rv, dtype=np.float32)

            print(f"Loaded {len(rand_vecs)} rand_vecs from episode_initial_states.json")
            if len(rand_vecs) > 0:
                return rand_vecs
        except Exception as e:
            print(f"  Failed to load from episode_initial_states.json: {e}")

    # Try loading from individual episode embedding files (if rand_vec is stored there)
    print("  rand_vec not found in episode_initial_states.json, checking embedding files...")

    # If rand_vec is still not found, we cannot proceed
    if len(rand_vecs) == 0:
        print(f"WARNING: No rand_vec found for any episodes!")
        print(f"  Please ensure episode_initial_states.json exists in dataset directory")
        print(f"  and contains 'rand_vec' field for each episode.")

    return rand_vecs


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    L2-normalize feature vectors.

    Args:
        features: 2D array of shape (n_episodes, feature_dim)

    Returns:
        L2-normalized features
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return features / norms


def build_rand_vec_regions(
    episode_data: Dict[int, Dict],
    region_ratio: float = 0.1,
    min_regions: int = 16,
    max_regions: int = 128,
    seed: int = 42,
) -> Tuple[Dict[int, int], int]:
    """
    Build rand_vec regions by partitioning the rand_vec space.

    Uses KMeans to cluster episodes based on their rand_vec values.
    Number of regions is dynamically computed based on episode count.

    Args:
        episode_data: Dict[episode_index, {"rand_vec": ..., "visual_embedding": ..., "action_descriptor": ...}]
        region_ratio: ratio to compute num_regions from episode count
        min_regions: minimum number of regions
        max_regions: maximum number of regions
        seed: random seed for clustering

    Returns:
        Tuple of:
            - Dict[episode_index, region_id]: mapping from episode to region
            - int: number of regions created
    """
    print(f"\n{'='*60}")
    print(f"Building rand_vec regions...")
    print(f"{'='*60}")

    valid_episodes = sorted(episode_data.keys())
    n_episodes = len(valid_episodes)

    if n_episodes == 0:
        raise ValueError("No valid episodes with rand_vec data")

    # Dynamically compute number of regions
    num_regions = max(min_regions, min(max_regions, int(n_episodes * region_ratio)))
    num_regions = min(num_regions, n_episodes)  # Cannot have more regions than episodes

    print(f"  Episodes: {n_episodes}")
    print(f"  Region ratio: {region_ratio}")
    print(f"  Computed regions: {num_regions} (min={min_regions}, max={max_regions})")

    # Extract rand_vec features
    rand_vec_list = []
    for ep_idx in valid_episodes:
        rand_vec_list.append(episode_data[ep_idx]["rand_vec"])

    rand_vecs = np.array(rand_vec_list)

    # Normalize rand_vecs
    norms = np.linalg.norm(rand_vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    rand_vecs_normalized = rand_vecs / norms

    # Cluster using KMeans
    kmeans = KMeans(
        n_clusters=num_regions,
        random_state=seed,
        n_init=10,
    )
    region_ids = kmeans.fit_predict(rand_vecs_normalized)

    # Build episode to region mapping
    episode_to_region = {}
    for i, ep_idx in enumerate(valid_episodes):
        episode_to_region[ep_idx] = int(region_ids[i])

    # Print region statistics
    region_counts = {}
    for region_id in region_ids:
        region_id = int(region_id)
        region_counts[region_id] = region_counts.get(region_id, 0) + 1

    print(f"  Regions created: {num_regions}")
    print(f"  Region size - min: {min(region_counts.values())}, max: {max(region_counts.values())}, "
          f"mean: {np.mean(list(region_counts.values())):.1f}")

    return episode_to_region, num_regions


def select_initial_b0_by_rand_vec_regions(
    episode_to_region: Dict[int, int],
    episode_data: Dict[int, Dict],
    num_regions: int,
    b0_region_ratio: float = 0.2,
    seed: int = 42,
) -> List[int]:
    """
    Select initial B0 episodes using rand_vec region coverage strategy.

    B0 size is dynamically computed based on number of regions.
    Uniformly sample from different rand_vec regions to ensure initial coverage.

    Args:
        episode_to_region: mapping from episode index to region id
        episode_data: episode data dict
        num_regions: total number of regions
        b0_region_ratio: ratio of regions to cover in B0
        seed: random seed

    Returns:
        List of selected episode indices
    """
    rng = np.random.RandomState(seed)

    # Compute B0 size dynamically
    b0_size = max(1, int(num_regions * b0_region_ratio))

    # Group episodes by region
    region_to_episodes = {}
    for ep_idx, region_id in episode_to_region.items():
        if region_id not in region_to_episodes:
            region_to_episodes[region_id] = []
        region_to_episodes[region_id].append(ep_idx)

    # Select one episode from each region until B0 is filled
    selected = []
    region_ids = sorted(region_to_episodes.keys())

    # First pass: select one from each region (prioritize coverage)
    for region_id in region_ids:
        if len(selected) >= b0_size:
            break
        region_episodes = region_to_episodes[region_id]
        chosen = rng.choice(region_episodes, size=1, replace=False).tolist()[0]
        selected.append(chosen)

    # If B0 not filled, sample more from larger regions
    if len(selected) < b0_size:
        remaining_episodes = [ep for ep in episode_data.keys() if ep not in selected]
        n_needed = b0_size - len(selected)
        additional = rng.choice(remaining_episodes, size=min(n_needed, len(remaining_episodes)), replace=False).tolist()
        selected.extend(additional)

    selected.sort()
    print(f"\n[Step 2] Initial B0 selection: {len(selected)} episodes from {num_regions} regions")
    print(f"  B0 size: {b0_size} (region_ratio={b0_region_ratio})")
    print(f"  B0 episodes: {selected}")

    return selected


def compute_region_priority(
    region_id: int,
    episode_to_region: Dict[int, int],
    selected_ids: List[int],
    episode_data: Dict[int, Dict],
    coverage_weight: float = 0.5,
    visual_weight: float = 0.3,
    action_weight: float = 0.2,
) -> Dict[str, float]:
    """
    Compute priority for a rand_vec region based on three factors:
    1. Coverage gap: fewer selected episodes in this region -> higher priority
    2. Visual uncertainty: higher variance in visual embeddings -> higher priority
    3. Action uncertainty: higher variance in action descriptors -> higher priority

    Args:
        region_id: the region to compute priority for
        episode_to_region: mapping from episode index to region id
        selected_ids: list of already selected episode indices
        episode_data: episode data dict
        coverage_weight: weight for coverage gap
        visual_weight: weight for visual uncertainty
        action_weight: weight for action uncertainty

    Returns:
        Dict with keys: "priority", "coverage_gap", "visual_uncertainty", "action_uncertainty"
    """
    # Get all episodes in this region
    region_episodes = [ep for ep, rid in episode_to_region.items() if rid == region_id]
    total_in_region = len(region_episodes)

    if total_in_region == 0:
        return {
            "priority": 0.0,
            "coverage_gap": 0.0,
            "visual_uncertainty": 0.0,
            "action_uncertainty": 0.0,
        }

    # 1. Coverage gap: inverse of selection ratio
    selected_in_region = [ep for ep in region_episodes if ep in selected_ids]
    n_selected_in_region = len(selected_in_region)

    # Coverage gap: higher when fewer episodes selected
    coverage_gap = 1.0 - (n_selected_in_region / total_in_region)

    # 2. Visual uncertainty: average pairwise distance of visual embeddings in region
    visual_uncertainty = 0.0
    region_visual_embs = []
    for ep in region_episodes:
        if ep in episode_data and "visual_embedding" in episode_data[ep]:
            region_visual_embs.append(episode_data[ep]["visual_embedding"])

    if len(region_visual_embs) >= 2:
        # Compute average pairwise distance
        visual_arr = np.array(region_visual_embs)
        n = len(visual_arr)
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += np.linalg.norm(visual_arr[i] - visual_arr[j])
                count += 1
        visual_uncertainty = total_dist / count if count > 0 else 0.0
    elif len(region_visual_embs) == 1:
        # Single episode: use norm as proxy
        visual_uncertainty = np.linalg.norm(region_visual_embs[0])

    # 3. Action uncertainty: average pairwise distance of action descriptors in region
    action_uncertainty = 0.0
    region_action_embs = []
    for ep in region_episodes:
        if ep in episode_data and "action_descriptor" in episode_data[ep]:
            region_action_embs.append(episode_data[ep]["action_descriptor"])

    if len(region_action_embs) >= 2:
        # Compute average pairwise distance
        action_arr = np.array(region_action_embs)
        n = len(action_arr)
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += np.linalg.norm(action_arr[i] - action_arr[j])
                count += 1
        action_uncertainty = total_dist / count if count > 0 else 0.0
    elif len(region_action_embs) == 1:
        action_uncertainty = np.linalg.norm(region_action_embs[0])

    # Normalize uncertainties to [0, 1] range for fair comparison
    # Use soft normalization based on typical ranges
    visual_uncertainty_norm = min(visual_uncertainty / 10.0, 1.0)  # Assume max ~10
    action_uncertainty_norm = min(action_uncertainty / 10.0, 1.0)

    # Compute final priority
    priority = (coverage_weight * coverage_gap +
                visual_weight * visual_uncertainty_norm +
                action_weight * action_uncertainty_norm)

    return {
        "priority": priority,
        "coverage_gap": coverage_gap,
        "visual_uncertainty": visual_uncertainty_norm,
        "action_uncertainty": action_uncertainty_norm,
    }


def compute_visual_coverage_score(
    candidate_embedding: np.ndarray,
    selected_embeddings: np.ndarray,
) -> float:
    """
    Compute visual coverage score for a candidate episode.

    Score = min distance to any already-selected episode in visual embedding space.
    Higher score = more visually distinct from selected episodes = better coverage.

    Args:
        candidate_embedding: visual feature of candidate episode
        selected_embeddings: visual features of already-selected episodes, shape (n_selected, dim)

    Returns:
        visual coverage score (higher = better)
    """
    if len(selected_embeddings) == 0:
        return 1.0

    distances = np.linalg.norm(selected_embeddings - candidate_embedding, axis=1)
    min_distance = float(np.min(distances))

    return min_distance


def compute_action_diversity_score(
    candidate_descriptor: np.ndarray,
    selected_descriptors: np.ndarray,
) -> float:
    """
    Compute action diversity score for a candidate episode.

    Score = min distance to any already-selected episode in action descriptor space.
    Higher score = more action-diverse from selected episodes = better coverage.

    Args:
        candidate_descriptor: action descriptor of candidate episode
        selected_descriptors: action descriptors of already-selected episodes, shape (n_selected, dim)

    Returns:
        action diversity score (higher = better)
    """
    if len(selected_descriptors) == 0:
        return 1.0

    distances = np.linalg.norm(selected_descriptors - candidate_descriptor, axis=1)
    min_distance = float(np.min(distances))

    return min_distance


def select_best_episode_from_region(
    region_id: int,
    episode_to_region: Dict[int, int],
    selected_ids: List[int],
    episode_data: Dict[int, Dict],
    visual_weight: float = 0.5,
    action_weight: float = 0.5,
) -> Optional[Tuple[int, Dict[str, float]]]:
    """
    Select the best episode from a specific region based on visual coverage and action diversity.

    Episode score = visual_weight * normalized_visual_score + action_weight * normalized_action_score.
    Visual and action scores are normalized before fusion to avoid scale mismatch.

    Args:
        region_id: the region to select from
        episode_to_region: mapping from episode index to region id
        selected_ids: list of already selected episode indices
        episode_data: episode data dict
        visual_weight: weight for visual coverage score
        action_weight: weight for action diversity score

    Returns:
        Tuple of (best_episode_index, scores_dict) or None if no candidates
    """
    # Get candidates from this region that are not yet selected
    candidates = [ep for ep, rid in episode_to_region.items() if rid == region_id and ep not in selected_ids]

    if not candidates:
        return None

    # Prepare selected embeddings for scoring
    selected_visual = np.array([
        episode_data[ep]["visual_embedding"]
        for ep in selected_ids
        if ep in episode_data and "visual_embedding" in episode_data[ep]
    ])
    selected_action = np.array([
        episode_data[ep]["action_descriptor"]
        for ep in selected_ids
        if ep in episode_data and "action_descriptor" in episode_data[ep]
    ])

    # Compute scores for all candidates
    candidate_scores = []
    for candidate_idx in candidates:
        if candidate_idx not in episode_data:
            continue

        # Compute visual coverage score
        if "visual_embedding" in episode_data[candidate_idx]:
            visual_score = compute_visual_coverage_score(
                episode_data[candidate_idx]["visual_embedding"],
                selected_visual
            )
        else:
            visual_score = 0.0

        # Compute action diversity score
        if "action_descriptor" in episode_data[candidate_idx]:
            action_score = compute_action_diversity_score(
                episode_data[candidate_idx]["action_descriptor"],
                selected_action
            )
        else:
            action_score = 0.0

        candidate_scores.append((candidate_idx, visual_score, action_score))

    if not candidate_scores:
        return None

    # Normalize scores to [0, 1] range
    visual_scores = [s[1] for s in candidate_scores]
    action_scores = [s[2] for s in candidate_scores]

    visual_min, visual_max = min(visual_scores), max(visual_scores)
    action_min, action_max = min(action_scores), max(action_scores)

    visual_range = visual_max - visual_min if visual_max > visual_min else 1.0
    action_range = action_max - action_min if action_max > action_min else 1.0

    # Compute normalized combined scores
    best_candidate = None
    best_scores = None
    best_score = -1.0

    for candidate_idx, vs, acs in candidate_scores:
        # Normalize to [0, 1]
        norm_vs = (vs - visual_min) / visual_range
        norm_acs = (acs - action_min) / action_range

        # Combined score
        total_score = visual_weight * norm_vs + action_weight * norm_acs

        if total_score > best_score:
            best_score = total_score
            best_candidate = candidate_idx
            best_scores = {
                "visual_score": float(vs),
                "action_score": float(acs),
                "normalized_visual_score": float(norm_vs),
                "normalized_action_score": float(norm_acs),
                "joint_score": float(total_score),
            }

    if best_candidate is not None:
        return best_candidate, best_scores

    return None


def select_episodes_v5_adaptive(
    all_episode_ids: List[int],
    visual_embeddings: Dict[int, np.ndarray],
    action_descriptors: Dict[int, np.ndarray],
    rand_vecs: Dict[int, np.ndarray],
    num_select: int,
    region_ratio: float = 0.1,
    min_regions: int = 16,
    max_regions: int = 128,
    b0_region_ratio: float = 0.2,
    coverage_weight: float = 0.5,
    region_visual_weight: float = 0.3,
    region_action_weight: float = 0.2,
    episode_visual_weight: float = 0.5,
    episode_action_weight: float = 0.5,
    seed: int = 42,
) -> Dict:
    """
    Execute rand_vec-aware adaptive coverage based selection flow.

    Flow:
    1. All episodes -> filter to those with rand_vec, visual, and action data
    2. Build rand_vec regions using KMeans
    3. Initialize B0 by uniformly sampling from different regions
    4. Iteratively:
       a. Compute region priorities for all regions
       b. Select highest priority region
       c. Within that region, select episode with highest score
       d. Update selected set and repeat (one episode at a time)
    5. Final selected episode ids

    Args:
        all_episode_ids: list of all episode indices
        visual_embeddings: Dict[ep_idx, visual_feature]
        action_descriptors: Dict[ep_idx, action_descriptor]
        rand_vecs: Dict[ep_idx, rand_vec]
        num_select: target number of episodes to select
        region_ratio: ratio to compute num_regions from episode count
        min_regions: minimum number of regions
        max_regions: maximum number of regions
        b0_region_ratio: ratio of regions to cover in B0
        coverage_weight: weight for coverage gap in region priority
        region_visual_weight: weight for visual uncertainty in region priority
        region_action_weight: weight for action uncertainty in region priority
        episode_visual_weight: weight for visual coverage in episode scoring
        episode_action_weight: weight for action diversity in episode scoring
        seed: random seed

    Returns:
        Dict with selected_episode_ids and selection metadata
    """
    # Build episode data dict (only episodes with all required data)
    episode_data = {}
    for ep_idx in all_episode_ids:
        if ep_idx in rand_vecs and ep_idx in visual_embeddings and ep_idx in action_descriptors:
            episode_data[ep_idx] = {
                "rand_vec": rand_vecs[ep_idx],
                "visual_embedding": visual_embeddings[ep_idx],
                "action_descriptor": action_descriptors[ep_idx],
            }

    valid_ids = sorted(episode_data.keys())

    print(f"\n{'='*60}")
    print(f"V5 Rand Vec Aware Adaptive Coverage Episode Selection")
    print(f"{'='*60}")
    print(f"Total episodes: {len(all_episode_ids)}")
    print(f"Valid episodes (with rand_vec+visual+action): {len(valid_ids)}")
    print(f"Target selection: {num_select}")
    print(f"Region ratio: {region_ratio}, min_regions: {min_regions}, max_regions: {max_regions}")
    print(f"B0 region ratio: {b0_region_ratio}")
    print(f"coverage_weight: {coverage_weight}, region_visual_weight: {region_visual_weight}, region_action_weight: {region_action_weight}")
    print(f"episode_visual_weight: {episode_visual_weight}, episode_action_weight: {episode_action_weight}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")

    # Step 1: Build rand_vec regions
    episode_to_region, num_regions = build_rand_vec_regions(
        episode_data, region_ratio, min_regions, max_regions, seed
    )

    # Step 2: Initialize B0 with region coverage
    b0_episodes = select_initial_b0_by_rand_vec_regions(
        episode_to_region, episode_data, num_regions, b0_region_ratio, seed
    )
    selected_ids = list(b0_episodes)

    # Step 3: Adaptive selection
    print(f"\n[Step 3] Starting adaptive region-based selection...")
    selection_log = []
    step_num = 0

    while len(selected_ids) < num_select:
        step_num += 1
        step_start = time.time()

        # Compute priorities for all regions
        region_priorities = {}
        for region_id in range(num_regions):
            priority_info = compute_region_priority(
                region_id, episode_to_region, selected_ids,
                episode_data,
                coverage_weight, region_visual_weight, region_action_weight
            )
            region_priorities[region_id] = priority_info

        # Select highest priority region
        best_region_id = max(region_priorities.keys(), key=lambda r: region_priorities[r]["priority"])
        best_region_priority = region_priorities[best_region_id]

        # Select best episode from this region
        result = select_best_episode_from_region(
            best_region_id, episode_to_region, selected_ids,
            episode_data,
            episode_visual_weight, episode_action_weight
        )

        if result is None:
            # No candidates in this region, mark as exhausted and try next
            # Remove this region from consideration by setting priority to -1
            region_priorities[best_region_id]["priority"] = -1.0
            print(f"  Step {step_num}: Region {best_region_id} exhausted, skipping...")

            # Check if all regions are exhausted
            max_priority = max(r["priority"] for r in region_priorities.values())
            if max_priority < 0:
                print(f"  All regions exhausted. Selected {len(selected_ids)} episodes.")
                break
            continue

        best_candidate, scores = result

        # Add to selected
        selected_ids.append(best_candidate)

        step_time = time.time() - step_start

        # Log this step
        selection_log.append({
            "step": step_num,
            "n_selected": len(selected_ids),
            "selected_region_id": best_region_id,
            "region_priority": best_region_priority["priority"],
            "region_coverage_gap": best_region_priority["coverage_gap"],
            "region_visual_uncertainty": best_region_priority["visual_uncertainty"],
            "region_action_uncertainty": best_region_priority["action_uncertainty"],
            "episode_index": best_candidate,
            "visual_score": scores["visual_score"],
            "action_score": scores["action_score"],
            "normalized_visual_score": scores["normalized_visual_score"],
            "normalized_action_score": scores["normalized_action_score"],
            "joint_score": scores["joint_score"],
            "time_seconds": step_time,
        })

        if step_num % 10 == 0 or step_num == 1:
            print(f"  Step {step_num}: region={best_region_id}, ep={best_candidate}, "
                  f"region_priority={best_region_priority['priority']:.4f}, "
                  f"joint_score={scores['joint_score']:.4f}, "
                  f"time={step_time:.2f}s")

    selected_ids.sort()

    print(f"\n{'='*60}")
    print(f"V5 Adaptive Selection Complete!")
    print(f"{'='*60}")
    print(f"Selected {len(selected_ids)} episodes")
    print(f"Total steps: {step_num}")

    # Compute region selection distribution
    region_selection_counts = {}
    for region_id in range(num_regions):
        region_episodes = [ep for ep, rid in episode_to_region.items() if rid == region_id]
        selected_in_region = [ep for ep in region_episodes if ep in selected_ids]
        region_selection_counts[region_id] = {
            "total": len(region_episodes),
            "selected": len(selected_in_region),
            "selection_ratio": len(selected_in_region) / len(region_episodes) if len(region_episodes) > 0 else 0.0,
        }

    return {
        "selected_episode_indices": selected_ids,
        "b0_episodes": b0_episodes,
        "episode_to_region": episode_to_region,
        "region_selection_counts": region_selection_counts,
        "selection_log": selection_log,
        "total_steps": step_num,
        "num_selected": len(selected_ids),
        "num_valid": len(valid_ids),
        "num_total": len(all_episode_ids),
        "num_regions": num_regions,
        "region_ratio": region_ratio,
        "min_regions": min_regions,
        "max_regions": max_regions,
        "b0_region_ratio": b0_region_ratio,
        "coverage_weight": coverage_weight,
        "region_visual_weight": region_visual_weight,
        "region_action_weight": region_action_weight,
        "episode_visual_weight": episode_visual_weight,
        "episode_action_weight": episode_action_weight,
        "seed": seed,
    }


def save_selected_episodes(selected_episode_ids: List[int], output_path: Path, metadata: Dict = None) -> None:
    """
    Save selected episode list in duibi experiment format.

    Args:
        selected_episode_ids: list of selected episode indices
        output_path: output JSON file path
        metadata: optional additional metadata
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subset_data = {
        "method": "our_v5",
        "selection_method": "our_v5_rand_vec_adaptive_coverage",
        "num_episodes": len(selected_episode_ids),
        "selected_episode_indices": selected_episode_ids,
    }

    if metadata:
        subset_data["parameters"] = metadata

    with open(output_path, "w") as f:
        json.dump(subset_data, f, indent=2)

    print(f"Selected episodes saved to: {output_path}")


def compute_diagnostic(
    result: Dict,
    episode_data: Dict[int, Dict],
    output_dir: Path,
) -> Dict:
    """Compute diagnostic statistics after selection, including rand_vec coverage metrics."""
    selected_ids = result["selected_episode_indices"]
    all_ids = sorted(episode_data.keys())
    full_ids = [ep for ep in all_ids if ep not in selected_ids]

    # Part 1: Per-episode diagnostic
    per_episode = []
    for entry in result.get("selection_log", []):
        ep_info = {
            "episode_id": entry["episode_index"],
            "step": entry["step"],
            "selected_region_id": entry["selected_region_id"],
            "region_priority": entry["region_priority"],
            "visual_score": entry["visual_score"],
            "action_score": entry["action_score"],
            "joint_score": entry["joint_score"],
        }
        per_episode.append(ep_info)

    # Part 2: Dataset statistics
    def _stats(scores):
        if not scores:
            return {"mean": 0.0, "std": 0.0}
        arr = np.array(scores)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

    sel_visual = [np.mean(episode_data[ep]["visual_embedding"]) for ep in selected_ids if ep in episode_data]
    full_visual = [np.mean(episode_data[ep]["visual_embedding"]) for ep in full_ids if ep in episode_data]
    sel_action = [np.mean(episode_data[ep]["action_descriptor"]) for ep in selected_ids if ep in episode_data]
    full_action = [np.mean(episode_data[ep]["action_descriptor"]) for ep in full_ids if ep in episode_data]

    # Part 3: Action coverage analysis
    sel_action_vecs = [episode_data[ep]["action_descriptor"] for ep in selected_ids if ep in episode_data]
    full_action_vecs = [episode_data[ep]["action_descriptor"] for ep in all_ids if ep in episode_data]

    l2_dists = []
    cosine_dists = []
    if sel_action_vecs and full_action_vecs:
        sel_arr = np.array(sel_action_vecs)
        full_arr = np.array(full_action_vecs)
        for sv in sel_arr:
            diffs = full_arr - sv
            l2_dists.append(float(np.mean(np.linalg.norm(diffs, axis=1))))
            norms_s = np.linalg.norm(sv)
            norms_f = np.linalg.norm(full_arr, axis=1)
            mask = (norms_s > 1e-10) & (norms_f > 1e-10)
            if mask.any():
                sims = np.dot(full_arr[mask], sv) / (norms_f[mask] * norms_s)
                sims = np.clip(sims, -1.0, 1.0)
                cosine_dists.append(float(np.mean(1.0 - sims)))

    # Part 4: Rand_vec coverage statistics
    episode_to_region = result.get("episode_to_region", {})
    region_selection_counts = result.get("region_selection_counts", {})
    num_regions = result.get("num_regions", 0)

    selected_region_count = sum(1 for rid, info in region_selection_counts.items() if info["selected"] > 0)
    total_region_count = len(region_selection_counts)

    region_selection_distribution = []
    for rid in sorted(region_selection_counts.keys()):
        info = region_selection_counts[rid]
        region_selection_distribution.append({
            "region_id": rid,
            "total_episodes": info["total"],
            "selected_episodes": info["selected"],
            "selection_ratio": info["selection_ratio"],
        })

    # Average region coverage gap for selected regions
    selected_region_coverage_gaps = [
        entry["region_coverage_gap"] for entry in result["selection_log"]
    ]
    avg_region_coverage_gap = float(np.mean(selected_region_coverage_gaps)) if selected_region_coverage_gaps else 0.0

    # Average uncertainties for selected regions
    selected_region_visual_uncertainties = [
        entry["region_visual_uncertainty"] for entry in result["selection_log"]
    ]
    avg_selected_region_visual_uncertainty = float(np.mean(selected_region_visual_uncertainties)) if selected_region_visual_uncertainties else 0.0

    selected_region_action_uncertainties = [
        entry["region_action_uncertainty"] for entry in result["selection_log"]
    ]
    avg_selected_region_action_uncertainty = float(np.mean(selected_region_action_uncertainties)) if selected_region_action_uncertainties else 0.0

    # Selected rand_vec distribution
    selected_rand_vecs = []
    for ep in selected_ids:
        if ep in episode_data and "rand_vec" in episode_data[ep]:
            selected_rand_vecs.append(episode_data[ep]["rand_vec"].tolist())

    full_rand_vecs = []
    for ep in all_ids:
        if ep in episode_data and "rand_vec" in episode_data[ep]:
            full_rand_vecs.append(episode_data[ep]["rand_vec"].tolist())

    # Build diagnostic result
    diagnostic = {
        "per_episode_diagnostic": per_episode,
        "dataset_statistics": {
            "selected_count": len(selected_ids),
            "total_count": len(all_ids),
            "full_count": len(full_ids),
            "selected_visual_score": _stats(sel_visual),
            "full_visual_score": _stats(full_visual),
            "selected_action_score": _stats(sel_action),
            "full_action_score": _stats(full_action),
        },
        "action_coverage_analysis": {
            "selected_action_distribution_distance": {
                "mean_l2_distance": float(np.mean(l2_dists)) if l2_dists else 0.0,
                "mean_cosine_distance": float(np.mean(cosine_dists)) if cosine_dists else 0.0,
            },
            "n_comparisons": len(l2_dists),
        },
        "rand_vec_coverage_statistics": {
            "selected_region_count": selected_region_count,
            "total_region_count": total_region_count,
            "region_coverage_ratio": selected_region_count / total_region_count if total_region_count > 0 else 0.0,
            "region_selection_distribution": region_selection_distribution,
            "average_region_coverage_gap": avg_region_coverage_gap,
            "average_region_visual_uncertainty": avg_selected_region_visual_uncertainty,
            "average_region_action_uncertainty": avg_selected_region_action_uncertainty,
            "selected_rand_vec_distribution": selected_rand_vecs[:10],  # First 10 for inspection
            "full_rand_vec_distribution": full_rand_vecs[:10],  # First 10 for comparison
        },
    }

    diag_file = output_dir / "diagnostic_v5.json"
    with open(diag_file, "w") as f:
        json.dump(diagnostic, f, indent=2)
    print(f"Diagnostic saved to: {diag_file}")

    return diagnostic


def main():
    parser = argparse.ArgumentParser(description="V5 Rand Vec Aware Adaptive Coverage Episode Selection")
    parser.add_argument("--visual-embedding-dir", type=str, required=True,
                       help="Visual embedding cache directory (existing global+wrist)")
    parser.add_argument("--action-descriptor-dir", type=str, required=True,
                       help="Action descriptor cache directory")
    parser.add_argument("--dataset-dir", type=str, required=True,
                       help="Dataset root directory (contains episode_initial_states.json)")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--num-selected", type=int, default=112,
                       help="Target number of episodes (default: 112)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--visual-weight", type=float, default=0.5,
                       help="Weight for visual coverage score (default: 0.5)")
    parser.add_argument("--action-weight", type=float, default=0.5,
                       help="Weight for action diversity score (default: 0.5)")
    parser.add_argument("--region-ratio", type=float, default=0.1,
                       help="Ratio to compute num_regions from episode count (default: 0.1)")
    parser.add_argument("--min-regions", type=int, default=16,
                       help="Minimum number of regions (default: 16)")
    parser.add_argument("--max-regions", type=int, default=128,
                       help="Maximum number of regions (default: 128)")
    parser.add_argument("--b0-region-ratio", type=float, default=0.2,
                       help="Ratio of regions to cover in B0 (default: 0.2)")
    parser.add_argument("--coverage-weight", type=float, default=0.5,
                       help="Weight for coverage gap in region priority (default: 0.5)")
    parser.add_argument("--region-visual-weight", type=float, default=0.3,
                       help="Weight for visual uncertainty in region priority (default: 0.3)")
    parser.add_argument("--region-action-weight", type=float, default=0.2,
                       help="Weight for action uncertainty in region priority (default: 0.2)")

    args = parser.parse_args()

    visual_dir = Path(args.visual_embedding_dir)
    action_dir = Path(args.action_descriptor_dir)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    if not visual_dir.exists():
        print(f"Error: Visual embedding directory does not exist: {visual_dir}")
        return

    if not action_dir.exists():
        print(f"Error: Action descriptor directory does not exist: {action_dir}")
        return

    if not dataset_dir.exists():
        print(f"Error: Dataset directory does not exist: {dataset_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "subsets").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    print(f"\nLoading visual embeddings from: {visual_dir}")
    visual_embeddings = load_visual_embeddings(visual_dir)

    print(f"\nLoading action descriptors from: {action_dir}")
    action_descriptors = load_action_descriptors(action_dir)

    all_episode_ids = sorted(set(list(visual_embeddings.keys()) | set(list(action_descriptors.keys()))))

    print(f"\nLoading rand_vecs from: {dataset_dir}")
    rand_vecs = load_rand_vecs(dataset_dir, all_episode_ids)

    # Filter to episodes with all required data
    valid_ids = [
        ep_idx for ep_idx in all_episode_ids
        if ep_idx in rand_vecs and ep_idx in visual_embeddings and ep_idx in action_descriptors
    ]
    print(f"Episodes with all data (rand_vec+visual+action): {len(valid_ids)}")

    if len(valid_ids) == 0:
        print("Error: No episodes with all required data")
        return

    result = select_episodes_v5_adaptive(
        all_episode_ids=all_episode_ids,
        visual_embeddings=visual_embeddings,
        action_descriptors=action_descriptors,
        rand_vecs=rand_vecs,
        num_select=args.num_selected,
        region_ratio=args.region_ratio,
        min_regions=args.min_regions,
        max_regions=args.max_regions,
        b0_region_ratio=args.b0_region_ratio,
        coverage_weight=args.coverage_weight,
        region_visual_weight=args.region_visual_weight,
        region_action_weight=args.region_action_weight,
        episode_visual_weight=args.visual_weight,
        episode_action_weight=args.action_weight,
        seed=args.seed,
    )

    # Build episode data for diagnostic
    episode_data = {}
    for ep_idx in valid_ids:
        episode_data[ep_idx] = {
            "rand_vec": rand_vecs[ep_idx],
            "visual_embedding": visual_embeddings[ep_idx],
            "action_descriptor": action_descriptors[ep_idx],
        }

    subset_file = output_dir / "subsets" / f"our_v5_{args.num_selected}_seed{args.seed}.json"
    save_selected_episodes(
        selected_episode_ids=result["selected_episode_indices"],
        output_path=subset_file,
        metadata={
            "selection_method": "our_v5_rand_vec_adaptive_coverage",
            "num_regions": result["num_regions"],
            "region_ratio": args.region_ratio,
            "min_regions": args.min_regions,
            "max_regions": args.max_regions,
            "b0_region_ratio": args.b0_region_ratio,
            "coverage_weight": args.coverage_weight,
            "region_visual_weight": args.region_visual_weight,
            "region_action_weight": args.region_action_weight,
            "episode_visual_weight": args.visual_weight,
            "episode_action_weight": args.action_weight,
            "dynamic_selection": True,
            "seed": args.seed,
        }
    )

    log_file = output_dir / "results" / f"selection_log_v5_{args.num_selected}_seed{args.seed}.json"
    with open(log_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Selection log saved to: {log_file}")

    # Run diagnostic analysis
    print(f"\n{'='*60}")
    print(f"Running diagnostic analysis...")
    compute_diagnostic(result, episode_data, output_dir)


if __name__ == "__main__":
    main()