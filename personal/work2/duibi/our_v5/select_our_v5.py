#!/usr/bin/env python
"""
Our V5 Episode Selection - Adaptive Latent Coverage Selection

Selects episodes based on adaptive latent coverage using:
- Visual embeddings (global+wrist concatenated)
- Action descriptors

Core idea:
- Build latent clusters using KMeans on concatenated visual+action features
- Initialize B0 by uniformly sampling from different clusters
- Iteratively select episodes based on cluster priority:
  - coverage_need: fewer selected episodes in cluster -> higher priority
  - visual_uncertainty: higher visual embedding variance -> higher priority
  - action_uncertainty: higher action descriptor variance -> higher priority
- Within selected cluster, choose episode with highest gain

Does NOT use: success, grasp_success, eval results, attention results, test results,
obj_init_pos, goal_pose, environment_state, or any simulation state information.

Usage:
    python select_our_v5.py \
        --visual-embedding-dir /path/to/visual/embeddings \
        --action-descriptor-dir /path/to/action/descriptors \
        --output-dir /path/to/output \
        --num-selected 112 \
        --seed 42 \
        --num-clusters 32 \
        --coverage-weight 0.5 \
        --cluster-visual-weight 0.3 \
        --cluster-action-weight 0.2
"""

import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import MiniBatchKMeans

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


def build_latent_space_clusters(
    visual_embeddings: Dict[int, np.ndarray],
    action_descriptors: Dict[int, np.ndarray],
    num_clusters: int = 32,
    seed: int = 42,
) -> Tuple[Dict[int, int], np.ndarray]:
    """
    Build latent space clusters by clustering episodes in joint visual+action space.

    Concatenates normalized visual features and action descriptors for each episode,
    then uses MiniBatchKMeans to partition into latent clusters.

    Args:
        visual_embeddings: Dict[episode_index, visual_feature]
        action_descriptors: Dict[episode_index, action_descriptor]
        num_clusters: number of latent clusters to create
        seed: random seed for clustering

    Returns:
        Tuple of:
            - Dict[episode_index, cluster_id]: mapping from episode to cluster
            - np.ndarray: cluster centers for diagnostics
    """
    print(f"\n{'='*60}")
    print(f"Building {num_clusters} latent space clusters...")
    print(f"{'='*60}")

    # Get valid episodes (have both visual and action data)
    valid_episodes = sorted(set(visual_embeddings.keys()) & set(action_descriptors.keys()))
    n_episodes = len(valid_episodes)

    if n_episodes == 0:
        raise ValueError("No valid episodes with both visual embeddings and action descriptors")

    # Extract and concatenate features
    feature_list = []
    for ep_idx in valid_episodes:
        visual_feat = visual_embeddings[ep_idx]
        action_feat = action_descriptors[ep_idx]

        # Concatenate visual and action features
        combined = np.concatenate([visual_feat, action_feat])
        feature_list.append(combined)

    features = np.array(feature_list)

    # L2 normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    features_normalized = features / norms

    # Cluster using MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=seed,
        batch_size=min(256, n_episodes),
        n_init=10,
    )
    cluster_ids = kmeans.fit_predict(features_normalized)

    # Build episode to cluster mapping
    episode_to_cluster = {}
    for i, ep_idx in enumerate(valid_episodes):
        episode_to_cluster[ep_idx] = int(cluster_ids[i])

    # Print cluster statistics
    cluster_counts = {}
    for cluster_id in cluster_ids:
        cluster_id = int(cluster_id)
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

    print(f"  Episodes clustered: {n_episodes}")
    print(f"  Clusters created: {num_clusters}")
    print(f"  Cluster size - min: {min(cluster_counts.values())}, max: {max(cluster_counts.values())}, "
          f"mean: {np.mean(list(cluster_counts.values())):.1f}")

    return episode_to_cluster, kmeans.cluster_centers_


def select_initial_b0_by_clusters(
    episode_to_cluster: Dict[int, int],
    valid_episodes: List[int],
    b0_size: int,
    num_clusters: int,
    seed: int = 42,
) -> List[int]:
    """
    Select initial B0 episodes using latent cluster coverage strategy.

    Instead of random sampling, uniformly sample from different latent clusters
    to ensure initial coverage of the latent space.

    Args:
        episode_to_cluster: mapping from episode index to cluster id
        valid_episodes: list of all valid episode indices
        b0_size: number of initial episodes to select
        num_clusters: total number of latent clusters
        seed: random seed

    Returns:
        List of selected episode indices
    """
    rng = np.random.RandomState(seed)

    # Group episodes by cluster
    cluster_to_episodes = {}
    for ep_idx in valid_episodes:
        cluster_id = episode_to_cluster.get(ep_idx)
        if cluster_id is not None:
            if cluster_id not in cluster_to_episodes:
                cluster_to_episodes[cluster_id] = []
            cluster_to_episodes[cluster_id].append(ep_idx)

    # Calculate episodes per cluster
    episodes_per_cluster = b0_size // num_clusters
    remainder = b0_size % num_clusters

    selected = []

    # First, select uniformly from each cluster
    for cluster_id in range(num_clusters):
        if cluster_id in cluster_to_episodes:
            cluster_episodes = cluster_to_episodes[cluster_id]
            n_select = episodes_per_cluster + (1 if cluster_id < remainder else 0)
            n_select = min(n_select, len(cluster_episodes))

            if n_select > 0:
                chosen = rng.choice(cluster_episodes, size=n_select, replace=False).tolist()
                selected.extend(chosen)

    # If we still need more, sample from remaining episodes
    if len(selected) < b0_size:
        remaining = [ep for ep in valid_episodes if ep not in selected]
        n_needed = b0_size - len(selected)
        additional = rng.choice(remaining, size=min(n_needed, len(remaining)), replace=False).tolist()
        selected.extend(additional)

    selected.sort()
    return selected


def compute_cluster_priority(
    cluster_id: int,
    episode_to_cluster: Dict[int, int],
    selected_ids: List[int],
    visual_embeddings: Dict[int, np.ndarray],
    action_descriptors: Dict[int, np.ndarray],
    coverage_weight: float = 0.5,
    visual_weight: float = 0.3,
    action_weight: float = 0.2,
) -> Dict[str, float]:
    """
    Compute priority for a latent cluster based on three factors:
    1. Coverage need: fewer selected episodes in this cluster -> higher priority
    2. Visual uncertainty: higher variance in visual embeddings -> higher priority
    3. Action uncertainty: higher variance in action descriptors -> higher priority

    Args:
        cluster_id: the cluster to compute priority for
        episode_to_cluster: mapping from episode index to cluster id
        selected_ids: list of already selected episode indices
        visual_embeddings: visual embeddings dict
        action_descriptors: action descriptors dict
        coverage_weight: weight for coverage need
        visual_weight: weight for visual uncertainty
        action_weight: weight for action uncertainty

    Returns:
        Dict with keys: "priority", "coverage_need", "visual_uncertainty", "action_uncertainty"
    """
    # Get all episodes in this cluster
    cluster_episodes = [ep for ep, cid in episode_to_cluster.items() if cid == cluster_id]
    total_in_cluster = len(cluster_episodes)

    if total_in_cluster == 0:
        return {
            "priority": 0.0,
            "coverage_need": 0.0,
            "visual_uncertainty": 0.0,
            "action_uncertainty": 0.0,
        }

    # 1. Coverage need: inverse of selection ratio
    selected_in_cluster = [ep for ep in cluster_episodes if ep in selected_ids]
    n_selected_in_cluster = len(selected_in_cluster)

    # Coverage need: higher when fewer episodes selected
    coverage_need = 1.0 - (n_selected_in_cluster / total_in_cluster)

    # 2. Visual uncertainty: average pairwise distance of visual embeddings in cluster
    visual_uncertainty = 0.0
    cluster_visual_embs = []
    for ep in cluster_episodes:
        if ep in visual_embeddings:
            cluster_visual_embs.append(visual_embeddings[ep])

    if len(cluster_visual_embs) >= 2:
        # Compute average pairwise distance
        visual_arr = np.array(cluster_visual_embs)
        n = len(visual_arr)
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += np.linalg.norm(visual_arr[i] - visual_arr[j])
                count += 1
        visual_uncertainty = total_dist / count if count > 0 else 0.0
    elif len(cluster_visual_embs) == 1:
        # Single episode: use norm as proxy
        visual_uncertainty = np.linalg.norm(cluster_visual_embs[0])

    # 3. Action uncertainty: average pairwise distance of action descriptors in cluster
    action_uncertainty = 0.0
    cluster_action_embs = []
    for ep in cluster_episodes:
        if ep in action_descriptors:
            cluster_action_embs.append(action_descriptors[ep])

    if len(cluster_action_embs) >= 2:
        # Compute average pairwise distance
        action_arr = np.array(cluster_action_embs)
        n = len(action_arr)
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += np.linalg.norm(action_arr[i] - action_arr[j])
                count += 1
        action_uncertainty = total_dist / count if count > 0 else 0.0
    elif len(cluster_action_embs) == 1:
        action_uncertainty = np.linalg.norm(cluster_action_embs[0])

    # Normalize uncertainties to [0, 1] range for fair comparison
    # Use soft normalization based on typical ranges
    visual_uncertainty_norm = min(visual_uncertainty / 10.0, 1.0)  # Assume max ~10
    action_uncertainty_norm = min(action_uncertainty / 10.0, 1.0)

    # Compute final priority
    priority = (coverage_weight * coverage_need +
                visual_weight * visual_uncertainty_norm +
                action_weight * action_uncertainty_norm)

    return {
        "priority": priority,
        "coverage_need": coverage_need,
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


def select_best_episode_from_cluster(
    cluster_id: int,
    selected_ids: List[int],
    cluster_episode_ids: List[int],
    visual_embeddings: Dict[int, np.ndarray],
    action_descriptors: Dict[int, np.ndarray],
    visual_weight_episode: float = 0.5,
    action_weight_episode: float = 0.5,
) -> Optional[Tuple[int, Dict[str, float]]]:
    """
    Select the best episode from a specific cluster based on visual coverage and action diversity.

    Args:
        cluster_id: the cluster to select from
        selected_ids: list of already selected episode indices
        cluster_episode_ids: list of episode indices in this cluster
        visual_embeddings: visual embeddings dict
        action_descriptors: action descriptors dict
        visual_weight_episode: weight for visual coverage score
        action_weight_episode: weight for action diversity score

    Returns:
        Tuple of (best_episode_index, scores_dict) or None if no candidates
    """
    # Get candidates from this cluster that are not yet selected
    candidates = [ep for ep in cluster_episode_ids if ep not in selected_ids]

    if not candidates:
        return None

    # Prepare selected embeddings for scoring
    selected_visual = np.array([visual_embeddings[ep] for ep in selected_ids if ep in visual_embeddings])
    selected_action = np.array([action_descriptors[ep] for ep in selected_ids if ep in action_descriptors])

    best_candidate = None
    best_scores = None
    best_score = -1.0

    for candidate_idx in candidates:
        if candidate_idx not in visual_embeddings or candidate_idx not in action_descriptors:
            continue

        # Compute visual coverage score
        visual_score = compute_visual_coverage_score(
            visual_embeddings[candidate_idx],
            selected_visual
        )

        # Compute action diversity score
        action_score = compute_action_diversity_score(
            action_descriptors[candidate_idx],
            selected_action
        )

        # Combined score
        total_score = visual_weight_episode * visual_score + action_weight_episode * action_score

        if total_score > best_score:
            best_score = total_score
            best_candidate = candidate_idx
            best_scores = {
                "visual_score": visual_score,
                "action_score": action_score,
                "joint_score": total_score,
            }

    if best_candidate is not None:
        return best_candidate, best_scores

    return None


def select_episodes_v5_adaptive(
    all_episode_ids: List[int],
    visual_embeddings: Dict[int, np.ndarray],
    action_descriptors: Dict[int, np.ndarray],
    num_select: int,
    num_clusters: int = 32,
    coverage_weight: float = 0.5,
    cluster_visual_weight: float = 0.3,
    cluster_action_weight: float = 0.2,
    episode_visual_weight: float = 0.5,
    episode_action_weight: float = 0.5,
    seed: int = 42,
    b0_size: int = 18,
) -> Dict:
    """
    Execute adaptive latent coverage based selection flow.

    Flow:
    1. All episodes -> filter to those with both visual and action data
    2. Build latent clusters using KMeans on concatenated features
    3. Initialize B0 by uniformly sampling from different clusters
    4. Iteratively:
       a. Compute cluster priorities for all clusters
       b. Select highest priority cluster
       c. Within that cluster, select episode with highest gain
       d. Update selected set and repeat (one episode at a time)
    5. Final selected episode ids

    Args:
        all_episode_ids: list of all episode indices
        visual_embeddings: Dict[ep_idx, visual_feature]
        action_descriptors: Dict[ep_idx, action_descriptor]
        num_select: target number of episodes to select
        num_clusters: number of latent clusters
        coverage_weight: weight for coverage need in cluster priority
        cluster_visual_weight: weight for visual uncertainty in cluster priority
        cluster_action_weight: weight for action uncertainty in cluster priority
        episode_visual_weight: weight for visual coverage in episode scoring
        episode_action_weight: weight for action diversity in episode scoring
        seed: random seed
        b0_size: initial B0 size

    Returns:
        Dict with selected_episode_ids and selection metadata
    """
    valid_ids = sorted([
        ep_id for ep_id in all_episode_ids
        if ep_id in visual_embeddings and ep_id in action_descriptors
    ])

    print(f"\n{'='*60}")
    print(f"V5 Adaptive Latent Coverage Episode Selection")
    print(f"{'='*60}")
    print(f"Total episodes: {len(all_episode_ids)}")
    print(f"Valid episodes (with both visual+action): {len(valid_ids)}")
    print(f"Target selection: {num_select}")
    print(f"Number of clusters: {num_clusters}")
    print(f"B0 size: {b0_size}")
    print(f"coverage_weight: {coverage_weight}, cluster_visual_weight: {cluster_visual_weight}, cluster_action_weight: {cluster_action_weight}")
    print(f"episode_visual_weight: {episode_visual_weight}, episode_action_weight: {episode_action_weight}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")

    # Step 1: Build latent clusters
    episode_to_cluster, cluster_centers = build_latent_space_clusters(
        visual_embeddings, action_descriptors, num_clusters, seed
    )

    # Step 2: Initialize B0 with cluster coverage
    print(f"\n[Step 2] Selecting initial B0 with cluster coverage ({b0_size} episodes)...")
    b0_episodes = select_initial_b0_by_clusters(
        episode_to_cluster, valid_ids, b0_size, num_clusters, seed
    )
    selected_ids = list(b0_episodes)
    remaining_ids = [ep for ep in valid_ids if ep not in selected_ids]

    print(f"  B0 episodes: {b0_episodes}")

    # Step 3: Adaptive selection
    print(f"\n[Step 3] Starting adaptive cluster-based selection...")
    selection_log = []
    step_num = 0

    while len(selected_ids) < num_select and remaining_ids:
        step_num += 1
        step_start = time.time()

        # Compute priorities for all clusters
        cluster_priorities = {}
        for cluster_id in range(num_clusters):
            priority_info = compute_cluster_priority(
                cluster_id, episode_to_cluster, selected_ids,
                visual_embeddings, action_descriptors,
                coverage_weight, cluster_visual_weight, cluster_action_weight
            )
            cluster_priorities[cluster_id] = priority_info

        # Select highest priority cluster
        best_cluster_id = max(cluster_priorities.keys(), key=lambda c: cluster_priorities[c]["priority"])
        best_cluster_priority = cluster_priorities[best_cluster_id]

        # Get episodes in this cluster
        cluster_episodes = [ep for ep, cid in episode_to_cluster.items() if cid == best_cluster_id]

        # Select best episode from this cluster
        result = select_best_episode_from_cluster(
            best_cluster_id, selected_ids, cluster_episodes,
            visual_embeddings, action_descriptors,
            episode_visual_weight, episode_action_weight
        )

        if result is None:
            # No candidates in this cluster, mark as exhausted and try next
            print(f"  Step {step_num}: Cluster {best_cluster_id} exhausted, skipping...")
            continue

        best_candidate, scores = result

        # Add to selected
        selected_ids.append(best_candidate)
        if best_candidate in remaining_ids:
            remaining_ids.remove(best_candidate)

        step_time = time.time() - step_start

        # Log this step
        selection_log.append({
            "step": step_num,
            "n_selected": len(selected_ids),
            "selected_cluster_id": best_cluster_id,
            "cluster_priority": best_cluster_priority["priority"],
            "cluster_coverage_need": best_cluster_priority["coverage_need"],
            "cluster_visual_uncertainty": best_cluster_priority["visual_uncertainty"],
            "cluster_action_uncertainty": best_cluster_priority["action_uncertainty"],
            "episode_index": best_candidate,
            "visual_score": scores["visual_score"],
            "action_score": scores["action_score"],
            "joint_score": scores["joint_score"],
            "time_seconds": step_time,
        })

        if step_num % 10 == 0 or step_num == 1:
            print(f"  Step {step_num}: cluster={best_cluster_id}, ep={best_candidate}, "
                  f"cluster_priority={best_cluster_priority['priority']:.4f}, "
                  f"joint_score={scores['joint_score']:.4f}, "
                  f"time={step_time:.2f}s")

    selected_ids.sort()

    print(f"\n{'='*60}")
    print(f"V5 Adaptive Selection Complete!")
    print(f"{'='*60}")
    print(f"Selected {len(selected_ids)} episodes")
    print(f"Total steps: {step_num}")

    # Compute cluster selection distribution
    cluster_selection_counts = {}
    for cluster_id in range(num_clusters):
        cluster_episodes = [ep for ep, cid in episode_to_cluster.items() if cid == cluster_id]
        selected_in_cluster = [ep for ep in cluster_episodes if ep in selected_ids]
        cluster_selection_counts[cluster_id] = {
            "total": len(cluster_episodes),
            "selected": len(selected_in_cluster),
            "selection_ratio": len(selected_in_cluster) / len(cluster_episodes) if len(cluster_episodes) > 0 else 0.0,
        }

    return {
        "selected_episode_indices": selected_ids,
        "b0_episodes": b0_episodes,
        "episode_to_cluster": episode_to_cluster,
        "cluster_selection_counts": cluster_selection_counts,
        "selection_log": selection_log,
        "total_steps": step_num,
        "num_selected": len(selected_ids),
        "num_valid": len(valid_ids),
        "num_total": len(all_episode_ids),
        "num_clusters": num_clusters,
        "coverage_weight": coverage_weight,
        "cluster_visual_weight": cluster_visual_weight,
        "cluster_action_weight": cluster_action_weight,
        "episode_visual_weight": episode_visual_weight,
        "episode_action_weight": episode_action_weight,
        "seed": seed,
        "b0_size": b0_size,
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
        "selection_method": "our_v5_adaptive_latent_coverage",
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
    visual_embeddings: Dict[int, np.ndarray],
    action_descriptors: Dict[int, np.ndarray],
    output_dir: Path,
) -> Dict:
    """Compute diagnostic statistics after selection, including adaptive coverage metrics."""
    selected_ids = result["selected_episode_indices"]
    all_ids = sorted(set(list(visual_embeddings.keys()) | set(list(action_descriptors.keys()))))
    full_ids = [ep for ep in all_ids if ep not in selected_ids]

    # Part 1: Per-episode diagnostic
    per_episode = []
    for entry in result.get("selection_log", []):
        ep_info = {
            "episode_id": entry["episode_index"],
            "step": entry["step"],
            "selected_cluster_id": entry["selected_cluster_id"],
            "cluster_priority": entry["cluster_priority"],
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

    sel_visual = [np.mean(visual_embeddings[ep]) for ep in selected_ids if ep in visual_embeddings]
    full_visual = [np.mean(visual_embeddings[ep]) for ep in full_ids if ep in visual_embeddings]
    sel_action = [np.mean(action_descriptors[ep]) for ep in selected_ids if ep in action_descriptors]
    full_action = [np.mean(action_descriptors[ep]) for ep in full_ids if ep in action_descriptors]

    # Part 3: Action coverage analysis
    sel_action_vecs = [action_descriptors[ep] for ep in selected_ids if ep in action_descriptors]
    full_action_vecs = [action_descriptors[ep] for ep in all_ids if ep in action_descriptors]

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

    # Part 4: Adaptive coverage statistics
    episode_to_cluster = result.get("episode_to_cluster", {})
    cluster_selection_counts = result.get("cluster_selection_counts", {})
    num_clusters = result.get("num_clusters", 0)

    selected_cluster_count = sum(1 for cid, info in cluster_selection_counts.items() if info["selected"] > 0)
    total_cluster_count = len(cluster_selection_counts)

    cluster_selection_distribution = []
    for cid in sorted(cluster_selection_counts.keys()):
        info = cluster_selection_counts[cid]
        cluster_selection_distribution.append({
            "cluster_id": cid,
            "total_episodes": info["total"],
            "selected_episodes": info["selected"],
            "selection_ratio": info["selection_ratio"],
        })

    # Average cluster priority for selected clusters
    selected_cluster_priorities = [
        entry["cluster_priority"] for entry in result["selection_log"]
    ]
    avg_cluster_priority = float(np.mean(selected_cluster_priorities)) if selected_cluster_priorities else 0.0

    # Average uncertainties for selected clusters
    selected_cluster_visual_uncertainties = [
        entry["cluster_visual_uncertainty"] for entry in result["selection_log"]
    ]
    avg_selected_cluster_visual_uncertainty = float(np.mean(selected_cluster_visual_uncertainties)) if selected_cluster_visual_uncertainties else 0.0

    selected_cluster_action_uncertainties = [
        entry["cluster_action_uncertainty"] for entry in result["selection_log"]
    ]
    avg_selected_cluster_action_uncertainty = float(np.mean(selected_cluster_action_uncertainties)) if selected_cluster_action_uncertainties else 0.0

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
        "adaptive_coverage_statistics": {
            "selected_cluster_count": selected_cluster_count,
            "total_cluster_count": total_cluster_count,
            "cluster_coverage_ratio": selected_cluster_count / total_cluster_count if total_cluster_count > 0 else 0.0,
            "cluster_selection_distribution": cluster_selection_distribution,
            "average_cluster_priority": avg_cluster_priority,
            "average_selected_cluster_visual_uncertainty": avg_selected_cluster_visual_uncertainty,
            "average_selected_cluster_action_uncertainty": avg_selected_cluster_action_uncertainty,
        },
    }

    diag_file = output_dir / "diagnostic_v5.json"
    with open(diag_file, "w") as f:
        json.dump(diagnostic, f, indent=2)
    print(f"Diagnostic saved to: {diag_file}")

    return diagnostic


def main():
    parser = argparse.ArgumentParser(description="V5 Adaptive Latent Coverage Episode Selection")
    parser.add_argument("--visual-embedding-dir", type=str, required=True,
                       help="Visual embedding cache directory (existing global+wrist)")
    parser.add_argument("--action-descriptor-dir", type=str, required=True,
                       help="Action descriptor cache directory")
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
    parser.add_argument("--b0-size", type=int, default=18,
                       help="Initial B0 size (default: 18)")
    parser.add_argument("--num-clusters", type=int, default=32,
                       help="Number of latent clusters (default: 32)")
    parser.add_argument("--coverage-weight", type=float, default=0.5,
                       help="Weight for coverage need in cluster priority (default: 0.5)")
    parser.add_argument("--cluster-visual-weight", type=float, default=0.3,
                       help="Weight for visual uncertainty in cluster priority (default: 0.3)")
    parser.add_argument("--cluster-action-weight", type=float, default=0.2,
                       help="Weight for action uncertainty in cluster priority (default: 0.2)")

    args = parser.parse_args()

    visual_dir = Path(args.visual_embedding_dir)
    action_dir = Path(args.action_descriptor_dir)
    output_dir = Path(args.output_dir)

    if not visual_dir.exists():
        print(f"Error: Visual embedding directory does not exist: {visual_dir}")
        return

    if not action_dir.exists():
        print(f"Error: Action descriptor directory does not exist: {action_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "subsets").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    print(f"\nLoading visual embeddings from: {visual_dir}")
    visual_embeddings = load_visual_embeddings(visual_dir)

    print(f"\nLoading action descriptors from: {action_dir}")
    action_descriptors = load_action_descriptors(action_dir)

    all_episode_ids = sorted(set(list(visual_embeddings.keys()) | set(list(action_descriptors.keys()))))

    result = select_episodes_v5_adaptive(
        all_episode_ids=all_episode_ids,
        visual_embeddings=visual_embeddings,
        action_descriptors=action_descriptors,
        num_select=args.num_selected,
        num_clusters=args.num_clusters,
        coverage_weight=args.coverage_weight,
        cluster_visual_weight=args.cluster_visual_weight,
        cluster_action_weight=args.cluster_action_weight,
        episode_visual_weight=args.visual_weight,
        episode_action_weight=args.action_weight,
        seed=args.seed,
        b0_size=args.b0_size,
    )

    subset_file = output_dir / "subsets" / f"our_v5_{args.num_selected}_seed{args.seed}.json"
    save_selected_episodes(
        selected_episode_ids=result["selected_episode_indices"],
        output_path=subset_file,
        metadata={
            "selection_method": "our_v5_adaptive_latent_coverage",
            "num_clusters": args.num_clusters,
            "coverage_weight": args.coverage_weight,
            "cluster_visual_weight": args.cluster_visual_weight,
            "cluster_action_weight": args.cluster_action_weight,
            "episode_visual_weight": args.visual_weight,
            "episode_action_weight": args.action_weight,
            "b0_size": args.b0_size,
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
    compute_diagnostic(result, visual_embeddings, action_descriptors, output_dir)


if __name__ == "__main__":
    main()