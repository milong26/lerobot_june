#!/usr/bin/env python
"""
Ours-v5 episode selector for Robomme pipeline.

Adapted from personal/work2/our_v5/select_our_v5.py

For Robomme:
- Configuration-space coverage uses initial-configuration features (not rand_vec)
- Visual embeddings use real phi_global + phi_wrist from SmolVLM
- Action descriptors use the same V5 definition as original Ours-v5
- Region priority, coverage gap, visual uncertainty, action uncertainty,
  and region-internal visual+action novelty selection follow the original logic

Usage:
    python select_ours_v5.py \
        --dataset-root /path/to/dataset \
        --task-name MoveCube_easy \
        --num-episodes 28 \
        --seed 42 \
        --output-dir /path/to/output
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Allow direct execution (python script.py) to find sibling modules
_selectors_dir = Path(__file__).resolve().parent
if str(_selectors_dir) not in sys.path:
    sys.path.insert(0, str(_selectors_dir))

from configuration_features import (
    load_configuration_features,
    TASK_FEATURE_DIMS,
)
from robomme_embeddings import (
    load_cached_visual_embeddings,
    load_cached_action_descriptors,
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
# Configuration-based region building (replaces rand_vec regions)
# ---------------------------------------------------------------------------

def build_configuration_regions(
    config_features: Dict[int, np.ndarray],
    region_ratio: float = 0.1,
    min_regions: int = 16,
    max_regions: int = 128,
    seed: int = 42,
) -> Tuple[Dict[int, int], int]:
    """
    Build configuration-space regions by partitioning the configuration feature space.

    Replaces the old build_rand_vec_regions() but keeps the same interface.

    Args:
        config_features: dict mapping episode_index -> configuration feature array
        region_ratio: ratio of episodes to regions
        min_regions: minimum number of regions
        max_regions: maximum number of regions
        seed: random seed for KMeans

    Returns:
        (episode_to_region, num_regions)
    """
    print(f"\n{'='*60}")
    print(f"Building configuration regions...")
    print(f"{'='*60}")

    valid_episodes = sorted(config_features.keys())
    n_episodes = len(valid_episodes)

    if n_episodes == 0:
        raise ValueError("No valid episodes with configuration features")

    num_regions = max(min_regions, min(max_regions, int(n_episodes * region_ratio)))
    num_regions = min(num_regions, n_episodes)

    print(f"  Episodes: {n_episodes}")
    print(f"  Computed regions: {num_regions}")

    config_list = []
    for ep_idx in valid_episodes:
        config_list.append(config_features[ep_idx])

    config_array = np.array(config_list)

    # L2 normalize configuration features for clustering
    norms = np.linalg.norm(config_array, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    config_normalized = config_array / norms

    kmeans = KMeans(n_clusters=num_regions, random_state=seed, n_init=10)
    region_ids = kmeans.fit_predict(config_normalized)

    episode_to_region = {}
    for i, ep_idx in enumerate(valid_episodes):
        episode_to_region[ep_idx] = int(region_ids[i])

    region_counts = {}
    for region_id in region_ids:
        region_id = int(region_id)
        region_counts[region_id] = region_counts.get(region_id, 0) + 1

    print(f"  Regions created: {num_regions}")
    print(f"  Region size - min: {min(region_counts.values())}, max: {max(region_counts.values())}")

    return episode_to_region, num_regions


# ---------------------------------------------------------------------------
# Initial B0 selection
# ---------------------------------------------------------------------------

def select_initial_b0(
    episode_to_region: Dict[int, int],
    episode_data: Dict[int, Dict],
    num_regions: int,
    b0_region_ratio: float = 0.2,
    seed: int = 42,
) -> List[int]:
    """Select initial B0 episodes using configuration region coverage."""
    rng = np.random.RandomState(seed)
    b0_size = max(1, int(num_regions * b0_region_ratio))

    region_to_episodes = {}
    for ep_idx, region_id in episode_to_region.items():
        if region_id not in region_to_episodes:
            region_to_episodes[region_id] = []
        region_to_episodes[region_id].append(ep_idx)

    selected = []
    region_ids = sorted(region_to_episodes.keys())

    for region_id in region_ids:
        if len(selected) >= b0_size:
            break
        region_episodes = region_to_episodes[region_id]
        chosen = rng.choice(region_episodes, size=1, replace=False).tolist()[0]
        selected.append(chosen)

    if len(selected) < b0_size:
        remaining_episodes = [ep for ep in episode_data.keys() if ep not in selected]
        n_needed = b0_size - len(selected)
        additional = rng.choice(remaining_episodes, size=min(n_needed, len(remaining_episodes)), replace=False).tolist()
        selected.extend(additional)

    selected.sort()
    print(f"\n[Step 2] Initial B0: {len(selected)} episodes from {num_regions} regions")
    return selected


# ---------------------------------------------------------------------------
# Region priority computation
# ---------------------------------------------------------------------------

def compute_region_priority(
    region_id: int,
    episode_to_region: Dict[int, int],
    selected_ids: List[int],
    episode_data: Dict[int, Dict],
    coverage_weight: float = 0.5,
    visual_weight: float = 0.3,
    action_weight: float = 0.2,
) -> Dict:
    """Compute priority for a configuration region."""
    region_episodes = [ep for ep, rid in episode_to_region.items() if rid == region_id]
    total_in_region = len(region_episodes)

    if total_in_region == 0:
        return {"priority": 0.0, "coverage_gap": 0.0, "visual_uncertainty": 0.0, "action_uncertainty": 0.0}

    selected_in_region = [ep for ep in region_episodes if ep in selected_ids]
    n_selected_in_region = len(selected_in_region)
    coverage_gap = 1.0 - (n_selected_in_region / total_in_region)

    # Visual uncertainty: pairwise distance of visual embeddings in region
    visual_uncertainty = 0.0
    region_visual_embs = []
    for ep in region_episodes:
        if ep in episode_data and "visual_embedding" in episode_data[ep]:
            region_visual_embs.append(episode_data[ep]["visual_embedding"])

    if len(region_visual_embs) >= 2:
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
        visual_uncertainty = np.linalg.norm(region_visual_embs[0])

    # Action uncertainty: pairwise distance of action descriptors in region
    action_uncertainty = 0.0
    region_action_embs = []
    for ep in region_episodes:
        if ep in episode_data and "action_descriptor" in episode_data[ep]:
            region_action_embs.append(episode_data[ep]["action_descriptor"])

    if len(region_action_embs) >= 2:
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

    # Normalize
    visual_uncertainty_norm = min(visual_uncertainty / 10.0, 1.0)
    action_uncertainty_norm = min(action_uncertainty / 10.0, 1.0)

    priority = (coverage_weight * coverage_gap +
                visual_weight * visual_uncertainty_norm +
                action_weight * action_uncertainty_norm)

    return {
        "priority": priority,
        "coverage_gap": coverage_gap,
        "visual_uncertainty": visual_uncertainty_norm,
        "action_uncertainty": action_uncertainty_norm,
    }


# ---------------------------------------------------------------------------
# Episode selection within a region
# ---------------------------------------------------------------------------

def select_best_episode_from_region(
    region_id: int,
    episode_to_region: Dict[int, int],
    selected_ids: List[int],
    episode_data: Dict[int, Dict],
    visual_weight: float = 0.5,
    action_weight: float = 0.5,
) -> Optional[Tuple[int, Dict]]:
    """Select the best episode from a specific region based on visual+action novelty."""
    candidates = [ep for ep, rid in episode_to_region.items() if rid == region_id and ep not in selected_ids]

    if not candidates:
        return None

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

    candidate_scores = []
    for candidate_idx in candidates:
        if candidate_idx not in episode_data:
            continue

        if "visual_embedding" in episode_data[candidate_idx]:
            if len(selected_visual) > 0:
                distances = np.linalg.norm(selected_visual - episode_data[candidate_idx]["visual_embedding"], axis=1)
                visual_score = float(np.min(distances))
            else:
                visual_score = 1.0
        else:
            visual_score = 0.0

        if "action_descriptor" in episode_data[candidate_idx]:
            if len(selected_action) > 0:
                distances = np.linalg.norm(selected_action - episode_data[candidate_idx]["action_descriptor"], axis=1)
                action_score = float(np.min(distances))
            else:
                action_score = 1.0
        else:
            action_score = 0.0

        candidate_scores.append((candidate_idx, visual_score, action_score))

    if not candidate_scores:
        return None

    visual_scores = [s[1] for s in candidate_scores]
    action_scores = [s[2] for s in candidate_scores]

    visual_min, visual_max = min(visual_scores), max(visual_scores)
    action_min, action_max = min(action_scores), max(action_scores)

    visual_range = visual_max - visual_min if visual_max > visual_min else 1.0
    action_range = action_max - action_min if action_max > action_min else 1.0

    best_candidate = None
    best_scores = None
    best_score = -1.0

    for candidate_idx, vs, acs in candidate_scores:
        norm_vs = (vs - visual_min) / visual_range
        norm_acs = (acs - action_min) / action_range
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


# ---------------------------------------------------------------------------
# Main selection loop
# ---------------------------------------------------------------------------

def select_episodes_v5(
    all_episode_ids: List[int],
    visual_embeddings: Dict[int, np.ndarray],
    action_descriptors: Dict[int, np.ndarray],
    config_features: Dict[int, np.ndarray],
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
    """Execute configuration-aware adaptive coverage selection."""
    episode_data = {}
    for ep_idx in all_episode_ids:
        if ep_idx in config_features and ep_idx in visual_embeddings and ep_idx in action_descriptors:
            episode_data[ep_idx] = {
                "config_feature": config_features[ep_idx],
                "visual_embedding": visual_embeddings[ep_idx],
                "action_descriptor": action_descriptors[ep_idx],
            }

    valid_ids = sorted(episode_data.keys())

    print(f"\n{'='*60}")
    print(f"V5 Configuration-Aware Adaptive Coverage Episode Selection")
    print(f"{'='*60}")
    print(f"Total episodes: {len(all_episode_ids)}")
    print(f"Valid episodes: {len(valid_ids)}")
    print(f"Target selection: {num_select}")
    print(f"Seed: {seed}")

    episode_to_region, num_regions = build_configuration_regions(
        config_features, region_ratio, min_regions, max_regions, seed
    )

    b0_episodes = select_initial_b0(
        episode_to_region, episode_data, num_regions, b0_region_ratio, seed
    )
    selected_ids = list(b0_episodes)

    print(f"\n[Step 3] Starting adaptive region-based selection...")
    selection_log = []
    step_num = 0

    while len(selected_ids) < num_select:
        step_num += 1
        step_start = time.time()

        region_priorities = {}
        for region_id in range(num_regions):
            priority_info = compute_region_priority(
                region_id, episode_to_region, selected_ids, episode_data,
                coverage_weight, region_visual_weight, region_action_weight
            )
            region_priorities[region_id] = priority_info

        best_region_id = max(region_priorities.keys(), key=lambda r: region_priorities[r]["priority"])
        best_region_priority = region_priorities[best_region_id]

        result = select_best_episode_from_region(
            best_region_id, episode_to_region, selected_ids, episode_data,
            episode_visual_weight, episode_action_weight
        )

        if result is None:
            region_priorities[best_region_id]["priority"] = -1.0
            max_priority = max(r["priority"] for r in region_priorities.values())
            if max_priority < 0:
                print(f"  All regions exhausted. Selected {len(selected_ids)} episodes.")
                break
            continue

        best_candidate, scores = result
        selected_ids.append(best_candidate)

        step_time = time.time() - step_start
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
                  f"priority={best_region_priority['priority']:.4f}, score={scores['joint_score']:.4f}")

    selected_ids.sort()

    print(f"\n{'='*60}")
    print(f"V5 Selection Complete: {len(selected_ids)} episodes selected")
    print(f"{'='*60}")

    return {
        "selected_episode_indices": selected_ids,
        "b0_episodes": b0_episodes,
        "episode_to_region": episode_to_region,
        "selection_log": selection_log,
        "num_selected": len(selected_ids),
        "num_valid": len(valid_ids),
        "num_total": len(all_episode_ids),
        "num_regions": num_regions,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def select_ours_v5_episodes(
    num_episodes: int,
    seed: int,
    dataset_root: str,
    task_name: str,
    output_dir: Optional[str] = None,
) -> List[int]:
    """Main entry point for Ours-v5 selection on Robomme datasets."""
    print(f"Loading LeRobotDataset from {dataset_root}")
    dataset = LeRobotDataset(repo_id="1/2", root=dataset_root)

    total_episodes = dataset.num_episodes
    print(f"Total episodes: {total_episodes}")

    all_episode_ids = list(range(total_episodes))

    # Load configuration features
    print(f"\nLoading configuration features...")
    config_features = load_configuration_features(dataset_root, task_name)

    # Load visual embeddings from cache (auto-extract if missing)
    print(f"\nLoading visual embeddings from cache...")
    visual_embeddings = load_cached_visual_embeddings(dataset_root, task_name)
    if visual_embeddings is None:
        print(f"Visual embeddings not found in cache. Extracting now...")
        print(f"This requires loading the SmolVLM model and may take a while.")
        import torch
        from robomme_embeddings import ensure_visual_embeddings, load_smolvla_feature_extractor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor = load_smolvla_feature_extractor(device)
        visual_embeddings = ensure_visual_embeddings(
            dataset_root, task_name, model, processor, device
        )
        print(f"Extracted {len(visual_embeddings)} visual embeddings")
    else:
        print(f"Loaded {len(visual_embeddings)} visual embeddings from cache")

    # Load action descriptors from cache (auto-extract if missing)
    print(f"\nLoading action descriptors from cache...")
    action_descriptors = load_cached_action_descriptors(dataset_root, task_name)
    if action_descriptors is None:
        print(f"Action descriptors not found in cache. Extracting now...")
        from robomme_embeddings import ensure_action_descriptors
        action_descriptors = ensure_action_descriptors(dataset_root, task_name)
        print(f"Extracted {len(action_descriptors)} action descriptors")
    else:
        print(f"Loaded {len(action_descriptors)} action descriptors from cache")

    valid_ids = [
        ep_idx for ep_idx in all_episode_ids
        if ep_idx in config_features and ep_idx in visual_embeddings and ep_idx in action_descriptors
    ]
    print(f"Episodes with all data: {len(valid_ids)}")

    if len(valid_ids) == 0:
        raise RuntimeError("No episodes with all required data (config+visual+action)")

    result = select_episodes_v5(
        all_episode_ids=all_episode_ids,
        visual_embeddings=visual_embeddings,
        action_descriptors=action_descriptors,
        config_features=config_features,
        num_select=num_episodes,
        seed=seed,
    )

    selected_ids = result["selected_episode_indices"]

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_file = output_path / f"ours_v5_{num_episodes}_seed{seed}.json"
        subset_data = {
            "method": "ours_v5",
            "selection_method": "ours_v5_configuration_adaptive_coverage",
            "num_episodes": len(selected_ids),
            "selected_episode_indices": selected_ids,
            "parameters": {
                "seed": seed,
                "num_regions": result["num_regions"],
                "num_valid": result["num_valid"],
            }
        }
        with open(subset_file, "w") as f:
            json.dump(subset_data, f, indent=2, cls=NumpyEncoder)
        print(f"Saved {len(selected_ids)} ours_v5 episodes (seed={seed}) to {subset_file}")

    return selected_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V5 Configuration-Aware Adaptive Coverage Episode Selection")
    parser.add_argument("--num-episodes", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--task-name", type=str, required=True,
                       choices=["MoveCube_easy", "PatternLock_medium", "RouteStick_hard"])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--region-ratio", type=float, default=0.1)
    parser.add_argument("--min-regions", type=int, default=16)
    parser.add_argument("--max-regions", type=int, default=128)
    parser.add_argument("--b0-region-ratio", type=float, default=0.2)
    parser.add_argument("--coverage-weight", type=float, default=0.5)
    parser.add_argument("--region-visual-weight", type=float, default=0.3)
    parser.add_argument("--region-action-weight", type=float, default=0.2)
    parser.add_argument("--visual-weight", type=float, default=0.5)
    parser.add_argument("--action-weight", type=float, default=0.5)
    args = parser.parse_args()

    select_ours_v5_episodes(
        num_episodes=args.num_episodes,
        seed=args.seed,
        dataset_root=args.dataset_root,
        task_name=args.task_name,
        output_dir=args.output_dir,
    )