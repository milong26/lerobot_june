#!/usr/bin/env python
"""
Ours-v5 episode selector for Robomme pipeline.
Adapted from personal/work2/our_v5/select_our_v5.py

For Robomme, we use rand_vec from episode_initial_states.json and
state/action features extracted from the LeRobotDataset.

Usage:
    python select_ours_v5.py \
        --dataset-root /path/to/dataset \
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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_rand_vecs(dataset_root: Path, all_episode_ids: List[int]) -> Dict[int, np.ndarray]:
    """Load rand_vec for each episode from episode_initial_states.json."""
    rand_vecs = {}
    metadata_file = dataset_root / "episode_initial_states.json"
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
            print(f"Loaded {len(rand_vecs)} rand_vecs")
        except Exception as e:
            print(f"Failed to load rand_vecs: {e}")
    else:
        print(f"WARNING: episode_initial_states.json not found at {metadata_file}")
    return rand_vecs


def extract_episode_features(dataset, episode_idx):
    """Extract state and action features for an episode."""
    ep_data = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_idx)
    if len(ep_data) == 0:
        return None, None

    states = np.array([frame["observation.state"] for frame in ep_data], dtype=np.float32)
    actions = np.array([frame["action"] for frame in ep_data], dtype=np.float32)

    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0)
    action_mean = np.mean(actions, axis=0)
    action_std = np.std(actions, axis=0)

    visual_feature = np.concatenate([state_mean, state_std])
    action_descriptor = np.concatenate([action_mean, action_std])
    return visual_feature, action_descriptor


def build_rand_vec_regions(episode_data, region_ratio=0.1, min_regions=16, max_regions=128, seed=42):
    """Build rand_vec regions by partitioning the rand_vec space."""
    print(f"\n{'='*60}")
    print(f"Building rand_vec regions...")
    print(f"{'='*60}")

    valid_episodes = sorted(episode_data.keys())
    n_episodes = len(valid_episodes)

    if n_episodes == 0:
        raise ValueError("No valid episodes with rand_vec data")

    num_regions = max(min_regions, min(max_regions, int(n_episodes * region_ratio)))
    num_regions = min(num_regions, n_episodes)

    print(f"  Episodes: {n_episodes}")
    print(f"  Computed regions: {num_regions}")

    rand_vec_list = []
    for ep_idx in valid_episodes:
        rand_vec_list.append(episode_data[ep_idx]["rand_vec"])

    rand_vecs = np.array(rand_vec_list)
    norms = np.linalg.norm(rand_vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    rand_vecs_normalized = rand_vecs / norms

    kmeans = KMeans(n_clusters=num_regions, random_state=seed, n_init=10)
    region_ids = kmeans.fit_predict(rand_vecs_normalized)

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


def select_initial_b0(episode_to_region, episode_data, num_regions, b0_region_ratio=0.2, seed=42):
    """Select initial B0 episodes using rand_vec region coverage."""
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


def compute_region_priority(region_id, episode_to_region, selected_ids, episode_data,
                           coverage_weight=0.5, visual_weight=0.3, action_weight=0.2):
    """Compute priority for a rand_vec region."""
    region_episodes = [ep for ep, rid in episode_to_region.items() if rid == region_id]
    total_in_region = len(region_episodes)

    if total_in_region == 0:
        return {"priority": 0.0, "coverage_gap": 0.0, "visual_uncertainty": 0.0, "action_uncertainty": 0.0}

    selected_in_region = [ep for ep in region_episodes if ep in selected_ids]
    n_selected_in_region = len(selected_in_region)
    coverage_gap = 1.0 - (n_selected_in_region / total_in_region)

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


def select_best_episode_from_region(region_id, episode_to_region, selected_ids, episode_data,
                                   visual_weight=0.5, action_weight=0.5):
    """Select the best episode from a specific region."""
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


def select_episodes_v5(all_episode_ids, visual_embeddings, action_descriptors, rand_vecs,
                      num_select, region_ratio=0.1, min_regions=16, max_regions=128,
                      b0_region_ratio=0.2, coverage_weight=0.5, region_visual_weight=0.3,
                      region_action_weight=0.2, episode_visual_weight=0.5, episode_action_weight=0.5,
                      seed=42):
    """Execute rand_vec-aware adaptive coverage selection."""
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
    print(f"Valid episodes: {len(valid_ids)}")
    print(f"Target selection: {num_select}")
    print(f"Seed: {seed}")

    episode_to_region, num_regions = build_rand_vec_regions(
        episode_data, region_ratio, min_regions, max_regions, seed
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


def select_ours_v5_episodes(num_episodes, seed, dataset_root, output_dir=None):
    """Main entry point for Ours-v5 selection on Robomme datasets."""
    print(f"Loading LeRobotDataset from {dataset_root}")
    dataset = LeRobotDataset(repo_id="1/2", root=dataset_root)

    total_episodes = dataset.num_episodes
    print(f"Total episodes: {total_episodes}")

    all_episode_ids = list(range(total_episodes))

    print(f"\nLoading rand_vecs from: {dataset_root}")
    rand_vecs = load_rand_vecs(Path(dataset_root), all_episode_ids)

    print(f"\nExtracting episode features...")
    visual_embeddings = {}
    action_descriptors = {}
    for ep_idx in all_episode_ids:
        vis_feat, act_feat = extract_episode_features(dataset, ep_idx)
        if vis_feat is not None:
            visual_embeddings[ep_idx] = vis_feat
        if act_feat is not None:
            action_descriptors[ep_idx] = act_feat

    valid_ids = [
        ep_idx for ep_idx in all_episode_ids
        if ep_idx in rand_vecs and ep_idx in visual_embeddings and ep_idx in action_descriptors
    ]
    print(f"Episodes with all data: {len(valid_ids)}")

    if len(valid_ids) == 0:
        raise RuntimeError("No episodes with all required data (rand_vec+visual+action)")

    episode_data = {}
    for ep_idx in valid_ids:
        episode_data[ep_idx] = {
            "rand_vec": rand_vecs[ep_idx],
            "visual_embedding": visual_embeddings[ep_idx],
            "action_descriptor": action_descriptors[ep_idx],
        }

    result = select_episodes_v5(
        all_episode_ids=all_episode_ids,
        visual_embeddings=visual_embeddings,
        action_descriptors=action_descriptors,
        rand_vecs=rand_vecs,
        num_select=num_episodes,
        seed=seed,
    )

    selected_ids = result["selected_episode_indices"]

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_file = output_path / f"our_v5_{num_episodes}_seed{seed}.json"
        subset_data = {
            "method": "our_v5",
            "selection_method": "our_v5_rand_vec_adaptive_coverage",
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
        print(f"Saved {len(selected_ids)} our_v5 episodes (seed={seed}) to {subset_file}")

    return selected_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V5 Rand Vec Aware Adaptive Coverage Episode Selection")
    parser.add_argument("--num-episodes", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-root", type=str, required=True)
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
        output_dir=args.output_dir,
    )