#!/usr/bin/env python
"""
Our V5 Ablation Episode Selection - Unified Entry Point

Supports three ablation modes:
  - full: complete our_v5 flow (visual + action + region hierarchy)
  - wo_action: remove action descriptor, only use visual representation
  - wo_region: remove configuration region hierarchy, use global embedding selection

Usage:
    python select_our_v5_ablation.py \
        --visual-embedding-dir /path/to/visual/embeddings \
        --action-descriptor-dir /path/to/action/descriptors \
        --dataset-dir /path/to/dataset \
        --output-dir /path/to/output \
        --num-selected 112 \
        --seed 42 \
        --ablation-mode full
"""

import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WORK2_ROOT = Path(__file__).resolve().parent.parent
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))


# ============================================================================
# Shared utility functions (extracted from select_our_v5.py)
# ============================================================================

def load_visual_embeddings(embedding_path: Path) -> Dict[int, np.ndarray]:
    """Load existing global+wrist visual embeddings from cache directory."""
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
    """Load action descriptors from cache directory."""
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
    """Load rand_vec for each episode from episode_initial_states.json."""
    rand_vecs = {}
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

    print("  rand_vec not found in episode_initial_states.json, checking embedding files...")
    if len(rand_vecs) == 0:
        print(f"WARNING: No rand_vec found for any episodes!")
        print(f"  Please ensure episode_initial_states.json exists in dataset directory")
        print(f"  and contains 'rand_vec' field for each episode.")
    return rand_vecs


def compute_visual_coverage_score(candidate_embedding: np.ndarray, selected_embeddings: np.ndarray) -> float:
    """Compute visual coverage score = min distance to selected episodes."""
    if len(selected_embeddings) == 0:
        return 1.0
    distances = np.linalg.norm(selected_embeddings - candidate_embedding, axis=1)
    return float(np.min(distances))


def compute_action_diversity_score(candidate_descriptor: np.ndarray, selected_descriptors: np.ndarray) -> float:
    """Compute action diversity score = min distance to selected episodes."""
    if len(selected_descriptors) == 0:
        return 1.0
    distances = np.linalg.norm(selected_descriptors - candidate_descriptor, axis=1)
    return float(np.min(distances))


def save_selected_episodes(selected_episode_ids: List[int], output_path: Path, metadata: Dict = None, method_name: str = "our_v5") -> None:
    """Save selected episode list in duibi experiment format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset_data = {
        "method": method_name,
        "selection_method": f"our_v5_ablation_{method_name}",
        "num_episodes": len(selected_episode_ids),
        "selected_episode_indices": selected_episode_ids,
    }
    if metadata:
        subset_data["parameters"] = metadata
    with open(output_path, "w") as f:
        json.dump(subset_data, f, indent=2, cls=NumpyEncoder)
    print(f"Selected episodes saved to: {output_path}")


# ============================================================================
# Full mode: original our_v5 selection logic (visual + action + region)
# ============================================================================

def build_rand_vec_regions(episode_data, region_ratio=0.1, min_regions=16, max_regions=128, seed=42):
    """Build rand_vec regions by partitioning the rand_vec space using KMeans."""
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
    print(f"  Region ratio: {region_ratio}")
    print(f"  Computed regions: {num_regions} (min={min_regions}, max={max_regions})")
    rand_vec_list = [episode_data[ep_idx]["rand_vec"] for ep_idx in valid_episodes]
    rand_vecs = np.array(rand_vec_list)
    norms = np.linalg.norm(rand_vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    rand_vecs_normalized = rand_vecs / norms
    kmeans = KMeans(n_clusters=num_regions, random_state=seed, n_init=10)
    region_ids = kmeans.fit_predict(rand_vecs_normalized)
    episode_to_region = {ep_idx: int(region_ids[i]) for i, ep_idx in enumerate(valid_episodes)}
    region_counts = {}
    for rid in region_ids:
        rid = int(rid)
        region_counts[rid] = region_counts.get(rid, 0) + 1
    print(f"  Regions created: {num_regions}")
    print(f"  Region size - min: {min(region_counts.values())}, max: {max(region_counts.values())}, "
          f"mean: {np.mean(list(region_counts.values())):.1f}")
    return episode_to_region, num_regions


def select_initial_b0(episode_to_region, episode_data, num_regions, b0_region_ratio=0.2, seed=42):
    """Select initial B0 episodes using rand_vec region coverage strategy."""
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
    print(f"\n[Step 2] Initial B0 selection: {len(selected)} episodes from {num_regions} regions")
    print(f"  B0 size: {b0_size} (region_ratio={b0_region_ratio})")
    print(f"  B0 episodes: {selected}")
    return selected


def compute_region_priority(region_id, episode_to_region, selected_ids, episode_data,
                            coverage_weight=0.5, visual_weight=0.3, action_weight=0.2):
    """Compute priority for a rand_vec region based on coverage gap, visual and action uncertainty."""
    region_episodes = [ep for ep, rid in episode_to_region.items() if rid == region_id]
    total_in_region = len(region_episodes)
    if total_in_region == 0:
        return {"priority": 0.0, "coverage_gap": 0.0, "visual_uncertainty": 0.0, "action_uncertainty": 0.0}
    selected_in_region = [ep for ep in region_episodes if ep in selected_ids]
    n_selected_in_region = len(selected_in_region)
    coverage_gap = 1.0 - (n_selected_in_region / total_in_region)
    visual_uncertainty = 0.0
    region_visual_embs = [episode_data[ep]["visual_embedding"] for ep in region_episodes
                          if ep in episode_data and "visual_embedding" in episode_data[ep]]
    if len(region_visual_embs) >= 2:
        visual_arr = np.array(region_visual_embs)
        n = len(visual_arr)
        total_dist = sum(np.linalg.norm(visual_arr[i] - visual_arr[j])
                         for i in range(n) for j in range(i + 1, n))
        visual_uncertainty = total_dist / (n * (n - 1) / 2)
    elif len(region_visual_embs) == 1:
        visual_uncertainty = np.linalg.norm(region_visual_embs[0])
    action_uncertainty = 0.0
    region_action_embs = [episode_data[ep]["action_descriptor"] for ep in region_episodes
                          if ep in episode_data and "action_descriptor" in episode_data[ep]]
    if len(region_action_embs) >= 2:
        action_arr = np.array(region_action_embs)
        n = len(action_arr)
        total_dist = sum(np.linalg.norm(action_arr[i] - action_arr[j])
                         for i in range(n) for j in range(i + 1, n))
        action_uncertainty = total_dist / (n * (n - 1) / 2)
    elif len(region_action_embs) == 1:
        action_uncertainty = np.linalg.norm(region_action_embs[0])
    visual_uncertainty_norm = min(visual_uncertainty / 10.0, 1.0)
    action_uncertainty_norm = min(action_uncertainty / 10.0, 1.0)
    priority = coverage_weight * coverage_gap + visual_weight * visual_uncertainty_norm + action_weight * action_uncertainty_norm
    return {"priority": priority, "coverage_gap": coverage_gap,
            "visual_uncertainty": visual_uncertainty_norm, "action_uncertainty": action_uncertainty_norm}


def select_best_from_region(region_id, episode_to_region, selected_ids, episode_data,
                            visual_weight=0.5, action_weight=0.5):
    """Select the best episode from a specific region based on visual coverage and action diversity."""
    candidates = [ep for ep, rid in episode_to_region.items() if rid == region_id and ep not in selected_ids]
    if not candidates:
        return None
    selected_visual = np.array([episode_data[ep]["visual_embedding"] for ep in selected_ids
                                if ep in episode_data and "visual_embedding" in episode_data[ep]])
    selected_action = np.array([episode_data[ep]["action_descriptor"] for ep in selected_ids
                                if ep in episode_data and "action_descriptor" in episode_data[ep]])
    candidate_scores = []
    for candidate_idx in candidates:
        if candidate_idx not in episode_data:
            continue
        if "visual_embedding" in episode_data[candidate_idx]:
            visual_score = compute_visual_coverage_score(episode_data[candidate_idx]["visual_embedding"], selected_visual)
        else:
            visual_score = 0.0
        if "action_descriptor" in episode_data[candidate_idx]:
            action_score = compute_action_diversity_score(episode_data[candidate_idx]["action_descriptor"], selected_action)
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
            best_scores = {"visual_score": float(vs), "action_score": float(acs),
                           "normalized_visual_score": float(norm_vs), "normalized_action_score": float(norm_acs),
                           "joint_score": float(total_score)}
    return (best_candidate, best_scores) if best_candidate else None


def select_full_mode(all_episode_ids, visual_embeddings, action_descriptors, rand_vecs,
                     num_select, region_ratio=0.1, min_regions=16, max_regions=128,
                     b0_region_ratio=0.2, coverage_weight=0.5, region_visual_weight=0.3,
                     region_action_weight=0.2, episode_visual_weight=0.5, episode_action_weight=0.5, seed=42):
    """Full mode: complete our_v5 flow with visual + action + region hierarchy."""
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
    print(f"V5 Full Mode: Rand Vec Aware Adaptive Coverage Episode Selection")
    print(f"{'='*60}")
    print(f"Total episodes: {len(all_episode_ids)}")
    print(f"Valid episodes (with rand_vec+visual+action): {len(valid_ids)}")
    print(f"Target selection: {num_select}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")
    episode_to_region, num_regions = build_rand_vec_regions(episode_data, region_ratio, min_regions, max_regions, seed)
    b0_episodes = select_initial_b0(episode_to_region, episode_data, num_regions, b0_region_ratio, seed)
    selected_ids = list(b0_episodes)
    print(f"\n[Step 3] Starting adaptive region-based selection...")
    selection_log = []
    step_num = 0
    while len(selected_ids) < num_select:
        step_num += 1
        step_start = time.time()
        region_priorities = {}
        for region_id in range(num_regions):
            region_priorities[region_id] = compute_region_priority(
                region_id, episode_to_region, selected_ids, episode_data,
                coverage_weight, region_visual_weight, region_action_weight)
        best_region_id = max(region_priorities.keys(), key=lambda r: region_priorities[r]["priority"])
        best_region_priority = region_priorities[best_region_id]
        result = select_best_from_region(best_region_id, episode_to_region, selected_ids, episode_data,
                                         episode_visual_weight, episode_action_weight)
        if result is None:
            region_priorities[best_region_id]["priority"] = -1.0
            print(f"  Step {step_num}: Region {best_region_id} exhausted, skipping...")
            max_priority = max(r["priority"] for r in region_priorities.values())
            if max_priority < 0:
                print(f"  All regions exhausted. Selected {len(selected_ids)} episodes.")
                break
            continue
        best_candidate, scores = result
        selected_ids.append(best_candidate)
        step_time = time.time() - step_start
        selection_log.append({"step": step_num, "n_selected": len(selected_ids),
                              "selected_region_id": best_region_id,
                              "region_priority": best_region_priority["priority"],
                              "region_coverage_gap": best_region_priority["coverage_gap"],
                              "region_visual_uncertainty": best_region_priority["visual_uncertainty"],
                              "region_action_uncertainty": best_region_priority["action_uncertainty"],
                              "episode_index": best_candidate,
                              "visual_score": scores["visual_score"], "action_score": scores["action_score"],
                              "normalized_visual_score": scores["normalized_visual_score"],
                              "normalized_action_score": scores["normalized_action_score"],
                              "joint_score": scores["joint_score"], "time_seconds": step_time})
        if step_num % 10 == 0 or step_num == 1:
            print(f"  Step {step_num}: region={best_region_id}, ep={best_candidate}, "
                  f"region_priority={best_region_priority['priority']:.4f}, "
                  f"joint_score={scores['joint_score']:.4f}, time={step_time:.2f}s")
    selected_ids.sort()
    print(f"\n{'='*60}")
    print(f"V5 Full Mode Selection Complete!")
    print(f"{'='*60}")
    print(f"Selected {len(selected_ids)} episodes")
    print(f"Total steps: {step_num}")
    region_selection_counts = {}
    for region_id in range(num_regions):
        region_episodes = [ep for ep, rid in episode_to_region.items() if rid == region_id]
        selected_in_region = [ep for ep in region_episodes if ep in selected_ids]
        region_selection_counts[region_id] = {
            "total": len(region_episodes), "selected": len(selected_in_region),
            "selection_ratio": len(selected_in_region) / len(region_episodes) if len(region_episodes) > 0 else 0.0}
    return {"selected_episode_indices": selected_ids, "b0_episodes": b0_episodes,
            "episode_to_region": episode_to_region, "region_selection_counts": region_selection_counts,
            "selection_log": selection_log, "total_steps": step_num, "num_selected": len(selected_ids),
            "num_valid": len(valid_ids), "num_total": len(all_episode_ids), "num_regions": num_regions,
            "region_ratio": region_ratio, "min_regions": min_regions, "max_regions": max_regions,
            "b0_region_ratio": b0_region_ratio, "coverage_weight": coverage_weight,
            "region_visual_weight": region_visual_weight, "region_action_weight": region_action_weight,
            "episode_visual_weight": episode_visual_weight, "episode_action_weight": episode_action_weight,
            "seed": seed, "ablation_mode": "full", "visual_coverage_enabled": True,
            "action_diversity_enabled": True, "region_decision_enabled": True}


# ============================================================================
# wo_action mode: no action descriptor, only visual embedding
# ============================================================================

def select_wo_action_mode(all_episode_ids, visual_embeddings, rand_vecs,
                          num_select, region_ratio=0.1, min_regions=16, max_regions=128,
                          b0_region_ratio=0.2, coverage_weight=0.5, region_visual_weight=0.5,
                          episode_visual_weight=1.0, seed=42):
    """wo_action mode: remove action descriptor, only use visual representation."""
    episode_data = {}
    for ep_idx in all_episode_ids:
        if ep_idx in rand_vecs and ep_idx in visual_embeddings:
            episode_data[ep_idx] = {"rand_vec": rand_vecs[ep_idx], "visual_embedding": visual_embeddings[ep_idx]}
    valid_ids = sorted(episode_data.keys())
    print(f"\n{'='*60}")
    print(f"V5 wo_action Mode: Visual-Only Adaptive Coverage Selection")
    print(f"{'='*60}")
    print(f"Total episodes: {len(all_episode_ids)}")
    print(f"Valid episodes (with rand_vec+visual, NO action): {len(valid_ids)}")
    print(f"Target selection: {num_select}")
    print(f"Seed: {seed}")
    print(f"Action descriptor: DISABLED")
    print(f"{'='*60}")
    episode_to_region, num_regions = build_rand_vec_regions(episode_data, region_ratio, min_regions, max_regions, seed)
    b0_episodes = select_initial_b0(episode_to_region, episode_data, num_regions, b0_region_ratio, seed)
    selected_ids = list(b0_episodes)
    print(f"\n[Step 3] Starting visual-only adaptive region-based selection...")
    selection_log = []
    step_num = 0
    while len(selected_ids) < num_select:
        step_num += 1
        step_start = time.time()
        region_priorities = {}
        for region_id in range(num_regions):
            region_episodes = [ep for ep, rid in episode_to_region.items() if rid == region_id]
            total_in_region = len(region_episodes)
            if total_in_region == 0:
                region_priorities[region_id] = {"priority": 0.0, "coverage_gap": 0.0, "visual_uncertainty": 0.0}
                continue
            selected_in_region = [ep for ep in region_episodes if ep in selected_ids]
            coverage_gap = 1.0 - (len(selected_in_region) / total_in_region)
            visual_uncertainty = 0.0
            region_visual_embs = [episode_data[ep]["visual_embedding"] for ep in region_episodes
                                  if ep in episode_data and "visual_embedding" in episode_data[ep]]
            if len(region_visual_embs) >= 2:
                visual_arr = np.array(region_visual_embs)
                n = len(visual_arr)
                total_dist = sum(np.linalg.norm(visual_arr[i] - visual_arr[j])
                                 for i in range(n) for j in range(i + 1, n))
                visual_uncertainty = total_dist / (n * (n - 1) / 2)
            elif len(region_visual_embs) == 1:
                visual_uncertainty = np.linalg.norm(region_visual_embs[0])
            visual_uncertainty_norm = min(visual_uncertainty / 10.0, 1.0)
            priority = coverage_weight * coverage_gap + region_visual_weight * visual_uncertainty_norm
            region_priorities[region_id] = {"priority": priority, "coverage_gap": coverage_gap,
                                            "visual_uncertainty": visual_uncertainty_norm}
        best_region_id = max(region_priorities.keys(), key=lambda r: region_priorities[r]["priority"])
        best_region_priority = region_priorities[best_region_id]
        candidates = [ep for ep, rid in episode_to_region.items() if rid == best_region_id and ep not in selected_ids]
        if not candidates:
            region_priorities[best_region_id]["priority"] = -1.0
            print(f"  Step {step_num}: Region {best_region_id} exhausted, skipping...")
            max_priority = max(r["priority"] for r in region_priorities.values())
            if max_priority < 0:
                print(f"  All regions exhausted. Selected {len(selected_ids)} episodes.")
                break
            continue
        selected_visual = np.array([episode_data[ep]["visual_embedding"] for ep in selected_ids
                                    if ep in episode_data and "visual_embedding" in episode_data[ep]])
        candidate_scores = []
        for candidate_idx in candidates:
            if candidate_idx not in episode_data:
                continue
            visual_score = compute_visual_coverage_score(episode_data[candidate_idx]["visual_embedding"], selected_visual)
            candidate_scores.append((candidate_idx, visual_score))
        if not candidate_scores:
            region_priorities[best_region_id]["priority"] = -1.0
            continue
        visual_scores = [s[1] for s in candidate_scores]
        visual_min, visual_max = min(visual_scores), max(visual_scores)
        visual_range = visual_max - visual_min if visual_max > visual_min else 1.0
        best_candidate = None
        best_scores = None
        best_score = -1.0
        for candidate_idx, vs in candidate_scores:
            norm_vs = (vs - visual_min) / visual_range
            if norm_vs > best_score:
                best_score = norm_vs
                best_candidate = candidate_idx
                best_scores = {"visual_score": float(vs), "normalized_visual_score": float(norm_vs),
                               "joint_score": float(norm_vs)}
        if best_candidate is None:
            continue
        selected_ids.append(best_candidate)
        step_time = time.time() - step_start
        selection_log.append({"step": step_num, "n_selected": len(selected_ids),
                              "selected_region_id": best_region_id,
                              "region_priority": best_region_priority["priority"],
                              "region_coverage_gap": best_region_priority["coverage_gap"],
                              "region_visual_uncertainty": best_region_priority["visual_uncertainty"],
                              "episode_index": best_candidate,
                              "visual_score": best_scores["visual_score"],
                              "action_score": 0.0,
                              "normalized_visual_score": best_scores["normalized_visual_score"],
                              "normalized_action_score": 0.0,
                              "joint_score": best_scores["joint_score"], "time_seconds": step_time})
        if step_num % 10 == 0 or step_num == 1:
            print(f"  Step {step_num}: region={best_region_id}, ep={best_candidate}, "
                  f"region_priority={best_region_priority['priority']:.4f}, "
                  f"joint_score={best_scores['joint_score']:.4f}, time={step_time:.2f}s")
    selected_ids.sort()
    print(f"\n{'='*60}")
    print(f"V5 wo_action Mode Selection Complete!")
    print(f"{'='*60}")
    print(f"Selected {len(selected_ids)} episodes")
    print(f"Total steps: {step_num}")
    region_selection_counts = {}
    for region_id in range(num_regions):
        region_episodes = [ep for ep, rid in episode_to_region.items() if rid == region_id]
        selected_in_region = [ep for ep in region_episodes if ep in selected_ids]
        region_selection_counts[region_id] = {
            "total": len(region_episodes), "selected": len(selected_in_region),
            "selection_ratio": len(selected_in_region) / len(region_episodes) if len(region_episodes) > 0 else 0.0}
    return {"selected_episode_indices": selected_ids, "b0_episodes": b0_episodes,
            "episode_to_region": episode_to_region, "region_selection_counts": region_selection_counts,
            "selection_log": selection_log, "total_steps": step_num, "num_selected": len(selected_ids),
            "num_valid": len(valid_ids), "num_total": len(all_episode_ids), "num_regions": num_regions,
            "region_ratio": region_ratio, "min_regions": min_regions, "max_regions": max_regions,
            "b0_region_ratio": b0_region_ratio, "coverage_weight": coverage_weight,
            "region_visual_weight": region_visual_weight, "episode_visual_weight": episode_visual_weight,
            "seed": seed, "ablation_mode": "wo_action", "visual_coverage_enabled": True,
            "action_diversity_enabled": False, "region_decision_enabled": True}


# ============================================================================
# wo_region mode: no region hierarchy, global embedding selection
# ============================================================================

def select_wo_region_mode(all_episode_ids, visual_embeddings, action_descriptors,
                          num_select, episode_visual_weight=0.5, episode_action_weight=0.5, seed=42):
    """wo_region mode: remove configuration region hierarchy, use global embedding selection."""
    episode_data = {}
    for ep_idx in all_episode_ids:
        if ep_idx in visual_embeddings and ep_idx in action_descriptors:
            episode_data[ep_idx] = {
                "visual_embedding": visual_embeddings[ep_idx],
                "action_descriptor": action_descriptors[ep_idx],
            }
    valid_ids = sorted(episode_data.keys())
    print(f"\n{'='*60}")
    print(f"V5 wo_region Mode: Global Embedding Selection (No Region Hierarchy)")
    print(f"{'='*60}")
    print(f"Total episodes: {len(all_episode_ids)}")
    print(f"Valid episodes (with visual+action, NO region): {len(valid_ids)}")
    print(f"Target selection: {num_select}")
    print(f"Seed: {seed}")
    print(f"Region hierarchy: DISABLED")
    print(f"{'='*60}")
    selected_ids = []
    selection_log = []
    step_num = 0
    print(f"\n[Step 2] Starting global embedding selection (no regions)...")
    while len(selected_ids) < num_select:
        step_num += 1
        step_start = time.time()
        selected_visual = np.array([episode_data[ep]["visual_embedding"] for ep in selected_ids
                                    if ep in episode_data and "visual_embedding" in episode_data[ep]])
        selected_action = np.array([episode_data[ep]["action_descriptor"] for ep in selected_ids
                                    if ep in episode_data and "action_descriptor" in episode_data[ep]])
        candidates = [ep for ep in episode_data.keys() if ep not in selected_ids]
        if not candidates:
            print(f"  All episodes selected. Selected {len(selected_ids)} episodes.")
            break
        candidate_scores = []
        for candidate_idx in candidates:
            if candidate_idx not in episode_data:
                continue
            visual_score = compute_visual_coverage_score(episode_data[candidate_idx]["visual_embedding"], selected_visual)
            action_score = compute_action_diversity_score(episode_data[candidate_idx]["action_descriptor"], selected_action)
            candidate_scores.append((candidate_idx, visual_score, action_score))
        if not candidate_scores:
            break
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
            total_score = episode_visual_weight * norm_vs + episode_action_weight * norm_acs
            if total_score > best_score:
                best_score = total_score
                best_candidate = candidate_idx
                best_scores = {"visual_score": float(vs), "action_score": float(acs),
                               "normalized_visual_score": float(norm_vs), "normalized_action_score": float(norm_acs),
                               "joint_score": float(total_score)}
        if best_candidate is None:
            break
        selected_ids.append(best_candidate)
        step_time = time.time() - step_start
        selection_log.append({"step": step_num, "n_selected": len(selected_ids),
                              "selected_region_id": -1,
                              "region_priority": 0.0, "region_coverage_gap": 0.0,
                              "region_visual_uncertainty": 0.0, "region_action_uncertainty": 0.0,
                              "episode_index": best_candidate,
                              "visual_score": best_scores["visual_score"], "action_score": best_scores["action_score"],
                              "normalized_visual_score": best_scores["normalized_visual_score"],
                              "normalized_action_score": best_scores["normalized_action_score"],
                              "joint_score": best_scores["joint_score"], "time_seconds": step_time})
        if step_num % 10 == 0 or step_num == 1:
            print(f"  Step {step_num}: ep={best_candidate}, joint_score={best_scores['joint_score']:.4f}, time={step_time:.2f}s")
    selected_ids.sort()
    print(f"\n{'='*60}")
    print(f"V5 wo_region Mode Selection Complete!")
    print(f"{'='*60}")
    print(f"Selected {len(selected_ids)} episodes")
    print(f"Total steps: {step_num}")
    return {"selected_episode_indices": selected_ids, "b0_episodes": selected_ids[:min(10, len(selected_ids))],
            "episode_to_region": {}, "region_selection_counts": {},
            "selection_log": selection_log, "total_steps": step_num, "num_selected": len(selected_ids),
            "num_valid": len(valid_ids), "num_total": len(all_episode_ids), "num_regions": 0,
            "region_ratio": 0.0, "min_regions": 0, "max_regions": 0, "b0_region_ratio": 0.0,
            "coverage_weight": 0.0, "region_visual_weight": 0.0, "region_action_weight": 0.0,
            "episode_visual_weight": episode_visual_weight, "episode_action_weight": episode_action_weight,
            "seed": seed, "ablation_mode": "wo_region", "visual_coverage_enabled": True,
            "action_diversity_enabled": True, "region_decision_enabled": False}


# ============================================================================
# Main entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="V5 Ablation Episode Selection")
    parser.add_argument("--visual-embedding-dir", type=str, required=True,
                       help="Visual embedding cache directory")
    parser.add_argument("--action-descriptor-dir", type=str, required=False, default=None,
                       help="Action descriptor cache directory (not used in wo_action mode)")
    parser.add_argument("--dataset-dir", type=str, required=True,
                       help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--num-selected", type=int, default=112,
                       help="Target number of episodes (default: 112)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--ablation-mode", type=str, required=True, choices=["full", "wo_action", "wo_region"],
                       help="Ablation mode: full, wo_action, wo_region")
    parser.add_argument("--visual-weight", type=float, default=0.5,
                       help="Weight for visual coverage score (default: 0.5)")
    parser.add_argument("--action-weight", type=float, default=0.5,
                       help="Weight for action diversity score (default: 0.5)")
    parser.add_argument("--region-ratio", type=float, default=0.1,
                       help="Ratio to compute num_regions (default: 0.1)")
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
    action_dir = Path(args.action_descriptor_dir) if args.action_descriptor_dir else None
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    if not visual_dir.exists():
        print(f"Error: Visual embedding directory does not exist: {visual_dir}")
        return
    if not dataset_dir.exists():
        print(f"Error: Dataset directory does not exist: {dataset_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "subsets").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    print(f"\nLoading visual embeddings from: {visual_dir}")
    visual_embeddings = load_visual_embeddings(visual_dir)

    action_descriptors = {}
    if args.ablation_mode == "wo_action":
        print(f"\n[wo_action mode] Action descriptor loading DISABLED")
    else:
        if action_dir and action_dir.exists():
            print(f"\nLoading action descriptors from: {action_dir}")
            action_descriptors = load_action_descriptors(action_dir)
        else:
            print(f"\nWarning: Action descriptor directory not found: {action_dir}")
            print(f"  Proceeding with empty action descriptors for {args.ablation_mode} mode")

    all_episode_ids = sorted(set(visual_embeddings.keys()) | set(action_descriptors.keys()))

    print(f"\nLoading rand_vecs from: {dataset_dir}")
    rand_vecs = load_rand_vecs(dataset_dir, all_episode_ids)

    if args.ablation_mode == "full":
        valid_ids = [ep_idx for ep_idx in all_episode_ids
                     if ep_idx in rand_vecs and ep_idx in visual_embeddings and ep_idx in action_descriptors]
        print(f"Episodes with all data (rand_vec+visual+action): {len(valid_ids)}")
        if len(valid_ids) == 0:
            print("Error: No episodes with all required data")
            return
        result = select_full_mode(
            all_episode_ids=all_episode_ids, visual_embeddings=visual_embeddings,
            action_descriptors=action_descriptors, rand_vecs=rand_vecs,
            num_select=args.num_selected, region_ratio=args.region_ratio,
            min_regions=args.min_regions, max_regions=args.max_regions,
            b0_region_ratio=args.b0_region_ratio, coverage_weight=args.coverage_weight,
            region_visual_weight=args.region_visual_weight, region_action_weight=args.region_action_weight,
            episode_visual_weight=args.visual_weight, episode_action_weight=args.action_weight,
            seed=args.seed)
    elif args.ablation_mode == "wo_action":
        valid_ids = [ep_idx for ep_idx in all_episode_ids
                     if ep_idx in rand_vecs and ep_idx in visual_embeddings]
        print(f"Episodes with rand_vec+visual (no action): {len(valid_ids)}")
        if len(valid_ids) == 0:
            print("Error: No episodes with required data")
            return
        result = select_wo_action_mode(
            all_episode_ids=all_episode_ids, visual_embeddings=visual_embeddings,
            rand_vecs=rand_vecs, num_select=args.num_selected,
            region_ratio=args.region_ratio, min_regions=args.min_regions, max_regions=args.max_regions,
            b0_region_ratio=args.b0_region_ratio, coverage_weight=args.coverage_weight,
            region_visual_weight=args.region_visual_weight + args.region_action_weight,
            episode_visual_weight=1.0, seed=args.seed)
    elif args.ablation_mode == "wo_region":
        valid_ids = [ep_idx for ep_idx in all_episode_ids
                     if ep_idx in visual_embeddings and ep_idx in action_descriptors]
        print(f"Episodes with visual+action (no region): {len(valid_ids)}")
        if len(valid_ids) == 0:
            print("Error: No episodes with required data")
            return
        result = select_wo_region_mode(
            all_episode_ids=all_episode_ids, visual_embeddings=visual_embeddings,
            action_descriptors=action_descriptors, num_select=args.num_selected,
            episode_visual_weight=args.visual_weight, episode_action_weight=args.action_weight,
            seed=args.seed)
    else:
        print(f"Error: Unknown ablation mode: {args.ablation_mode}")
        return

    episode_data_for_diag = {}
    for ep_idx in valid_ids:
        entry = {}
        if ep_idx in rand_vecs:
            entry["rand_vec"] = rand_vecs[ep_idx]
        if ep_idx in visual_embeddings:
            entry["visual_embedding"] = visual_embeddings[ep_idx]
        if ep_idx in action_descriptors:
            entry["action_descriptor"] = action_descriptors[ep_idx]
        if entry:
            episode_data_for_diag[ep_idx] = entry

    method_name = args.ablation_mode
    subset_file = output_dir / "subsets" / f"our_v5_{method_name}_{args.num_selected}_seed{args.seed}.json"
    save_selected_episodes(
        selected_episode_ids=result["selected_episode_indices"],
        output_path=subset_file,
        metadata={
            "selection_method": f"our_v5_ablation_{method_name}",
            "ablation_mode": method_name,
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
            "visual_coverage_enabled": result["visual_coverage_enabled"],
            "action_diversity_enabled": result["action_diversity_enabled"],
            "region_decision_enabled": result["region_decision_enabled"],
            "dynamic_selection": True,
            "seed": args.seed,
        },
        method_name=method_name)

    log_file = output_dir / "results" / f"selection_log_v5_{method_name}_{args.num_selected}_seed{args.seed}.json"
    with open(log_file, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"Selection log saved to: {log_file}")

    print(f"\n{'='*60}")
    print(f"Ablation mode '{method_name}' complete!")
    print(f"  Visual coverage: {'ENABLED' if result['visual_coverage_enabled'] else 'DISABLED'}")
    print(f"  Action diversity: {'ENABLED' if result['action_diversity_enabled'] else 'DISABLED'}")
    print(f"  Region decision: {'ENABLED' if result['region_decision_enabled'] else 'DISABLED'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()