#!/usr/bin/env python
"""
Our V5 Episode Selection - Adaptive Latent Coverage Based Selection

Loads V5 action-aware embeddings (global_embedding, wrist_embedding, action_descriptor)
and runs adaptive latent coverage based episode selection.

Core idea:
- Build latent cells using KMeans on concatenated embeddings
- Select episodes adaptively based on cell priority (coverage need + visual uncertainty + action uncertainty)
- Within each selected cell, choose the episode with highest gain (visual coverage + action diversity)

Usage:
    python select_episodes_v5.py \
        --embeddings-dir /path/to/v5/embeddings \
        --output-dir /path/to/output \
        --target-size 112 \
        --num-cells 32 \
        --seed 42 \
        --coverage-weight 0.5 \
        --visual-weight 0.3 \
        --action-weight 0.2
"""

import sys
import os
import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import MiniBatchKMeans

sys.stdout.reconfigure(line_buffering=True)

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WORK2_ROOT = Path(__file__).resolve().parents[2]
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

from our.core.anchors import AnchorSystem
from our.core.sic import compute_sic_score


def load_v5_embeddings(embeddings_dir: Path) -> Dict[int, Dict]:
    """
    Load V5 action-aware embeddings with separate visual and action representations.
    
    V5 uses 'global_embedding', 'wrist_embedding', and 'action_descriptor' keys.
    
    Args:
        embeddings_dir: V5 embedding cache directory
    
    Returns:
        Dict[episode_index, {"phi_global": ..., "phi_wrist": ..., "action_descriptor": ...}]
    """
    embeddings = {}
    
    for f in sorted(embeddings_dir.glob("*.npy")):
        try:
            data = np.load(f, allow_pickle=True).item()
            ep_idx = data.get("episode_index")
            if ep_idx is None:
                continue
            
            global_embedding = data["global_embedding"]
            wrist_embedding = data["wrist_embedding"]
            action_descriptor = data["action_descriptor"]
            
            embeddings[ep_idx] = {
                "phi_global": global_embedding,
                "phi_wrist": wrist_embedding,
                "action_descriptor": action_descriptor,
            }
        except Exception as e:
            print(f"  Skipping invalid file: {f.name} ({e})")
    
    print(f"Loaded {len(embeddings)} V5 action-aware episode embeddings")
    return embeddings


def build_latent_cells(
    embeddings: Dict[int, Dict],
    num_cells: int = 32,
    seed: int = 42
) -> Tuple[Dict[int, int], np.ndarray]:
    """
    Build latent cells by clustering episodes in the joint embedding space.
    
    Concatenates normalized global_embedding, wrist_embedding, and action_descriptor
    for each episode, then uses MiniBatchKMeans to partition into latent cells.
    
    Args:
        embeddings: Dict[episode_index, {"phi_global": ..., "phi_wrist": ..., "action_descriptor": ...}]
        num_cells: number of latent cells to create
        seed: random seed for clustering
    
    Returns:
        Tuple of:
            - Dict[episode_index, cell_id]: mapping from episode to cell
            - np.ndarray: cluster centers for diagnostics
    """
    print(f"\n{'='*60}")
    print(f"Building {num_cells} latent cells...")
    print(f"{'='*60}")
    
    episode_indices = sorted(embeddings.keys())
    n_episodes = len(episode_indices)
    
    # Extract and concatenate features
    feature_list = []
    for ep_idx in episode_indices:
        phi_global = embeddings[ep_idx]["phi_global"]
        phi_wrist = embeddings[ep_idx]["phi_wrist"]
        action_desc = embeddings[ep_idx]["action_descriptor"]
        
        # Concatenate all features
        combined = np.concatenate([phi_global, phi_wrist, action_desc])
        feature_list.append(combined)
    
    features = np.array(feature_list)
    
    # L2 normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    features_normalized = features / norms
    
    # Cluster using MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters=num_cells,
        random_state=seed,
        batch_size=min(256, n_episodes),
        n_init=10,
    )
    cell_ids = kmeans.fit_predict(features_normalized)
    
    # Build episode to cell mapping
    episode_to_cell = {}
    for i, ep_idx in enumerate(episode_indices):
        episode_to_cell[ep_idx] = int(cell_ids[i])
    
    # Print cell statistics
    cell_counts = {}
    for cell_id in cell_ids:
        cell_id = int(cell_id)
        cell_counts[cell_id] = cell_counts.get(cell_id, 0) + 1
    
    print(f"  Episodes clustered: {n_episodes}")
    print(f"  Cells created: {num_cells}")
    print(f"  Cell size - min: {min(cell_counts.values())}, max: {max(cell_counts.values())}, "
          f"mean: {np.mean(list(cell_counts.values())):.1f}")
    
    return episode_to_cell, kmeans.cluster_centers_


def compute_cell_priority(
    cell_id: int,
    episode_to_cell: Dict[int, int],
    selected_episodes: List[int],
    embeddings: Dict[int, Dict],
    coverage_weight: float = 0.5,
    visual_weight: float = 0.3,
    action_weight: float = 0.2,
) -> Dict[str, float]:
    """
    Compute priority for a latent cell based on three factors:
    1. Coverage need: fewer selected episodes in this cell -> higher priority
    2. Visual uncertainty: higher variance in visual embeddings -> higher priority
    3. Action uncertainty: higher variance in action descriptors -> higher priority
    
    Args:
        cell_id: the cell to compute priority for
        episode_to_cell: mapping from episode index to cell id
        selected_episodes: list of already selected episode indices
        embeddings: episode embeddings dict
        coverage_weight: weight for coverage need
        visual_weight: weight for visual uncertainty
        action_weight: weight for action uncertainty
    
    Returns:
        Dict with keys: "priority", "coverage_need", "visual_uncertainty", "action_uncertainty"
    """
    # Get all episodes in this cell
    cell_episodes = [ep for ep, cid in episode_to_cell.items() if cid == cell_id]
    total_in_cell = len(cell_episodes)
    
    if total_in_cell == 0:
        return {
            "priority": 0.0,
            "coverage_need": 0.0,
            "visual_uncertainty": 0.0,
            "action_uncertainty": 0.0,
        }
    
    # 1. Coverage need: inverse of selection ratio
    selected_in_cell = [ep for ep in cell_episodes if ep in selected_episodes]
    n_selected_in_cell = len(selected_in_cell)
    
    # Coverage need: higher when fewer episodes selected
    coverage_need = 1.0 - (n_selected_in_cell / total_in_cell)
    
    # 2. Visual uncertainty: average pairwise distance of visual embeddings in cell
    visual_uncertainty = 0.0
    cell_global_embs = []
    cell_wrist_embs = []
    for ep in cell_episodes:
        if ep in embeddings:
            cell_global_embs.append(embeddings[ep]["phi_global"])
            cell_wrist_embs.append(embeddings[ep]["phi_wrist"])
    
    if len(cell_global_embs) >= 2:
        # Compute average pairwise distance for global embeddings
        global_arr = np.array(cell_global_embs)
        n = len(global_arr)
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += np.linalg.norm(global_arr[i] - global_arr[j])
                count += 1
        visual_uncertainty_global = total_dist / count if count > 0 else 0.0
        
        # Compute average pairwise distance for wrist embeddings
        wrist_arr = np.array(cell_wrist_embs)
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += np.linalg.norm(wrist_arr[i] - wrist_arr[j])
                count += 1
        visual_uncertainty_wrist = total_dist / count if count > 0 else 0.0
        
        # Combine both visual uncertainties
        visual_uncertainty = (visual_uncertainty_global + visual_uncertainty_wrist) / 2.0
    elif len(cell_global_embs) == 1:
        # Single episode: use norm as proxy
        visual_uncertainty = np.linalg.norm(cell_global_embs[0])
    
    # 3. Action uncertainty: average pairwise distance of action descriptors in cell
    action_uncertainty = 0.0
    cell_action_embs = []
    for ep in cell_episodes:
        if ep in embeddings:
            cell_action_embs.append(embeddings[ep]["action_descriptor"])
    
    if len(cell_action_embs) >= 2:
        action_arr = np.array(cell_action_embs)
        n = len(action_arr)
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += np.linalg.norm(action_arr[i] - action_arr[j])
                count += 1
        action_uncertainty = total_dist / count if count > 0 else 0.0
    elif len(cell_action_embs) == 1:
        action_uncertainty = np.linalg.norm(cell_action_embs[0])
    
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


def select_candidate_from_cell(
    cell_id: int,
    episode_to_cell: Dict[int, int],
    selected_episodes: List[int],
    remaining_episodes: List[int],
    embeddings: Dict[int, Dict],
    alpha: float = 1.0,
    lambda_wrist: float = 1.0,
    visual_weight: float = 0.7,
    action_weight: float = 0.3,
) -> Optional[Tuple[int, Dict[str, float]]]:
    """
    Select the best candidate episode from a specific cell based on gain.
    
    candidate_gain = visual_coverage_gain + action_diversity_gain
    
    Args:
        cell_id: the cell to select from
        episode_to_cell: mapping from episode index to cell id
        selected_episodes: list of already selected episode indices
        remaining_episodes: list of remaining candidate episodes
        embeddings: episode embeddings dict
        alpha: SIC smoothing coefficient
        lambda_wrist: wrist camera weight
        visual_weight: weight for visual coverage
        action_weight: weight for action diversity
    
    Returns:
        Tuple of (best_episode_index, scores_dict) or None if no candidates
    """
    # Get candidates from this cell that are still remaining
    candidates = [ep for ep in remaining_episodes if episode_to_cell.get(ep) == cell_id]
    
    if not candidates:
        return None
    
    best_candidate = None
    best_scores = None
    best_gain = -1.0
    
    for candidate_idx in candidates:
        # Compute visual coverage gain using existing SIC function
        visual_gain = compute_visual_coverage_score(
            selected_episodes, candidate_idx, embeddings, alpha, lambda_wrist
        )
        
        # Compute action diversity gain using existing function
        action_gain = compute_action_diversity_score(
            selected_episodes, candidate_idx, embeddings
        )
        
        # Combined gain
        total_gain = visual_weight * visual_gain + action_weight * action_gain
        
        if total_gain > best_gain:
            best_gain = total_gain
            best_candidate = candidate_idx
            best_scores = {
                "visual_score": visual_gain,
                "action_score": action_gain,
                "joint_score": total_gain,
            }
    
    if best_candidate is not None:
        return best_candidate, best_scores
    
    return None


def compute_visual_coverage_score(
    selected_episodes: List[int],
    candidate_idx: int,
    embeddings: Dict[int, Dict],
    alpha: float = 1.0,
    lambda_wrist: float = 1.0
) -> float:
    """
    Compute visual coverage gain using only phi_global and phi_wrist.
    
    Uses the existing AnchorSystem and SIC visual coverage calculation.
    
    Args:
        selected_episodes: list of already selected episode indices
        candidate_idx: candidate episode index to evaluate
        embeddings: episode embeddings dict
        alpha: SIC smoothing coefficient
        lambda_wrist: wrist camera weight
    
    Returns:
        float: visual coverage score
    """
    candidate_set = {}
    for ep_idx in selected_episodes:
        candidate_set[ep_idx] = candidate_set.get(ep_idx, 0) + 1
    
    candidate_set[candidate_idx] = candidate_set.get(candidate_idx, 0) + 1
    
    anchor_system = AnchorSystem()
    for ep_idx in set(list(selected_episodes) + [candidate_idx]):
        if ep_idx in embeddings:
            anchor_system.add_anchor(
                ep_idx,
                embeddings[ep_idx]["phi_global"],
                embeddings[ep_idx]["phi_wrist"]
            )
    
    anchor_system.compute_dbar()
    
    return compute_sic_score(candidate_set, anchor_system, alpha, lambda_wrist)


def compute_action_diversity_score(
    selected_episodes: List[int],
    candidate_idx: int,
    embeddings: Dict[int, Dict],
) -> float:
    """
    Compute action diversity gain using only action_descriptor.
    
    Calculates the average distance between candidate action_descriptor
    and selected episodes' action_descriptors. Larger distance means
    higher action diversity contribution.
    
    Uses normalized cosine distance.
    
    Args:
        selected_episodes: list of already selected episode indices
        candidate_idx: candidate episode index to evaluate
        embeddings: episode embeddings dict
    
    Returns:
        float: action diversity score (higher = more diverse)
    """
    if candidate_idx not in embeddings:
        return 0.0
    
    candidate_action = embeddings[candidate_idx]["action_descriptor"]
    
    if not selected_episodes:
        return 1.0
    
    distances = []
    for ep_idx in selected_episodes:
        if ep_idx not in embeddings:
            continue
        
        selected_action = embeddings[ep_idx]["action_descriptor"]
        
        candidate_norm = np.linalg.norm(candidate_action)
        selected_norm = np.linalg.norm(selected_action)
        
        if candidate_norm < 1e-8 or selected_norm < 1e-8:
            cosine_sim = 0.0
        else:
            cosine_sim = np.dot(candidate_action, selected_action) / (candidate_norm * selected_norm)
            cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        
        cosine_dist = 1.0 - cosine_sim
        distances.append(cosine_dist)
    
    if not distances:
        return 1.0
    
    return np.mean(distances)


def select_initial_b0_latent(
    all_episode_indices: List[int],
    episode_to_cell: Dict[int, int],
    b0_size: int,
    num_cells: int,
    seed: int = 42
) -> List[int]:
    """
    Select initial B0 episodes using latent coverage strategy.
    
    Instead of random sampling, uniformly sample from different latent cells
    to ensure initial coverage of the latent space.
    
    Args:
        all_episode_indices: list of all episode indices
        episode_to_cell: mapping from episode index to cell id
        b0_size: number of initial episodes to select
        num_cells: total number of latent cells
        seed: random seed
    
    Returns:
        List of selected episode indices
    """
    rng = np.random.RandomState(seed)
    
    # Group episodes by cell
    cell_to_episodes = {}
    for ep_idx in all_episode_indices:
        cell_id = episode_to_cell.get(ep_idx)
        if cell_id is not None:
            if cell_id not in cell_to_episodes:
                cell_to_episodes[cell_id] = []
            cell_to_episodes[cell_id].append(ep_idx)
    
    # Calculate episodes per cell
    episodes_per_cell = b0_size // num_cells
    remainder = b0_size % num_cells
    
    selected = []
    
    # First, select uniformly from each cell
    for cell_id in range(num_cells):
        if cell_id in cell_to_episodes:
            cell_episodes = cell_to_episodes[cell_id]
            n_select = episodes_per_cell + (1 if cell_id < remainder else 0)
            n_select = min(n_select, len(cell_episodes))
            
            if n_select > 0:
                chosen = rng.choice(cell_episodes, size=n_select, replace=False).tolist()
                selected.extend(chosen)
    
    # If we still need more, sample from remaining episodes
    if len(selected) < b0_size:
        remaining = [ep for ep in all_episode_indices if ep not in selected]
        n_needed = b0_size - len(selected)
        additional = rng.choice(remaining, size=min(n_needed, len(remaining)), replace=False).tolist()
        selected.extend(additional)
    
    selected.sort()
    return selected


def iterative_select_episodes_adaptive(
    embeddings: Dict[int, Dict],
    b0_size: int = 18,
    target_size: int = 112,
    num_cells: int = 32,
    seed: int = 42,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0,
    coverage_weight: float = 0.5,
    visual_weight: float = 0.3,
    action_weight: float = 0.2,
) -> Dict:
    """
    Adaptive latent coverage based episode selection.
    
    Flow:
    1. Build latent cells using KMeans on concatenated embeddings
    2. Initialize B0 with latent coverage (uniform sampling from cells)
    3. Iteratively:
       a. Compute cell priorities for all cells
       b. Select highest priority cell
       c. Within that cell, select episode with highest gain
       d. Update selected set and repeat
    """
    all_episode_indices = sorted(embeddings.keys())
    n_total = len(all_episode_indices)
    
    print(f"\n{'='*60}")
    print(f"V5 Adaptive Latent Coverage Episode Selection")
    print(f"{'='*60}")
    print(f"Total episodes: {n_total}")
    print(f"Number of latent cells: {num_cells}")
    print(f"Initial B0 size: {b0_size}")
    print(f"Target size: {target_size}")
    print(f"coverage_weight: {coverage_weight}, visual_weight: {visual_weight}, action_weight: {action_weight}")
    print(f"alpha: {alpha}, lambda_wrist: {lambda_wrist}")
    print(f"{'='*60}")
    
    # Step 1: Build latent cells
    episode_to_cell, cell_centers = build_latent_cells(embeddings, num_cells, seed)
    
    # Step 2: Initialize B0 with latent coverage
    print(f"\n[Step 2] Selecting initial B0 with latent coverage ({b0_size} episodes)...")
    b0_episodes = select_initial_b0_latent(
        all_episode_indices, episode_to_cell, b0_size, num_cells, seed
    )
    selected_episodes = list(b0_episodes)
    remaining_episodes = [ep for ep in all_episode_indices if ep not in selected_episodes]
    
    print(f"  B0 episodes: {b0_episodes}")
    
    # Compute initial SIC
    candidate_set_b0 = {ep: 1 for ep in selected_episodes}
    anchor_system_b0 = AnchorSystem()
    for ep_idx in selected_episodes:
        if ep_idx in embeddings:
            anchor_system_b0.add_anchor(
                ep_idx,
                embeddings[ep_idx]["phi_global"],
                embeddings[ep_idx]["phi_wrist"]
            )
    anchor_system_b0.compute_dbar()
    initial_sic = compute_sic_score(candidate_set_b0, anchor_system_b0, alpha, lambda_wrist)
    
    print(f"  Initial SIC score: {initial_sic:.4f}")
    
    # Step 3: Adaptive selection
    print(f"\n[Step 3] Starting adaptive latent coverage selection...")
    selection_log = []
    sic_history = [initial_sic]
    step_num = 0
    
    while len(selected_episodes) < target_size and remaining_episodes:
        step_num += 1
        step_start = time.time()
        
        # Compute priorities for all cells
        cell_priorities = {}
        for cell_id in range(num_cells):
            priority_info = compute_cell_priority(
                cell_id, episode_to_cell, selected_episodes, embeddings,
                coverage_weight, visual_weight, action_weight
            )
            cell_priorities[cell_id] = priority_info
        
        # Select highest priority cell
        best_cell_id = max(cell_priorities.keys(), key=lambda c: cell_priorities[c]["priority"])
        best_cell_priority = cell_priorities[best_cell_id]
        
        # Select best candidate from this cell
        result = select_candidate_from_cell(
            best_cell_id, episode_to_cell, selected_episodes, remaining_episodes,
            embeddings, alpha, lambda_wrist, visual_weight, action_weight
        )
        
        if result is None:
            # No candidates in this cell, mark as exhausted and try next
            print(f"  Step {step_num}: Cell {best_cell_id} exhausted, skipping...")
            continue
        
        best_candidate, scores = result
        
        # Add to selected
        selected_episodes.append(best_candidate)
        remaining_episodes.remove(best_candidate)
        
        # Recompute SIC
        candidate_set_current = {ep: 1 for ep in selected_episodes}
        anchor_system_current = AnchorSystem()
        for ep_idx in selected_episodes:
            if ep_idx in embeddings:
                anchor_system_current.add_anchor(
                    ep_idx,
                    embeddings[ep_idx]["phi_global"],
                    embeddings[ep_idx]["phi_wrist"]
                )
        anchor_system_current.compute_dbar()
        current_sic = compute_sic_score(candidate_set_current, anchor_system_current, alpha, lambda_wrist)
        sic_history.append(current_sic)
        
        step_time = time.time() - step_start
        
        # Log this step
        selection_log.append({
            "step": step_num,
            "n_selected": len(selected_episodes),
            "selected_cell_id": best_cell_id,
            "cell_priority": best_cell_priority["priority"],
            "cell_coverage_need": best_cell_priority["coverage_need"],
            "cell_visual_uncertainty": best_cell_priority["visual_uncertainty"],
            "cell_action_uncertainty": best_cell_priority["action_uncertainty"],
            "episode_index": best_candidate,
            "visual_score": scores["visual_score"],
            "action_score": scores["action_score"],
            "joint_score": scores["joint_score"],
            "sic_score": current_sic,
            "sic_gain": current_sic - sic_history[-2] if len(sic_history) > 1 else 0,
            "time_seconds": step_time,
        })
        
        if step_num % 10 == 0 or step_num == 1:
            print(f"  Step {step_num}: cell={best_cell_id}, ep={best_candidate}, "
                  f"cell_priority={best_cell_priority['priority']:.4f}, "
                  f"joint_score={scores['joint_score']:.4f}, "
                  f"SIC={current_sic:.4f} (gain={current_sic - sic_history[-2]:.4f}), "
                  f"time={step_time:.2f}s")
    
    # Step 4: Output results
    print(f"\n{'='*60}")
    print(f"Adaptive selection complete!")
    print(f"{'='*60}")
    print(f"Final selected episodes: {len(selected_episodes)}")
    print(f"Final SIC score: {sic_history[-1]:.4f}")
    print(f"SIC gain: {sic_history[-1] - initial_sic:.4f}")
    print(f"Total steps: {step_num}")
    
    # Compute cell selection distribution
    cell_selection_counts = {}
    for cell_id in range(num_cells):
        cell_episodes = [ep for ep, cid in episode_to_cell.items() if cid == cell_id]
        selected_in_cell = [ep for ep in cell_episodes if ep in selected_episodes]
        cell_selection_counts[cell_id] = {
            "total": len(cell_episodes),
            "selected": len(selected_in_cell),
            "selection_ratio": len(selected_in_cell) / len(cell_episodes) if len(cell_episodes) > 0 else 0.0,
        }
    
    return {
        "selected_episodes": sorted(selected_episodes),
        "b0_episodes": b0_episodes,
        "episode_to_cell": episode_to_cell,
        "cell_selection_counts": cell_selection_counts,
        "selection_log": selection_log,
        "sic_history": sic_history,
        "initial_sic": initial_sic,
        "final_sic": sic_history[-1],
        "total_steps": step_num,
        "target_size": target_size,
        "b0_size": b0_size,
        "num_cells": num_cells,
        "alpha": alpha,
        "lambda_wrist": lambda_wrist,
        "coverage_weight": coverage_weight,
        "visual_weight": visual_weight,
        "action_weight": action_weight,
        "seed": seed,
    }


def compute_action_distribution_distance(
    selected_action_vecs: List[np.ndarray],
    full_action_vecs: List[np.ndarray],
) -> float:
    """
    Compute action distribution distance between selected subset and full dataset.
    
    Uses average pairwise cosine distance between selected action_descriptors
    and all action_descriptors to measure distribution shift.
    
    Args:
        selected_action_vecs: list of action_descriptor vectors for selected episodes
        full_action_vecs: list of action_descriptor vectors for all episodes
    
    Returns:
        float: mean cosine distance (higher = more distribution shift)
    """
    if not selected_action_vecs or not full_action_vecs:
        return 0.0
    
    sel_arr = np.array(selected_action_vecs)
    full_arr = np.array(full_action_vecs)
    
    cosine_dists = []
    for sv in sel_arr:
        sv_norm = np.linalg.norm(sv)
        if sv_norm < 1e-10:
            continue
        full_norms = np.linalg.norm(full_arr, axis=1)
        mask = full_norms > 1e-10
        if not mask.any():
            continue
        sims = np.dot(full_arr[mask], sv) / (full_norms[mask] * sv_norm)
        sims = np.clip(sims, -1.0, 1.0)
        cosine_dists.append(float(np.mean(1.0 - sims)))
    
    return float(np.mean(cosine_dists)) if cosine_dists else 0.0


def compute_diagnostic(
    result: Dict,
    embeddings: Dict[int, Dict],
    output_dir: Path,
) -> Dict:
    """
    Compute comprehensive diagnostic statistics after selection.
    
    Produces episode-level scores, dataset-level statistics, action distribution
    coverage analysis, selection bias analysis, and latent coverage statistics.
    """
    selected_ids = result["selected_episodes"]
    all_ids = sorted(embeddings.keys())
    
    # Part 1: Episode-level selection info
    episode_scores = []
    for entry in result["selection_log"]:
        vs = float(entry["visual_score"])
        acs = float(entry["action_score"])
        js = float(entry["joint_score"])
        score_ratio_action = acs / (vs + acs + 1e-8)
        score_ratio_visual = vs / (vs + acs + 1e-8)
        episode_scores.append({
            "episode_index": entry["episode_index"],
            "step": entry["step"],
            "selected_cell_id": entry["selected_cell_id"],
            "cell_priority": entry["cell_priority"],
            "visual_score": vs,
            "action_score": acs,
            "joint_score": js,
            "score_ratio_action": score_ratio_action,
            "score_ratio_visual": score_ratio_visual,
        })
    
    # Part 2: Selected subset vs full dataset embedding statistics
    selected_global_vecs = [embeddings[ep]["phi_global"] for ep in selected_ids if ep in embeddings]
    full_global_vecs = [embeddings[ep]["phi_global"] for ep in all_ids if ep in embeddings]
    selected_action_vecs = [embeddings[ep]["action_descriptor"] for ep in selected_ids if ep in embeddings]
    full_action_vecs = [embeddings[ep]["action_descriptor"] for ep in all_ids if ep in embeddings]
    
    def _compute_vec_stats(vecs: List[np.ndarray]) -> Dict[str, float]:
        if not vecs:
            return {"mean": 0.0, "std": 0.0}
        arr = np.array(vecs)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
    
    selected_visual_stats = _compute_vec_stats(selected_global_vecs)
    full_visual_stats = _compute_vec_stats(full_global_vecs)
    selected_action_stats = _compute_vec_stats(selected_action_vecs)
    full_action_stats = _compute_vec_stats(full_action_vecs)
    
    # Part 3: Action distribution coverage analysis
    action_distribution_distance = compute_action_distribution_distance(
        selected_action_vecs, full_action_vecs
    )
    
    # Part 4: Selection bias analysis
    visual_dominant_count = 0
    action_dominant_count = 0
    balanced_count = 0
    for ep_info in episode_scores:
        vs = ep_info["visual_score"]
        acs = ep_info["action_score"]
        ratio = abs(vs - acs) / (vs + acs + 1e-8)
        if vs > acs:
            visual_dominant_count += 1
        elif acs > vs:
            action_dominant_count += 1
        if ratio < 0.1:
            balanced_count += 1
    
    # Part 5: Latent coverage statistics
    episode_to_cell = result.get("episode_to_cell", {})
    cell_selection_counts = result.get("cell_selection_counts", {})
    num_cells = result.get("num_cells", 0)
    
    selected_cell_count = sum(1 for cid, info in cell_selection_counts.items() if info["selected"] > 0)
    total_cell_count = len(cell_selection_counts)
    
    cell_selection_distribution = []
    for cid in sorted(cell_selection_counts.keys()):
        info = cell_selection_counts[cid]
        cell_selection_distribution.append({
            "cell_id": cid,
            "total_episodes": info["total"],
            "selected_episodes": info["selected"],
            "selection_ratio": info["selection_ratio"],
        })
    
    # Average cell priority for selected cells
    selected_cell_priorities = [
        entry["cell_priority"] for entry in result["selection_log"]
    ]
    avg_cell_priority = float(np.mean(selected_cell_priorities)) if selected_cell_priorities else 0.0
    
    # Average uncertainties for selected cells
    selected_cell_visual_uncertainties = [
        entry["cell_visual_uncertainty"] for entry in result["selection_log"]
    ]
    avg_selected_cell_visual_uncertainty = float(np.mean(selected_cell_visual_uncertainties)) if selected_cell_visual_uncertainties else 0.0
    
    selected_cell_action_uncertainties = [
        entry["cell_action_uncertainty"] for entry in result["selection_log"]
    ]
    avg_selected_cell_action_uncertainty = float(np.mean(selected_cell_action_uncertainties)) if selected_cell_action_uncertainties else 0.0
    
    # Part 6: Build and save diagnostic result
    diagnostic = {
        "episode_scores": episode_scores,
        "selected_count": len(selected_ids),
        "total_count": len(all_ids),
        "selected_visual_mean": selected_visual_stats["mean"],
        "full_visual_mean": full_visual_stats["mean"],
        "selected_visual_std": selected_visual_stats["std"],
        "full_visual_std": full_visual_stats["std"],
        "selected_action_mean": selected_action_stats["mean"],
        "full_action_mean": full_action_stats["mean"],
        "selected_action_std": selected_action_stats["std"],
        "full_action_std": full_action_stats["std"],
        "selected_action_distribution_distance": action_distribution_distance,
        "visual_dominant_count": visual_dominant_count,
        "action_dominant_count": action_dominant_count,
        "balanced_count": balanced_count,
        # Latent coverage statistics
        "latent_coverage_statistics": {
            "selected_cell_count": selected_cell_count,
            "total_cell_count": total_cell_count,
            "cell_coverage_ratio": selected_cell_count / total_cell_count if total_cell_count > 0 else 0.0,
            "cell_selection_distribution": cell_selection_distribution,
            "average_cell_priority": avg_cell_priority,
            "average_selected_cell_visual_uncertainty": avg_selected_cell_visual_uncertainty,
            "average_selected_cell_action_uncertainty": avg_selected_cell_action_uncertainty,
        },
    }
    
    diag_file = output_dir / "selection_diagnostics.json"
    with open(diag_file, "w") as f:
        json.dump(diagnostic, f, indent=2)
    print(f"Diagnostic saved to: {diag_file}")
    
    return diagnostic


def main():
    parser = argparse.ArgumentParser(description="V5 Adaptive Latent Coverage Episode Selection")
    parser.add_argument("--embeddings-dir", type=str, required=True, help="V5 embedding cache directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--b0-size", type=int, default=18, help="Initial B0 size (default: 18)")
    parser.add_argument("--target-size", type=int, default=112, help="Target episode count (default: 112)")
    parser.add_argument("--num-cells", type=int, default=32, help="Number of latent cells (default: 32)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", type=float, default=1.0, help="SIC smoothing coefficient")
    parser.add_argument("--lambda-wrist", type=float, default=1.0, help="Wrist weight")
    parser.add_argument("--coverage-weight", type=float, default=0.5, help="Weight for coverage need (default: 0.5)")
    parser.add_argument("--visual-weight", type=float, default=0.3, help="Weight for visual uncertainty (default: 0.3)")
    parser.add_argument("--action-weight", type=float, default=0.2, help="Weight for action uncertainty (default: 0.2)")
    
    args = parser.parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    
    if not embeddings_dir.exists():
        print(f"Error: Embeddings directory does not exist: {embeddings_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading V5 embeddings from: {embeddings_dir}")
    embeddings = load_v5_embeddings(embeddings_dir)
    
    if not embeddings:
        print("Error: No valid V5 embeddings found")
        return
    
    result = iterative_select_episodes_adaptive(
        embeddings=embeddings,
        b0_size=args.b0_size,
        target_size=args.target_size,
        num_cells=args.num_cells,
        seed=args.seed,
        alpha=args.alpha,
        lambda_wrist=args.lambda_wrist,
        coverage_weight=args.coverage_weight,
        visual_weight=args.visual_weight,
        action_weight=args.action_weight,
    )
    
    # Compute visual and action contribution statistics
    all_visual_scores = [entry["visual_score"] for entry in result["selection_log"]]
    all_action_scores = [entry["action_score"] for entry in result["selection_log"]]
    
    visual_contribution_stats = {
        "mean": float(np.mean(all_visual_scores)) if all_visual_scores else 0.0,
        "std": float(np.std(all_visual_scores)) if all_visual_scores else 0.0,
        "min": float(np.min(all_visual_scores)) if all_visual_scores else 0.0,
        "max": float(np.max(all_visual_scores)) if all_visual_scores else 0.0,
    }
    action_contribution_stats = {
        "mean": float(np.mean(all_action_scores)) if all_action_scores else 0.0,
        "std": float(np.std(all_action_scores)) if all_action_scores else 0.0,
        "min": float(np.min(all_action_scores)) if all_action_scores else 0.0,
        "max": float(np.max(all_action_scores)) if all_action_scores else 0.0,
    }
    
    # Save subset file
    subset_file = output_dir / "subset.json"
    subset_data = {
        "method": "our_v5_action_aware",
        "selection_method": "our_v5_adaptive_latent_coverage",
        "num_episodes": args.target_size,
        "seed": args.seed,
        "visual_weight": args.visual_weight,
        "action_weight": args.action_weight,
        "uses_action_descriptor": True,
        "selected_episode_indices": result["selected_episodes"],
        "parameters": {
            "b0_size": args.b0_size,
            "num_cells": args.num_cells,
            "alpha": args.alpha,
            "lambda_wrist": args.lambda_wrist,
            "coverage_weight": args.coverage_weight,
            "visual_weight": args.visual_weight,
            "action_weight": args.action_weight,
            "dynamic_addition": True,
        },
        "contribution_stats": {
            "visual_contribution": visual_contribution_stats,
            "action_contribution": action_contribution_stats,
        },
    }
    with open(subset_file, "w") as f:
        json.dump(subset_data, f, indent=2)
    print(f"\nSubset saved to: {subset_file}")
    
    # Save full selection log
    log_file = output_dir / "selection_log_v5.json"
    with open(log_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Selection log saved to: {log_file}")

    # Run diagnostic analysis
    print(f"\n{'='*60}")
    print(f"Running diagnostic analysis...")
    compute_diagnostic(result, embeddings, output_dir)


if __name__ == "__main__":
    main()