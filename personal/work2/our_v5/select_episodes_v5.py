#!/usr/bin/env python
"""
Our V5 Episode Selection

Loads V5 action-aware embeddings (global_embedding, wrist_embedding, action_descriptor)
and runs iterative joint visual-coverage + action-diversity SIC selection.

Usage:
    python select_episodes_v5.py \
        --embeddings-dir /path/to/v5/embeddings \
        --output-dir /path/to/output \
        --target-size 112 \
        --seed 42
"""

import sys
import os
import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List

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


def select_initial_b0(all_episode_indices: List[int], b0_size: int, seed: int = 42) -> List[int]:
    """Uniformly sample initial B0 episodes."""
    rng = np.random.RandomState(seed)
    selected = rng.choice(all_episode_indices, size=b0_size, replace=False).tolist()
    selected.sort()
    return selected


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


def compute_sic_for_candidate(
    selected_episodes: List[int],
    candidate_idx: int,
    embeddings: Dict[int, Dict],
    alpha: float = 1.0,
    lambda_wrist: float = 1.0,
    visual_weight: float = 0.7,
    action_weight: float = 0.3,
) -> Dict[str, float]:
    """
    Compute joint SIC score combining visual coverage and action diversity.
    
    Args:
        selected_episodes: list of already selected episode indices
        candidate_idx: candidate episode index to evaluate
        embeddings: episode embeddings dict
        alpha: SIC smoothing coefficient
        lambda_wrist: wrist camera weight
        visual_weight: weight for visual coverage score
        action_weight: weight for action diversity score
    
    Returns:
        Dict with keys: "visual_score", "action_score", "joint_score"
    """
    visual_score = compute_visual_coverage_score(
        selected_episodes, candidate_idx, embeddings, alpha, lambda_wrist
    )
    action_score = compute_action_diversity_score(
        selected_episodes, candidate_idx, embeddings
    )
    
    joint_score = visual_weight * visual_score + action_weight * action_score
    
    return {
        "visual_score": visual_score,
        "action_score": action_score,
        "joint_score": joint_score,
    }


def iterative_select_episodes(
    embeddings: Dict[int, Dict],
    b0_size: int = 18,
    target_size: int = 112,
    n_add_per_round: int = 9,
    seed: int = 42,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0,
    visual_weight: float = 0.7,
    action_weight: float = 0.3,
) -> Dict:
    """Iterative SIC episode selection using V5 action-aware embeddings."""
    all_episode_indices = sorted(embeddings.keys())
    n_total = len(all_episode_indices)
    
    print(f"\n{'='*60}")
    print(f"V5 Action-Aware Iterative SIC Episode Selection")
    print(f"{'='*60}")
    print(f"Total episodes: {n_total}")
    print(f"Initial B0 size: {b0_size}")
    print(f"Target size: {target_size}")
    print(f"Add per round: {n_add_per_round}")
    print(f"alpha: {alpha}, lambda_wrist: {lambda_wrist}")
    print(f"visual_weight: {visual_weight}, action_weight: {action_weight}")
    print(f"{'='*60}")
    
    # Step 1: Select initial B0
    print(f"\n[Step 1] Selecting initial B0 ({b0_size} episodes)...")
    b0_episodes = select_initial_b0(all_episode_indices, b0_size, seed)
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
    
    # Step 2: Iterative selection
    print(f"\n[Step 2] Starting iterative selection...")
    selection_log = []
    sic_history = [initial_sic]
    round_num = 0
    
    while len(selected_episodes) < target_size and remaining_episodes:
        round_num += 1
        round_start = time.time()
        
        n_to_add = min(n_add_per_round, target_size - len(selected_episodes))
        
        print(f"\n  Round {round_num}: selected {len(selected_episodes)}, need to add {n_to_add}")
        
        # Compute joint score for each candidate
        candidate_scores = []
        
        for candidate_idx in remaining_episodes:
            scores = compute_sic_for_candidate(
                selected_episodes, candidate_idx, embeddings,
                alpha, lambda_wrist, visual_weight, action_weight
            )
            candidate_scores.append((candidate_idx, scores))
        
        # Sort by joint score, select best n_to_add
        candidate_scores.sort(key=lambda x: x[1]["joint_score"], reverse=True)
        
        best_candidates = []
        for i in range(n_to_add):
            if i >= len(candidate_scores):
                break
            best_idx, scores = candidate_scores[i]
            selected_episodes.append(best_idx)
            remaining_episodes.remove(best_idx)
            best_candidates.append({
                "episode_index": best_idx,
                "visual_score": scores["visual_score"],
                "action_score": scores["action_score"],
                "joint_score": scores["joint_score"],
            })
        
        # Recompute current SIC
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
        
        round_time = time.time() - round_start
        
        avg_visual = np.mean([c["visual_score"] for c in best_candidates])
        avg_action = np.mean([c["action_score"] for c in best_candidates])
        avg_joint = np.mean([c["joint_score"] for c in best_candidates])
        
        selection_log.append({
            "round": round_num,
            "n_selected": len(selected_episodes),
            "n_added": len(best_candidates),
            "sic_score": current_sic,
            "sic_gain": current_sic - sic_history[-2] if len(sic_history) > 1 else 0,
            "avg_visual_score": float(avg_visual),
            "avg_action_score": float(avg_action),
            "avg_joint_score": float(avg_joint),
            "time_seconds": round_time,
            "added_episodes": best_candidates,
        })
        
        print(f"    Added {len(best_candidates)} episodes")
        print(f"    Current SIC: {current_sic:.4f} (gain: {current_sic - sic_history[-2]:.4f})")
        print(f"    Avg visual: {avg_visual:.4f}, Avg action: {avg_action:.4f}, Avg joint: {avg_joint:.4f}")
        print(f"    Time: {round_time:.2f}s")
    
    # Step 3: Output results
    print(f"\n{'='*60}")
    print(f"Iterative selection complete!")
    print(f"{'='*60}")
    print(f"Final selected episodes: {len(selected_episodes)}")
    print(f"Final SIC score: {sic_history[-1]:.4f}")
    print(f"SIC gain: {sic_history[-1] - initial_sic:.4f}")
    print(f"Total rounds: {round_num}")
    
    return {
        "selected_episodes": sorted(selected_episodes),
        "b0_episodes": b0_episodes,
        "selection_log": selection_log,
        "sic_history": sic_history,
        "initial_sic": initial_sic,
        "final_sic": sic_history[-1],
        "total_rounds": round_num,
        "target_size": target_size,
        "b0_size": b0_size,
        "n_add_per_round": n_add_per_round,
        "alpha": alpha,
        "lambda_wrist": lambda_wrist,
        "visual_weight": visual_weight,
        "action_weight": action_weight,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description="V5 Action-Aware Episode Selection with SIC")
    parser.add_argument("--embeddings-dir", type=str, required=True, help="V5 embedding cache directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--b0-size", type=int, default=18, help="Initial B0 size (default: 18)")
    parser.add_argument("--target-size", type=int, default=112, help="Target episode count (default: 112)")
    parser.add_argument("--n-add-per-round", type=int, default=9, help="Episodes to add per round (default: 9)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", type=float, default=1.0, help="SIC smoothing coefficient")
    parser.add_argument("--lambda-wrist", type=float, default=1.0, help="Wrist weight")
    parser.add_argument("--visual-weight", type=float, default=0.7, help="Weight for visual coverage score (default: 0.7)")
    parser.add_argument("--action-weight", type=float, default=0.3, help="Weight for action diversity score (default: 0.3)")
    
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
    
    result = iterative_select_episodes(
        embeddings=embeddings,
        b0_size=args.b0_size,
        target_size=args.target_size,
        n_add_per_round=args.n_add_per_round,
        seed=args.seed,
        alpha=args.alpha,
        lambda_wrist=args.lambda_wrist,
        visual_weight=args.visual_weight,
        action_weight=args.action_weight,
    )
    
    # Compute visual and action contribution statistics
    all_visual_scores = []
    all_action_scores = []
    for entry in result["selection_log"]:
        for ep in entry.get("added_episodes", []):
            all_visual_scores.append(ep["visual_score"])
            all_action_scores.append(ep["action_score"])
    
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
        "selection_method": "our_v5_action_aware",
        "num_episodes": args.target_size,
        "seed": args.seed,
        "visual_weight": args.visual_weight,
        "action_weight": args.action_weight,
        "uses_action_descriptor": True,
        "selected_episode_indices": result["selected_episodes"],
        "parameters": {
            "b0_size": args.b0_size,
            "n_add_per_round": args.n_add_per_round,
            "alpha": args.alpha,
            "lambda_wrist": args.lambda_wrist,
            "visual_weight": args.visual_weight,
            "action_weight": args.action_weight,
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


if __name__ == "__main__":
    main()