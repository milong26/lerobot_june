#!/usr/bin/env python
"""
Our V5 Episode Selection

Loads V5 vision-action embeddings (episode_embedding) and runs iterative SIC selection.
Adapts V5 single-vector embeddings to the format expected by the existing SIC selection logic.

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
    Load V5 embeddings and adapt to phi_global/phi_wrist format.
    
    V5 uses 'episode_embedding' key (single vector).
    We map it to both phi_global and phi_wrist for compatibility with SIC.
    
    Args:
        embeddings_dir: V5 embedding cache directory
    
    Returns:
        Dict[episode_index, {"phi_global": ..., "phi_wrist": ...}]
    """
    embeddings = {}
    
    for f in sorted(embeddings_dir.glob("*.npy")):
        try:
            data = np.load(f, allow_pickle=True).item()
            ep_idx = data.get("episode_index")
            if ep_idx is None:
                continue
            
            episode_embedding = data["episode_embedding"]
            
            embeddings[ep_idx] = {
                "phi_global": episode_embedding,
                "phi_wrist": episode_embedding,
            }
        except Exception as e:
            print(f"  Skipping invalid file: {f.name} ({e})")
    
    print(f"Loaded {len(embeddings)} V5 episode embeddings")
    return embeddings


def select_initial_b0(all_episode_indices: List[int], b0_size: int, seed: int = 42) -> List[int]:
    """Uniformly sample initial B0 episodes."""
    rng = np.random.RandomState(seed)
    selected = rng.choice(all_episode_indices, size=b0_size, replace=False).tolist()
    selected.sort()
    return selected


def compute_sic_for_candidate(
    selected_episodes: List[int],
    candidate_idx: int,
    embeddings: Dict[int, Dict],
    alpha: float = 1.0,
    lambda_wrist: float = 1.0
) -> float:
    """Compute SIC score when adding a candidate episode."""
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


def iterative_select_episodes(
    embeddings: Dict[int, Dict],
    b0_size: int = 18,
    target_size: int = 112,
    n_add_per_round: int = 9,
    seed: int = 42,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0
) -> Dict:
    """Iterative SIC episode selection using V5 embeddings."""
    all_episode_indices = sorted(embeddings.keys())
    n_total = len(all_episode_indices)
    
    print(f"\n{'='*60}")
    print(f"V5 Iterative SIC Episode Selection")
    print(f"{'='*60}")
    print(f"Total episodes: {n_total}")
    print(f"Initial B0 size: {b0_size}")
    print(f"Target size: {target_size}")
    print(f"Add per round: {n_add_per_round}")
    print(f"alpha: {alpha}, lambda_wrist: {lambda_wrist}")
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
        
        # Compute SIC gain for each candidate
        candidate_scores = []
        
        for candidate_idx in remaining_episodes:
            sic_with_candidate = compute_sic_for_candidate(
                selected_episodes, candidate_idx, embeddings, alpha, lambda_wrist
            )
            candidate_scores.append((candidate_idx, sic_with_candidate))
        
        # Sort by SIC score, select best n_to_add
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_candidates = []
        for i in range(n_to_add):
            if i >= len(candidate_scores):
                break
            best_idx, best_sic = candidate_scores[i]
            selected_episodes.append(best_idx)
            remaining_episodes.remove(best_idx)
            best_candidates.append({
                "episode_index": best_idx,
                "sic_score": best_sic
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
        
        selection_log.append({
            "round": round_num,
            "n_selected": len(selected_episodes),
            "n_added": len(best_candidates),
            "sic_score": current_sic,
            "sic_gain": current_sic - sic_history[-2] if len(sic_history) > 1 else 0,
            "time_seconds": round_time,
            "added_episodes": best_candidates
        })
        
        print(f"    Added {len(best_candidates)} episodes")
        print(f"    Current SIC: {current_sic:.4f} (gain: {current_sic - sic_history[-2]:.4f})")
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
        "seed": seed
    }


def main():
    parser = argparse.ArgumentParser(description="V5 Episode Selection with SIC")
    parser.add_argument("--embeddings-dir", type=str, required=True, help="V5 embedding cache directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--b0-size", type=int, default=18, help="Initial B0 size (default: 18)")
    parser.add_argument("--target-size", type=int, default=112, help="Target episode count (default: 112)")
    parser.add_argument("--n-add-per-round", type=int, default=9, help="Episodes to add per round (default: 9)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", type=float, default=1.0, help="SIC smoothing coefficient")
    parser.add_argument("--lambda-wrist", type=float, default=1.0, help="Wrist weight")
    
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
    )
    
    # Save subset file
    subset_file = output_dir / "subset.json"
    subset_data = {
        "method": "our_v5",
        "num_episodes": args.target_size,
        "seed": args.seed,
        "selected_episode_indices": result["selected_episodes"],
        "parameters": {
            "b0_size": args.b0_size,
            "n_add_per_round": args.n_add_per_round,
            "alpha": args.alpha,
            "lambda_wrist": args.lambda_wrist,
        }
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