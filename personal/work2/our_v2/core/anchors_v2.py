"""
V2 Sequential Greedy Selection Module

Implements sequential greedy episode selection with marginal gain recomputation
after each selection step.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_v2.core.sic_v2 import (
    compute_bandwidth,
    compute_sic_total,
    compute_candidate_score,
)


def initialize_seed_set(
    all_indices: List[int],
    seed_size: int,
    seed: int = 42,
) -> List[int]:
    """
    Initialize a random seed set for greedy selection.

    Args:
        all_indices: all available episode indices.
        seed_size: number of seed episodes.
        seed: random seed.

    Returns:
        List of seed episode indices.
    """
    rng = np.random.RandomState(seed)
    selected = rng.choice(all_indices, size=seed_size, replace=False).tolist()
    selected.sort()
    return selected


def select_next_episode(
    selected_indices: List[int],
    candidate_indices: List[int],
    embeddings: Dict[int, np.ndarray],
    bandwidth: float,
    k: int = 5,
) -> tuple:
    """
    Select the next best episode from candidates based on score.

    Args:
        selected_indices: currently selected episode indices.
        candidate_indices: remaining candidate episode indices.
        embeddings: episode embedding dict.
        bandwidth: kernel bandwidth.
        k: kNN parameter for redundancy.

    Returns:
        (best_idx, best_score, marginal_gain, redundancy_penalty)
    """
    best_idx = None
    best_score = -float("inf")
    best_gain = 0.0
    best_redundancy = 0.0

    for cand_idx in candidate_indices:
        from our_v2.core.sic_v2 import compute_sic_gain, compute_redundancy_penalty

        gain = compute_sic_gain(selected_indices, cand_idx, embeddings, bandwidth)
        redundancy = compute_redundancy_penalty(cand_idx, selected_indices, embeddings, k)
        score = gain / (1.0 + redundancy)

        if score > best_score:
            best_score = score
            best_idx = cand_idx
            best_gain = gain
            best_redundancy = redundancy

    return best_idx, best_score, best_gain, best_redundancy


def sequential_greedy_selection(
    embeddings: Dict[int, np.ndarray],
    initial_selected: List[int],
    target_size: int,
    k: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Sequential greedy selection: pick one episode at a time, recompute gains.

    Args:
        embeddings: episode embedding dict.
        initial_selected: initial seed episode indices.
        target_size: target number of selected episodes.
        k: kNN parameter for redundancy.
        verbose: whether to print progress.

    Returns:
        Dict with selected_episode_indices and selection log.
    """
    all_indices = sorted(embeddings.keys())
    selected = list(initial_selected)
    remaining = [i for i in all_indices if i not in selected]

    bandwidth = compute_bandwidth(embeddings, k=k)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Sequential Greedy Selection (v2)")
        print(f"{'='*60}")
        print(f"Total episodes: {len(all_indices)}")
        print(f"Initial seed: {len(selected)}")
        print(f"Target size: {target_size}")
        print(f"Bandwidth: {bandwidth:.4f}")
        print(f"{'='*60}")

    current_sic = compute_sic_total(selected, embeddings, bandwidth)
    selection_log = []
    step = 0

    while len(selected) < target_size and remaining:
        step += 1
        best_idx, best_score, best_gain, best_redundancy = select_next_episode(
            selected, remaining, embeddings, bandwidth, k
        )

        if best_idx is None:
            if verbose:
                print(f"  No valid candidate found, stopping early.")
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

        new_sic = compute_sic_total(selected, embeddings, bandwidth)
        sic_gain = new_sic - current_sic

        log_entry = {
            "step": step,
            "selected_episode_index": int(best_idx),
            "marginal_gain": float(best_gain),
            "redundancy_penalty": float(best_redundancy),
            "score": float(best_score),
            "current_sic": float(new_sic),
            "sic_gain": float(sic_gain),
            "n_selected": len(selected),
        }
        selection_log.append(log_entry)
        current_sic = new_sic

        if verbose:
            print(
                f"  Step {step:3d}: "
                f"selected ep={best_idx:4d}, "
                f"gain={best_gain:.4f}, "
                f"redundancy={best_redundancy:.4f}, "
                f"SIC={new_sic:.4f}"
            )

    selected.sort()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Selection complete!")
        print(f"Final selected: {len(selected)} episodes")
        print(f"Final SIC: {current_sic:.4f}")
        print(f"{'='*60}")

    return {
        "selected_episode_indices": selected,
        "selection_log": selection_log,
        "final_sic": float(current_sic),
        "bandwidth": float(bandwidth),
        "target_size": target_size,
        "n_selected": len(selected),
    }


def save_subset(result: dict, output_path: Path):
    """
    Save subset in JSON format compatible with train/eval code.

    Args:
        result: result dict from sequential_greedy_selection.
        output_path: output JSON file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subset_data = {
        "selected_episode_indices": result["selected_episode_indices"],
        "num_episodes": result["n_selected"],
        "selection_method": "dynamic_anchor_v2",
        "parameters": {
            "target_size": result["target_size"],
            "bandwidth": result["bandwidth"],
        },
    }

    with open(output_path, "w") as f:
        json.dump(subset_data, f, indent=2)

    print(f"Subset saved to: {output_path}")