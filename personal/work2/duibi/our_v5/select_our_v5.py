#!/usr/bin/env python
"""
Our V5 Episode Selection - Visual Coverage + Action Diversity

Selects episodes based on joint scoring of:
- Visual coverage score (from existing global+wrist embeddings)
- Action diversity score (from action descriptors)

Does NOT use: success, grasp_success, eval results, attention results, test results.

Usage:
    python select_our_v5.py \
        --visual-embedding-dir /path/to/visual/embeddings \
        --action-descriptor-dir /path/to/action/descriptors \
        --output-dir /path/to/output \
        --num-selected 112 \
        --seed 42 \
        --visual-weight 0.5 \
        --action-weight 0.5
"""

import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

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


def compute_v5_selection_score(
    candidate_visual_embedding: np.ndarray,
    candidate_action_descriptor: np.ndarray,
    selected_visual_embeddings: np.ndarray,
    selected_action_descriptors: np.ndarray,
    visual_weight: float = 0.5,
    action_weight: float = 0.5,
) -> float:
    """
    Compute combined V5 selection score.

    Score = visual_weight * visual_coverage_score + action_weight * action_diversity_score

    Args:
        candidate_visual_embedding: candidate visual feature
        candidate_action_descriptor: candidate action descriptor
        selected_visual_embeddings: selected visual features
        selected_action_descriptors: selected action descriptors
        visual_weight: weight for visual coverage
        action_weight: weight for action diversity

    Returns:
        combined selection score
    """
    visual_score = compute_visual_coverage_score(candidate_visual_embedding, selected_visual_embeddings)
    action_score = compute_action_diversity_score(candidate_action_descriptor, selected_action_descriptors)

    return visual_weight * visual_score + action_weight * action_score


def select_episodes_v5(
    all_episode_ids: List[int],
    visual_embeddings: Dict[int, np.ndarray],
    action_descriptors: Dict[int, np.ndarray],
    num_select: int,
    visual_weight: float = 0.5,
    action_weight: float = 0.5,
    seed: int = 42,
    b0_size: int = 18,
    n_add_per_round: int = 9,
) -> Dict:
    """
    Execute complete V5 selection flow.

    Flow:
    1. All episodes -> filter to those with both visual and action data
    2. Visual coverage candidate selection (initial B0 via uniform sampling)
    3. Action diversity reranking (iterative greedy selection)
    4. Final selected episode ids

    Args:
        all_episode_ids: list of all episode indices
        visual_embeddings: Dict[ep_idx, visual_feature]
        action_descriptors: Dict[ep_idx, action_descriptor]
        num_select: target number of episodes to select
        visual_weight: weight for visual coverage score
        action_weight: weight for action diversity score
        seed: random seed
        b0_size: initial uniform sample size
        n_add_per_round: episodes to add per iteration round

    Returns:
        Dict with selected_episode_ids and selection metadata
    """
    valid_ids = [
        ep_id for ep_id in all_episode_ids
        if ep_id in visual_embeddings and ep_id in action_descriptors
    ]
    valid_ids.sort()

    print(f"\n{'='*60}")
    print(f"V5 Episode Selection - Visual Coverage + Action Diversity")
    print(f"{'='*60}")
    print(f"Total episodes: {len(all_episode_ids)}")
    print(f"Valid episodes (with both visual+action): {len(valid_ids)}")
    print(f"Target selection: {num_select}")
    print(f"Visual weight: {visual_weight}, Action weight: {action_weight}")
    print(f"B0 size: {b0_size}, Add per round: {n_add_per_round}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")

    rng = np.random.RandomState(seed)

    b0_candidates = rng.choice(valid_ids, size=min(b0_size, len(valid_ids)), replace=False).tolist()
    b0_candidates.sort()

    selected_ids = list(b0_candidates)
    remaining_ids = [ep for ep in valid_ids if ep not in selected_ids]

    selection_log = []
    round_num = 0

    while len(selected_ids) < num_select and remaining_ids:
        round_num += 1
        round_start = time.time()

        n_to_add = min(n_add_per_round, num_select - len(selected_ids))

        selected_visual = np.array([visual_embeddings[ep] for ep in selected_ids])
        selected_action = np.array([action_descriptors[ep] for ep in selected_ids])

        candidate_scores = []
        for candidate_id in remaining_ids:
            score = compute_v5_selection_score(
                visual_embeddings[candidate_id],
                action_descriptors[candidate_id],
                selected_visual,
                selected_action,
                visual_weight,
                action_weight,
            )
            candidate_scores.append((candidate_id, score))

        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        added_episodes = []
        for i in range(n_to_add):
            if i >= len(candidate_scores):
                break
            best_id, best_score = candidate_scores[i]
            selected_ids.append(best_id)
            remaining_ids.remove(best_id)
            added_episodes.append({
                "episode_index": best_id,
                "selection_score": float(best_score),
            })

        round_time = time.time() - round_start

        selection_log.append({
            "round": round_num,
            "n_selected": len(selected_ids),
            "n_added": len(added_episodes),
            "time_seconds": round_time,
            "added_episodes": added_episodes,
        })

        print(f"  Round {round_num}: added {len(added_episodes)} episodes, total={len(selected_ids)}, time={round_time:.2f}s")

    selected_ids.sort()

    print(f"\n{'='*60}")
    print(f"V5 Selection Complete!")
    print(f"{'='*60}")
    print(f"Selected {len(selected_ids)} episodes")
    print(f"Total rounds: {round_num}")

    return {
        "selected_episode_indices": selected_ids,
        "b0_episodes": b0_candidates,
        "selection_log": selection_log,
        "total_rounds": round_num,
        "num_selected": len(selected_ids),
        "num_valid": len(valid_ids),
        "num_total": len(all_episode_ids),
        "visual_weight": visual_weight,
        "action_weight": action_weight,
        "seed": seed,
        "b0_size": b0_size,
        "n_add_per_round": n_add_per_round,
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
        "num_episodes": len(selected_episode_ids),
        "selected_episode_indices": selected_episode_ids,
    }

    if metadata:
        subset_data["parameters"] = metadata

    with open(output_path, "w") as f:
        json.dump(subset_data, f, indent=2)

    print(f"Selected episodes saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="V5 Episode Selection - Visual Coverage + Action Diversity")
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
                       help="Initial B0 uniform sample size (default: 18)")
    parser.add_argument("--n-add-per-round", type=int, default=9,
                       help="Episodes to add per round (default: 9)")

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

    result = select_episodes_v5(
        all_episode_ids=all_episode_ids,
        visual_embeddings=visual_embeddings,
        action_descriptors=action_descriptors,
        num_select=args.num_selected,
        visual_weight=args.visual_weight,
        action_weight=args.action_weight,
        seed=args.seed,
        b0_size=args.b0_size,
        n_add_per_round=args.n_add_per_round,
    )

    subset_file = output_dir / "subsets" / f"our_v5_{args.num_selected}_seed{args.seed}.json"
    save_selected_episodes(
        selected_episode_ids=result["selected_episode_indices"],
        output_path=subset_file,
        metadata={
            "visual_weight": args.visual_weight,
            "action_weight": args.action_weight,
            "b0_size": args.b0_size,
            "n_add_per_round": args.n_add_per_round,
            "seed": args.seed,
        }
    )

    log_file = output_dir / "results" / f"selection_log_v5_{args.num_selected}_seed{args.seed}.json"
    with open(log_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Selection log saved to: {log_file}")


if __name__ == "__main__":
    main()