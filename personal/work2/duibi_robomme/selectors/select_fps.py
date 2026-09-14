#!/usr/bin/env python
"""
Visual-FPS (Farthest Point Sampling) for Robomme pipeline.

Adapted from personal/work2/duibi/fps/select_visual_fps.py

For Robomme, we use real visual embeddings (phi_global + phi_wrist) from
the shared embedding cache. This is the same cache used by Ours-v5.

Algorithm (standard greedy Farthest Point Sampling):
  1. Randomly select first point from ALL candidates using np.random.RandomState(seed)
  2. For each remaining candidate, compute min_dist[i] = ||v_i - v_first||_2
  3. Iteratively select argmax_i min_dist[i] (farthest from current selected set)
  4. Incrementally update: min_dist[i] = min(min_dist[i], ||v_i - v_new||_2)
  5. Tie-break: smaller episode index wins (deterministic)

Feature definition: visual_feature = np.concatenate([phi_global, phi_wrist])
  - Same as Ours-v5's visual representation
  - No L2 normalization, standardization, whitening, re-PCA, or cosine normalization
  - Euclidean L2 distance

FPS does NOT use: configuration feature, action descriptor, observation.state,
reward, success, or any other data.

Usage:
    python select_fps.py \
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

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Allow direct execution (python script.py) to find sibling modules
_selectors_dir = Path(__file__).resolve().parent
if str(_selectors_dir) not in sys.path:
    sys.path.insert(0, str(_selectors_dir))

from robomme_embeddings import load_cached_visual_embeddings


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def farthest_point_sampling(
    candidate_ids: List[int],
    features: Dict[int, np.ndarray],
    num_selected: int,
    seed: int,
) -> Dict:
    """
    Standard greedy Farthest Point Sampling.

    Algorithm:
      1. Select first point uniformly at random from all candidates using seed
      2. Initialize min_dist[i] = ||v_i - v_first||_2 for all remaining
      3. Each round: select argmax_i min_dist[i], update min_dist incrementally
      4. Tie-break: smaller episode index
    """
    rng = np.random.RandomState(seed)
    n_candidates = len(candidate_ids)

    if num_selected <= 0:
        raise ValueError("num_selected must be > 0")
    if num_selected > n_candidates:
        raise ValueError(f"num_selected ({num_selected}) > candidate_count ({n_candidates})")

    selection_order = []
    selection_trace = []
    remaining = set(range(n_candidates))

    # Step 1: Random first point
    first_local_idx = int(rng.choice(n_candidates))
    first_ep_id = candidate_ids[first_local_idx]
    selection_order.append(first_ep_id)
    remaining.remove(first_local_idx)

    selection_trace.append({
        "step": 0,
        "episode_index": int(first_ep_id),
        "farthest_min_distance": None,
        "remaining_candidates": len(remaining),
    })

    print(f"  FPS seed point: episode {first_ep_id} (local index {first_local_idx})")

    # Initialize min_dist for all remaining candidates
    first_feature = features[first_ep_id]
    min_dist = np.full(n_candidates, np.inf)

    for local_idx in remaining:
        ep_id = candidate_ids[local_idx]
        dist = np.linalg.norm(features[ep_id] - first_feature)
        min_dist[local_idx] = dist

    # Steps 2..N: Greedy farthest point selection
    for step in range(1, num_selected):
        best_local_idx = None
        best_dist = -1.0

        remaining_sorted = sorted(remaining)
        for local_idx in remaining_sorted:
            d = min_dist[local_idx]
            if d > best_dist + 1e-12:
                best_dist = d
                best_local_idx = local_idx
            elif abs(d - best_dist) <= 1e-12:
                if best_local_idx is None or candidate_ids[local_idx] < candidate_ids[best_local_idx]:
                    best_local_idx = local_idx
                    best_dist = d

        if best_local_idx is None:
            raise RuntimeError(f"No candidate found at step {step}")

        selected_ep_id = candidate_ids[best_local_idx]
        selection_order.append(selected_ep_id)
        remaining.remove(best_local_idx)

        selection_trace.append({
            "step": step,
            "episode_index": int(selected_ep_id),
            "farthest_min_distance": float(best_dist),
            "remaining_candidates": len(remaining),
        })

        if step % 10 == 0 or step == 1:
            print(f"  Step {step}: selected episode {selected_ep_id}, "
                  f"farthest_min_distance={best_dist:.6f}, "
                  f"remaining={len(remaining)}")

        # Incremental update
        new_feature = features[selected_ep_id]
        for local_idx in remaining:
            ep_id = candidate_ids[local_idx]
            new_distance = np.linalg.norm(features[ep_id] - new_feature)
            if new_distance < min_dist[local_idx]:
                min_dist[local_idx] = new_distance

    selected_ids_sorted = sorted(selection_order)

    return {
        "selection_order": selection_order,
        "selected_episode_indices": selected_ids_sorted,
        "selection_trace": selection_trace,
    }


def select_fps_episodes(
    num_episodes: int,
    seed: int,
    dataset_root: str,
    task_name: str,
    output_dir: Optional[str] = None,
) -> List[int]:
    """Main entry point for Visual-FPS selection on Robomme datasets."""
    print(f"Loading LeRobotDataset from {dataset_root}")
    dataset = LeRobotDataset(repo_id="1/2", root=dataset_root)

    total_episodes = dataset.num_episodes
    print(f"Total episodes: {total_episodes}")

    candidate_ids = list(range(total_episodes))

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

    # Filter candidates to those with valid visual embeddings
    valid_candidates = [ep for ep in candidate_ids if ep in visual_embeddings]
    print(f"Valid candidates with visual embeddings: {len(valid_candidates)}")

    if len(valid_candidates) == 0:
        raise RuntimeError("No valid visual embeddings found for any episode")

    print(f"\n{'='*60}")
    print(f"Running Visual Farthest Point Sampling...")
    print(f"{'='*60}")
    start_time = time.time()

    result = farthest_point_sampling(
        candidate_ids=valid_candidates,
        features=visual_embeddings,
        num_selected=num_episodes,
        seed=seed,
    )

    elapsed_time = time.time() - start_time
    print(f"\nFPS completed in {elapsed_time:.2f}s")

    selected_episode_indices = result["selected_episode_indices"]

    # Verify we got exactly K distinct episodes
    assert len(selected_episode_indices) == num_episodes, (
        f"Selected {len(selected_episode_indices)} episodes, expected {num_episodes}"
    )
    assert len(set(selected_episode_indices)) == num_episodes, "Duplicate episodes selected"

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_file = output_path / f"fps_{num_episodes}_seed{seed}.json"
        subset_data = {
            "method": "visual_fps",
            "selection_method": "visual_farthest_point_sampling",
            "num_episodes": num_episodes,
            "seed": seed,
            "candidate_count": len(valid_candidates),
            "feature_definition": "concat(phi_global, phi_wrist)",
            "distance_metric": "euclidean",
            "feature_dim": next(iter(visual_embeddings.values())).shape[0],
            "initial_episode": int(result["selection_order"][0]),
            "selection_order": result["selection_order"],
            "selected_episode_indices": selected_episode_indices,
        }
        with open(subset_file, "w") as f:
            json.dump(subset_data, f, indent=2, cls=NumpyEncoder)
        print(f"Saved {num_episodes} FPS episodes (seed={seed}) to {subset_file}")

    return selected_episode_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual-FPS Episode Selection")
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--task-name", type=str, required=True,
                       choices=["MoveCube_easy", "PatternLock_medium", "RouteStick_hard"])
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    select_fps_episodes(
        num_episodes=args.num_episodes,
        seed=args.seed,
        dataset_root=args.dataset_root,
        task_name=args.task_name,
        output_dir=args.output_dir,
    )