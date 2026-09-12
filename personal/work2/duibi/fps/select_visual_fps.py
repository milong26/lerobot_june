#!/usr/bin/env python
"""
Visual-FPS (Farthest Point Sampling) Baseline

Selects episodes based purely on global feature-space diversity using existing
visual embeddings. This is a strict control-variable baseline to answer:
"Is the performance improvement merely from selecting more dispersed visual data?"

FPS is defined as:
  - Only uses existing visual embedding cache (phi_global + phi_wrist)
  - Does NOT use: rand_vec, configuration region, obj_init_pos, goal_pose,
    environment_state, action descriptor, robot action sequence, success,
    grasp_success, reward, policy evaluation, attention, checkpoint performance,
    or any test-stage information
  - Does NOT implement: action-FPS, multimodal-FPS, configuration-FPS
  - Does NOT use: region-level priority, coverage gap, visual uncertainty,
    action uncertainty, or B0 region initialization from our_v5

Algorithm (standard greedy Farthest Point Sampling):
  1. Randomly select first point from ALL candidates using np.random.RandomState(seed)
  2. For each remaining candidate, compute min_dist[i] = ||v_i - v_first||_2
  3. Iteratively select argmax_i min_dist[i] (farthest from current selected set)
  4. Incrementally update: min_dist[i] = min(min_dist[i], ||v_i - v_new||_2)
  5. Tie-break: smaller episode index wins (deterministic)

Feature definition: visual_feature = np.concatenate([phi_global, phi_wrist])
  - Same as our_v5's visual representation
  - No L2 normalization, standardization, whitening, re-PCA, or cosine normalization
  - Euclidean L2 distance (same geometry as our_v5 visual novelty)

Cache path: resolved via get_shared_embedding_dir() from embedding_utils.cache
  Default: /data/.../shared_embeddings/{dataset_name}/smolvlm2-500m_last-hidden-tokenmean_global-first5_wrist-20to70_temporal-mean_pca32_v1/

Usage:
    python select_visual_fps.py \
        --dataset-dir /data/.../dataset_view/pick_place_corner \
        --dataset-name pick_place_corner \
        --output-dir /path/to/output \
        --num-selected 112 \
        --seed 42 \
        --pca-dim 32
"""

import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in path for embedding_utils imports
# File is at: personal/work2/duibi/fps/select_visual_fps.py
# WORK2_ROOT should be: personal/work2 (parent of embedding_utils)
WORK2_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

from embedding_utils.cache import (
    get_shared_embedding_dir,
    get_dataset_episode_indices,
    validate_embedding_file,
)
from embedding_utils.config import (
    DEFAULT_PCA_DIM,
    build_extraction_method_name,
)


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


def load_episode_visual_feature(ep_file: Path, expected_ep_idx: int) -> Optional[np.ndarray]:
    """
    Load and validate a single episode visual feature from cache.

    Returns concatenated [phi_global, phi_wrist] or None if invalid.
    """
    valid, issues = validate_embedding_file(ep_file, expected_ep_idx)
    if not valid:
        for issue in issues:
            print(f"  WARNING: {issue}")
        return None

    data = np.load(str(ep_file), allow_pickle=True).item()

    if "episode_index" in data:
        file_ep_idx = int(data["episode_index"])
        if file_ep_idx != expected_ep_idx:
            print(f"  WARNING: episode_index mismatch in {ep_file.name}: "
                  f"expected {expected_ep_idx}, got {file_ep_idx}")
            return None

    phi_global = data["phi_global"]
    phi_wrist = data["phi_wrist"]

    if not isinstance(phi_global, np.ndarray) or phi_global.ndim != 1 or len(phi_global) == 0:
        print(f"  WARNING: phi_global in {ep_file.name} is invalid")
        return None
    if not isinstance(phi_wrist, np.ndarray) or phi_wrist.ndim != 1 or len(phi_wrist) == 0:
        print(f"  WARNING: phi_wrist in {ep_file.name} is invalid")
        return None
    if not np.all(np.isfinite(phi_global)):
        print(f"  WARNING: phi_global in {ep_file.name} contains non-finite values")
        return None
    if not np.all(np.isfinite(phi_wrist)):
        print(f"  WARNING: phi_wrist in {ep_file.name} contains non-finite values")
        return None

    visual_feature = np.concatenate([phi_global, phi_wrist])
    return visual_feature


def load_all_visual_features(
    cache_dir: Path,
    candidate_ids: List[int],
) -> Tuple[Dict[int, np.ndarray], int]:
    """
    Load visual features for all candidate episodes from shared cache.

    Strict validation:
      - Every candidate must have a corresponding ({index}).npy file
      - Missing files cause fail-fast exit
      - All features must have consistent dimensions after concatenation

    Returns:
        (features_dict, feature_dim)
    """
    features = {}
    missing_ids = []

    for ep_idx in candidate_ids:
        ep_file = cache_dir / f"({ep_idx}).npy"
        if not ep_file.exists():
            missing_ids.append(ep_idx)
            continue

        feat = load_episode_visual_feature(ep_file, ep_idx)
        if feat is None:
            print(f"  ERROR: Failed to load valid feature for episode {ep_idx}")
            sys.exit(1)
        features[ep_idx] = feat

    if missing_ids:
        print(f"\n{'='*60}")
        print(f"FAIL-FAST: Missing cache files for {len(missing_ids)} episodes")
        print(f"Missing episode IDs: {missing_ids}")
        print(f"Cache directory: {cache_dir}")
        print(f"Expected files: ({', '.join(str(i) for i in missing_ids)}).npy")
        print(f"{'='*60}")
        sys.exit(1)

    if len(features) == 0:
        print(f"ERROR: No valid features loaded from {cache_dir}")
        sys.exit(1)

    feature_dim = None
    for ep_idx, feat in features.items():
        if feature_dim is None:
            feature_dim = feat.shape[0]
        elif feat.shape[0] != feature_dim:
            print(f"ERROR: Feature dimension mismatch for episode {ep_idx}: "
                  f"expected {feature_dim}, got {feat.shape[0]}")
            sys.exit(1)

    return features, feature_dim


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

    Args:
        candidate_ids: list of candidate episode indices
        features: Dict[ep_idx, visual_feature_array]
        num_selected: target number of episodes to select
        seed: random seed for first point selection

    Returns:
        Dict with selection_order, selected_episode_indices, selection_trace
    """
    rng = np.random.RandomState(seed)
    n_candidates = len(candidate_ids)

    if num_selected <= 0:
        print("ERROR: num_selected must be > 0")
        sys.exit(1)
    if num_selected > n_candidates:
        print(f"ERROR: num_selected ({num_selected}) > candidate_count ({n_candidates})")
        sys.exit(1)
    if len(set(candidate_ids)) != n_candidates:
        print("ERROR: candidate_ids contains duplicates")
        sys.exit(1)

    id_to_idx = {ep_id: i for i, ep_id in enumerate(candidate_ids)}

    selection_order = []
    selection_trace = []

    remaining = set(range(n_candidates))

    # Step 1: Random first point
    first_local_idx = rng.choice(n_candidates)
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
        # Find argmax min_dist among remaining, with tie-break by smaller episode index
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
            print(f"ERROR: No candidate found at step {step}")
            sys.exit(1)

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

        # Incremental update: compute distance from new point to all remaining
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


def validate_selection(
    selection_order: List[int],
    selected_episode_indices: List[int],
    candidate_ids: List[int],
    num_selected: int,
):
    """Validate selection results for correctness."""
    if len(selection_order) != num_selected:
        print(f"ERROR: selection_order length {len(selection_order)} != num_selected {num_selected}")
        sys.exit(1)
    if len(selected_episode_indices) != num_selected:
        print(f"ERROR: selected_episode_indices length {len(selected_episode_indices)} != num_selected {num_selected}")
        sys.exit(1)
    if len(set(selected_episode_indices)) != num_selected:
        print("ERROR: selected_episode_indices contains duplicates")
        sys.exit(1)
    if len(set(selection_order)) != num_selected:
        print("ERROR: selection_order contains duplicates")
        sys.exit(1)

    candidate_set = set(candidate_ids)
    for ep_id in selected_episode_indices:
        if ep_id not in candidate_set:
            print(f"ERROR: selected episode {ep_id} not in candidate set")
            sys.exit(1)

    print(f"  Validation passed: {num_selected} episodes selected, no duplicates, all in candidate set")


def check_monotonicity(selection_trace: List[Dict], tolerance: float = 1e-9):
    """
    Check that farthest_min_distance is non-increasing from step 1 onwards.

    As the selected set grows, each remaining point's min distance to the
    selected set should not increase (monotonically non-increasing).
    """
    distances = []
    for entry in selection_trace:
        if entry["farthest_min_distance"] is not None:
            distances.append(entry["farthest_min_distance"])

    if len(distances) < 2:
        return True, []

    violations = []
    for i in range(1, len(distances)):
        if distances[i] > distances[i - 1] + tolerance:
            violations.append({
                "step_prev": i - 1,
                "step_curr": i,
                "dist_prev": distances[i - 1],
                "dist_curr": distances[i],
                "increase": distances[i] - distances[i - 1],
            })

    if violations:
        print(f"\n  WARNING: farthest_min_distance monotonicity violations detected:")
        for v in violations:
            print(f"    Step {v['step_prev']}->{v['step_curr']}: "
                  f"{v['dist_prev']:.6f} -> {v['dist_curr']:.6f} "
                  f"(increase: {v['increase']:.2e})")
        return False, violations

    return True, []


def compute_pairwise_statistics(
    selected_features: List[np.ndarray],
) -> Dict:
    """
    Compute pairwise Euclidean L2 distance statistics for the selected subset.

    Returns mean, std, min, max of all pairwise distances.
    """
    n = len(selected_features)
    if n < 2:
        return {"mean": None, "std": None, "min": None, "max": None, "n_pairs": 0}

    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(selected_features[i] - selected_features[j])
            distances.append(d)

    arr = np.array(distances)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_pairs": len(distances),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Visual-FPS (Farthest Point Sampling) Episode Selection"
    )
    parser.add_argument("--dataset-dir", type=str, required=True,
                       help="Dataset root directory")
    parser.add_argument("--dataset-name", type=str, required=True,
                       help="Dataset name (e.g., pick_place_corner)")
    parser.add_argument("--visual-embedding-dir", type=str, default=None,
                       help="Visual embedding cache directory (auto-resolved if not provided)")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--num-selected", type=int, default=112,
                       help="Target number of episodes to select (default: 112)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for first FPS point (default: 42)")
    parser.add_argument("--pca-dim", type=int, default=DEFAULT_PCA_DIM,
                       help=f"PCA dimension (default: {DEFAULT_PCA_DIM})")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    num_selected = args.num_selected
    seed = args.seed
    pca_dim = args.pca_dim

    print(f"{'='*60}")
    print(f"Visual-FPS Episode Selection")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Num selected: {num_selected}")
    print(f"Seed: {seed}")
    print(f"PCA dim: {pca_dim}")

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "subsets").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    # Resolve visual embedding cache path
    # Note: shared_embeddings uses names like 'pick_place_corner' while dataset_view
    # uses 'pick_place-v3_corner'. Strip '-v3' suffix for shared embedding lookup.
    shared_embedding_dataset_name = args.dataset_name.replace("-v3", "")
    
    if args.visual_embedding_dir:
        visual_cache_dir = Path(args.visual_embedding_dir)
        print(f"\nUsing provided visual embedding dir: {visual_cache_dir}")
    else:
        visual_cache_dir = get_shared_embedding_dir(shared_embedding_dataset_name, pca_dim)
        print(f"\nDataset name (dataset_view): {args.dataset_name}")
        print(f"Dataset name (shared_embeddings): {shared_embedding_dataset_name}")
        print(f"Auto-resolved visual embedding dir: {visual_cache_dir}")

    embedding_method = build_extraction_method_name(pca_dim)
    print(f"Embedding method: {embedding_method}")

    if not visual_cache_dir.exists():
        print(f"\n{'='*60}")
        print(f"FAIL-FAST: Visual embedding cache does not exist")
        print(f"Expected path: {visual_cache_dir}")
        print(f"Dataset: {args.dataset_name}")
        print(f"{'='*60}")
        sys.exit(1)

    # Get candidate episode IDs from dataset
    candidate_ids = get_dataset_episode_indices(str(dataset_dir))
    candidate_ids = sorted(candidate_ids)
    print(f"\nCandidate episodes from dataset: {len(candidate_ids)}")

    # Filter to only episodes that exist in visual cache
    available_ids = []
    missing_in_cache = []
    for ep_idx in candidate_ids:
        ep_file = visual_cache_dir / f"({ep_idx}).npy"
        if ep_file.exists():
            available_ids.append(ep_idx)
        else:
            missing_in_cache.append(ep_idx)

    if missing_in_cache:
        print(f"  WARNING: {len(missing_in_cache)} episodes missing from cache: {missing_in_cache}")
        print(f"  Using only {len(available_ids)} episodes with valid cache files")

    candidate_ids = available_ids
    print(f"  Final candidate pool: {len(candidate_ids)} episodes")
    print(f"  IDs: {candidate_ids}")

    if len(candidate_ids) == 0:
        print(f"\n{'='*60}")
        print(f"FAIL-FAST: No valid candidate episodes with cache files")
        print(f"Cache dir: {visual_cache_dir}")
        print(f"{'='*60}")
        sys.exit(1)

    # Load visual features
    print(f"\nLoading visual features from cache...")
    features, feature_dim = load_all_visual_features(visual_cache_dir, candidate_ids)
    print(f"  Loaded {len(features)} features, dimension: {feature_dim}")

    # Verify phi_global and phi_wrist dimensions
    sample_ep = candidate_ids[0]
    sample_file = visual_cache_dir / f"({sample_ep}).npy"
    sample_data = np.load(str(sample_file), allow_pickle=True).item()
    phi_global_dim = sample_data["phi_global"].shape[0]
    phi_wrist_dim = sample_data["phi_wrist"].shape[0]
    print(f"  phi_global dim: {phi_global_dim}")
    print(f"  phi_wrist dim: {phi_wrist_dim}")
    print(f"  Concatenated feature dim: {feature_dim}")

    # Run FPS
    print(f"\n{'='*60}")
    print(f"Running Farthest Point Sampling...")
    print(f"{'='*60}")
    start_time = time.time()

    result = farthest_point_sampling(
        candidate_ids=candidate_ids,
        features=features,
        num_selected=num_selected,
        seed=seed,
    )

    elapsed_time = time.time() - start_time
    print(f"\nFPS completed in {elapsed_time:.2f}s")

    selection_order = result["selection_order"]
    selected_episode_indices = result["selected_episode_indices"]

    # Validate selection
    print(f"\nValidating selection results...")
    validate_selection(selection_order, selected_episode_indices, candidate_ids, num_selected)

    # Check monotonicity
    print(f"\nChecking farthest_min_distance monotonicity...")
    is_monotonic, violations = check_monotonicity(result["selection_trace"])
    if is_monotonic:
        print(f"  Monotonicity check PASSED")
    else:
        print(f"  Monotonicity check: {len(violations)} violation(s) found")

    # Print first 10 selection steps
    print(f"\nFirst 10 selection steps:")
    for entry in result["selection_trace"][:10]:
        dist_str = f"{entry['farthest_min_distance']:.6f}" if entry["farthest_min_distance"] is not None else "N/A (seed)"
        print(f"  Step {entry['step']}: episode {entry['episode_index']}, "
              f"farthest_min_distance={dist_str}")

    # Compute pairwise statistics for selected subset
    selected_feats = [features[ep] for ep in selected_episode_indices]
    pairwise_stats = compute_pairwise_statistics(selected_feats)
    print(f"\nPairwise distance statistics (selected subset):")
    print(f"  Mean: {pairwise_stats['mean']:.6f}" if pairwise_stats['mean'] else "  Mean: N/A")
    print(f"  Std:  {pairwise_stats['std']:.6f}" if pairwise_stats['std'] else "  Std: N/A")
    print(f"  Min:  {pairwise_stats['min']:.6f}" if pairwise_stats['min'] else "  Min: N/A")
    print(f"  Max:  {pairwise_stats['max']:.6f}" if pairwise_stats['max'] else "  Max: N/A")

    # Save subset JSON
    subset_file = output_dir / "subsets" / f"fps_{num_selected}_seed{seed}.json"
    subset_data = {
        "method": "visual_fps",
        "selection_method": "visual_farthest_point_sampling",
        "num_episodes": num_selected,
        "seed": seed,
        "dataset_name": args.dataset_name,
        "dataset_root": str(dataset_dir.resolve()),
        "candidate_count": len(candidate_ids),
        "visual_embedding_dir": str(visual_cache_dir.resolve()),
        "embedding_method": embedding_method,
        "pca_dim": pca_dim,
        "feature_definition": "concat(phi_global, phi_wrist)",
        "distance_metric": "euclidean",
        "feature_dim": feature_dim,
        "initial_episode": int(selection_order[0]),
        "selection_order": selection_order,
        "selected_episode_indices": selected_episode_indices,
    }

    with open(subset_file, "w") as f:
        json.dump(subset_data, f, indent=2, cls=NumpyEncoder)
    print(f"\nSubset saved to: {subset_file}")

    # Save selection trace
    trace_file = output_dir / "results" / f"fps_selection_trace.json"
    trace_data = {
        "selection_trace": result["selection_trace"],
        "selection_order": selection_order,
        "selected_episode_indices": selected_episode_indices,
        "candidate_count": len(candidate_ids),
        "selected_count": len(selected_episode_indices),
        "feature_dim": feature_dim,
        "phi_global_dim": phi_global_dim,
        "phi_wrist_dim": phi_wrist_dim,
        "selection_elapsed_time_seconds": elapsed_time,
        "pairwise_distance_statistics": pairwise_stats,
        "monotonicity_check": {
            "passed": is_monotonic,
            "violations": violations,
        },
    }

    with open(trace_file, "w") as f:
        json.dump(trace_data, f, indent=2, cls=NumpyEncoder)
    print(f"Selection trace saved to: {trace_file}")

    print(f"\n{'='*60}")
    print(f"Visual-FPS Selection Complete!")
    print(f"{'='*60}")
    print(f"  Selected {num_selected} episodes from {len(candidate_ids)} candidates")
    print(f"  Seed: {seed}")
    print(f"  First episode (seed point): {selection_order[0]}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Elapsed time: {elapsed_time:.2f}s")
    print(f"  Subset: {subset_file}")
    print(f"  Trace: {trace_file}")


if __name__ == "__main__":
    main()