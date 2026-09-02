#!/usr/bin/env python
"""
SubZeroCore Episode Selection

Main entry point for SubZeroCore facility-location based episode selection.

Usage:
    python select_episodes_subzerocore.py \
        --dataset-root /path/to/dataset \
        --embedding-dir /path/to/embeddings \
        --output-dir /path/to/output \
        --num-selected 112 \
        --coverage-gamma 0.6 \
        --seed 42

Output:
    subsets/subzerocore_{num_selected}_seed{seed}.json
    results/selection_log_subzerocore_{num_selected}_seed{seed}.json
"""

import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SubZeroCore.config import COVERAGE_GAMMA, SEED, GLOBAL_WEIGHT, WRIST_WEIGHT
from SubZeroCore.core.embedding_loader import load_all_visual_embeddings
from SubZeroCore.core.subzerocore import run_subzerocore
from SubZeroCore.core.validation import validate_selection_result


def parse_args():
    parser = argparse.ArgumentParser(description="SubZeroCore Episode Selection")
    parser.add_argument(
        "--dataset-root", type=str, required=True,
        help="Dataset root directory (contains episode_initial_states.json)"
    )
    parser.add_argument(
        "--embedding-dir", type=str, required=True,
        help="Visual embedding cache directory"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--num-selected", type=int, default=112,
        help="Target number of episodes to select"
    )
    parser.add_argument(
        "--coverage-gamma", type=float, default=COVERAGE_GAMMA,
        help=f"Coverage gamma parameter (default: {COVERAGE_GAMMA})"
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Override K for KNN radius (default: auto-computed via coverage probability inversion)"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED})"
    )
    return parser.parse_args()


def build_subset_output(
    result: Dict,
    args: argparse.Namespace,
    embedding_dim: int,
    candidate_pool_size: int,
) -> Dict:
    """
    Build subset dictionary compatible with existing training pipeline.
    Matches the format used by random/uniform/ours selection scripts.
    """
    return {
        "selected_episode_indices": result["selected_episode_indices"],
        "num_episodes": args.num_selected,
        "selection_method": "SubZeroCore",
        "parameters": {
            "coverage_gamma": result["coverage_gamma"],
            "k": result["k"],
            "seed": args.seed,
            "embedding_dim": embedding_dim,
            "candidate_pool_size": candidate_pool_size,
            "offline_full_pool_access": True,
            "embedding_type": "visual_global_wrist",
            "global_weight": GLOBAL_WEIGHT,
            "wrist_weight": WRIST_WEIGHT,
            "similarity_type": "cosine",
            "density_weight_type": "gaussian_knn_radius",
        },
    }


def build_selection_log(
    result: Dict,
    args: argparse.Namespace,
    embedding_dim: int,
    candidate_pool_size: int,
    validation: Dict,
    runtime_seconds: float,
) -> Dict:
    """Build detailed selection log."""
    knn_radius = result["knn_radius"]
    density_weights = result["density_weights"]

    return {
        "candidate_pool_size": candidate_pool_size,
        "selected_size": len(result["selected_episode_indices"]),
        "embedding_dim": embedding_dim,
        "coverage_gamma": result["coverage_gamma"],
        "k": result["k"],
        "k_selection_method": "coverage_probability_inversion",
        "density_weight_formula": "gaussian_knn_radius",
        "knn_radius_stats": {
            "mean": float(np.mean(knn_radius)),
            "std": float(np.std(knn_radius)),
            "min": float(np.min(knn_radius)),
            "max": float(np.max(knn_radius)),
        },
        "density_weights_stats": {
            "mean": float(np.mean(density_weights)),
            "std": float(np.std(density_weights)),
            "min": float(np.min(density_weights)),
            "max": float(np.max(density_weights)),
        },
        "selection_order_episode_indices": result["selection_order_episode_indices"],
        "selected_episode_indices": result["selected_episode_indices"],
        "marginal_gains": result["marginal_gains"],
        "final_objective": result["final_objective"],
        "runtime_seconds": runtime_seconds,
        "validation": validation,
    }


def save_json_outputs(
    subset_data: Dict,
    selection_log: Dict,
    output_dir: Path,
    num_selected: int,
    seed: int,
) -> None:
    """Save subset and selection log to JSON files."""
    subsets_dir = output_dir / "subsets"
    results_dir = output_dir / "results"
    subsets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    subset_file = subsets_dir / f"subzerocore_{num_selected}_seed{seed}.json"
    with open(subset_file, "w") as f:
        json.dump(subset_data, f, indent=2)

    log_file = results_dir / f"selection_log_subzerocore_{num_selected}_seed{seed}.json"
    with open(log_file, "w") as f:
        json.dump(selection_log, f, indent=2)


def print_summary(
    result: Dict,
    validation: Dict,
    embedding_dim: int,
    candidate_pool_size: int,
    runtime_seconds: float,
) -> None:
    """Print human-readable summary of selection results."""
    print(f"\n{'='*60}")
    print(f"SubZeroCore Selection Summary")
    print(f"{'='*60}")

    print(f"Candidate pool size: {candidate_pool_size}")
    print(f"Target subset size: {result['target_size']}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"K (KNN): {result['k']}")
    print(f"K selection method: coverage_probability_inversion")
    print(f"Coverage gamma: {result['coverage_gamma']}")
    print(f"Density weight method: gaussian_knn_radius")

    knn_radius = result["knn_radius"]
    print(f"KNN radius: mean={np.mean(knn_radius):.4f}, "
          f"std={np.std(knn_radius):.4f}, "
          f"min={np.min(knn_radius):.4f}, "
          f"max={np.max(knn_radius):.4f}")

    density_weights = result["density_weights"]
    print(f"Density weights: mean={np.mean(density_weights):.4f}, "
          f"std={np.std(density_weights):.4f}, "
          f"min={np.min(density_weights):.4f}, "
          f"max={np.max(density_weights):.4f}")

    print(f"Selected episodes: {len(result['selected_episode_indices'])}")

    if result["marginal_gains"]:
        print(f"Initial marginal gain: {result['marginal_gains'][0]:.4f}")
        print(f"Final marginal gain: {result['marginal_gains'][-1]:.4f}")

    print(f"Final objective: {result['final_objective']:.4f}")
    print(f"Runtime: {runtime_seconds:.2f}s")

    print(f"\nValidation: {'PASSED' if validation['valid'] else 'FAILED'}")
    if validation["issues"]:
        for issue in validation["issues"]:
            print(f"  Issue: {issue}")

    print(f"{'='*60}")


def main():
    args = parse_args()

    dataset_root = args.dataset_root
    embedding_dir = args.embedding_dir
    output_dir = Path(args.output_dir)
    num_selected = args.num_selected
    seed = args.seed
    coverage_gamma = args.coverage_gamma
    k_override = args.k

    if not Path(dataset_root).exists():
        print(f"Error: Dataset root does not exist: {dataset_root}")
        sys.exit(1)

    if not Path(embedding_dir).exists():
        print(f"Error: Embedding directory does not exist: {embedding_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SubZeroCore Episode Selection")
    print(f"{'='*60}")
    print(f"Dataset root: {dataset_root}")
    print(f"Embedding dir: {embedding_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Target episodes: {num_selected}")
    print(f"Coverage gamma: {coverage_gamma}")
    print(f"K override: {k_override}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")

    start_time = time.time()

    print("\nLoading visual embeddings...")
    embedding_matrix, episode_indices = load_all_visual_embeddings(
        dataset_root, embedding_dir
    )
    candidate_pool_size = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    print(f"Loaded {candidate_pool_size} episodes, embedding dim={embedding_dim}")

    print("\nRunning SubZeroCore selection...")
    result = run_subzerocore(
        X=embedding_matrix,
        episode_indices=episode_indices,
        target_size=num_selected,
        coverage_gamma=coverage_gamma,
        k_override=k_override,
        seed=seed,
    )

    runtime_seconds = time.time() - start_time

    print("\nValidating selection result...")
    validation = validate_selection_result(
        result,
        episode_indices,
        num_selected,
        embedding_matrix=embedding_matrix,
        similarity_matrix=result["similarity_matrix"],
        weighted_similarity_matrix=result["weighted_similarity_matrix"],
    )

    if not validation["valid"]:
        print_summary(result, validation, embedding_dim, candidate_pool_size, runtime_seconds)
        print(f"\nVALIDATION FAILED. Exiting with error.")
        for issue in validation["issues"]:
            print(f"  Issue: {issue}")
        sys.exit(1)

    subset_data = build_subset_output(result, args, embedding_dim, candidate_pool_size)
    selection_log = build_selection_log(
        result, args, embedding_dim, candidate_pool_size, validation, runtime_seconds
    )

    save_json_outputs(subset_data, selection_log, output_dir, num_selected, seed)

    print_summary(result, validation, embedding_dim, candidate_pool_size, runtime_seconds)

    print(f"\nAll checks passed.")


if __name__ == "__main__":
    main()