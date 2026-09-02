#!/usr/bin/env python
"""
V3 No-Action Episode Selection

Main entry point for adaptive grid-based episode selection.
Simulates real robot data collection from scratch with:
- Stage 1: coarse uniform coverage (7x4 = 28 episodes)
- Stage 2: adaptive acquisition based on spatial need + visual disagreement

Usage:
    python select_episodes_v3_no_action.py \
        --dataset-root /path/to/dataset \
        --embedding-dir /path/to/embeddings \
        --output-dir /path/to/output \
        --num-selected 112 \
        --seed 42

Output:
    subsets/dynamicgrid_v3_no_action_112_seed42.json
    results/acquisition_log_v3_no_action_112_seed42.json
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Set

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_v3_no_action.core.planner import V3Planner
from our_v3_no_action.config import (
    TOTAL_BUDGET, INITIAL_GRID_X, INITIAL_GRID_Y, INITIAL_BUDGET,
    MAX_DEPTH, SPATIAL_WEIGHT, VISUAL_WEIGHT, SEED,
)


def validate_causal_access(planner: V3Planner) -> bool:
    """
    Validate that no unacquired episode embeddings were accessed during selection.
    """
    return planner.validate_causal_access()


def validate_results(result: Dict, expected_budget: int) -> Dict:
    """
    Validate the selection results:
    - All episode indices are unique
    - Total count matches budget
    - Stage 1 covers all coarse cells
    - Stage 2 episodes are distinct from Stage 1
    """
    selected = result["selected_episode_indices"]
    initial = result["initial_stage_indices"]
    adaptive = result["adaptive_stage_indices"]

    issues = []

    # Check uniqueness
    if len(selected) != len(set(selected)):
        issues.append(f"Duplicate episodes found: {len(selected)} selected, {len(set(selected))} unique")

    # Check total count
    if len(selected) != expected_budget:
        issues.append(f"Expected {expected_budget} episodes, got {len(selected)}")

    # Check Stage 1 vs Stage 2 disjoint
    overlap = set(initial) & set(adaptive)
    if overlap:
        issues.append(f"Stage 1 and Stage 2 overlap: {overlap}")

    # Check Stage 1 + Stage 2 = Total
    if len(initial) + len(adaptive) != len(selected):
        issues.append(f"Stage 1 ({len(initial)}) + Stage 2 ({len(adaptive)}) != Total ({len(selected)})")

    # Check Stage 1 covers all coarse cells
    grid_x = result["parameters"]["initial_grid_x"]
    grid_y = result["parameters"]["initial_grid_y"]
    expected_coarse_cells = grid_x * grid_y
    if len(initial) != expected_coarse_cells:
        issues.append(f"Stage 1 should cover {expected_coarse_cells} coarse cells, got {len(initial)}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "n_selected": len(selected),
        "n_unique": len(set(selected)),
        "n_initial": len(initial),
        "n_adaptive": len(adaptive),
    }


def print_summary(result: Dict, validation: Dict):
    """Print a human-readable summary of the selection results."""
    print(f"\n{'='*60}")
    print(f"V3 No-Action Selection Summary")
    print(f"{'='*60}")

    params = result["parameters"]
    print(f"Grid: {params['initial_grid_x']} x {params['initial_grid_y']}")
    print(f"Total budget: {params['total_budget']}")
    print(f"Stage 1 (coarse uniform): {validation['n_initial']} episodes")
    print(f"Stage 2 (adaptive): {validation['n_adaptive']} episodes")
    print(f"Total selected: {validation['n_selected']}")
    print(f"Unique episodes: {validation['n_unique']}")

    # Check if adaptive stage shows differential density
    if validation['n_adaptive'] > 0:
        log = result["acquisition_log"]
        adaptive_logs = [l for l in log if l["stage"] == "adaptive"]
        cell_counts = {}
        for l in adaptive_logs:
            cid = l["selected_cell_id"]
            cell_counts[cid] = cell_counts.get(cid, 0) + 1

        if cell_counts:
            max_count = max(cell_counts.values())
            min_count = min(cell_counts.values())
            print(f"\nAdaptive stage cell visit distribution:")
            print(f"  Most visited cell: {max_count} times")
            print(f"  Least visited cell: {min_count} times")
            if max_count > min_count:
                print(f"  -> Differential density confirmed (some regions encrypted, others sparse)")
            else:
                print(f"  -> Uniform distribution (no adaptive behavior)")

    print(f"\nValidation: {'PASSED' if validation['valid'] else 'FAILED'}")
    if validation['issues']:
        for issue in validation['issues']:
            print(f"  Issue: {issue}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="V3 No-Action Episode Selection")
    parser.add_argument("--dataset-root", type=str, required=True,
                       help="Dataset root directory (contains episode_initial_states.json)")
    parser.add_argument("--embedding-dir", type=str, required=True,
                       help="Visual embedding cache directory")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--num-selected", type=int, default=TOTAL_BUDGET,
                       help=f"Target number of episodes (default: {TOTAL_BUDGET})")
    parser.add_argument("--seed", type=int, default=SEED,
                       help=f"Random seed (default: {SEED})")
    parser.add_argument("--grid-x", type=int, default=INITIAL_GRID_X,
                       help=f"Initial grid X resolution (default: {INITIAL_GRID_X})")
    parser.add_argument("--grid-y", type=int, default=INITIAL_GRID_Y,
                       help=f"Initial grid Y resolution (default: {INITIAL_GRID_Y})")
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH,
                       help=f"Maximum cell split depth (default: {MAX_DEPTH})")
    parser.add_argument("--spatial-weight", type=float, default=SPATIAL_WEIGHT,
                       help=f"Spatial need weight (default: {SPATIAL_WEIGHT})")
    parser.add_argument("--visual-weight", type=float, default=VISUAL_WEIGHT,
                       help=f"Visual disagreement weight (default: {VISUAL_WEIGHT})")
    parser.add_argument("--selection-only", action="store_true",
                       help="Only run selection, skip training")

    args = parser.parse_args()

    dataset_root = args.dataset_root
    embedding_dir = args.embedding_dir
    output_dir = Path(args.output_dir)
    num_selected = args.num_selected
    seed = args.seed

    # Validate inputs
    if not Path(dataset_root).exists():
        print(f"Error: Dataset root does not exist: {dataset_root}")
        return

    if not Path(embedding_dir).exists():
        print(f"Error: Embedding directory does not exist: {embedding_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "subsets").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"V3 No-Action Adaptive Grid Episode Selection")
    print(f"{'='*60}")
    print(f"Dataset root: {dataset_root}")
    print(f"Embedding dir: {embedding_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Target episodes: {num_selected}")
    print(f"Grid: {args.grid_x} x {args.grid_y}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")

    # Create planner
    planner = V3Planner(
        dataset_root=dataset_root,
        embedding_dir=embedding_dir,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        total_budget=num_selected,
        initial_budget=args.grid_x * args.grid_y,
        max_depth=args.max_depth,
        spatial_weight=args.spatial_weight,
        visual_weight=args.visual_weight,
        seed=seed,
    )

    # Run collection
    start_time = time.time()
    result = planner.run_adaptive_collection(total_budget=num_selected)
    elapsed = time.time() - start_time

    # Validate causal access
    print(f"\nValidating causal access...")
    try:
        causal_ok = validate_causal_access(planner)
        print(f"  Causal access validation: {'PASSED' if causal_ok else 'FAILED'}")
    except RuntimeError as e:
        print(f"  Causal access validation: FAILED - {e}")
        result["causal_validation"] = {"passed": False, "error": str(e)}
    else:
        result["causal_validation"] = {"passed": True}

    # Validate results
    validation = validate_results(result, num_selected)
    result["validation"] = validation

    # Save subset JSON
    subset_file = output_dir / "subsets" / f"dynamicgrid_v3_no_action_{num_selected}_seed{seed}.json"
    subset_data = {
        "selected_episode_indices": result["selected_episode_indices"],
        "target_init_positions": result["target_init_positions"],
        "actual_init_positions": result["actual_init_positions"],
        "mapping_distances": result["mapping_distances"],
        "initial_stage_indices": result["initial_stage_indices"],
        "adaptive_stage_indices": result["adaptive_stage_indices"],
        "selection_method": result["selection_method"],
        "parameters": result["parameters"],
        "validation": validation,
        "causal_validation": result["causal_validation"],
    }
    with open(subset_file, "w") as f:
        json.dump(subset_data, f, indent=2)
    print(f"\nSubset saved to: {subset_file}")

    # Save acquisition log
    log_file = output_dir / "results" / f"acquisition_log_v3_no_action_{num_selected}_seed{seed}.json"
    with open(log_file, "w") as f:
        json.dump(result["acquisition_log"], f, indent=2)
    print(f"Acquisition log saved to: {log_file}")

    # Print summary
    print_summary(result, validation)

    print(f"\nTotal time: {elapsed:.2f}s")

    return result


if __name__ == "__main__":
    main()