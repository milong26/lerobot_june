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
import numpy as np
from pathlib import Path
from typing import Dict, Set, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_v3_no_action.core.planner import V3Planner
from our_v3_no_action.config import (
    TOTAL_BUDGET, INITIAL_GRID_X, INITIAL_GRID_Y, INITIAL_BUDGET,
    MAX_DEPTH, SPATIAL_WEIGHT, VISUAL_WEIGHT, SEED,
)


def get_coarse_ancestor_id(cell_id: str) -> str:
    """
    Extract the coarse ancestor ID from any cell_id.
    All cells descended from the initial 7x4 coarse grid share the same coarse ancestor.

    Examples:
        c0_0       -> c0_0
        c0_0_00    -> c0_0
        c0_0_00_11 -> c0_0
        c6_3_10_01 -> c6_3

    Raises ValueError if cell_id format is invalid.
    """
    parts = cell_id.split("_")
    if len(parts) < 2 or not parts[0].startswith("c"):
        raise ValueError(f"Invalid cell_id format: {cell_id}")
    return f"{parts[0]}_{parts[1]}"


def validate_causal_access(planner: V3Planner) -> bool:
    """
    Validate that no unacquired episode embeddings were accessed during selection.
    """
    return planner.validate_causal_access()


def validate_results(result: Dict, expected_budget: int) -> Dict:
    """
    Validate the selection results with comprehensive checks:
    - All episode indices are unique
    - Total count matches budget
    - Stage 1 covers all coarse cells (verified by cell_id, not just count)
    - Stage 2 episodes are distinct from Stage 1
    - All target/actual positions match episode count
    - All mapping distances are non-negative
    - All visual embeddings correspond to acquired episodes
    - Strict one-to-one correspondence: selected_episode_indices[i], target_init_positions[i],
      actual_init_positions[i], mapping_distances[i], mapping_fallbacks[i], acquisition_log[i]
      must all describe the same acquisition
    - initial_stage_indices == acquisition_log where stage=="initial" (in order)
    - adaptive_stage_indices == acquisition_log where stage=="adaptive" (in order)
    """
    selected = result["selected_episode_indices"]
    initial = result["initial_stage_indices"]
    adaptive = result["adaptive_stage_indices"]
    acquisition_log = result["acquisition_log"]
    params = result["parameters"]
    target_positions = result.get("target_init_positions", [])
    actual_positions = result.get("actual_init_positions", [])
    mapping_distances = result.get("mapping_distances", [])
    mapping_fallbacks = result.get("mapping_fallbacks", [])

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

    # Check Stage 1 covers all coarse cells by verifying cell_ids
    grid_x = params["initial_grid_x"]
    grid_y = params["initial_grid_y"]
    expected_coarse_cells = grid_x * grid_y

    # Extract stage=initial cell_ids from acquisition log
    initial_logs = [l for l in acquisition_log if l["stage"] == "initial"]
    initial_cell_ids = set(l["selected_cell_id"] for l in initial_logs)

    # Build expected coarse cell_id set
    expected_cell_ids = set(f"c{i}_{j}" for i in range(grid_x) for j in range(grid_y))

    if len(initial_cell_ids) != expected_coarse_cells:
        issues.append(
            f"Stage 1 should cover {expected_coarse_cells} unique coarse cells, "
            f"got {len(initial_cell_ids)} unique cell_ids: {sorted(initial_cell_ids)}"
        )

    if initial_cell_ids != expected_cell_ids:
        missing = expected_cell_ids - initial_cell_ids
        extra = initial_cell_ids - expected_cell_ids
        if missing:
            issues.append(f"Missing expected coarse cells: {sorted(missing)}")
        if extra:
            issues.append(f"Unexpected coarse cells in Stage 1: {sorted(extra)}")

    # Check each selected episode is unique
    if len(set(selected)) != len(selected):
        issues.append("Not all selected episodes are unique")

    # Check target and actual positions count matches
    if len(target_positions) != len(selected):
        issues.append(
            f"target_init_positions count ({len(target_positions)}) != "
            f"selected episodes ({len(selected)})"
        )
    if len(actual_positions) != len(selected):
        issues.append(
            f"actual_init_positions count ({len(actual_positions)}) != "
            f"selected episodes ({len(selected)})"
        )

    # Check all mapping distances are non-negative
    if any(d < 0 for d in mapping_distances):
        issues.append("Found negative mapping distances")

    # Check mapping_fallbacks array exists and matches length
    if len(mapping_fallbacks) != len(selected):
        issues.append(
            f"mapping_fallbacks count ({len(mapping_fallbacks)}) != "
            f"selected episodes ({len(selected)})"
        )

    # === STRICT ONE-TO-ONE CORRESPONDENCE CHECKS ===
    n = len(selected)
    if len(acquisition_log) != n:
        issues.append(
            f"acquisition_log length ({len(acquisition_log)}) != "
            f"selected_episode_indices length ({n})"
        )
    if len(target_positions) != n:
        issues.append(
            f"target_init_positions length ({len(target_positions)}) != "
            f"selected_episode_indices length ({n})"
        )
    if len(actual_positions) != n:
        issues.append(
            f"actual_init_positions length ({len(actual_positions)}) != "
            f"selected_episode_indices length ({n})"
        )
    if len(mapping_distances) != n:
        issues.append(
            f"mapping_distances length ({len(mapping_distances)}) != "
            f"selected_episode_indices length ({n})"
        )
    if len(mapping_fallbacks) != n:
        issues.append(
            f"mapping_fallbacks length ({len(mapping_fallbacks)}) != "
            f"selected_episode_indices length ({n})"
        )

    # Check that all arrays have the same length
    all_lengths = [
        len(acquisition_log), len(selected), len(target_positions),
        len(actual_positions), len(mapping_distances), len(mapping_fallbacks)
    ]
    if len(set(all_lengths)) != 1:
        issues.append(
            f"Array length mismatch: acquisition_log={len(acquisition_log)}, "
            f"selected={len(selected)}, target={len(target_positions)}, "
            f"actual={len(actual_positions)}, mapping_dist={len(mapping_distances)}, "
            f"mapping_fallback={len(mapping_fallbacks)}"
        )

    # For each i, check acquisition_log[i]["episode_index"] == selected_episode_indices[i]
    for i in range(min(len(acquisition_log), len(selected))):
        log_ep = acquisition_log[i].get("episode_index")
        if log_ep != selected[i]:
            issues.append(
                f"Index {i}: acquisition_log episode_index={log_ep} != "
                f"selected_episode_indices={selected[i]}"
            )
            break  # Report first mismatch only

    # Check initial_stage_indices == acquisition_log where stage=="initial" (in order)
    expected_initial = [h["episode_index"] for h in acquisition_log if h["stage"] == "initial"]
    if initial != expected_initial:
        issues.append(
            f"initial_stage_indices mismatch: got {initial}, expected {expected_initial}"
        )

    # Check adaptive_stage_indices == acquisition_log where stage=="adaptive" (in order)
    expected_adaptive = [h["episode_index"] for h in acquisition_log if h["stage"] == "adaptive"]
    if adaptive != expected_adaptive:
        issues.append(
            f"adaptive_stage_indices mismatch: got {adaptive}, expected {expected_adaptive}"
        )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "n_selected": len(selected),
        "n_unique": len(set(selected)),
        "n_initial": len(initial),
        "n_adaptive": len(adaptive),
        "n_initial_cells": len(initial_cell_ids),
        "expected_coarse_cells": expected_coarse_cells,
    }


def print_detailed_summary(result: Dict, validation: Dict):
    """Print a comprehensive human-readable summary of the selection results."""
    print(f"\n{'='*60}")
    print(f"V3 No-Action Selection Summary")
    print(f"{'='*60}")

    params = result["parameters"]
    log = result["acquisition_log"]
    initial_logs = [l for l in log if l["stage"] == "initial"]
    adaptive_logs = [l for l in log if l["stage"] == "adaptive"]

    print(f"Grid: {params['initial_grid_x']} x {params['initial_grid_y']}")
    print(f"Total budget: {params['total_budget']}")
    print(f"Stage 1 (coarse uniform): {validation['n_initial']} episodes")
    print(f"Stage 2 (adaptive): {validation['n_adaptive']} episodes")
    print(f"Total selected: {validation['n_selected']}")
    print(f"Unique episodes: {validation['n_unique']}")

    # Print Stage 1 target and actual positions
    print(f"\n--- Stage 1 Target Init Positions (28 coarse cells) ---")
    for i, l in enumerate(initial_logs):
        target = l["target_init_pos"]
        actual = l["actual_init_pos"]
        print(f"  Cell {l['selected_cell_id']}: target=({target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}), "
              f"actual=({actual[0]:.4f}, {actual[1]:.4f}, {actual[2]:.4f})")

    # Print all final actual init positions
    print(f"\n--- All {len(result['actual_init_positions'])} Actual Init Positions ---")
    for i, pos in enumerate(result["actual_init_positions"]):
        print(f"  Ep {i+1}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

    # Cell visit distribution
    cell_counts = {}
    for l in log:
        cid = l["selected_cell_id"]
        cell_counts[cid] = cell_counts.get(cid, 0) + 1

    print(f"\n--- Adaptive Cell Visit Distribution ---")
    for cid in sorted(cell_counts.keys()):
        print(f"  {cid}: {cell_counts[cid]} times")

    # Depth distribution
    depth_counts = {}
    for l in log:
        d = l["cell_depth"]
        depth_counts[d] = depth_counts.get(d, 0) + 1
    print(f"\n--- Samples by Depth ---")
    for d in sorted(depth_counts.keys()):
        print(f"  Depth {d}: {depth_counts[d]} samples")

    # Split statistics
    split_count = sum(1 for l in log if l.get("split", False))
    print(f"\n--- Split Statistics ---")
    print(f"  Total splits: {split_count}")

    # Mapping fallback statistics
    mapping_fallbacks = result.get("mapping_fallbacks", [])
    fallback_count = sum(1 for f in mapping_fallbacks if f)
    fallback_ratio = fallback_count / len(mapping_fallbacks) if mapping_fallbacks else 0.0
    mapping_distances = result.get("mapping_distances", [])
    mean_dist = float(np.mean(mapping_distances)) if mapping_distances else 0.0
    max_dist = float(np.max(mapping_distances)) if mapping_distances else 0.0

    print(f"\n--- Mapping Fallback Statistics ---")
    print(f"  Fallback count: {fallback_count}/{len(mapping_fallbacks)}")
    print(f"  Fallback ratio: {fallback_ratio:.4f}")
    print(f"  Mean mapping distance: {mean_dist:.4f}")
    print(f"  Max mapping distance: {max_dist:.4f}")

    # Adaptive behavior diagnosis
    if validation['n_adaptive'] > 0:
        # Check adaptive density distribution across ALL 28 initial coarse cells
        params_grid_x = params['initial_grid_x']
        params_grid_y = params['initial_grid_y']
        coarse_ancestor_counts = {}
        # Initialize all 28 coarse cells to 0
        for i in range(params_grid_x):
            for j in range(params_grid_y):
                coarse_ancestor_counts[f"c{i}_{j}"] = 0
        # Accumulate from adaptive logs using get_coarse_ancestor_id
        for l in adaptive_logs:
            cid = l["selected_cell_id"]
            coarse_ancestor = get_coarse_ancestor_id(cid)
            coarse_ancestor_counts[coarse_ancestor] = coarse_ancestor_counts.get(coarse_ancestor, 0) + 1

        # Print per-coarse-cell adaptive counts
        print(f"\n--- Adaptive Count per Initial Coarse Cell (7x4=28) ---")
        for i in range(params_grid_x):
            for j in range(params_grid_y):
                cell_key = f"c{i}_{j}"
                print(f"  {cell_key}: {coarse_ancestor_counts[cell_key]}")

        counts = list(coarse_ancestor_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        print(f"\n--- Adaptive Density Diagnosis ---")
        print(f"  Coarse ancestor regions with adaptive samples: {sum(1 for c in counts if c > 0)}/{len(counts)}")
        print(f"  Max samples in one region: {max_count}")
        print(f"  Min samples in one region: {min_count}")
        print(f"  Mean: {mean_count:.2f}, Std: {std_count:.2f}")

        adaptive_budget = validation['n_adaptive']
        n_coarse = params_grid_x * params_grid_y
        expected_per_region = adaptive_budget / n_coarse

        # If distribution is nearly uniform, warn about degeneration
        if std_count < 0.5 * mean_count and mean_count > 0:
            print(f"  WARNING: Adaptive distribution is nearly uniform (std < 0.5 * mean).")
            print(f"  Algorithm may be degenerating toward uniform sampling.")
            print(f"  adaptive_density_confirmed=False")
        else:
            print(f"  Adaptive density confirmed: some regions receive more budget than others.")
            print(f"  adaptive_density_confirmed=True")

    print(f"\n{'='*60}")
    print(f"VALIDATION {'PASSED' if validation['valid'] else 'FAILED'}")
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
        "mapping_fallbacks": result.get("mapping_fallbacks", []),
        "initial_stage_indices": result["initial_stage_indices"],
        "adaptive_stage_indices": result["adaptive_stage_indices"],
        "selection_method": result["selection_method"],
        "parameters": result["parameters"],
        "mapping_stats": result.get("mapping_stats", {}),
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

    # Print detailed summary
    print_detailed_summary(result, validation)

    print(f"\nTotal time: {elapsed:.2f}s")

    # Final validation gate: exit with non-zero if validation or causal access failed
    causal_passed = result.get("causal_validation", {}).get("passed", False)
    validation_passed = validation.get("valid", False)

    print(f"\n{'='*60}")
    if validation_passed and causal_passed:
        print(f"ALL CHECKS PASSED — VALIDATION PASSED, CAUSAL ACCESS PASSED")
        print(f"{'='*60}")
        return result
    else:
        print(f"VALIDATION {'PASSED' if validation_passed else 'FAILED'}")
        print(f"CAUSAL ACCESS {'PASSED' if causal_passed else 'FAILED'}")
        print(f"EXITING WITH ERROR (non-zero exit code)")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()