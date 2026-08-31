#!/usr/bin/env python
"""
Corner3 offline sanity check for V2

复用已有 corner3 embedding cache，运行 V2 selection，
并与现有 V1 subset 做公平比较。

Usage:
    python sanity_check_corner3.py \
        --embeddings-dir /path/to/corner3/embeddings \
        --v1-subset /path/to/v1/subset.json \
        --output-dir /path/to/output
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent))

from sic_v2 import FixedAnchorSIC
from iterative_select_episodes_v2 import (
    load_embeddings,
    select_initial_b0_random,
    sequential_greedy_select,
    compute_fixed_universe_sic,
    compute_pairwise_redundancy,
    compute_mean_nearest_selected_distance
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Corner3 V2 sanity check")
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--v1-subset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--b0-size", type=int, default=18)
    parser.add_argument("--target-size", type=int, default=112)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda-wrist", type=float, default=1.0)

    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Corner3 V2 Sanity Check")
    print(f"{'='*60}")

    print(f"\n[1/6] Loading embeddings...")
    embeddings = load_embeddings(embeddings_dir)

    if not embeddings:
        print("错误: 没有找到嵌入文件")
        return

    all_episode_indices = sorted(embeddings.keys())
    phi_globals = np.array([embeddings[ep]["phi_global"] for ep in all_episode_indices])
    phi_wrists = np.array([embeddings[ep]["phi_wrist"] for ep in all_episode_indices])
    episode_to_idx = {ep: i for i, ep in enumerate(all_episode_indices)}

    print(f"  Total episodes: {len(all_episode_indices)}")

    print(f"\n[2/6] Building FixedAnchorSIC calculator...")
    sic_calc = FixedAnchorSIC(
        episode_indices=all_episode_indices,
        phi_globals=phi_globals,
        phi_wrists=phi_wrists,
        alpha=args.alpha,
        lambda_wrist=args.lambda_wrist
    )

    print(f"  Reference anchor count: {sic_calc.reference_anchor_count}")
    print(f"  dbar_global: {sic_calc.dbar_global:.6f}")
    print(f"  dbar_wrist: {sic_calc.dbar_wrist:.6f}")
    print(f"  dbar_fallback_used: {sic_calc.dbar_fallback_used}")

    print(f"\n[3/6] Selecting B0 (random, seed={args.seed})...")
    b0_episodes = select_initial_b0_random(all_episode_indices, args.b0_size, args.seed)
    print(f"  B0 episodes: {b0_episodes}")

    print(f"\n[4/6] Running V2 sequential greedy selection...")
    overall_start = time.time()

    result = sequential_greedy_select(
        sic_calculator=sic_calc,
        all_episode_indices=all_episode_indices,
        b0_episodes=b0_episodes,
        target_size=args.target_size,
        n_add_per_round=9
    )

    overall_time = time.time() - overall_start
    result["total_runtime_seconds"] = overall_time

    print(f"\n  V2 selected {len(result['selected_episodes'])} episodes")
    print(f"  Final SIC: {result['final_sic']:.4f}")
    print(f"  Normalized SIC: {result['normalized_sic']:.4f}")
    print(f"  Total runtime: {overall_time:.2f}s")

    print(f"\n[5/6] Loading V1 subset and computing fixed-universe SIC...")
    with open(args.v1_subset) as f:
        v1_data = json.load(f)

    v1_episodes = v1_data["selected_episode_indices"]

    v1_sic_info = compute_fixed_universe_sic(
        v1_episodes, all_episode_indices, phi_globals, phi_wrists,
        args.alpha, args.lambda_wrist
    )

    v2_episodes = result["selected_episodes"]
    v2_sic_info = compute_fixed_universe_sic(
        v2_episodes, all_episode_indices, phi_globals, phi_wrists,
        args.alpha, args.lambda_wrist
    )

    print(f"\n  --- Fixed-Universe SIC (fair comparison) ---")
    print(f"  V1 fixed-universe SIC: {v1_sic_info['fixed_universe_sic']:.4f}")
    print(f"  V1 normalized SIC: {v1_sic_info['normalized_sic']:.4f}")
    print(f"  V2 fixed-universe SIC: {v2_sic_info['fixed_universe_sic']:.4f}")
    print(f"  V2 normalized SIC: {v2_sic_info['normalized_sic']:.4f}")

    print(f"\n[6/6] Computing overlap and redundancy metrics...")

    v1_set = set(v1_episodes)
    v2_set = set(v2_episodes)
    overlap = v1_set & v2_set

    print(f"  Overlap: {len(overlap)} / {len(v1_episodes)} ({len(overlap)/len(v1_episodes)*100:.1f}%)")

    from sic_v2 import compute_dbar_from_embeddings, build_kernel_matrices
    dbar_g, dbar_w, _ = compute_dbar_from_embeddings(phi_globals, phi_wrists)
    K_global, K_wrist = build_kernel_matrices(phi_globals, phi_wrists, dbar_g, dbar_w)

    v1_redundancy = compute_pairwise_redundancy(v1_episodes, episode_to_idx, K_global, K_wrist)
    v2_redundancy = compute_pairwise_redundancy(v2_episodes, episode_to_idx, K_global, K_wrist)

    print(f"\n  --- Pairwise Redundancy ---")
    print(f"  V1 global: {v1_redundancy['mean_global_redundancy']:.4f}, wrist: {v1_redundancy['mean_wrist_redundancy']:.4f}")
    print(f"  V2 global: {v2_redundancy['mean_global_redundancy']:.4f}, wrist: {v2_redundancy['mean_wrist_redundancy']:.4f}")

    v1_coverage = compute_mean_nearest_selected_distance(
        all_episode_indices, v1_episodes, episode_to_idx, K_global, K_wrist
    )
    v2_coverage = compute_mean_nearest_selected_distance(
        all_episode_indices, v2_episodes, episode_to_idx, K_global, K_wrist
    )

    print(f"\n  --- Coverage ---")
    print(f"  V1 global: {v1_coverage['mean_nearest_global']:.4f}, wrist: {v1_coverage['mean_nearest_wrist']:.4f}")
    print(f"  V2 global: {v2_coverage['mean_nearest_global']:.4f}, wrist: {v2_coverage['mean_nearest_wrist']:.4f}")

    comparison = {
        "v1_episodes_count": len(v1_episodes),
        "v2_episodes_count": len(v2_episodes),
        "overlap_count": len(overlap),
        "overlap_ratio": len(overlap) / len(v1_episodes) if v1_episodes else 0,
        "v1_fixed_universe_sic": v1_sic_info["fixed_universe_sic"],
        "v1_normalized_sic": v1_sic_info["normalized_sic"],
        "v2_fixed_universe_sic": v2_sic_info["fixed_universe_sic"],
        "v2_normalized_sic": v2_sic_info["normalized_sic"],
        "v1_global_redundancy": v1_redundancy["mean_global_redundancy"],
        "v1_wrist_redundancy": v1_redundancy["mean_wrist_redundancy"],
        "v2_global_redundancy": v2_redundancy["mean_global_redundancy"],
        "v2_wrist_redundancy": v2_redundancy["mean_wrist_redundancy"],
        "v1_global_coverage": v1_coverage["mean_nearest_global"],
        "v1_wrist_coverage": v1_coverage["mean_nearest_wrist"],
        "v2_global_coverage": v2_coverage["mean_nearest_global"],
        "v2_wrist_coverage": v2_coverage["mean_nearest_wrist"],
        "reference_anchor_count": v1_sic_info["reference_anchor_count"],
        "dbar_global": v1_sic_info["dbar_global"],
        "dbar_wrist": v1_sic_info["dbar_wrist"],
        "v2_runtime_seconds": overall_time,
        "v2_selected_episodes": v2_episodes,
        "v2_b0_episodes": b0_episodes
    }

    comparison_file = output_dir / "corner3_v1_vs_v2_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n保存比较结果到: {comparison_file}")

    v2_result_file = output_dir / "v2_selection_result.json"
    with open(v2_result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"保存 V2 选择结果到: {v2_result_file}")

    v2_subset_file = output_dir / "v2_subset.json"
    subset_data = {
        "selected_episode_indices": v2_episodes,
        "b0_episodes": b0_episodes,
        "num_episodes": len(v2_episodes),
        "selection_method": "fixed_anchor_sequential_sic_v2",
        "parameters": {
            "b0_size": args.b0_size,
            "target_size": args.target_size,
            "seed": args.seed,
            "alpha": args.alpha,
            "lambda_wrist": args.lambda_wrist,
            "b0_strategy": "random"
        }
    }
    with open(v2_subset_file, "w") as f:
        json.dump(subset_data, f, indent=2)
    print(f"保存 V2 subset 到: {v2_subset_file}")

    print(f"\n{'='*60}")
    print(f"Corner3 Sanity Check Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()