#!/usr/bin/env python
"""Run paper-faithful causal Our-V7 acquisition on one benchmark dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORK2_ROOT = Path(__file__).resolve().parents[2]
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

from our_v7.config import (
    B0_REGION_RATIO,
    DEFAULT_VISUAL_VARIANT,
    MAX_REGIONS,
    MIN_REGIONS,
    REGION_RATIO,
    SEED,
    TOTAL_BUDGET,
    VISUAL_VARIANTS,
)
from our_v7.core.planner import V7Planner
from our_v7.core.robomme_planner import RoboMMEV7Planner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Our-V7 paper-faithful causal demonstration acquisition"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--benchmark",
        choices=("metaworld", "robomme"),
        default="metaworld",
        help="Configuration/dataset adapter. Default keeps the original MetaWorld behavior.",
    )
    parser.add_argument("--total-budget", type=int, default=TOTAL_BUDGET)
    parser.add_argument("--visual-variant", choices=VISUAL_VARIANTS, default=DEFAULT_VISUAL_VARIANT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--region-ratio", type=float, default=REGION_RATIO)
    parser.add_argument("--min-regions", type=int, default=MIN_REGIONS)
    parser.add_argument("--max-regions", type=int, default=MAX_REGIONS)
    parser.add_argument("--b0-region-ratio", type=float, default=B0_REGION_RATIO)
    parser.add_argument(
        "--ablation",
        choices=("full", "wo_action", "wo_adaptive_priority"),
        default="full",
    )
    parser.add_argument(
        "--raw-visual-cache-root",
        default=None,
        help="Optional shared raw VLM cache root; cache access remains acquired-gated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("Our-V7 paper-faithful causal demonstration acquisition")
    print(f"benchmark={args.benchmark}")
    print(f"dataset={args.dataset_name}")
    print(f"budget={args.total_budget}")
    print(f"visual_variant={args.visual_variant}")
    print(f"ablation={args.ablation}")
    if args.benchmark == "metaworld":
        print("configuration source: episode_initial_states.json -> rand_vec")
        print("configuration normalization: predefined MetaWorld reset bounds")
        planner_cls = V7Planner
    else:
        print("configuration source: episode_initial_states.json -> initial_configuration")
        print(
            "configuration fields: movable_objects/randomized_targets/articulations/task_config "
            "(scene_state excluded)"
        )
        print("configuration normalization: task-relevant admissible-support bounds")
        planner_cls = RoboMMEV7Planner
    print("KMeans init: deterministic centroid-nearest + farthest-point traversal")
    print("B0 init: deterministic centroid-nearest + farthest-point traversal")
    print("priority: g_k + minmax(u_visual) + minmax(u_action)")
    print("=" * 76)

    planner = planner_cls(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        total_budget=args.total_budget,
        visual_variant=args.visual_variant,
        device=args.device,
        seed=args.seed,
        region_ratio=args.region_ratio,
        min_regions=args.min_regions,
        max_regions=args.max_regions,
        b0_region_ratio=args.b0_region_ratio,
        ablation=args.ablation,
        raw_visual_cache_root=args.raw_visual_cache_root,
    )
    result = planner.run()
    planner.validate_causal_access()
    result["benchmark"] = args.benchmark

    output_file = output_dir / "selected_episodes_v7.json"
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=" * 76)
    print(f"selection written to: {output_file}")
    print(f"selected={result['num_selected']}")
    print(f"regions={result['num_regions']}")
    print(f"initial={len(result['initial_episode_indices'])}")
    print(f"adaptive={len(result['adaptive_episode_indices'])}")
    print(f"raw visual cache={result['raw_visual_cache']}")
    print("causal validation: PASS")
    print("=" * 76)


if __name__ == "__main__":
    main()
