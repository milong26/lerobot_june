#!/usr/bin/env python
"""Run causal Our-V6 demonstration acquisition on one LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORK2_ROOT = Path(__file__).resolve().parents[2]
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

from our_v6.config import (
    B0_REGION_RATIO,
    COVERAGE_WEIGHT,
    DEFAULT_VISUAL_VARIANT,
    MAX_REGIONS,
    MIN_REGIONS,
    REGION_ACTION_WEIGHT,
    REGION_RATIO,
    REGION_VISUAL_WEIGHT,
    SEED,
    TOTAL_BUDGET,
    VISUAL_VARIANTS,
)
from our_v6.core.planner import V6Planner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Our-V6 causal acquisition: configuration regions + acquired-only multimodal feedback"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-budget", type=int, default=TOTAL_BUDGET)
    parser.add_argument("--visual-variant", choices=VISUAL_VARIANTS, default=DEFAULT_VISUAL_VARIANT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--region-ratio", type=float, default=REGION_RATIO)
    parser.add_argument("--min-regions", type=int, default=MIN_REGIONS)
    parser.add_argument("--max-regions", type=int, default=MAX_REGIONS)
    parser.add_argument("--b0-region-ratio", type=float, default=B0_REGION_RATIO)
    parser.add_argument("--coverage-weight", type=float, default=COVERAGE_WEIGHT)
    parser.add_argument("--visual-weight", type=float, default=REGION_VISUAL_WEIGHT)
    parser.add_argument("--action-weight", type=float, default=REGION_ACTION_WEIGHT)
    parser.add_argument(
        "--ablation",
        choices=("full", "wo_action", "wo_adaptive_priority"),
        default="full",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Our-V6 causal demonstration acquisition")
    print(f"dataset={args.dataset_name}")
    print(f"budget={args.total_budget}")
    print(f"visual_variant={args.visual_variant}")
    print(f"ablation={args.ablation}")
    print("pre-acquisition information: episode_index + rand_vec only")
    print("=" * 72)

    planner = V6Planner(
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
        coverage_weight=args.coverage_weight,
        visual_weight=args.visual_weight,
        action_weight=args.action_weight,
        ablation=args.ablation,
    )
    result = planner.run()
    planner.validate_causal_access()

    output_file = output_dir / "selected_episodes_v6.json"
    with output_file.open("w") as f:
        json.dump(result, f, indent=2)

    print("=" * 72)
    print(f"selection written to: {output_file}")
    print(f"selected={result['num_selected']}")
    print(f"regions={result['num_regions']}")
    print(f"initial={len(result['initial_episode_indices'])}")
    print(f"adaptive={len(result['adaptive_episode_indices'])}")
    print("causal validation: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
