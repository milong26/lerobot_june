"""
Select Episodes using V6: Adaptive Grid with V5 Action Descriptors

Combines V4's from-scratch acquisition strategy with V5's action descriptor approach.
- Stage 1: coarse uniform coverage (one episode per coarse cell)
- Stage 2: adaptive acquisition based on spatial need + visual disagreement + action disagreement
- Action descriptors use V5's pre-computed cache (causal access maintained)
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_v6.core.planner import V6Planner


def main():
    parser = argparse.ArgumentParser(description="Select episodes using V6 (AdaptiveGrid + V5 Action)")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to LeRobot dataset root")
    parser.add_argument("--embedding_dir", type=str, required=True, help="Path to visual embedding cache directory")
    parser.add_argument("--action_descriptor_dir", type=str, required=True, help="Path to V5 action descriptor directory")
    parser.add_argument("--total_budget", type=int, default=112, help="Total number of episodes to select")
    parser.add_argument("--initial_grid_x", type=int, default=7, help="Initial grid X resolution")
    parser.add_argument("--initial_grid_y", type=int, default=4, help="Initial grid Y resolution")
    parser.add_argument("--max_depth", type=int, default=3, help="Maximum cell splitting depth")
    parser.add_argument("--spatial_weight", type=float, default=1.0, help="Weight for spatial need")
    parser.add_argument("--visual_weight", type=float, default=1.0, help="Weight for visual disagreement")
    parser.add_argument("--action_weight", type=float, default=0.5, help="Weight for action disagreement")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="output_v6", help="Output directory for results")

    args = parser.parse_args()

    print("="*60)
    print("V6 Episode Selection: AdaptiveGrid + V5 Action Descriptors")
    print("="*60)
    print(f"Dataset: {args.dataset_root}")
    print(f"Visual embeddings: {args.embedding_dir}")
    print(f"Action descriptors: {args.action_descriptor_dir}")
    print(f"Budget: {args.total_budget}")
    print(f"Grid: {args.initial_grid_x}x{args.initial_grid_y}")
    print(f"Weights: spatial={args.spatial_weight}, visual={args.visual_weight}, action={args.action_weight}")
    print(f"Seed: {args.seed}")
    print("="*60)

    planner = V6Planner(
        dataset_root=args.dataset_root,
        embedding_dir=args.embedding_dir,
        action_descriptor_dir=args.action_descriptor_dir,
        grid_x=args.initial_grid_x,
        grid_y=args.initial_grid_y,
        total_budget=args.total_budget,
        initial_budget=args.initial_grid_x * args.initial_grid_y,
        max_depth=args.max_depth,
        spatial_weight=args.spatial_weight,
        visual_weight=args.visual_weight,
        action_weight=args.action_weight,
        seed=args.seed,
    )

    result = planner.run_adaptive_collection(total_budget=args.total_budget)

    planner.validate_causal_access()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "selected_episodes_v6.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Selected {len(result['selected_episode_indices'])} episodes")
    print(f"  Initial stage: {len(result['initial_stage_indices'])} episodes")
    print(f"  Adaptive stage: {len(result['adaptive_stage_indices'])} episodes")
    print(f"  Mapping fallback ratio: {result['mapping_stats']['fallback_ratio']:.2%}")

    return result


if __name__ == "__main__":
    main()