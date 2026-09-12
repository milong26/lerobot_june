#!/usr/bin/env python
"""
Verify Ablation Selection Consistency

Validates the three selection mode outputs (full, wo_action, wo_region) for consistency.
Checks:
  - selected_episode_indices count matches expected num_episodes
  - seed values are consistent across modes
  - episode indices are within valid range
  - outputs selection statistics for paper ablation table

Usage:
    python verify_ablation.py \
        --full-subset /path/to/our_v5_full_112_seed42.json \
        --wo-action-subset /path/to/our_v5_wo_action_112_seed42.json \
        --wo-region-subset /path/to/our_v5_wo_region_112_seed42.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_subset(file_path: Path) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


def verify_subset(subset_data: dict, expected_mode: str, expected_num: int, expected_seed: int) -> dict:
    issues = []
    stats = {}

    actual_num = subset_data.get("num_episodes", 0)
    if actual_num != expected_num:
        issues.append(f"num_episodes mismatch: expected {expected_num}, got {actual_num}")
    stats["num_episodes"] = actual_num

    indices = subset_data.get("selected_episode_indices", [])
    if len(indices) != expected_num:
        issues.append(f"selected_episode_indices count mismatch: expected {expected_num}, got {len(indices)}")
    stats["num_indices"] = len(indices)

    params = subset_data.get("parameters", {})
    actual_seed = params.get("seed", -1)
    if actual_seed != expected_seed:
        issues.append(f"seed mismatch: expected {expected_seed}, got {actual_seed}")
    stats["seed"] = actual_seed

    actual_mode = params.get("ablation_mode", "unknown")
    if actual_mode != expected_mode:
        issues.append(f"ablation_mode mismatch: expected {expected_mode}, got {actual_mode}")
    stats["ablation_mode"] = actual_mode

    if indices:
        min_idx = min(indices)
        max_idx = max(indices)
        stats["min_episode_index"] = min_idx
        stats["max_episode_index"] = max_idx
        if min_idx < 0:
            issues.append(f"negative episode index found: {min_idx}")
    else:
        stats["min_episode_index"] = None
        stats["max_episode_index"] = None
        issues.append("empty selected_episode_indices")

    visual_enabled = params.get("visual_coverage_enabled", False)
    action_enabled = params.get("action_diversity_enabled", False)
    region_enabled = params.get("region_decision_enabled", False)
    stats["visual_coverage_enabled"] = visual_enabled
    stats["action_diversity_enabled"] = action_enabled
    stats["region_decision_enabled"] = region_enabled

    return {"issues": issues, "stats": stats}


def compare_selections(full_indices: list, wo_action_indices: list, wo_region_indices: list) -> dict:
    full_set = set(full_indices)
    wo_action_set = set(wo_action_indices)
    wo_region_set = set(wo_region_indices)

    full_wo_action_overlap = len(full_set & wo_action_set)
    full_wo_region_overlap = len(full_set & wo_region_set)
    wo_action_wo_region_overlap = len(wo_action_set & wo_region_set)
    all_three_overlap = len(full_set & wo_action_set & wo_region_set)

    return {
        "full_vs_wo_action_overlap": full_wo_action_overlap,
        "full_vs_wo_region_overlap": full_wo_region_overlap,
        "wo_action_vs_wo_region_overlap": wo_action_wo_region_overlap,
        "all_three_overlap": all_three_overlap,
        "full_unique": len(full_set - wo_action_set - wo_region_set),
        "wo_action_unique": len(wo_action_set - full_set - wo_region_set),
        "wo_region_unique": len(wo_region_set - full_set - wo_action_set),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify Ablation Selection Consistency")
    parser.add_argument("--full-subset", type=str, required=True,
                       help="Path to full mode subset JSON file")
    parser.add_argument("--wo-action-subset", type=str, required=True,
                       help="Path to wo_action mode subset JSON file")
    parser.add_argument("--wo-region-subset", type=str, required=True,
                       help="Path to wo_region mode subset JSON file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for verification report (optional)")
    args = parser.parse_args()

    full_path = Path(args.full_subset)
    wo_action_path = Path(args.wo_action_subset)
    wo_region_path = Path(args.wo_region_subset)

    for p in [full_path, wo_action_path, wo_region_path]:
        if not p.exists():
            print(f"Error: Subset file not found: {p}")
            return

    print(f"\n{'='*60}")
    print(f"Ablation Selection Verification")
    print(f"{'='*60}")
    print(f"Full subset:      {full_path}")
    print(f"wo_action subset: {wo_action_path}")
    print(f"wo_region subset: {wo_region_path}")

    full_data = load_subset(full_path)
    wo_action_data = load_subset(wo_action_path)
    wo_region_data = load_subset(wo_region_path)

    full_num = full_data.get("num_episodes", 0)
    full_seed = full_data.get("parameters", {}).get("seed", 42)

    print(f"\nExpected: num_episodes={full_num}, seed={full_seed}")

    full_result = verify_subset(full_data, "full", full_num, full_seed)
    wo_action_result = verify_subset(wo_action_data, "wo_action", full_num, full_seed)
    wo_region_result = verify_subset(wo_region_data, "wo_region", full_num, full_seed)

    print(f"\n{'='*60}")
    print(f"Full Mode Verification")
    print(f"{'='*60}")
    if full_result["issues"]:
        for issue in full_result["issues"]:
            print(f"  ISSUE: {issue}")
    else:
        print(f"  PASS: All checks passed")
    for k, v in full_result["stats"].items():
        print(f"  {k}: {v}")

    print(f"\n{'='*60}")
    print(f"wo_action Mode Verification")
    print(f"{'='*60}")
    if wo_action_result["issues"]:
        for issue in wo_action_result["issues"]:
            print(f"  ISSUE: {issue}")
    else:
        print(f"  PASS: All checks passed")
    for k, v in wo_action_result["stats"].items():
        print(f"  {k}: {v}")

    print(f"\n{'='*60}")
    print(f"wo_region Mode Verification")
    print(f"{'='*60}")
    if wo_region_result["issues"]:
        for issue in wo_region_result["issues"]:
            print(f"  ISSUE: {issue}")
    else:
        print(f"  PASS: All checks passed")
    for k, v in wo_region_result["stats"].items():
        print(f"  {k}: {v}")

    full_indices = full_data.get("selected_episode_indices", [])
    wo_action_indices = wo_action_data.get("selected_episode_indices", [])
    wo_region_indices = wo_region_data.get("selected_episode_indices", [])

    comparison = compare_selections(full_indices, wo_action_indices, wo_region_indices)

    print(f"\n{'='*60}")
    print(f"Selection Overlap Analysis")
    print(f"{'='*60}")
    for k, v in comparison.items():
        print(f"  {k}: {v}")

    all_issues = full_result["issues"] + wo_action_result["issues"] + wo_region_result["issues"]

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    if all_issues:
        print(f"  Total issues found: {len(all_issues)}")
        for issue in all_issues:
            print(f"    - {issue}")
    else:
        print(f"  All verifications passed!")

    report = {
        "full_verification": full_result,
        "wo_action_verification": wo_action_result,
        "wo_region_verification": wo_region_result,
        "comparison": comparison,
        "total_issues": len(all_issues),
        "all_issues": all_issues,
    }

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "ablation_verification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        print(f"\nVerification report saved to: {report_path}")


if __name__ == "__main__":
    main()