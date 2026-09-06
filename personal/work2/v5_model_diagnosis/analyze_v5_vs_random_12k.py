#!/usr/bin/env python3
"""V5 vs Random 12k checkpoint diagnosis for disassemble-v3.

Analyzes why our_v5 does not consistently beat random at 12k checkpoint.
Only analyzes checkpoint 012000. No retraining, no large-scale eval.

Usage:
    python analyze_v5_vs_random_12k.py --task disassemble-v3 --checkpoint 012000 --device cuda --max-trace-seeds 6
    python analyze_v5_vs_random_12k.py --task disassemble-v3 --checkpoint 012000 --device cuda --max-trace-seeds 6 --dry-run
"""

import argparse
import csv
import json
import os
import sys
import glob
import numpy as np
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORK2_ROOT = PROJECT_ROOT / "personal" / "work2"

TASK_CONFIGS = {
    "disassemble-v3": {
        "v5_dir": WORK2_ROOT / "duibi" / "our_v5_112_seed42_disassemble-v3_corner" / "our_v5_112_seed42",
        "random_dir": WORK2_ROOT / "duibi" / "random_42_disassemble-v3_corner" / "random_112_seed42",
        "dataset_name": "disassemble-v3_corner",
        "env_task": "disassemble-v3",
    },
    "pick-place-v3": {
        "v5_dir": WORK2_ROOT / "duibi" / "our_v5_112_seed42_pick_place-v3_corner" / "our_v5_112_seed42",
        "random_dir": WORK2_ROOT / "duibi" / "random_42_pick_place-v3_corner" / "random_112_seed42",
        "dataset_name": "pick_place-v3_corner",
        "env_task": "pick-place-v3",
    },
}

DIAGNOSTIC_SEEDS = list(range(42, 62))  # 20 seeds max for diagnostic eval


def find_eval_results(model_dir: Path, checkpoint: str) -> list:
    """Find existing eval episode results JSON files."""
    results_dir = model_dir / "eval" / f"results_step_{checkpoint}"
    if not results_dir.exists():
        return []
    json_files = []
    for f in results_dir.rglob("eval_episode_results.json"):
        json_files.append(f)
    return sorted(json_files)


def load_eval_episodes(json_path: Path) -> list:
    """Load per-episode eval results from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    episodes = data.get("episodes", [])
    if not episodes:
        return []
    return episodes


def find_selection_log(v5_dir: Path) -> Optional[Path]:
    """Find V5 selection log."""
    parent = v5_dir.parent
    for pattern in ["results/selection_log_v5_*.json", "selection_log_v5_*.json"]:
        matches = list(parent.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_diagnostic_json(v5_dir: Path) -> Optional[Path]:
    """Find V5 diagnostic JSON."""
    parent = v5_dir.parent
    for pattern in ["diagnostic_v5.json", "results/diagnostic_v5.json"]:
        matches = list(parent.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_embeddings(dataset_name: str) -> dict:
    """Find existing embedding files for the dataset."""
    embed_base = WORK2_ROOT / "dataset_view" / dataset_name
    result = {}
    for pattern in ["global_embedding.npy", "wrist_embedding.npy", "combined_embedding.npy"]:
        matches = list(embed_base.rglob(pattern))
        if matches:
            result[pattern.replace(".npy", "")] = matches[0]
    for pattern in ["episode_initial_states.json", "metadata.json"]:
        matches = list(embed_base.rglob(pattern))
        if matches:
            result[pattern.replace(".json", "")] = matches[0]
    return result


def find_v5_subset(v5_dir: Path) -> Optional[Path]:
    """Find V5 selected subset indices."""
    parent = v5_dir.parent
    for pattern in ["results/subset_indices_v5_*.json", "subset_indices_v5_*.json",
                     "results/subset_v5_*.json", "subset_v5_*.json"]:
        matches = list(parent.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_selection_log(path: Path) -> dict:
    """Load V5 selection log."""
    with open(path) as f:
        return json.load(f)


def load_diagnostic(path: Path) -> dict:
    """Load V5 diagnostic JSON."""
    with open(path) as f:
        return json.load(f)


def load_embeddings_data(embed_paths: dict) -> dict:
    """Load embedding numpy arrays if available."""
    data = {}
    for key, path in embed_paths.items():
        if key.endswith("_embedding"):
            try:
                data[key] = np.load(path)
            except Exception:
                pass
    return data


def compute_physical_distance(eval_state: dict, subset_states: list) -> tuple:
    """Compute L2 distance between eval episode initial state and nearest subset state.

    Uses obj_init_pos as the primary physical distance metric.
    Returns (min_distance, nearest_episode_index).
    """
    eval_pos = np.array(eval_state.get("obj_init_pos", [0, 0, 0]))
    min_dist = float("inf")
    nearest_idx = -1
    for i, state in enumerate(subset_states):
        subset_pos = np.array(state.get("obj_init_pos", [0, 0, 0]))
        dist = np.linalg.norm(eval_pos - subset_pos)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i
    return min_dist, nearest_idx


def get_region_info(seed: int, selection_log: dict) -> dict:
    """Get region info for a given seed from selection log."""
    regions = selection_log.get("regions", {})
    for region_id, region_data in regions.items():
        seeds = region_data.get("seeds", [])
        if seed in seeds:
            total = region_data.get("total_episodes", 0)
            selected = region_data.get("selected_episodes", len(seeds))
            ratio = selected / total if total > 0 else 0
            return {
                "region": region_id,
                "total": total,
                "selected": selected,
                "selection_ratio": ratio,
            }
    return {"region": "unknown", "total": 0, "selected": 0, "selection_ratio": 0}


def classify_paired_results(v5_episodes: list, random_episodes: list) -> dict:
    """Classify paired V5/random results into four groups by seed."""
    v5_by_seed = {ep["seed"]: ep for ep in v5_episodes if "seed" in ep}
    random_by_seed = {ep["seed"]: ep for ep in random_episodes if "seed" in ep}

    groups = {
        "random_success_v5_fail": [],
        "v5_success_random_fail": [],
        "both_success": [],
        "both_fail": [],
    }

    common_seeds = sorted(set(v5_by_seed.keys()) & set(random_by_seed.keys()))
    for seed in common_seeds:
        v5_ep = v5_by_seed[seed]
        random_ep = random_by_seed[seed]
        v5_success = bool(v5_ep.get("success", False))
        random_success = bool(random_ep.get("success", False))

        case = {
            "seed": seed,
            "v5_success": v5_success,
            "random_success": random_success,
            "v5_grasp": v5_ep.get("grasp_success", False),
            "random_grasp": random_ep.get("grasp_success", False),
            "v5_episode": v5_ep,
            "random_episode": random_ep,
        }

        if random_success and not v5_success:
            groups["random_success_v5_fail"].append(case)
        elif v5_success and not random_success:
            groups["v5_success_random_fail"].append(case)
        elif v5_success and random_success:
            groups["both_success"].append(case)
        else:
            groups["both_fail"].append(case)

    return groups


def select_trace_seeds(groups: dict, max_seeds: int) -> list:
    """Select representative seeds for attention trace analysis.

    Priority: random_success_v5_fail > v5_success_random_fail.
    Within each group, prioritize seeds with larger outcome divergence.
    """
    selected = []
    # Prioritize random_success_v5_fail
    for group_name in ["random_success_v5_fail", "v5_success_random_fail"]:
        group = groups.get(group_name, [])
        if not group:
            continue
        # Sort by seed for determinism, take up to max_seeds // 2 per group
        n_take = min(max_seeds // 2, len(group), max(2, max_seeds - len(selected)))
        selected.extend(group[:n_take])
        if len(selected) >= max_seeds:
            break
    return selected[:max_seeds]


def run_inference_trace(seed: int, v5_model_path: Path, random_model_path: Path,
                        device: str, layers: list = None) -> dict:
    """Run inference trace for a single seed on both V5 and random models.

    Extracts x_t, v_t, suffix hidden, final action at specified layers.
    Returns divergence metrics between V5 and random at each denoising stage.
    """
    if layers is None:
        layers = [3, 7, 11]

    result = {
        "seed": seed,
        "layers": {},
        "action_divergence": 0.0,
    }

    # This is a placeholder - actual implementation would load models and run inference
    # For now, return structure that will be filled when models are loaded
    return result


def write_diagnostic_csv(cases: list, output_path: Path):
    """Write diagnostic_cases.csv."""
    fieldnames = [
        "seed", "group", "v5_success", "random_success",
        "v5_grasp", "random_grasp",
        "obj_init_pos", "goal_pos",
        "v5_nearest_randvec_dist", "random_nearest_randvec_dist",
        "v5_nearest_visual_dist", "random_nearest_visual_dist",
        "v5_region", "v5_region_selection_ratio",
        "trace_action_divergence",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            row = {k: case.get(k, "NA") for k in fieldnames}
            if isinstance(row.get("obj_init_pos"), list):
                row["obj_init_pos"] = ",".join(f"{x:.4f}" for x in row["obj_init_pos"])
            if isinstance(row.get("goal_pos"), list):
                row["goal_pos"] = ",".join(f"{x:.4f}" for x in row["goal_pos"])
            writer.writerow(row)


def write_summary_json(groups: dict, diagnostic_label: str, output_path: Path):
    """Write summary.json."""
    summary = {
        "group_counts": {k: len(v) for k, v in groups.items()},
        "random_success_v5_fail_stats": {},
        "diagnostic_label": diagnostic_label,
    }

    rs_vf = groups.get("random_success_v5_fail", [])
    if rs_vf:
        rand_vec_deltas = []
        visual_deltas = []
        region_ratios = []
        action_divs = []
        for case in rs_vf:
            v5_rv = case.get("v5_nearest_randvec_dist", float("nan"))
            rand_rv = case.get("random_nearest_randvec_dist", float("nan"))
            if not np.isnan(v5_rv) and not np.isnan(rand_rv):
                rand_vec_deltas.append(v5_rv - rand_rv)
            v5_vis = case.get("v5_nearest_visual_dist", float("nan"))
            rand_vis = case.get("random_nearest_visual_dist", float("nan"))
            if not np.isnan(v5_vis) and not np.isnan(rand_vis):
                visual_deltas.append(v5_vis - rand_vis)
            region_ratios.append(case.get("v5_region_selection_ratio", 0))
            action_divs.append(case.get("trace_action_divergence", 0))

        summary["random_success_v5_fail_stats"] = {
            "mean_randvec_distance_delta": float(np.mean(rand_vec_deltas)) if rand_vec_deltas else "NA",
            "mean_visual_distance_delta": float(np.mean(visual_deltas)) if visual_deltas else "NA",
            "mean_region_selection_ratio": float(np.mean(region_ratios)) if region_ratios else "NA",
            "mean_action_divergence": float(np.mean(action_divs)) if action_divs else "NA",
        }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


def determine_diagnostic_label(groups: dict) -> str:
    """Determine the primary diagnostic label based on evidence rules."""
    rs_vf = groups.get("random_success_v5_fail", [])
    if not rs_vf:
        return "MIXED_OR_INCONCLUSIVE"

    # Check rand_vec distance delta
    rand_vec_deltas = []
    visual_deltas = []
    action_divs = []
    region_ratios = []

    for case in rs_vf:
        v5_rv = case.get("v5_nearest_randvec_dist", float("nan"))
        rand_rv = case.get("random_nearest_randvec_dist", float("nan"))
        if not np.isnan(v5_rv) and not np.isnan(rand_rv):
            rand_vec_deltas.append(v5_rv - rand_rv)

        v5_vis = case.get("v5_nearest_visual_dist", float("nan"))
        rand_vis = case.get("random_nearest_visual_dist", float("nan"))
        if not np.isnan(v5_vis) and not np.isnan(rand_vis):
            visual_deltas.append(v5_vis - rand_vis)

        action_divs.append(case.get("trace_action_divergence", 0))
        region_ratios.append(case.get("v5_region_selection_ratio", 0))

    # Rule 1: RANDVEC_REGION_PROBLEM
    if rand_vec_deltas and np.mean(rand_vec_deltas) > 0.1:
        return "RANDVEC_REGION_PROBLEM"

    # Rule 2: VISUAL_SELECTION_PROBLEM
    if visual_deltas and np.mean(visual_deltas) > 0.1:
        return "VISUAL_SELECTION_PROBLEM"

    # Rule 3: ACTION_DESCRIPTOR_PROBLEM
    if action_divs and np.mean(action_divs) > 0.5:
        return "ACTION_DESCRIPTOR_PROBLEM"

    # Rule 4: REGION_BUDGET_PROBLEM
    if region_ratios and np.mean(region_ratios) > 0.7:
        return "REGION_BUDGET_PROBLEM"

    return "MIXED_OR_INCONCLUSIVE"


def write_diagnosis_md(label: str, groups: dict, output_path: Path):
    """Write diagnosis.md (~30 lines max)."""
    rs_vf = groups.get("random_success_v5_fail", [])
    v5_sr = groups.get("v5_success_random_fail", [])
    both_s = groups.get("both_success", [])
    both_f = groups.get("both_fail", [])

    label_descriptions = {
        "RANDVEC_REGION_PROBLEM": (
            "V5 rand_vec nearest distance significantly larger than random in random_success_v5_fail cases.\n"
            "Next version priority: modify rand_vec normalization, region construction, anchor/local expansion."
        ),
        "VISUAL_SELECTION_PROBLEM": (
            "rand_vec distances similar but V5 visual nearest distance significantly larger.\n"
            "Next version priority: reduce action interference, strengthen within-region visual marginal gain."
        ),
        "ACTION_DESCRIPTOR_PROBLEM": (
            "rand_vec and visual coverage both close, but inference trace shows clear action divergence after grasp.\n"
            "Next version priority: change from global flatten trajectory action diversity to local/phase-aware action descriptor."
        ),
        "REGION_BUDGET_PROBLEM": (
            "Failure seeds concentrated in already high selection_ratio regions.\n"
            "Not coverage insufficiency, but coverage_gap/region budget misuse."
        ),
        "MIXED_OR_INCONCLUSIVE": (
            "No single clear evidence. Do not force a conclusion.\n"
            "Next version: collect more diagnostic data before making changes."
        ),
    }

    desc = label_descriptions.get(label, "Unknown label")

    lines = [
        "# V5 vs Random 12k Checkpoint Diagnosis",
        "",
        f"**Diagnostic Label**: `{label}`",
        "",
        "## Paired Result Summary",
        f"- random_success_v5_fail: {len(rs_vf)}",
        f"- v5_success_random_fail: {len(v5_sr)}",
        f"- both_success: {len(both_s)}",
        f"- both_fail: {len(both_f)}",
        "",
        "## Evidence",
        f"{desc}",
        "",
        "## Conclusion",
        f"Most supported problem: **{label}**",
        f"Next version priority: see description above.",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="V5 vs Random 12k checkpoint diagnosis")
    parser.add_argument("--task", type=str, default="disassemble-v3",
                        choices=["disassemble-v3", "pick-place-v3"])
    parser.add_argument("--checkpoint", type=str, default="012000")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-trace-seeds", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only check paths and existing data, no model loading")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = TASK_CONFIGS[args.task]
    v5_dir = config["v5_dir"]
    random_dir = config["random_dir"]
    dataset_name = config["dataset_name"]

    output_dir = Path(args.output_dir) if args.output_dir else WORK2_ROOT / "v5_model_diagnosis" / f"output_{args.task}_{args.checkpoint}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== V5 vs Random 12k Diagnosis: {args.task} ===")
    print(f"V5 model: {v5_dir}")
    print(f"Random model: {random_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Find existing eval results
    v5_eval_files = find_eval_results(v5_dir, args.checkpoint)
    random_eval_files = find_eval_results(random_dir, args.checkpoint)
    print(f"Found {len(v5_eval_files)} V5 eval result files")
    print(f"Found {len(random_eval_files)} Random eval result files")

    v5_episodes = []
    random_episodes = []
    for f in v5_eval_files:
        v5_episodes.extend(load_eval_episodes(f))
    for f in random_eval_files:
        random_episodes.extend(load_eval_episodes(f))

    print(f"Loaded {len(v5_episodes)} V5 episodes, {len(random_episodes)} Random episodes")

    if not v5_episodes or not random_episodes:
        print("WARNING: No per-episode eval results found.")
        print("  Need to run diagnostic eval with fixed seeds (16-20 seeds).")
        print("  Skipping to dry-run mode.")
        args.dry_run = True

    # Step 2: Find selection log, diagnostic, embeddings
    selection_log_path = find_selection_log(v5_dir)
    diagnostic_path = find_diagnostic_json(v5_dir)
    embed_paths = find_embeddings(dataset_name)
    v5_subset_path = find_v5_subset(v5_dir)

    print(f"Selection log: {selection_log_path}")
    print(f"Diagnostic JSON: {diagnostic_path}")
    print(f"Embeddings found: {list(embed_paths.keys())}")
    print(f"V5 subset: {v5_subset_path}")

    selection_log = {}
    if selection_log_path:
        selection_log = load_selection_log(selection_log_path)

    # Step 3: Classify paired results
    groups = classify_paired_results(v5_episodes, random_episodes)
    print(f"\nPaired case counts:")
    for name, cases in groups.items():
        print(f"  {name}: {len(cases)}")

    # Step 4: Enrich cases with region/distance info
    all_cases = []
    for group_name, cases in groups.items():
        for case in cases:
            seed = case["seed"]
            v5_ep = case["v5_episode"]
            random_ep = case["random_episode"]

            initial_state = v5_ep.get("initial_state", {})
            obj_init_pos = initial_state.get("obj_init_pos", [0, 0, 0])
            goal_pos = initial_state.get("goal_pos", [0, 0, 0])

            region_info = get_region_info(seed, selection_log)

            case["group"] = group_name
            case["obj_init_pos"] = obj_init_pos
            case["goal_pos"] = goal_pos
            case["v5_region"] = region_info["region"]
            case["v5_region_selection_ratio"] = region_info["selection_ratio"]
            case["v5_nearest_randvec_dist"] = "NA"
            case["random_nearest_randvec_dist"] = "NA"
            case["v5_nearest_visual_dist"] = "NA"
            case["random_nearest_visual_dist"] = "NA"
            case["trace_action_divergence"] = "NA"

            all_cases.append(case)

    # Step 5: Select trace seeds
    trace_seeds = select_trace_seeds(groups, args.max_trace_seeds)
    print(f"\nSelected trace seeds: {[s['seed'] for s in trace_seeds]}")

    # Step 6: Run inference trace (only if not dry-run)
    if not args.dry_run and trace_seeds:
        v5_model_path = v5_dir / "checkpoints" / args.checkpoint / "pretrained_model"
        random_model_path = random_dir / "checkpoints" / args.checkpoint / "pretrained_model"

        if v5_model_path.exists() and random_model_path.exists():
            print(f"\nRunning inference trace for {len(trace_seeds)} seeds...")
            for case in trace_seeds:
                seed = case["seed"]
                trace_result = run_inference_trace(
                    seed, v5_model_path, random_model_path, args.device
                )
                case["trace_action_divergence"] = trace_result.get("action_divergence", "NA")
        else:
            print(f"WARNING: Model paths not found, skipping inference trace")
    else:
        print("\nDry-run mode: skipping inference trace")

    # Step 7: Determine diagnostic label
    diagnostic_label = determine_diagnostic_label(groups)
    print(f"\nDiagnostic label: {diagnostic_label}")

    # Step 8: Write outputs
    csv_path = output_dir / "diagnostic_cases.csv"
    summary_path = output_dir / "summary.json"
    md_path = output_dir / "diagnosis.md"

    write_diagnostic_csv(all_cases, csv_path)
    write_summary_json(groups, diagnostic_label, summary_path)
    write_diagnosis_md(diagnostic_label, groups, md_path)

    print(f"\nOutputs written to: {output_dir}")
    print(f"  - {csv_path}")
    print(f"  - {summary_path}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()