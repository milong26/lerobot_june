#!/usr/bin/env python
"""
Aggregate evaluation results from three Robomme tasks into a single summary.

Reads structured JSON from each task's evaluation, computes per-task and mean
success rates, and updates the global results file.

Usage:
    python aggregate_results.py \
        --eval-dir /path/to/eval \
        --method ours_v5 \
        --seed 42 \
        --k 28 \
        --output-dir /path/to/output \
        --global-results /path/to/all_results.csv
"""

import argparse
import csv
import json
import os
from pathlib import Path

ROBOMME_TASKS = ["MoveCube_easy", "PatternLock_medium", "RouteStick_hard"]


def load_task_eval(eval_dir, task_name):
    """Load evaluation results for a single task."""
    task_dir = Path(eval_dir) / task_name
    result_file = task_dir / "eval_result.json"

    if not result_file.exists():
        raise FileNotFoundError(f"Evaluation result not found: {result_file}")

    with open(result_file, "r") as f:
        data = json.load(f)

    return data


def aggregate_results(eval_dir, method, seed, k, output_dir, global_results_path=None):
    """Aggregate results from all three tasks."""
    print(f"\n{'='*60}")
    print(f"Aggregating results for method={method}, seed={seed}, k={k}")
    print(f"{'='*60}")

    task_results = {}
    success_rates = {}

    for task_name in ROBOMME_TASKS:
        data = load_task_eval(eval_dir, task_name)
        task_results[task_name] = data

        num_episodes = data.get("num_episodes", 0)
        num_success = data.get("num_success", 0)
        success_rate = data.get("success_rate", 0.0)

        success_rates[task_name] = success_rate
        print(f"  {task_name}: {num_success}/{num_episodes} = {success_rate:.4f}")

    mean_success_rate = sum(success_rates.values()) / len(success_rates) if success_rates else 0.0
    print(f"\n  Mean success rate: {mean_success_rate:.4f}")

    summary = {
        "method": method,
        "seed": seed,
        "k_per_task": k,
        "total_selected": k * len(ROBOMME_TASKS),
        "MoveCube_easy_success_rate": success_rates.get("MoveCube_easy", 0.0),
        "PatternLock_medium_success_rate": success_rates.get("PatternLock_medium", 0.0),
        "RouteStick_hard_success_rate": success_rates.get("RouteStick_hard", 0.0),
        "mean_success_rate": mean_success_rate,
        "status": "completed",
        "task_details": {},
    }

    for task_name in ROBOMME_TASKS:
        data = task_results[task_name]
        summary["task_details"][task_name] = {
            "num_episodes": data.get("num_episodes", 0),
            "num_success": data.get("num_success", 0),
            "success_rate": data.get("success_rate", 0.0),
            "episode_results": data.get("episode_results", []),
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_dir / "summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_json_path}")

    summary_csv_path = output_dir / "summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "seed", "k_per_task", "total_selected",
            "MoveCube_easy_success_rate", "PatternLock_medium_success_rate",
            "RouteStick_hard_success_rate", "mean_success_rate", "status"
        ])
        writer.writerow([
            method, seed, k, k * len(ROBOMME_TASKS),
            success_rates.get("MoveCube_easy", 0.0),
            success_rates.get("PatternLock_medium", 0.0),
            success_rates.get("RouteStick_hard", 0.0),
            mean_success_rate, "completed"
        ])
    print(f"  Summary CSV saved to: {summary_csv_path}")

    if global_results_path:
        global_results_path = Path(global_results_path)
        global_results_path.parent.mkdir(parents=True, exist_ok=True)

        run_key = f"{method}_seed{seed}_k{k}"

        if global_results_path.exists():
            with open(global_results_path, "r") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
                fieldnames = reader.fieldnames
        else:
            existing_rows = []
            fieldnames = [
                "method", "seed", "k_per_task", "total_selected",
                "MoveCube_easy_episodes", "PatternLock_medium_episodes", "RouteStick_hard_episodes",
                "MoveCube_easy_success_rate", "PatternLock_medium_success_rate",
                "RouteStick_hard_success_rate", "mean_success_rate",
                "merged_dataset_path", "checkpoint_path",
                "status", "run_key"
            ]

        new_row = {
            "method": method,
            "seed": seed,
            "k_per_task": k,
            "total_selected": k * len(ROBOMME_TASKS),
            "MoveCube_easy_episodes": task_results["MoveCube_easy"].get("num_episodes", 0),
            "PatternLock_medium_episodes": task_results["PatternLock_medium"].get("num_episodes", 0),
            "RouteStick_hard_episodes": task_results["RouteStick_hard"].get("num_episodes", 0),
            "MoveCube_easy_success_rate": success_rates.get("MoveCube_easy", 0.0),
            "PatternLock_medium_success_rate": success_rates.get("PatternLock_medium", 0.0),
            "RouteStick_hard_success_rate": success_rates.get("RouteStick_hard", 0.0),
            "mean_success_rate": mean_success_rate,
            "merged_dataset_path": "",
            "checkpoint_path": "",
            "status": "completed",
            "run_key": run_key,
        }

        updated = False
        for i, row in enumerate(existing_rows):
            if row.get("run_key") == run_key:
                existing_rows[i] = {**row, **new_row}
                updated = True
                print(f"  Updated existing row for {run_key}")
                break

        if not updated:
            existing_rows.append(new_row)
            print(f"  Added new row for {run_key}")

        with open(global_results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
        print(f"  Global results updated: {global_results_path}")

        global_json_path = global_results_path.with_suffix(".json")
        with open(global_json_path, "w") as f:
            json.dump(existing_rows, f, indent=2)
        print(f"  Global JSON updated: {global_json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Aggregate Robomme evaluation results")
    parser.add_argument("--eval-dir", type=str, required=True,
                       help="Directory containing eval/MoveCube_easy/, eval/PatternLock_medium/, eval/RouteStick_hard/")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--global-results", type=str, default=None,
                       help="Path to global all_results.csv")
    parser.add_argument("--merged-dataset-path", type=str, default="")
    parser.add_argument("--checkpoint-path", type=str, default="")
    args = parser.parse_args()

    summary = aggregate_results(
        eval_dir=args.eval_dir,
        method=args.method,
        seed=args.seed,
        k=args.k,
        output_dir=args.output_dir,
        global_results_path=args.global_results,
    )

    print(f"\nAggregation complete!")
    print(f"  Mean success rate: {summary['mean_success_rate']:.4f}")


if __name__ == "__main__":
    main()