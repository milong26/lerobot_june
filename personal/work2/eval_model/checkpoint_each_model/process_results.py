#!/usr/bin/env python
"""Process eval results: convert eval_episode_results.json to per-seed CSV and per-checkpoint summary."""
import csv
import json
import sys
from pathlib import Path

def process(results_json_path, per_seed_csv_path, per_checkpoint_json_path, model_name, checkpoint, gpu_id, status="ok", exitcode=None):
    results_json = Path(results_json_path)
    if not results_json.is_file():
        return False

    with open(results_json) as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    n_episodes = len(episodes)

    per_seed_rows = []
    for ep in episodes:
        per_seed_rows.append({
            "model": model_name,
            "checkpoint": checkpoint,
            "episode_ix": ep["episode_ix"],
            "seed": ep.get("seed", ""),
            "success": 1 if ep.get("success", False) else 0,
            "grasp_success": 1 if ep.get("grasp_success", False) else 0,
            "sum_reward": ep.get("sum_reward", ""),
            "max_reward": ep.get("max_reward", ""),
        })

    per_seed_csv = Path(per_seed_csv_path)
    per_seed_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(per_seed_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "checkpoint", "episode_ix", "seed", "success", "grasp_success", "sum_reward", "max_reward"])
        writer.writeheader()
        writer.writerows(per_seed_rows)

    success_count = sum(1 for ep in episodes if ep.get("success", False))
    grasp_count = sum(1 for ep in episodes if ep.get("grasp_success", False))
    success_rate = success_count / max(1, n_episodes)
    grasp_rate = grasp_count / max(1, n_episodes)

    summary = {
        "model": model_name,
        "checkpoint": checkpoint,
        "gpu": gpu_id,
        "n_episodes": n_episodes,
        "success_count": success_count,
        "grasp_success_count": grasp_count,
        "success_rate": round(success_rate, 6),
        "grasp_success_rate": round(grasp_rate, 6),
        "pc_success": round(success_rate * 100, 4),
        "pc_grasp_success": round(grasp_rate * 100, 4),
        "status": status,
    }

    per_checkpoint = Path(per_checkpoint_json_path)
    per_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with open(per_checkpoint, "w") as f:
        json.dump(summary, f, indent=2)

    return True

def process_failed(per_checkpoint_json_path, model_name, checkpoint, gpu_id, exitcode, log_path):
    summary = {
        "model": model_name,
        "checkpoint": checkpoint,
        "gpu": gpu_id,
        "n_episodes": 0,
        "success_count": 0,
        "grasp_success_count": 0,
        "success_rate": 0.0,
        "grasp_success_rate": 0.0,
        "pc_success": 0.0,
        "pc_grasp_success": 0.0,
        "status": "failed",
        "exitcode": exitcode,
        "log_path": log_path,
    }
    per_checkpoint = Path(per_checkpoint_json_path)
    per_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with open(per_checkpoint, "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ok":
        results_json = sys.argv[2]
        per_seed_csv = sys.argv[3]
        per_checkpoint_json = sys.argv[4]
        model_name = sys.argv[5]
        checkpoint = sys.argv[6]
        gpu_id = sys.argv[7]
        ok = process(results_json, per_seed_csv, per_checkpoint_json, model_name, checkpoint, gpu_id)
        sys.exit(0 if ok else 1)
    elif mode == "failed":
        per_checkpoint_json = sys.argv[2]
        model_name = sys.argv[3]
        checkpoint = sys.argv[4]
        gpu_id = sys.argv[5]
        exitcode = sys.argv[6]
        log_path = sys.argv[7]
        process_failed(per_checkpoint_json, model_name, checkpoint, gpu_id, exitcode, log_path)
        sys.exit(0)