#!/usr/bin/env python3
"""Run paired V5/random eval with identical seeds and initial states.

Finds divergent cases (random_success_v5_fail, v5_success_random_fail) for diagnosis.

Usage:
    python run_paired_eval.py --task disassemble-v3 --checkpoint 012000 --device cuda --n-seeds 20 --start-seed 42
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORK2_ROOT = PROJECT_ROOT / "personal" / "work2"

TASK_CONFIGS = {
    "disassemble-v3": {
        "v5_dir": WORK2_ROOT / "duibi" / "our_v5_112_seed42_disassemble-v3_corner" / "our_v5_112_seed42",
        "random_dir": WORK2_ROOT / "duibi" / "random_42_disassemble-v3_corner" / "random_112_seed42",
        "env_task": "disassemble-v3",
        "rename_map": '{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}',
    },
    "pick-place-v3": {
        "v5_dir": WORK2_ROOT / "duibi" / "our_v5_112_seed42_pick_place-v3_corner" / "our_v5_112_seed42",
        "random_dir": WORK2_ROOT / "duibi" / "random_42_pick_place-v3_corner" / "random_112_seed42",
        "env_task": "pick-place-v3",
        "rename_map": '{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}',
    },
}


def build_eval_cmd(policy_path: Path, env_task: str, n_episodes: int, start_seed: int,
                   device: str, output_dir: Path, rename_map: str) -> list:
    """Build lerobot-eval command."""
    return [
        "lerobot-eval",
        f"--policy.path={policy_path}",
        "--env.type=metaworld",
        f"--env.task={env_task}",
        "--env.camera_name=corner,gripperPOV",
        "--env.use_self_mw=true",
        "--eval.batch_size=8",
        f"--eval.n_episodes={n_episodes}",
        f"--policy.device={device}",
        f"--seed={start_seed}",
        f"--output_dir={output_dir}",
        f"--rename_map={rename_map}",
    ]


def run_eval(cmd: list, label: str) -> dict:
    """Run eval command and return per-episode results."""
    print(f"\n{'='*60}")
    print(f"Running {label} eval...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"ERROR: {label} eval failed with return code {result.returncode}")
        print(f"stderr: {result.stderr[-500:]}")
        return {"episodes": []}

    # Parse per-episode results from the output JSON
    # The eval script saves to results_dir/eval_episode_results.json
    # We need to find the output directory from the command
    return {"episodes": [], "stdout": result.stdout[-1000:], "stderr": result.stderr[-500:]}


def load_existing_episodes(model_dir: Path, checkpoint: str) -> list:
    """Load existing eval episodes from multiple possible locations."""
    episodes = []
    # Search in multiple possible result directories
    search_dirs = [
        model_dir / "eval" / f"results_step_{checkpoint}" / "paired" / "eval_results",
        model_dir / "eval" / f"results_step_{checkpoint}",
        PROJECT_ROOT / "outputs" / "eval",
    ]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for f in sorted(search_dir.rglob("eval_episode_results.json")):
            with open(f) as fh:
                data = json.load(fh)
            episodes.extend(data.get("episodes", []))
    # Deduplicate by episode index
    seen = set()
    unique_episodes = []
    for ep in episodes:
        idx = ep.get("episode_index", ep.get("seed", id(ep)))
        if idx not in seen:
            seen.add(idx)
            unique_episodes.append(ep)
    return unique_episodes


def load_latest_eval_results(task: str, label: str) -> list:
    """Load the latest eval results from outputs/eval/ directory."""
    eval_root = PROJECT_ROOT / "outputs" / "eval"
    if not eval_root.exists():
        return []
    # Find the latest directory for this task
    latest_dir = None
    latest_time = ""
    for d in sorted(eval_root.iterdir()):
        if not d.is_dir():
            continue
        # Check if this directory contains results for our task
        for f in d.rglob("eval_episode_results.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                episodes = data.get("episodes", [])
                if episodes and "seed" in episodes[0]:
                    # This is a valid result file
                    dir_time = d.name
                    if dir_time > latest_time:
                        latest_time = dir_time
                        latest_dir = d
            except (json.JSONDecodeError, KeyError):
                continue
    if latest_dir is None:
        return []
    # Load results
    for f in sorted(latest_dir.rglob("eval_episode_results.json")):
        with open(f) as fh:
            data = json.load(fh)
        episodes = data.get("episodes", [])
        if episodes:
            return episodes
    return []


def main():
    parser = argparse.ArgumentParser(description="Run paired V5/random eval")
    parser.add_argument("--task", type=str, default="disassemble-v3",
                        choices=["disassemble-v3", "pick-place-v3"])
    parser.add_argument("--checkpoint", type=str, default="012000")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-seeds", type=int, default=20,
                        help="Number of paired seeds to evaluate")
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print commands, don't run")
    parser.add_argument("--cuda-visible-devices", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES value (e.g., '0' or '1')")
    args = parser.parse_args()

    # Set required environment variables for headless MuJoCo rendering
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    config = TASK_CONFIGS[args.task]
    v5_dir = config["v5_dir"]
    random_dir = config["random_dir"]
    env_task = config["env_task"]
    rename_map = config["rename_map"]

    v5_model = v5_dir / "checkpoints" / args.checkpoint / "pretrained_model"
    random_model = random_dir / "checkpoints" / args.checkpoint / "pretrained_model"

    print(f"=== Paired Eval Runner: {args.task} ===")
    print(f"V5 model: {v5_model}")
    print(f"Random model: {random_model}")
    print(f"Seeds: {args.start_seed} to {args.start_seed + args.n_seeds - 1}")
    print(f"Device: {args.device}")
    print()

    # Check existing results
    v5_existing = load_existing_episodes(v5_dir, args.checkpoint)
    random_existing = load_existing_episodes(random_dir, args.checkpoint)
    print(f"Existing V5 episodes: {len(v5_existing)}")
    print(f"Existing Random episodes: {len(random_existing)}")

    # Build commands
    v5_cmd = build_eval_cmd(v5_model, env_task, args.n_seeds, args.start_seed,
                            args.device, v5_dir / "eval" / f"results_step_{args.checkpoint}" / "paired", rename_map)
    random_cmd = build_eval_cmd(random_model, env_task, args.n_seeds, args.start_seed,
                                args.device, random_dir / "eval" / f"results_step_{args.checkpoint}" / "paired", rename_map)

    print(f"\nV5 eval command:")
    print("  " + " ".join(v5_cmd))
    print(f"\nRandom eval command:")
    print("  " + " ".join(random_cmd))

    if args.dry_run:
        print("\nDry-run mode: commands printed, not executed")
        return

    # Run V5 eval first
    print(f"\n{'='*60}")
    print("Starting V5 eval...")
    print(f"{'='*60}")
    sys.stdout.flush()
    v5_result = subprocess.run(v5_cmd, capture_output=True, text=True, timeout=600)
    if v5_result.returncode != 0:
        print(f"ERROR: V5 eval failed with return code {v5_result.returncode}")
        print(v5_result.stderr[-500:] if v5_result.stderr else v5_result.stdout[-500:])
        sys.exit(1)
    print("V5 eval completed")

    # Run Random eval
    print(f"\n{'='*60}")
    print("Starting Random eval...")
    print(f"{'='*60}")
    sys.stdout.flush()
    random_result = subprocess.run(random_cmd, capture_output=True, text=True, timeout=600)
    if random_result.returncode != 0:
        print(f"ERROR: Random eval failed with return code {random_result.returncode}")
        print(random_result.stderr[-500:] if random_result.stderr else random_result.stdout[-500:])
        sys.exit(1)
    print("Random eval completed")

    # Load and compare results
    v5_episodes = load_existing_episodes(v5_dir, args.checkpoint)
    random_episodes = load_existing_episodes(random_dir, args.checkpoint)

    print(f"\n=== Results ===")
    print(f"V5 episodes: {len(v5_episodes)}")
    print(f"Random episodes: {len(random_episodes)}")

    # Find divergent cases
    v5_by_seed = {ep["seed"]: ep for ep in v5_episodes if "seed" in ep}
    random_by_seed = {ep["seed"]: ep for ep in random_episodes if "seed" in ep}

    common_seeds = sorted(set(v5_by_seed.keys()) & set(random_by_seed.keys()))
    divergent = []
    for seed in common_seeds:
        v5_s = v5_by_seed[seed].get("success", False)
        rand_s = random_by_seed[seed].get("success", False)
        if v5_s != rand_s:
            divergent.append({
                "seed": seed,
                "v5_success": v5_s,
                "random_success": rand_s,
            })

    print(f"\nDivergent cases: {len(divergent)}")
    for d in divergent:
        direction = "V5✓/R✗" if d["v5_success"] else "V5✗/R✓"
        print(f"  seed={d['seed']}: {direction}")

    if not divergent:
        print("\nNo divergent cases found. Consider running more seeds.")
    else:
        print(f"\nFound {len(divergent)} divergent cases for diagnosis!")


if __name__ == "__main__":
    main()