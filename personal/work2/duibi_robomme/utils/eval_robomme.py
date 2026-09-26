#!/usr/bin/env python
"""
Robomme evaluation wrapper for the duibi_robomme pipeline.

Calls the repository's lerobot-eval infrastructure (src/lerobot/scripts/lerobot_eval.py)
with RoboMMEEnv and BenchmarkEnvBuilder. Does NOT implement custom policy loading
or raw rollout loops.

Task mapping:
  MoveCube_easy    -> MoveCube
  PatternLock_medium -> PatternLock
  RouteStick_hard  -> RouteStick

Usage:
    python eval_robomme.py \
        --checkpoint-path /path/to/pretrained_model \
        --task MoveCube_easy \
        --eval-task-ids 0,1,2,3,4 \
        --output-dir /path/to/eval_output \
        [--gpu-id 0] [--episode-length 300]
"""

import argparse
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

TASK_NAME_MAP = {
    "MoveCube_easy": "MoveCube",
    "PatternLock_medium": "PatternLock",
    "RouteStick_hard": "RouteStick",
}

DEPRECATED_EVAL_SEEDS = {
    "MoveCube_easy": [0, 1, 2, 3, 4],
    "PatternLock_medium": [0, 1, 2, 3, 4],
    "RouteStick_hard": [0, 1, 2, 3, 4],
}


def parse_eval_task_ids(task_ids_str: str) -> list[int]:
    """Parse comma-separated episode IDs into a list of ints."""
    return [int(x.strip()) for x in task_ids_str.split(",") if x.strip()]


def run_lerobot_eval(
    checkpoint_path: str,
    robomme_task: str,
    eval_task_ids: list[int],
    output_dir: str,
    gpu_id: int = 0,
    episode_length: int = 300,
) -> dict:
    """Run current lerobot-eval on exact RoboMME benchmark episode IDs.

    Each explicit task_id is created as a one-environment evaluation target and
    evaluated exactly once. This avoids repeating fixed RoboMME episodes when
    eval.n_episodes is larger than the vector size.
    """
    task_ids_arg = "[" + ",".join(str(x) for x in eval_task_ids) + "]"

    cmd = [
        "lerobot-eval",
        f"--policy.path={checkpoint_path}",
        f"--policy.device=cuda:{gpu_id}",
        "--policy.use_amp=false",
        "--env.type=robomme",
        f"--env.task={robomme_task}",
        "--env.action_space=joint_angle",
        "--env.dataset_split=test",
        f"--env.episode_length={episode_length}",
        f"--env.task_ids={task_ids_arg}",
        "--eval.batch_size=1",
        "--eval.n_episodes=1",
        f"--output_dir={output_dir}",
    ]

    print("Running lerobot-eval:")
    print("  " + " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent.parent.parent.parent),
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"lerobot-eval failed with return code {result.returncode}. "
            f"stderr: {result.stderr[:1000]}"
        )

    eval_info_path = Path(output_dir) / "eval_info.json"
    if not eval_info_path.exists():
        raise FileNotFoundError(
            f"lerobot-eval completed but did not produce {eval_info_path}"
        )
    with open(eval_info_path, "r") as f:
        return json.load(f)

def parse_eval_result(eval_info: dict, task_name: str, robomme_task: str,
                      eval_task_ids: list[int], checkpoint_path: str) -> dict:
    """Convert the current eval_policy_all schema to the pipeline result schema."""
    group_info = eval_info.get("per_group", {}).get(robomme_task, {})
    per_task = eval_info.get("per_task", [])

    episode_results = []
    num_success = 0
    for item in per_task:
        if item.get("task_group") != robomme_task:
            continue
        episode_id = int(item.get("task_id", 0))
        metrics = item.get("metrics", {})
        successes = metrics.get("successes", [])
        rewards = metrics.get("sum_rewards", [])
        success = bool(successes[0]) if successes else False
        if success:
            num_success += 1
        episode_results.append({
            "episode_index": episode_id,
            "success": success,
            "reward": float(rewards[0]) if rewards else 0.0,
        })

    episode_results.sort(key=lambda x: x["episode_index"])
    num_episodes = int(group_info.get("n_episodes", len(episode_results)))
    if not episode_results and num_episodes:
        pc_success = float(group_info.get("pc_success", 0.0))
        num_success = round(pc_success * num_episodes / 100.0)

    success_rate = num_success / num_episodes if num_episodes > 0 else 0.0

    return {
        "task": task_name,
        "robomme_task": robomme_task,
        "checkpoint": checkpoint_path,
        "eval_task_ids": eval_task_ids,
        "num_episodes": num_episodes,
        "num_success": num_success,
        "success_rate": success_rate,
        "episode_results": episode_results,
        "aggregated": group_info,
    }

def evaluate_task(
    checkpoint_path: str,
    task_name: str,
    eval_task_ids: list[int],
    output_dir: str,
    gpu_id: int = 0,
    episode_length: int = 300,
) -> dict:
    """Evaluate a single Robomme task using lerobot-eval."""
    robomme_task = TASK_NAME_MAP.get(task_name, task_name)

    print(f"\n{'='*60}")
    print(f"Evaluating task: {task_name} (RoboMME task: {robomme_task})")
    print(f"Test episode IDs: {eval_task_ids}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    eval_info = run_lerobot_eval(
        checkpoint_path=checkpoint_path,
        robomme_task=robomme_task,
        eval_task_ids=eval_task_ids,
        output_dir=str(output_path),
        gpu_id=gpu_id,
        episode_length=episode_length,
    )

    eval_result = parse_eval_result(
        eval_info=eval_info,
        task_name=task_name,
        robomme_task=robomme_task,
        eval_task_ids=eval_task_ids,
        checkpoint_path=checkpoint_path,
    )

    result_file = output_path / "eval_result.json"
    with open(result_file, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"  Result saved to: {result_file}")
    print(f"  Success rate: {eval_result['num_success']}/{eval_result['num_episodes']} = {eval_result['success_rate']:.4f}")

    return eval_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA on Robomme task via lerobot-eval")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                       help="Path to pretrained_model directory")
    parser.add_argument("--task", type=str, required=True,
                       choices=["MoveCube_easy", "PatternLock_medium", "RouteStick_hard"])
    parser.add_argument("--eval-task-ids", type=str, default=None,
                       help="Comma-separated benchmark test episode IDs (e.g., 0,1,2,3,4)")
    parser.add_argument("--eval-seeds", type=str, default=None,
                       help="[DEPRECATED] Comma-separated seeds; converted to fixed benchmark episode IDs")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=300)
    args = parser.parse_args()

    if args.eval_task_ids is not None:
        eval_task_ids = parse_eval_task_ids(args.eval_task_ids)
    elif args.eval_seeds is not None:
        warnings.warn(
            "--eval-seeds is deprecated. It will be converted to fixed benchmark episode IDs. "
            "Please use --eval-task-ids instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        eval_task_ids = DEPRECATED_EVAL_SEEDS.get(args.task, [0, 1, 2, 3, 4])
        print(f"WARNING: --eval-seeds is deprecated. Using fixed benchmark episode IDs: {eval_task_ids}")
    else:
        eval_task_ids = DEPRECATED_EVAL_SEEDS.get(args.task, [0, 1, 2, 3, 4])
        print(f"Using default benchmark episode IDs: {eval_task_ids}")

    evaluate_task(
        checkpoint_path=args.checkpoint_path,
        task_name=args.task,
        eval_task_ids=eval_task_ids,
        output_dir=args.output_dir,
        gpu_id=args.gpu_id,
        episode_length=args.episode_length,
    )


if __name__ == "__main__":
    main()