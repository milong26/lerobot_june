"""
MiniVLA Evaluation Script

Evaluates MiniVLA checkpoints using lerobot-eval.
Supports evaluating all checkpoints (e.g. 2k, 4k, 6k, ...) in a training run.

Output:
- eval_results.json: aggregated metrics (pc_success, pc_grasp_success, etc.)
- eval_episodes.json: per-episode detailed results
- Per-checkpoint result files
"""

import sys
import os
import ast
import json
import subprocess
import re
from pathlib import Path
from typing import Dict, Optional, List

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.minivla.configs.minivla_config import MiniVLAExperimentConfig


def find_all_checkpoints(checkpoint_dir: str) -> List[tuple]:
    """Find all checkpoint directories, sorted by step number."""
    cp_root = Path(checkpoint_dir)
    if not cp_root.exists():
        return []

    checkpoints = []
    for step_dir in cp_root.iterdir():
        if step_dir.is_dir() and step_dir.name.isdigit():
            pretrained = step_dir / "pretrained_model"
            if pretrained.exists():
                checkpoints.append((int(step_dir.name), str(pretrained)))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def run_eval_for_checkpoint(
    cfg: MiniVLAExperimentConfig,
    checkpoint_path: str,
    step_number: int,
) -> Dict:
    """
    Run evaluation for a single checkpoint.
    Returns evaluation results dict.
    """
    print(f"\nRunning Evaluation for checkpoint step_{step_number:06d}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"N episodes: {cfg.eval_n_episodes}")
    print(f"Seed: {cfg.eval_seed}")
    print(f"GPU: {cfg.gpu_id}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    eval_output_dir = Path(cfg.eval_results_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    eval_log = Path(cfg.logs_dir) / f"{cfg.exp_name}_eval_step_{step_number:06d}.log"

    # Extract task name from dataset name
    dataset_base = cfg.dataset_name
    for suffix in ["_corner", "_top", "_gripper", "_left", "_right", "_front", "_back", "_view"]:
        if dataset_base.endswith(suffix):
            dataset_base = dataset_base[:-len(suffix)]
            break
    task_base = dataset_base.replace("_", "-")
    if "v3" not in task_base and "v2" not in task_base:
        task_name = f"{task_base}-v3"
    else:
        task_name = task_base

    cmd = [
        "lerobot-eval",
        f"--policy.path={checkpoint_path}",
        "--env.type=metaworld",
        f"--env.task={task_name}",
        f"--env.camera_name={cfg.camera_names}",
        "--env.use_self_mw=true",
        f"--eval.batch_size={cfg.eval_batch_size}",
        f"--eval.n_episodes={cfg.eval_n_episodes}",
        "--policy.device=cuda",
        f"--rename_map={cfg.rename_map}",
    ]

    print(f"\nRunning: lerobot-eval ...")
    print(f"Eval log: {eval_log}")
    sys.stdout.flush()

    with open(eval_log, "w") as log_f:
        result = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parents[5]),
        )

    if result.returncode != 0:
        print(f"WARNING: Eval exited with code {result.returncode}")
        sys.stdout.flush()

    metrics, episode_details = parse_eval_log(str(eval_log))

    # Save per-checkpoint results
    results_file = eval_output_dir / f"eval_results_step_{step_number:06d}.json"
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)

    episodes_file = eval_output_dir / f"eval_episodes_step_{step_number:06d}.json"
    with open(episodes_file, "w") as f:
        json.dump(episode_details, f, indent=2)

    print(f"\nEval results for step_{step_number:06d}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to: {results_file}")
    sys.stdout.flush()

    return metrics


def parse_eval_log(log_path: str) -> tuple:
    """
    Parse evaluation log to extract metrics and per-episode details.
    Returns (aggregated_metrics, episode_details).
    """
    metric = None
    success_list = []
    grasp_success_list = []
    rewards_list = []

    try:
        with open(log_path, "r", errors="ignore") as f:
            content = f.read()

        for line in content.split("\n"):
            if "pc_success" not in line:
                continue
            start = line.find("{")
            end = line.rfind("}")
            if start < 0 or end < start:
                continue
            raw = line[start:end + 1]
            try:
                value = ast.literal_eval(raw)
            except Exception:
                continue
            if isinstance(value, dict) and "pc_success" in value:
                metric = value

        ep_pattern = re.compile(r'episode.*success.*:? *(true|false|1|0)', re.IGNORECASE)
        for line in content.split("\n"):
            if "episode" in line.lower() and ("success" in line.lower() or "grasp" in line.lower()):
                match = ep_pattern.search(line)
                if match:
                    val = match.group(1).lower()
                    success_list.append(val in ("true", "1"))

        if "grasp_success" in content.lower():
            grasp_pattern = re.compile(r'grasp.*success.*:? *(true|false|1|0)', re.IGNORECASE)
            for line in content.split("\n"):
                match = grasp_pattern.search(line)
                if match:
                    val = match.group(1).lower()
                    grasp_success_list.append(val in ("true", "1"))

        reward_pattern = re.compile(r'reward.*:? *([0-9.]+)')
        for line in content.split("\n"):
            if "reward" in line.lower():
                match = reward_pattern.search(line)
                if match:
                    rewards_list.append(float(match.group(1)))

    except Exception as e:
        print(f"Error parsing eval log: {e}")
        sys.stdout.flush()

    if metric is None:
        aggregated = {
            "pc_success": -1,
            "pc_grasp_success": -1,
            "parse_status": "failed",
        }
        episode_details = {
            "success_list": [],
            "grasp_success_list": [],
            "rewards_list": [],
            "n_episodes": 0,
        }
        return aggregated, episode_details

    aggregated = {
        "pc_success": metric.get("pc_success", -1),
        "pc_grasp_success": metric.get("pc_grasp_success", -1),
        "avg_sum_reward": metric.get("avg_sum_reward", -1),
        "avg_max_reward": metric.get("avg_max_reward", -1),
        "n_episodes": metric.get("n_episodes", -1),
        "parse_status": "ok",
    }

    episode_details = {
        "success_list": success_list if success_list else [],
        "grasp_success_list": grasp_success_list if grasp_success_list else [],
        "rewards_list": rewards_list if rewards_list else [],
        "n_episodes": len(success_list) if success_list else metric.get("n_episodes", 0),
    }

    return aggregated, episode_details


def run_eval_all_checkpoints(cfg: MiniVLAExperimentConfig, checkpoint_dir: str, latest_only: bool = False) -> List[Dict]:
    """
    Run evaluation for all checkpoints in the training directory.
    Returns list of evaluation results.
    """
    checkpoints = find_all_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No valid checkpoints found in {checkpoint_dir}")

    if latest_only:
        checkpoints = [checkpoints[-1]]
        print(f"\nEvaluating LATEST checkpoint only:")
    else:
        print(f"\nFound {len(checkpoints)} checkpoints to evaluate:")

    for step_num, ckpt_path in checkpoints:
        print(f"  step_{step_num:06d}: {ckpt_path}")
    sys.stdout.flush()

    all_results = []
    for step_num, ckpt_path in checkpoints:
        try:
            metrics = run_eval_for_checkpoint(cfg, ckpt_path, step_num)
            metrics["step"] = step_num
            metrics["checkpoint_path"] = ckpt_path
            all_results.append(metrics)
        except Exception as e:
            print(f"ERROR evaluating checkpoint step_{step_num:06d}: {e}")
            all_results.append({
                "step": step_num,
                "checkpoint_path": ckpt_path,
                "pc_success": -1,
                "pc_grasp_success": -1,
                "error": str(e),
            })

    # Save summary
    summary_file = Path(cfg.eval_results_dir) / "eval_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nEval Summary")
    for r in all_results:
        step = r.get("step", "?")
        success = r.get("pc_success", "N/A")
        grasp = r.get("pc_grasp_success", "N/A")
        print(f"  step_{step:06d}: pc_success={success}, pc_grasp_success={grasp}")
    print(f"\nSummary saved to: {summary_file}")
    sys.stdout.flush()

    return all_results