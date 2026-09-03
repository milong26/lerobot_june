"""
Evaluation Wrapper for DifferentVLM

Calls the existing lerobot-eval flow with fixed episode=200, seed=42.

Output:
- eval_results.json: aggregated metrics (pc_success, pc_grasp_success, etc.)
- eval_episodes.json: per-episode detailed results (success list, grasp_success list)
- Output directory isolated by vlm_name + camera
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

from differentvlm.configs.vlm_config import VLMExperimentConfig


def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the best (latest) checkpoint directory."""
    cp_root = Path(checkpoint_dir)
    if not cp_root.exists():
        return None

    checkpoints = []
    for step_dir in cp_root.iterdir():
        if step_dir.is_dir() and step_dir.name.isdigit():
            pretrained = step_dir / "pretrained_model"
            if pretrained.exists():
                checkpoints.append((int(step_dir.name), str(pretrained)))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def run_eval(cfg: VLMExperimentConfig, checkpoint_dir: str) -> Dict:
    """
    Run evaluation with fixed episode=200, seed=42.
    Returns evaluation results dict.
    """
    print(f"\n{'='*60}")
    print(f"Running Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"N episodes: {cfg.eval_n_episodes}")
    print(f"Seed: {cfg.eval_seed}")
    print(f"GPU: {cfg.gpu_id}")
    print(f"{'='*60}")

    checkpoint_path = find_best_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No valid checkpoint found in {checkpoint_dir}")

    print(f"Using checkpoint: {checkpoint_path}")
    sys.stdout.flush()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    eval_output_dir = Path(cfg.results_dir) / "eval_results"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    eval_log = Path(cfg.logs_dir) / f"{cfg.vlm_name}_{cfg.camera}_eval.log"

    cmd = [
        "lerobot-eval",
        f"--policy.path={checkpoint_path}",
        "--env.type=metaworld",
        "--env.task=pick-place-v3",
        f"--env.camera_name={cfg.camera_names}",
        "--env.use_self_mw=true",
        f"--eval.batch_size={cfg.eval_batch_size}",
        f"--eval.n_episodes={cfg.eval_n_episodes}",
        "--policy.device=cuda",
    ]

    print(f"\nRunning: lerobot-eval ...")
    print(f"Eval log: {eval_log}")
    sys.stdout.flush()

    with open(eval_log, "w") as log_f:
        result = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parents[4]),
        )

    if result.returncode != 0:
        print(f"WARNING: Eval exited with code {result.returncode}")
        sys.stdout.flush()

    metrics, episode_details = parse_eval_log(str(eval_log))

    results_file = eval_output_dir / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)

    episodes_file = eval_output_dir / "eval_episodes.json"
    with open(episodes_file, "w") as f:
        json.dump(episode_details, f, indent=2)

    print(f"\nEval results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to: {results_file}")
    print(f"Per-episode details saved to: {episodes_file}")
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