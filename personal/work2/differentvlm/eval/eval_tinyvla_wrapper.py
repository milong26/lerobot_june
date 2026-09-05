"""
TinyVLA Evaluation Wrapper for DifferentVLM

Calls the existing lerobot-eval flow with fixed episode=200, seed=42.

Output:
- eval_results.json: aggregated metrics (pc_success, pc_grasp_success, etc.)
- eval_episodes.json: per-episode detailed results (success list, grasp_success list)
- Output directory isolated by tinyvla policy type + camera
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


def run_tinyvla_eval(cfg: VLMExperimentConfig, checkpoint_dir: str) -> Dict:
    """
    Run evaluation with fixed episode=200, seed=42.
    Returns evaluation results dict.
    """
    print(f"\n{'='*60}")
    print(f"Running TinyVLA Evaluation")
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
        print(f"WARNING: Evaluation exited with code {result.returncode}")
        print(f"Check log: {eval_log}")
        sys.stdout.flush()
        return {"error": f"Evaluation failed with code {result.returncode}", "log_file": str(eval_log)}

    print(f"\nEvaluation complete.")
    print(f"Eval log: {eval_log}")
    sys.stdout.flush()

    results = parse_eval_log(eval_log)
    results_file = eval_output_dir / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Eval results saved to: {results_file}")
    sys.stdout.flush()

    return results


def parse_eval_log(log_file: Path) -> Dict:
    """Parse evaluation log to extract metrics."""
    with open(log_file, "r") as f:
        log_content = f.read()

    results = {"log_file": str(log_file)}

    success_match = re.search(r'success.*?(\d+\.?\d*)', log_content, re.IGNORECASE)
    if success_match:
        results["success_rate"] = float(success_match.group(1))

    grasp_match = re.search(r'grasp_success.*?(\d+\.?\d*)', log_content, re.IGNORECASE)
    if grasp_match:
        results["grasp_success_rate"] = float(grasp_match.group(1))

    return results