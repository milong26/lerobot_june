"""
MiniVLA Evaluation Wrapper for DifferentVLM

Calls the existing lerobot-eval flow.

Key changes from previous version:
- Each run creates independent log and output directory (timestamp-based)
- No appending to old logs, no reading from old eval_results.json
- On non-zero return code, missing results, or JSON parse failure, mark as failed
- Read aggregated.pc_success and aggregated.pc_grasp_success from eval output JSON
- Record checkpoint path, checkpoint step, episode count, return code, and completion time
"""

import sys
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.minivla.configs.minivla_config import MiniVLAExperimentConfig


def find_all_checkpoints(checkpoint_dir: str) -> List[tuple]:
    """Find all checkpoint directories, sorted by step number.
    
    Returns:
        List of (step_number, checkpoint_path) tuples.
    """
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


def run_minivla_eval(cfg: MiniVLAExperimentConfig, checkpoint_path: str, step_number: int, n_episodes: int | None = None) -> Dict:
    """
    Run evaluation for a single checkpoint with independent output directory.
    
    Args:
        cfg: Experiment configuration.
        checkpoint_path: Path to pretrained_model directory.
        step_number: Training step number of the checkpoint.
        n_episodes: Override episode count (uses cfg default if None).
    
    Returns:
        Evaluation results dict with status, metrics, and metadata.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    n_eps = n_episodes if n_episodes is not None else cfg.eval_n_episodes
    
    print(f"\n{'='*60}")
    print(f"Running MiniVLA Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path} (step {step_number})")
    print(f"N episodes: {n_eps}")
    print(f"Seed: {cfg.eval_seed}")
    print(f"GPU: {cfg.gpu_id}")
    print(f"Timestamp: {timestamp}")
    print(f"{'='*60}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    # Create independent output directory for this run
    eval_output_dir = Path(cfg.eval_results_dir) / f"minivla_step{step_number:06d}_{timestamp}"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Create independent log file for this run (write mode, not append)
    eval_log = Path(cfg.logs_dir) / f"minivla_eval_step{step_number:06d}_{timestamp}.log"

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
        f"--eval.n_episodes={n_eps}",
        "--policy.device=cuda",
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    print(f"Eval log: {eval_log}")
    print(f"Output dir: {eval_output_dir}")
    sys.stdout.flush()

    # Run eval with independent log (write mode, not append)
    with open(eval_log, "w") as log_f:
        result = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parents[4]),
        )

    completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build base result metadata
    base_result = {
        "timestamp": timestamp,
        "completion_time": completion_time,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_step": step_number,
        "n_episodes": n_eps,
        "returncode": result.returncode,
        "log_file": str(eval_log),
        "output_dir": str(eval_output_dir),
    }

    # Check for failure conditions
    if result.returncode != 0:
        print(f"EVALUATION FAILED: Exit code {result.returncode}")
        print(f"Check log: {eval_log}")
        sys.stdout.flush()
        
        failed_result = {
            **base_result,
            "status": "failed",
            "error": f"Evaluation failed with exit code {result.returncode}",
            "aggregated": {},
        }
        _save_results(failed_result, eval_output_dir)
        return failed_result

    # Parse results from eval output JSON files
    results = _parse_eval_results(eval_output_dir, base_result)
    _save_results(results, eval_output_dir)
    
    print(f"\nEvaluation complete.")
    print(f"Status: {results['status']}")
    if results.get("aggregated"):
        agg = results["aggregated"]
        print(f"pc_success: {agg.get('pc_success', 'N/A')}")
        print(f"pc_grasp_success: {agg.get('pc_grasp_success', 'N/A')}")
    print(f"Results saved to: {eval_output_dir / 'eval_results.json'}")
    sys.stdout.flush()

    return results


def _parse_eval_results(eval_output_dir: Path, base_result: Dict) -> Dict:
    """Parse evaluation results from JSON output files.
    
    Reads aggregated.pc_success and aggregated.pc_grasp_success directly
    from the eval output JSON. Does NOT use regex on log files.
    
    The lerobot-eval script writes:
    - eval_info.json: contains {"aggregated": {"pc_success": ..., "pc_grasp_success": ...}}
    - eval_episode_results.json: per-episode results
    """
    # Search in output dir and parent dirs for eval output files
    search_dirs = [eval_output_dir, eval_output_dir.parent, eval_output_dir.parent.parent]
    all_result_files = []
    for d in search_dirs:
        if d.exists():
            all_result_files.extend(d.glob("eval_info.json"))
            all_result_files.extend(d.glob("eval_episode_results.json"))
            all_result_files.extend(d.glob("eval_results.json"))
    
    all_result_files = sorted(set(all_result_files))
    
    for result_file in all_result_files:
        try:
            with open(result_file, "r") as f:
                data = json.load(f)
            
            # Look for aggregated metrics (top-level "aggregated" key)
            aggregated = data.get("aggregated", {})
            pc_success = aggregated.get("pc_success")
            pc_grasp_success = aggregated.get("pc_grasp_success")
            
            if pc_success is not None or pc_grasp_success is not None:
                return {
                    **base_result,
                    "status": "success",
                    "aggregated": {
                        "pc_success": pc_success,
                        "pc_grasp_success": pc_grasp_success,
                    },
                    "raw_data": data,
                    "source_file": str(result_file),
                }
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to parse {result_file}: {e}")
            continue
    
    # If no aggregated results found, mark as failed
    return {
        **base_result,
        "status": "failed",
        "error": "No aggregated results found in eval output JSON files",
        "aggregated": {},
        "searched_files": [str(f) for f in all_result_files],
    }


def _save_results(results: Dict, output_dir: Path):
    """Save evaluation results to output directory."""
    results_file = output_dir / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)