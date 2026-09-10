"""
TinyVLA Evaluation Wrapper for DifferentVLM

Calls the existing lerobot-eval flow.

Key changes from previous version:
- Each run creates independent log and output directory (timestamp-based)
- No appending to old logs, no reading from old eval_results.json
- On non-zero return code, missing results, or JSON parse failure, mark as failed
- Read aggregated.pc_success and aggregated.pc_grasp_success from eval output JSON
- Record checkpoint path, checkpoint step, n_action_steps, episode count, return code, and completion time
"""

import sys
import os
import json
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.configs.vlm_config import VLMExperimentConfig


def find_best_checkpoint(checkpoint_dir: str) -> Optional[tuple]:
    """Find the best (latest) checkpoint directory.
    
    Returns:
        Tuple of (checkpoint_step, checkpoint_path) or None if not found.
    """
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
    return checkpoints[0]


def run_tinyvla_eval(cfg: VLMExperimentConfig, checkpoint_dir: str, n_action_steps: int = 1) -> Dict:
    """
    Run evaluation with independent output directory per run.
    
    Args:
        cfg: Experiment configuration.
        checkpoint_dir: Directory containing checkpoint subdirectories.
        n_action_steps: Number of action steps per model invocation.
    
    Returns:
        Evaluation results dict with status, metrics, and metadata.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"Running TinyVLA Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"N episodes: {cfg.eval_n_episodes}")
    print(f"N action steps: {n_action_steps}")
    print(f"Seed: {cfg.eval_seed}")
    print(f"GPU: {cfg.gpu_id}")
    print(f"Timestamp: {timestamp}")
    print(f"{'='*60}")

    ckpt_info = find_best_checkpoint(checkpoint_dir)
    if ckpt_info is None:
        return {
            "status": "failed",
            "error": f"No valid checkpoint found in {checkpoint_dir}",
            "timestamp": timestamp,
        }
    
    checkpoint_step, checkpoint_path = ckpt_info
    print(f"Using checkpoint: {checkpoint_path} (step {checkpoint_step})")
    sys.stdout.flush()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    # Create independent output directory for this run
    eval_output_dir = Path(cfg.results_dir) / "eval_results" / f"tinyvla_{timestamp}"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Create independent log directory and file for this run (no append)
    Path(cfg.logs_dir).mkdir(parents=True, exist_ok=True)
    eval_log = Path(cfg.logs_dir) / f"tinyvla_eval_{timestamp}.log"

    cmd = [
        "lerobot-eval",
        f"--policy.path={checkpoint_path}",
        "--env.type=metaworld",
        f"--env.task={cfg.env_task}",
        f"--env.camera_name={cfg.camera_names}",
        "--env.use_self_mw=true",
        f"--eval.batch_size={cfg.eval_batch_size}",
        f"--eval.n_episodes={cfg.eval_n_episodes}",
        f"--policy.n_action_steps={n_action_steps}",
        "--policy.device=cuda",
        f"--rename_map={cfg.rename_map}",
        f"--output_dir={eval_output_dir}",
        f"--seed={cfg.eval_seed}",
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
        "checkpoint_step": checkpoint_step,
        "n_action_steps": n_action_steps,
        "n_episodes": cfg.eval_n_episodes,
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
    
    Reads pc_success and pc_grasp_success from eval_info.json.
    The lerobot-eval script writes:
    - eval_info.json: {"overall": {"pc_success": ..., "pc_grasp_success": ...}, "per_task": [...], ...}
    - eval_episode_results.json: per-episode results
    """
    all_result_files = []
    for result_file in eval_output_dir.rglob("eval_info.json"):
        all_result_files.append(result_file)
    for result_file in eval_output_dir.rglob("eval_episode_results.json"):
        all_result_files.append(result_file)
    
    all_result_files = sorted(set(all_result_files))
    
    for result_file in all_result_files:
        try:
            with open(result_file, "r") as f:
                data = json.load(f)
            
            pc_success = None
            pc_grasp_success = None
            
            # Try "overall" key (lerobot-eval structure)
            overall = data.get("overall", {})
            pc_success = overall.get("pc_success")
            pc_grasp_success = overall.get("pc_grasp_success")
            
            # Fallback: try "aggregated" key
            if pc_success is None and pc_grasp_success is None:
                aggregated = data.get("aggregated", {})
                pc_success = aggregated.get("pc_success")
                pc_grasp_success = aggregated.get("pc_grasp_success")
            
            # Fallback: try top-level keys directly
            if pc_success is None and pc_grasp_success is None:
                pc_success = data.get("pc_success")
                pc_grasp_success = data.get("pc_grasp_success")
            
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