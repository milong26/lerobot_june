"""
DifferentVLM Experiment Main Entry Point

Orchestrates the full experiment pipeline:
1. VLM embedding extraction
2. V4 episode selection
3. SmolVLA training
4. Evaluation

Complete experiment state tracking:
- start_time, end_time
- current stage, failed stage
- selection VLM model id
- embedding path
- selected_episode path
- checkpoint path
- eval results path
- On exception: saves experiment_summary.json before exit
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from differentvlm.configs.vlm_config import get_config
from differentvlm.vlm_extract.auto_extract_embedding import auto_extract_embedding
from differentvlm.selection.select_v4_wrapper import run_v4_selection
from differentvlm.train.train_smolvla_wrapper import run_smolvla_training
from differentvlm.eval.eval_wrapper import run_eval


def run_experiment(vlm_name: str, gpu_id: int, camera: str = "corner"):
    """Run the complete differentvlm experiment pipeline."""
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    overall_start = time.time()

    print(f"\n{'#'*60}")
    print(f"# DifferentVLM Experiment: {vlm_name}")
    print(f"# Camera: {camera}")
    print(f"# GPU: {gpu_id}")
    print(f"# Start time: {start_time}")
    print(f"{'#'*60}")

    cfg = get_config(vlm_name=vlm_name, gpu_id=gpu_id, camera=camera)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    experiment_state = {
        "vlm_name": vlm_name,
        "camera": camera,
        "gpu_id": gpu_id,
        "start_time": start_time,
        "end_time": None,
        "current_stage": "initializing",
        "failed_stage": None,
        "selection_vlm_model_id": cfg.selection_vlm_model_id,
        "selection_vlm_description": cfg.selection_vlm_description,
        "smolvla_policy_model": cfg.smolvla_policy_model,
        "config": {
            "pca_dim": cfg.pca_dim,
            "selection_num_episodes": cfg.selection_num_episodes,
            "selection_seed": cfg.selection_seed,
            "train_steps": cfg.train_steps,
            "train_batch_size": cfg.train_batch_size,
            "train_lr": cfg.train_lr,
            "eval_n_episodes": cfg.eval_n_episodes,
            "eval_seed": cfg.eval_seed,
        },
        "paths": {
            "embedding_dir": cfg.embedding_cache_dir,
            "subset_file": None,
            "checkpoint_dir": None,
            "eval_results_file": None,
            "eval_episodes_file": None,
            "results_dir": cfg.results_dir,
            "checkpoints_dir": cfg.checkpoints_dir,
            "logs_dir": cfg.logs_dir,
        },
        "stages": {},
        "final_metrics": {},
        "total_time_seconds": None,
    }

    def save_summary():
        experiment_state["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        experiment_state["total_time_seconds"] = round(time.time() - overall_start, 1)
        summary_file = Path(cfg.results_dir) / "experiment_summary.json"
        with open(summary_file, "w") as f:
            json.dump(experiment_state, f, indent=2)
        print(f"\nExperiment summary saved to: {summary_file}")
        sys.stdout.flush()

    try:
        print(f"\n{'='*60}")
        print(f"Stage 1: VLM Embedding Extraction")
        print(f"{'='*60}")
        experiment_state["current_stage"] = "embedding_extraction"
        stage_start = time.time()
        embedding_dir = auto_extract_embedding(cfg)
        stage_time = time.time() - stage_start
        experiment_state["paths"]["embedding_dir"] = embedding_dir
        experiment_state["stages"]["embedding_extraction"] = {
            "status": "success",
            "embedding_dir": embedding_dir,
            "time_seconds": round(stage_time, 1),
        }
        print(f"Stage 1 complete: {stage_time:.1f}s")

        print(f"\n{'='*60}")
        print(f"Stage 2: V4 Episode Selection")
        print(f"{'='*60}")
        experiment_state["current_stage"] = "episode_selection"
        stage_start = time.time()
        subset_file = run_v4_selection(cfg)
        stage_time = time.time() - stage_start
        experiment_state["paths"]["subset_file"] = subset_file
        experiment_state["stages"]["episode_selection"] = {
            "status": "success",
            "subset_file": subset_file,
            "time_seconds": round(stage_time, 1),
        }
        print(f"Stage 2 complete: {stage_time:.1f}s")

        print(f"\n{'='*60}")
        print(f"Stage 3: SmolVLA Training")
        print(f"{'='*60}")
        experiment_state["current_stage"] = "training"
        stage_start = time.time()
        checkpoint_dir = run_smolvla_training(cfg, subset_file)
        stage_time = time.time() - stage_start
        experiment_state["paths"]["checkpoint_dir"] = checkpoint_dir
        experiment_state["stages"]["training"] = {
            "status": "success",
            "checkpoint_dir": checkpoint_dir,
            "time_seconds": round(stage_time, 1),
        }
        print(f"Stage 3 complete: {stage_time:.1f}s")

        print(f"\n{'='*60}")
        print(f"Stage 4: Evaluation")
        print(f"{'='*60}")
        experiment_state["current_stage"] = "evaluation"
        stage_start = time.time()
        eval_metrics = run_eval(cfg, checkpoint_dir)
        stage_time = time.time() - stage_start
        experiment_state["paths"]["eval_results_file"] = str(Path(cfg.results_dir) / "eval_results" / "eval_results.json")
        experiment_state["paths"]["eval_episodes_file"] = str(Path(cfg.results_dir) / "eval_results" / "eval_episodes.json")
        experiment_state["stages"]["evaluation"] = {
            "status": "success",
            "time_seconds": round(stage_time, 1),
        }
        experiment_state["final_metrics"] = eval_metrics
        print(f"Stage 4 complete: {stage_time:.1f}s")

    except Exception as e:
        print(f"\nEXPERIMENT FAILED at stage: {experiment_state['current_stage']}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        experiment_state["failed_stage"] = experiment_state["current_stage"]
        experiment_state["error"] = str(e)
        save_summary()
        sys.exit(1)

    experiment_state["current_stage"] = "completed"
    save_summary()

    print(f"\n{'#'*60}")
    print(f"# Experiment Complete")
    print(f"# Total time: {experiment_state['total_time_seconds']/3600:.2f} hours ({experiment_state['total_time_seconds']/60:.1f} minutes)")
    print(f"# Summary: {Path(cfg.results_dir) / 'experiment_summary.json'}")
    if experiment_state["final_metrics"]:
        print(f"# pc_success: {experiment_state['final_metrics'].get('pc_success', 'N/A')}")
        print(f"# pc_grasp_success: {experiment_state['final_metrics'].get('pc_grasp_success', 'N/A')}")
    print(f"{'#'*60}")

    sys.stdout.flush()
    return experiment_state


def main():
    parser = argparse.ArgumentParser(description="DifferentVLM Experiment Runner")
    parser.add_argument("--vlm", type=str, required=True,
                       choices=["llava_pythia400m", "prismatic_qwen25_05b"],
                       help="VLM backbone name")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU ID")
    parser.add_argument("--camera", type=str, default="corner",
                       help="Camera configuration (default: corner)")
    args = parser.parse_args()

    run_experiment(vlm_name=args.vlm, gpu_id=args.gpu, camera=args.camera)


if __name__ == "__main__":
    main()