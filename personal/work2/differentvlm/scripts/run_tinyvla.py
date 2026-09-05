"""
TinyVLA Experiment Main Entry Point

Orchestrates the complete TinyVLA experiment pipeline:
1. VLM embedding extraction (using our_v5 method with LLaVA-Pythia backbone)
2. V4 episode selection (using our_v5 selection algorithm)
3. TinyVLA fine-tuning (with LoRA + diffusion action head)
4. Evaluation

Usage:
    python run_tinyvla.py --policy_type tinyvla_s --gpu 0 --camera corner --num_episodes 112
    python run_tinyvla.py --policy_type tinyvla_b --gpu 1 --camera corner --num_episodes 112

All intermediate results are saved in differentvlm/tinyvla{s|b}/ directory.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
WORK2_ROOT = Path(__file__).resolve().parents[2]
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

from differentvlm.configs.vlm_config import get_config
from differentvlm.vlm_extract.auto_extract_embedding import auto_extract_embedding
from differentvlm.selection.select_v4_wrapper import run_v4_selection
from differentvlm.train.train_tinyvla_wrapper import run_tinyvla_training
from differentvlm.eval.eval_tinyvla_wrapper import run_tinyvla_eval


def run_experiment(policy_type: str, gpu_id: int, camera: str = "corner", num_episodes: int = 112, dataset_root: str = None):
    """Run the complete TinyVLA experiment pipeline."""
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    overall_start = time.time()

    print(f"\n{'#'*60}")
    print(f"# TinyVLA Experiment: {policy_type}")
    print(f"# Camera: {camera}")
    print(f"# GPU: {gpu_id}")
    print(f"# Num Episodes: {num_episodes}")
    print(f"# Start time: {start_time}")
    print(f"{'#'*60}")

    cfg = get_config(vlm_name=policy_type, gpu_id=gpu_id, camera=camera)
    cfg.selection_num_episodes = num_episodes
    
    if dataset_root is not None:
        cfg.dataset_root = dataset_root
        cfg.ensure_dirs()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    experiment_state = {
        "policy_type": policy_type,
        "camera": camera,
        "gpu_id": gpu_id,
        "num_episodes": num_episodes,
        "dataset_root": cfg.dataset_root,
        "start_time": start_time,
        "end_time": None,
        "current_stage": "initializing",
        "failed_stage": None,
        "selection_vlm_model_id": cfg.selection_vlm_model_id,
        "selection_vlm_description": cfg.selection_vlm_description,
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
        # Stage 1: VLM Embedding Extraction
        experiment_state["current_stage"] = "embedding_extraction"
        print(f"\n{'='*60}")
        print(f"Stage 1: VLM Embedding Extraction")
        print(f"{'='*60}")
        print(f"Using VLM: {cfg.selection_vlm_model_id}")
        print(f"Embedding dir: {cfg.embedding_cache_dir}")
        sys.stdout.flush()

        stage_start = time.time()
        embedding_dir = auto_extract_embedding(cfg)
        stage_time = round(time.time() - stage_start, 1)

        experiment_state["stages"]["embedding_extraction"] = {
            "status": "completed",
            "time_seconds": stage_time,
            "embedding_dir": embedding_dir,
        }
        experiment_state["paths"]["embedding_dir"] = embedding_dir
        print(f"\nEmbedding extraction complete in {stage_time}s")
        print(f"Embedding dir: {embedding_dir}")
        sys.stdout.flush()

        # Stage 2: Episode Selection (V4)
        experiment_state["current_stage"] = "episode_selection"
        print(f"\n{'='*60}")
        print(f"Stage 2: Episode Selection (V4)")
        print(f"{'='*60}")
        print(f"Selecting {cfg.selection_num_episodes} episodes")
        print(f"Seed: {cfg.selection_seed}")
        sys.stdout.flush()

        stage_start = time.time()
        subset_file = run_v4_selection(cfg, embedding_dir)
        stage_time = round(time.time() - stage_start, 1)

        experiment_state["stages"]["episode_selection"] = {
            "status": "completed",
            "time_seconds": stage_time,
            "subset_file": subset_file,
            "num_selected_episodes": cfg.selection_num_episodes,
        }
        experiment_state["paths"]["subset_file"] = subset_file
        print(f"\nEpisode selection complete in {stage_time}s")
        print(f"Subset file: {subset_file}")
        sys.stdout.flush()

        # Stage 3: TinyVLA Fine-tuning
        experiment_state["current_stage"] = "training"
        print(f"\n{'='*60}")
        print(f"Stage 3: TinyVLA Fine-tuning")
        print(f"{'='*60}")
        print(f"Policy type: {cfg.tinyvla_policy_type}")
        print(f"Training steps: {cfg.train_steps}")
        print(f"Batch size: {cfg.train_batch_size}")
        print(f"Learning rate: {cfg.train_lr}")
        sys.stdout.flush()

        stage_start = time.time()
        checkpoint_dir = run_tinyvla_training(cfg, subset_file)
        stage_time = round(time.time() - stage_start, 1)

        experiment_state["stages"]["training"] = {
            "status": "completed",
            "time_seconds": stage_time,
            "checkpoint_dir": checkpoint_dir,
            "train_steps": cfg.train_steps,
        }
        experiment_state["paths"]["checkpoint_dir"] = checkpoint_dir
        print(f"\nTraining complete in {stage_time}s")
        print(f"Checkpoint dir: {checkpoint_dir}")
        sys.stdout.flush()

        # Stage 4: Evaluation
        experiment_state["current_stage"] = "evaluation"
        print(f"\n{'='*60}")
        print(f"Stage 4: Evaluation")
        print(f"{'='*60}")
        print(f"N episodes: {cfg.eval_n_episodes}")
        print(f"Seed: {cfg.eval_seed}")
        sys.stdout.flush()

        stage_start = time.time()
        eval_results = run_tinyvla_eval(cfg, checkpoint_dir)
        stage_time = round(time.time() - stage_start, 1)

        experiment_state["stages"]["evaluation"] = {
            "status": "completed",
            "time_seconds": stage_time,
            "eval_results": eval_results,
        }
        experiment_state["final_metrics"] = eval_results
        print(f"\nEvaluation complete in {stage_time}s")
        print(f"Eval results: {eval_results}")
        sys.stdout.flush()

        experiment_state["current_stage"] = "completed"

    except Exception as e:
        experiment_state["failed_stage"] = experiment_state["current_stage"]
        experiment_state["current_stage"] = "failed"
        experiment_state["error"] = str(e)
        print(f"\n{'='*60}")
        print(f"EXPERIMENT FAILED")
        print(f"{'='*60}")
        print(f"Failed stage: {experiment_state['failed_stage']}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

    finally:
        save_summary()

    return experiment_state


def main():
    parser = argparse.ArgumentParser(description="TinyVLA Experiment Launcher")
    parser.add_argument(
        "--policy_type",
        type=str,
        required=True,
        choices=["tinyvla_s", "tinyvla_b"],
        help="TinyVLA policy type: tinyvla_s (400M) or tinyvla_b (700M)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="corner",
        help="Camera configuration (default: corner)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=112,
        help="Number of episodes to select for training (default: 112)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Path to dataset root directory (default: use config default)",
    )

    args = parser.parse_args()

    run_experiment(
        policy_type=args.policy_type,
        gpu_id=args.gpu,
        camera=args.camera,
        num_episodes=args.num_episodes,
        dataset_root=args.dataset_root,
    )


if __name__ == "__main__":
    main()