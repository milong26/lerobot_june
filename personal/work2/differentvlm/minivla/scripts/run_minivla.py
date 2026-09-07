"""
MiniVLA Experiment Main Entry Point

Orchestrates the full MiniVLA experiment pipeline:
1. Episode selection (grid_uniform, random, etc.)
2. MiniVLA training
3. Evaluation

Complete experiment state tracking:
- start_time, end_time
- current stage, failed stage
- selected_episode path
- checkpoint path
- eval results (success/grasp_success)
- On exception: saves experiment_summary.json before exit
"""

import sys
import os
import time
import json
import argparse
import subprocess
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
WORK2_ROOT = Path(__file__).resolve().parents[3]
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

from differentvlm.minivla.configs.minivla_config import get_minivla_config, MiniVLAExperimentConfig
from differentvlm.minivla.train.train_minivla import run_minivla_training
from differentvlm.minivla.eval.eval_minivla import run_eval_all_checkpoints


def run_episode_selection(cfg: MiniVLAExperimentConfig) -> str:
    """
    Run episode selection using the duibi select_grid_uniform.py script.
    Returns the subset file path.
    """
    subset_file = Path(cfg.subsets_dir) / f"{cfg.selection_mode}_{cfg.selection_num_episodes}_seed{cfg.selection_seed}.json"

    if subset_file.exists():
        print(f"\nEpisode Selection (CACHED)")
        print(f"Subset file already exists: {subset_file}")
        sys.stdout.flush()
        return str(subset_file)

    print(f"\nRunning Episode Selection: {cfg.selection_mode}")
    print(f"Dataset: {cfg.dataset_name}")
    print(f"Num episodes: {cfg.selection_num_episodes}")
    print(f"Seed: {cfg.selection_seed}")

    select_script = PROJECT_ROOT / "personal" / "work2" / "duibi" / "train_and_eval_scripts" / "select_grid_uniform.py"

    if not select_script.exists():
        raise FileNotFoundError(f"Selection script not found: {select_script}")

    cmd = [
        sys.executable, str(select_script),
        "--num-episodes", str(cfg.selection_num_episodes),
        "--seed", str(cfg.selection_seed),
        "--dataset-root", cfg.dataset_root,
        "--output-dir", cfg.subsets_dir,
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        raise RuntimeError(f"Episode selection failed with exit code {result.returncode}")

    if not subset_file.exists():
        raise FileNotFoundError(f"Subset file not generated: {subset_file}")

    print(f"\nSelection complete. Subset file: {subset_file}")
    sys.stdout.flush()
    return str(subset_file)


def run_experiment(
    gpu_id: int = 0,
    dataset_name: str = "disassemble-v3_corner",
    num_episodes: int = 112,
    seed: int = 42,
    selection_mode: str = "grid_uniform",
    eval_all: bool = True,
):
    """Run the complete MiniVLA experiment pipeline."""
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    overall_start = time.time()

    cfg = get_minivla_config(
        gpu_id=gpu_id,
        dataset_name=dataset_name,
        num_episodes=num_episodes,
        seed=seed,
        selection_mode=selection_mode,
    )
    cfg.ensure_dirs()

    print(f"\n{'#'*60}")
    print(f"# MiniVLA Experiment: {cfg.exp_name}")
    print(f"# Dataset: {cfg.dataset_name}")
    print(f"# GPU: {cfg.gpu_id}")
    print(f"# Start time: {start_time}")
    print(f"{'#'*60}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    experiment_state = {
        "exp_name": cfg.exp_name,
        "dataset_name": cfg.dataset_name,
        "gpu_id": gpu_id,
        "start_time": start_time,
        "end_time": None,
        "current_stage": "initializing",
        "failed_stage": None,
        "config": {
            "selection_mode": cfg.selection_mode,
            "selection_num_episodes": cfg.selection_num_episodes,
            "selection_seed": cfg.selection_seed,
            "train_steps": cfg.train_steps,
            "train_batch_size": cfg.train_batch_size,
            "train_lr": cfg.train_lr,
            "eval_n_episodes": cfg.eval_n_episodes,
            "eval_seed": cfg.eval_seed,
            "vq_model_path": cfg.vq_model_path,
        },
        "paths": {
            "subset_file": None,
            "checkpoint_dir": None,
            "eval_results_dir": cfg.eval_results_dir,
        },
        "stages": {},
        "final_metrics": {},
        "total_time_seconds": None,
    }

    def save_summary():
        experiment_state["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        experiment_state["total_time_seconds"] = round(time.time() - overall_start, 1)
        summary_file = Path(cfg.experiment_dir) / "experiment_summary.json"
        with open(summary_file, "w") as f:
            json.dump(experiment_state, f, indent=2)
        print(f"\nExperiment summary saved to: {summary_file}")
        sys.stdout.flush()

    try:
        # Stage 1: Episode Selection
        print(f"\nStage 1: Episode Selection ({cfg.selection_mode})")
        experiment_state["current_stage"] = "episode_selection"
        stage_start = time.time()
        subset_file = run_episode_selection(cfg)
        stage_time = time.time() - stage_start
        experiment_state["paths"]["subset_file"] = subset_file
        experiment_state["stages"]["episode_selection"] = {
            "status": "success",
            "subset_file": subset_file,
            "time_seconds": round(stage_time, 1),
        }
        print(f"Stage 1 complete: {stage_time:.1f}s")

        # Stage 2: MiniVLA Training
        print(f"\nStage 2: MiniVLA Training")
        experiment_state["current_stage"] = "training"
        stage_start = time.time()
        checkpoint_dir = run_minivla_training(cfg, subset_file)
        stage_time = time.time() - stage_start
        experiment_state["paths"]["checkpoint_dir"] = checkpoint_dir
        experiment_state["stages"]["training"] = {
            "status": "success",
            "checkpoint_dir": checkpoint_dir,
            "time_seconds": round(stage_time, 1),
        }
        print(f"Stage 2 complete: {stage_time:.1f}s")

        # Stage 3: Evaluation
        if eval_all:
            print(f"\nStage 3: Evaluation (all checkpoints)")
            experiment_state["current_stage"] = "evaluation"
            stage_start = time.time()
            eval_results = run_eval_all_checkpoints(cfg, checkpoint_dir)
            stage_time = time.time() - stage_start
            experiment_state["stages"]["evaluation"] = {
                "status": "success",
                "time_seconds": round(stage_time, 1),
            }
            experiment_state["final_metrics"] = {
                "eval_results": eval_results,
                "best_pc_success": max(
                    [r.get("pc_success", -1) for r in eval_results if r.get("pc_success", -1) >= 0],
                    default=-1,
                ),
                "best_pc_grasp_success": max(
                    [r.get("pc_grasp_success", -1) for r in eval_results if r.get("pc_grasp_success", -1) >= 0],
                    default=-1,
                ),
            }
            print(f"Stage 3 complete: {stage_time:.1f}s")

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
    print(f"# MiniVLA Experiment Complete")
    print(f"# Total time: {experiment_state['total_time_seconds']/3600:.2f} hours ({experiment_state['total_time_seconds']/60:.1f} minutes)")
    print(f"# Summary: {Path(cfg.experiment_dir) / 'experiment_summary.json'}")
    if experiment_state["final_metrics"]:
        print(f"# pc_success: {experiment_state['final_metrics'].get('best_pc_success', 'N/A')}")
        print(f"# pc_grasp_success: {experiment_state['final_metrics'].get('best_pc_grasp_success', 'N/A')}")
    print(f"{'#'*60}")

    sys.stdout.flush()
    return experiment_state


def main():
    parser = argparse.ArgumentParser(description="MiniVLA Experiment Runner")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--dataset", type=str, default="disassemble-v3_corner",
                       help="Dataset name (e.g. disassemble-v3_corner, pick_place-v3_corner)")
    parser.add_argument("--num-episodes", type=int, default=112, help="Number of episodes to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--selection-mode", type=str, default="grid_uniform",
                       help="Episode selection mode (grid_uniform, random, etc.)")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation after training")
    args = parser.parse_args()

    run_experiment(
        gpu_id=args.gpu,
        dataset_name=args.dataset,
        num_episodes=args.num_episodes,
        seed=args.seed,
        selection_mode=args.selection_mode,
        eval_all=not args.no_eval,
    )


if __name__ == "__main__":
    main()