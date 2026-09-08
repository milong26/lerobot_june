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


def ensure_v5_embeddings_exist(cfg: MiniVLAExperimentConfig) -> tuple:
    """
    Ensure V5 visual embeddings and action descriptors exist.
    If not, extract them automatically.
    Returns (visual_embedding_dir, action_descriptor_dir).
    """
    print(f"\n{'='*60}")
    print(f"Stage 0: V5 Embedding Extraction")
    print(f"{'='*60}")
    print(f"Dataset: {cfg.dataset_name}")
    print(f"GPU: {cfg.gpu_id}")
    sys.stdout.flush()

    stage_start = time.time()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    from embedding_utils.ensure_embeddings_v5 import ensure_v5_embeddings_exist as _ensure_v5

    visual_dir, action_dir = _ensure_v5(
        dataset_root=cfg.dataset_root,
        dataset_name=cfg.dataset_name,
        gpu_id=cfg.gpu_id,
        pca_dim=32,
    )

    stage_time = time.time() - stage_start
    print(f"\nEmbedding extraction complete in {stage_time:.1f}s")
    print(f"Visual embeddings: {visual_dir}")
    print(f"Action descriptors: {action_dir}")
    sys.stdout.flush()

    return str(visual_dir), str(action_dir)


def run_episode_selection(cfg: MiniVLAExperimentConfig) -> str:
    """
    Run episode selection based on the selection_mode.
    Supported modes: grid_uniform, random, our_v5, deminf
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

    selection_dispatchers = {
        "grid_uniform": _select_grid_uniform,
        "random": _select_random,
        "our_v5": _select_our_v5,
        "deminf": _select_deminf,
    }

    if cfg.selection_mode not in selection_dispatchers:
        raise ValueError(
            f"Unknown selection mode '{cfg.selection_mode}'. "
            f"Supported modes: {', '.join(selection_dispatchers.keys())}"
        )

    selector = selection_dispatchers[cfg.selection_mode]
    return selector(cfg)


def _select_grid_uniform(cfg: MiniVLAExperimentConfig) -> str:
    """Select episodes using grid_uniform algorithm."""
    subset_file = Path(cfg.subsets_dir) / f"grid_uniform_{cfg.selection_num_episodes}_seed{cfg.selection_seed}.json"

    if subset_file.exists():
        print(f"\nEpisode Selection (CACHED)")
        print(f"Subset file already exists: {subset_file}")
        sys.stdout.flush()
        return str(subset_file)

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


def _select_random(cfg: MiniVLAExperimentConfig) -> str:
    """Select episodes randomly."""
    subset_file = Path(cfg.subsets_dir) / f"random_{cfg.selection_num_episodes}_seed{cfg.selection_seed}.json"

    if subset_file.exists():
        print(f"\nEpisode Selection (CACHED)")
        print(f"Subset file already exists: {subset_file}")
        sys.stdout.flush()
        return str(subset_file)

    select_script = PROJECT_ROOT / "personal" / "work2" / "duibi" / "train_and_eval_scripts" / "select_random_episodes.py"
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


def _select_our_v5(cfg: MiniVLAExperimentConfig) -> str:
    """Select episodes using our_v5 adaptive coverage algorithm."""
    subset_file = Path(cfg.subsets_dir) / f"our_v5_{cfg.selection_num_episodes}_seed{cfg.selection_seed}.json"

    if subset_file.exists():
        print(f"\nEpisode Selection (CACHED)")
        print(f"Subset file already exists: {subset_file}")
        sys.stdout.flush()
        return str(subset_file)

    select_script = PROJECT_ROOT / "personal" / "work2" / "our_v5" / "select_our_v5.py"
    if not select_script.exists():
        raise FileNotFoundError(f"Selection script not found: {select_script}")

    dataset_dir = Path(cfg.dataset_root)
    v5_output_dir = Path(cfg.subsets_dir).parent / f"our_v5_{cfg.selection_num_episodes}_seed{cfg.selection_seed}"

    # Auto-extract embeddings if they don't exist
    from embedding_utils.ensure_embeddings_v5 import ensure_v5_embeddings_exist as _ensure_v5
    from embedding_utils.config import DEFAULT_PCA_DIM

    print(f"\nEnsuring V5 embeddings exist...")
    visual_dir, action_dir = _ensure_v5(
        dataset_root=cfg.dataset_root,
        dataset_name=cfg.dataset_name,
        gpu_id=cfg.gpu_id,
        pca_dim=DEFAULT_PCA_DIM,
    )
    visual_embedding_dir = str(visual_dir)
    action_descriptor_dir = str(action_dir)
    print(f"Visual embeddings: {visual_embedding_dir}")
    print(f"Action descriptors: {action_descriptor_dir}")
    sys.stdout.flush()

    if not (dataset_dir / "episode_initial_states.json").exists():
        raise FileNotFoundError(
            f"Dataset directory missing episode_initial_states.json: {dataset_dir}\n"
            f"our_v5 requires rand_vec for each episode."
        )

    cmd = [
        sys.executable, str(select_script),
        "--visual-embedding-dir", str(visual_embedding_dir),
        "--action-descriptor-dir", str(action_descriptor_dir),
        "--dataset-dir", str(dataset_dir),
        "--output-dir", str(v5_output_dir),
        "--num-selected", str(cfg.selection_num_episodes),
        "--seed", str(cfg.selection_seed),
        "--visual-weight", "0.5",
        "--action-weight", "0.5",
        "--region-ratio", "0.1",
        "--min-regions", "16",
        "--max-regions", "128",
        "--b0-region-ratio", "0.2",
        "--coverage-weight", "0.5",
        "--region-visual-weight", "0.3",
        "--region-action-weight", "0.2",
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        raise RuntimeError(f"Episode selection failed with exit code {result.returncode}")

    v5_subset = v5_output_dir / "subsets" / f"our_v5_{cfg.selection_num_episodes}_seed{cfg.selection_seed}.json"
    if not v5_subset.exists():
        raise FileNotFoundError(f"Subset file not generated: {v5_subset}")

    import shutil
    shutil.copy(v5_subset, subset_file)

    print(f"\nSelection complete. Subset file: {subset_file}")
    sys.stdout.flush()
    return str(subset_file)


def _select_deminf(cfg: MiniVLAExperimentConfig) -> str:
    """Select episodes using DemInf mutual information algorithm."""
    subset_file = Path(cfg.subsets_dir) / f"deminf_{cfg.selection_num_episodes}_seed{cfg.selection_seed}.json"

    if subset_file.exists():
        print(f"\nEpisode Selection (CACHED)")
        print(f"Subset file already exists: {subset_file}")
        sys.stdout.flush()
        return str(subset_file)

    select_script = PROJECT_ROOT / "personal" / "work2" / "deminf" / "run_deminf.py"
    if not select_script.exists():
        raise FileNotFoundError(f"Selection script not found: {select_script}")

    deminf_output_dir = Path(cfg.subsets_dir).parent / f"deminf_{cfg.selection_num_episodes}_seed{cfg.selection_seed}"

    cmd = [
        sys.executable, str(select_script),
        "--dataset-path", cfg.dataset_root,
        "--output-dir", str(deminf_output_dir),
        "--target-episodes", str(cfg.selection_num_episodes),
        "--seed", str(cfg.selection_seed),
        "--vae-steps", "50000",
        "--vae-lr", "1e-4",
        "--vae-batch-size", "256",
        "--state-latent-dim", "12",
        "--action-latent-dim", "6",
        "--ks", "5", "6", "7",
        "--state-source", "observation.environment_state",
        "--quality-batch-size", "1024",
        "--quality-repeat", "4",
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        raise RuntimeError(f"Episode selection failed with exit code {result.returncode}")

    deminf_subset = deminf_output_dir / "subsets" / f"deminf_{cfg.selection_num_episodes}_seed{cfg.selection_seed}.json"
    if not deminf_subset.exists():
        raise FileNotFoundError(f"Subset file not generated: {deminf_subset}")

    import shutil
    shutil.copy(deminf_subset, subset_file)

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
    eval_latest_only: bool = False,
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

    print(f"MiniVLA Experiment: {cfg.exp_name}")
    print(f"Dataset: {cfg.dataset_name}")
    print(f"GPU: {cfg.gpu_id}")
    print(f"Start time: {start_time}")

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
            eval_results = run_eval_all_checkpoints(cfg, checkpoint_dir, latest_only=eval_latest_only)
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

    print(f"MiniVLA Experiment Complete")
    print(f"Total time: {experiment_state['total_time_seconds']/3600:.2f} hours ({experiment_state['total_time_seconds']/60:.1f} minutes)")
    print(f"Summary: {Path(cfg.experiment_dir) / 'experiment_summary.json'}")
    if experiment_state["final_metrics"]:
        print(f"pc_success: {experiment_state['final_metrics'].get('best_pc_success', 'N/A')}")
        print(f"pc_grasp_success: {experiment_state['final_metrics'].get('best_pc_grasp_success', 'N/A')}")

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
                       help="Episode selection mode (grid_uniform, random, our_v5, deminf)")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--eval-latest-only", action="store_true",
                       help="Only evaluate the latest checkpoint (default: evaluate all)")
    args = parser.parse_args()

    run_experiment(
        gpu_id=args.gpu,
        dataset_name=args.dataset,
        num_episodes=args.num_episodes,
        seed=args.seed,
        selection_mode=args.selection_mode,
        eval_all=not args.no_eval,
        eval_latest_only=args.eval_latest_only,
    )


if __name__ == "__main__":
    main()