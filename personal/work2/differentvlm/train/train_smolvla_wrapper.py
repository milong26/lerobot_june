"""
SmolVLA Training Wrapper for DifferentVLM

Calls the existing lerobot-train entry point with the selected dataset.
Does NOT modify training code.

Experiment isolation:
- Output directory: differentvlm/checkpoints/{vlm_name}_{camera}
- Log file: differentvlm/logs/{vlm_name}_{camera}/training.log
- Training uses ONLY selected episode dataset, NOT full dataset
- SmolVLA policy model is FIXED (not the selection VLM)
- Selection VLM is NOT passed to SmolVLA policy parameters
"""

import sys
import json
import subprocess
import os
import shutil
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.configs.vlm_config import VLMExperimentConfig


def run_smolvla_training(cfg: VLMExperimentConfig, subset_file: str) -> str:
    """
    Run SmolVLA training with the selected dataset.
    Reuses existing lerobot-train entry point.
    Returns the checkpoint directory path.
    """
    print(f"\n{'='*60}")
    print(f"Running SmolVLA Training")
    print(f"{'='*60}")

    with open(subset_file, "r") as f:
        subset_data = json.load(f)

    episode_indices = subset_data["selected_episode_indices"]
    episodes_str = "[" + ",".join(str(x) for x in episode_indices) + "]"

    exp_name = f"smolvla_{cfg.vlm_name}_{cfg.camera}"
    output_dir = Path(cfg.checkpoints_dir) / exp_name

    # Clean existing output dir to avoid lerobot-train resume conflict
    # Do NOT create the directory - lerobot-train will create it itself
    if output_dir.exists():
        print(f"Cleaning existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
        sys.stdout.flush()

    train_log = Path(cfg.logs_dir) / f"{exp_name}_training.log"

    print(f"Experiment name: {exp_name}")
    print(f"Selected episodes: {len(episode_indices)}")
    print(f"Episode indices: {episodes_str}")
    print(f"Dataset root: {cfg.dataset_root}")
    print(f"Output dir: {output_dir}")
    print(f"Checkpoint dir: {output_dir / 'checkpoints'}")
    print(f"Training log: {train_log}")
    print(f"SmolVLA policy model: {cfg.smolvla_policy_model}")
    print(f"Steps: {cfg.train_steps}")
    print(f"Batch size: {cfg.train_batch_size}")
    print(f"GPU: {cfg.gpu_id}")
    print(f"{'='*60}")
    sys.stdout.flush()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["LD_LIBRARY_PATH"] = f"{os.environ.get('CONDA_PREFIX', '')}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ["LD_PRELOAD"] = f"{os.environ.get('CONDA_PREFIX', '')}/lib/libstdc++.so.6"

    cmd = [
        "lerobot-train",
        "--policy.path=lerobot/smolvla_base",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=lerobot/metaworld_pick_place",
        f"--dataset.root={cfg.dataset_root}",
        f"--dataset.episodes={episodes_str}",
        "--dataset.eval_split=0.0",
        f"--rename_map={cfg.rename_map}",
        "--env.type=metaworld",
        "--env.task=pick-place-v3",
        f"--env.camera_name={cfg.camera_names}",
        f"--policy.vlm_model_name={cfg.smolvla_policy_model}",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.train_state_proj=false",
        f"--policy.optimizer_lr={cfg.train_lr}",
        f"--save_freq={cfg.train_save_freq}",
        f"--steps={cfg.train_steps}",
        f"--batch_size={cfg.train_batch_size}",
        f"--num_workers={cfg.train_num_workers}",
        f"--eval.n_episodes={cfg.eval_n_episodes}",
        f"--eval.batch_size={cfg.eval_batch_size}",
        f"--env_eval_freq={cfg.train_steps}",
        f"--seed={cfg.selection_seed}",
        f"--job_name={exp_name}",
        f"--output_dir={output_dir}",
        '--remove_features=["observation.environment_state"]',
        "--wandb.enable=true",
    ]

    print(f"\nRunning: lerobot-train ...")
    print(f"Output dir: {output_dir}")
    print(f"Log file: {train_log}")
    sys.stdout.flush()

    with open(train_log, "w") as log_f:
        result = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parents[4]),
        )

    if result.returncode != 0:
        print(f"WARNING: Training exited with code {result.returncode}")
        print(f"Check log: {train_log}")
        sys.stdout.flush()

    checkpoint_dir = output_dir / "checkpoints"
    print(f"\nTraining complete.")
    print(f"Checkpoint dir: {checkpoint_dir}")
    sys.stdout.flush()
    return str(checkpoint_dir)