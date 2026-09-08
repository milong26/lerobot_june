"""
MiniVLA Training Wrapper

Calls the existing lerobot-train entry point with MiniVLA policy configuration.
Does NOT modify training code.

Experiment isolation:
- Output directory: differentvlm/minivla/checkpoints/{exp_name}
- Log file: differentvlm/minivla/logs/{exp_name}_training.log
- Training uses ONLY selected episode dataset
"""

import sys
import json
import subprocess
import os
import shutil
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.minivla.configs.minivla_config import MiniVLAExperimentConfig


def run_minivla_training(cfg: MiniVLAExperimentConfig, subset_file: str) -> str:
    """
    Run MiniVLA training with the selected dataset.
    Reuses existing lerobot-train entry point.
    Returns the checkpoint directory path.
    """
    print(f"\nRunning MiniVLA Training")

    with open(subset_file, "r") as f:
        subset_data = json.load(f)

    episode_indices = subset_data["selected_episode_indices"]
    episodes_str = "[" + ",".join(str(x) for x in episode_indices) + "]"

    output_dir = Path(cfg.checkpoints_dir)

    # Auto-resume: check if output directory exists (LeRobot will check for checkpoints inside)
    resume = False
    training_complete = False
    if output_dir.exists():
        # Check if there are any saved checkpoints (numeric directories)
        existing_steps = sorted([
            d.name for d in output_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])
        if existing_steps:
            resume = True
            print(f"Found existing checkpoints: {existing_steps[-1]}")
            # Check if training already completed all steps
            last_step = int(existing_steps[-1])
            if last_step >= cfg.train_steps:
                training_complete = True
                print(f"Training already complete: step {last_step} >= {cfg.train_steps}")
            else:
                print(f"Training will resume from latest checkpoint (step {last_step}/{cfg.train_steps})")
        else:
            print(f"Output directory exists but no checkpoints found, starting fresh training")

    train_log = Path(cfg.logs_dir) / f"{cfg.exp_name}_training.log"

    print(f"Experiment name: {cfg.exp_name}")
    print(f"Selected episodes: {len(episode_indices)}")
    print(f"Episode indices: {episodes_str}")
    print(f"Dataset root: {cfg.dataset_root}")
    print(f"Output dir: {output_dir}")
    print(f"Training log: {train_log}")
    print(f"VQ model path: {cfg.vq_model_path}")
    print(f"Steps: {cfg.train_steps}")
    print(f"Batch size: {cfg.train_batch_size}")
    print(f"GPU: {cfg.gpu_id}")
    sys.stdout.flush()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["LD_LIBRARY_PATH"] = f"{os.environ.get('CONDA_PREFIX', '')}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ["LD_PRELOAD"] = f"{os.environ.get('CONDA_PREFIX', '')}/lib/libstdc++.so.6"

    # Extract task name from dataset name (same logic as duibi/train_and_eval.sh)
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

    print(f"Dataset name: {cfg.dataset_name}")
    print(f"Extracted task name: {task_name}")

    cmd = [
        "lerobot-train",
        "--policy.type=minivla",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=lerobot/metaworld_pick_place",
        f"--dataset.root={cfg.dataset_root}",
        f"--dataset.episodes={episodes_str}",
        "--dataset.eval_split=0.0",
        "--env.type=metaworld",
        f"--env.task={task_name}",
        f"--env.camera_name={cfg.camera_names}",
        f"--policy.vq_model_path={cfg.vq_model_path}",
        f"--policy.action_tokenizer_type=libero_vq_action_tokenizer",
        f"--policy.primary_image_key={cfg.primary_image_key}",
        f"--policy.wrist_image_key={cfg.wrist_image_key}",
        f"--policy.optimizer_lr={cfg.train_lr}",
        f"--save_freq={cfg.train_save_freq}",
        f"--steps={cfg.train_steps}",
        f"--batch_size={cfg.train_batch_size}",
        f"--num_workers={cfg.train_num_workers}",
        f"--eval.n_episodes={cfg.eval_n_episodes}",
        f"--eval.batch_size={cfg.eval_batch_size}",
        f"--env_eval_freq={cfg.train_steps}",
        f"--seed={cfg.selection_seed}",
        f"--job_name=minivla_{cfg.exp_name}",
        f"--output_dir={output_dir}",
        '--remove_features=["observation.environment_state"]',
        "--wandb.enable=true",
    ]

    # Auto-resume: add --resume flag if checkpoints exist
    if resume:
        cmd.append("--resume=true")
        print(f"Auto-resume enabled: adding --resume=true to command")

    # If training already complete, skip training
    if training_complete:
        print(f"\nSkipping training (already complete at step {cfg.train_steps})")
        sys.stdout.flush()
    else:
        print(f"\nRunning: lerobot-train ...")
        print(f"Output dir: {output_dir}")
        print(f"Log file: {train_log}")
        sys.stdout.flush()

        # Use append mode when resuming, write mode for fresh start
        log_mode = "a" if resume else "w"
        with open(train_log, log_mode) as log_f:
            if resume:
                log_f.write(f"\n\n{'='*60}\n")
                log_f.write(f"RESUMING TRAINING from existing checkpoints\n")
                log_f.write(f"{'='*60}\n\n")
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).resolve().parents[5]),
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