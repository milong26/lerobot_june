"""
TinyVLA Training Wrapper for DifferentVLM

Calls the existing lerobot-train entry point with TinyVLA policy configuration.
Supports resume from existing checkpoints and clean restart.

Experiment isolation:
- Output directory: differentvlm/tinyvla{s|b}/checkpoints/{policy_type}_{camera}
- Log file: differentvlm/tinyvla{s|b}/logs/{policy_type}_{camera}_training.log
- Training uses ONLY selected episode dataset, NOT full dataset
- TinyVLA policy uses LLaVA-Pythia backbone with LoRA fine-tuning
- Action head: droid_diffusion (UNet-based diffusion policy)

Resume / restart control:
- restart=True  : always clean output_dir and start fresh
- resume=True  : find latest checkpoint under output_dir/checkpoints/, resume from it
- Default (both False): if checkpoint exists, resume; otherwise start fresh
"""

import sys
import json
import subprocess
import os
import shutil
import math
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.configs.vlm_config import VLMExperimentConfig

TRAINING_STATE_DIR = "training_state"
LAST_CHECKPOINT_LINK = "last"
TRAINING_STEP_FILE = "training_step.json"


def _find_latest_checkpoint(checkpoints_dir: Path) -> tuple[Path | None, int]:
    """Find the latest numbered checkpoint directory and its step.

    Returns (checkpoint_dir, step) or (None, 0) if none found.
    """
    if not checkpoints_dir.exists():
        return None, 0

    numeric_dirs = []
    for d in checkpoints_dir.iterdir():
        if d.is_dir() and d.name.isdigit():
            numeric_dirs.append((int(d.name), d))

    if not numeric_dirs:
        return None, 0

    numeric_dirs.sort(key=lambda x: x[0])
    latest_step, latest_dir = numeric_dirs[-1]
    return latest_dir, latest_step


def _find_last_symlink(checkpoints_dir: Path) -> Path | None:
    """Follow the 'last' symlink if it exists."""
    last_link = checkpoints_dir / LAST_CHECKPOINT_LINK
    if last_link.is_symlink():
        target = last_link.resolve()
        if target.is_dir():
            return target
    return None


def _read_training_step(checkpoint_dir: Path) -> int:
    """Read the training step from a checkpoint's training_state."""
    step_file = checkpoint_dir / TRAINING_STATE_DIR / TRAINING_STEP_FILE
    if step_file.exists():
        with open(step_file, "r") as f:
            data = json.load(f)
        return data.get("step", 0)
    return 0


def _has_training_state(checkpoint_dir: Path) -> bool:
    """Check if a checkpoint has a valid training_state directory."""
    ts_dir = checkpoint_dir / TRAINING_STATE_DIR
    return ts_dir.is_dir() and (ts_dir / TRAINING_STEP_FILE).exists()


def run_tinyvla_training(
    cfg: VLMExperimentConfig,
    subset_file: str,
    *,
    restart: bool = False,
    resume: bool = False,
) -> str:
    """
    Run TinyVLA training with the selected dataset.
    Reuses existing lerobot-train entry point.
    Returns the checkpoint directory path.

    Args:
        cfg: Experiment configuration.
        subset_file: Path to JSON with selected episode indices.
        restart: If True, always clean output_dir and start fresh.
        resume: If True, find latest checkpoint and resume from it.
    """
    print(f"\n{'='*60}")
    print(f"Running TinyVLA Training")
    print(f"{'='*60}")

    with open(subset_file, "r") as f:
        subset_data = json.load(f)

    episode_indices = subset_data["selected_episode_indices"]
    episodes_str = "[" + ",".join(str(x) for x in episode_indices) + "]"

    exp_name = f"tinyvla_{cfg.tinyvla_policy_type}_{cfg.dataset_name}"
    output_dir = Path(cfg.checkpoints_dir) / exp_name
    checkpoints_dir = output_dir / "checkpoints"
    train_log = Path(cfg.logs_dir) / f"{exp_name}_training.log"

    # Scheduler: decay_steps = total training steps, warmup = 0.5% of total
    target_steps = cfg.train_steps
    scheduler_decay_steps = target_steps
    scheduler_warmup_steps = max(1, math.ceil(target_steps * 0.005))

    # Determine resume / restart behavior
    existing_checkpoint, existing_step = _find_latest_checkpoint(checkpoints_dir)
    last_symlink = _find_last_symlink(checkpoints_dir)

    if restart:
        print(f"[RESTART] Cleaning output directory: {output_dir}")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        actual_resume = False
        actual_start_step = 0
        print(f"[RESTART] Starting fresh training to {target_steps} steps")
    elif resume:
        if existing_checkpoint is None:
            print(f"[RESUME] No existing checkpoint found, starting fresh")
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            actual_resume = False
            actual_start_step = 0
        else:
            # Prefer the 'last' symlink if available
            resume_dir = last_symlink if last_symlink else existing_checkpoint
            actual_start_step = _read_training_step(resume_dir)

            if actual_start_step >= target_steps:
                print(f"[RESUME] Already trained {actual_start_step} steps >= target {target_steps}, skipping training")
                return str(checkpoints_dir)

            if not _has_training_state(resume_dir):
                print(f"[WARNING] Checkpoint {resume_dir} has no training_state, starting fresh")
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                checkpoints_dir.mkdir(parents=True, exist_ok=True)
                actual_resume = False
                actual_start_step = 0
            else:
                actual_resume = True
                print(f"[RESUME] Resuming from {resume_dir} (step {actual_start_step})")

                # Rebuild scheduler if extending beyond old decay_steps
                old_decay = 10000  # old default
                if actual_start_step >= old_decay:
                    print(f"[RESUME] Checkpoint was trained with decay_steps={old_decay}, "
                          f"rebuilding scheduler for remaining {target_steps - actual_start_step} steps")
    else:
        # Default: resume if checkpoint exists, otherwise start fresh
        if existing_checkpoint is not None and _has_training_state(existing_checkpoint):
            resume_dir = last_symlink if last_symlink else existing_checkpoint
            actual_start_step = _read_training_step(resume_dir)
            if actual_start_step >= target_steps:
                print(f"[INFO] Already trained {actual_start_step} steps >= target {target_steps}, skipping training")
                return str(checkpoints_dir)
            actual_resume = True
            print(f"[AUTO-RESUME] Found checkpoint at {resume_dir} (step {actual_start_step})")
        else:
            actual_resume = False
            actual_start_step = 0
            print(f"[AUTO-START] No valid checkpoint found, starting fresh")
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoints_dir.mkdir(parents=True, exist_ok=True)

    sys.stdout.flush()

    # Append to log instead of overwrite - preserve history
    print(f"Experiment name: {exp_name}")
    print(f"Policy type: {cfg.tinyvla_policy_type}")
    print(f"Selected episodes: {len(episode_indices)}")
    print(f"Episode indices: {episodes_str}")
    print(f"Dataset root: {cfg.dataset_root}")
    print(f"Output dir: {output_dir}")
    print(f"Checkpoint dir: {checkpoints_dir}")
    print(f"Training log: {train_log}")
    print(f"Target steps: {target_steps}")
    print(f"Resume: {actual_resume}, Start step: {actual_start_step}")
    print(f"Scheduler: warmup={scheduler_warmup_steps}, decay={scheduler_decay_steps}, "
          f"peak_lr={cfg.train_lr}, decay_lr=2.5e-6")
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
        f"--policy.type={cfg.tinyvla_policy_type}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={cfg.lerobot_repo_id}",
        f"--dataset.root={cfg.dataset_root}",
        f"--dataset.episodes={episodes_str}",
        "--dataset.eval_split=0.0",
        "--env.type=metaworld",
        f"--env.task={cfg.env_task}",
        f"--env.camera_name={cfg.camera_names}",
        "--env.use_self_mw=true",
        f"--policy.optimizer_lr={cfg.train_lr}",
        f"--policy.optimizer_weight_decay={0.0}",
        f"--policy.scheduler_warmup_steps={scheduler_warmup_steps}",
        f"--policy.scheduler_decay_steps={scheduler_decay_steps}",
        f"--policy.scheduler_decay_lr={2.5e-6}",
        f"--save_freq={cfg.train_save_freq}",
        f"--steps={target_steps}",
        f"--batch_size={cfg.train_batch_size}",
        f"--num_workers={cfg.train_num_workers}",
        f"--eval.n_episodes={cfg.eval_n_episodes}",
        f"--eval.batch_size={cfg.eval_batch_size}",
        f"--env_eval_freq={target_steps}",
        f"--seed={cfg.selection_seed}",
        f"--job_name={exp_name}",
        f"--output_dir={output_dir}",
        '--remove_features=["observation.environment_state"]',
        "--wandb.enable=true",
    ]

    if actual_resume:
        cmd.append("--resume=true")

    cmd.append("--log_freq=50")

    print(f"\nRunning: lerobot-train ...")
    print(f"Output dir: {output_dir}")
    print(f"Log file: {train_log}")
    sys.stdout.flush()

    with open(train_log, "a") as log_f:
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

    print(f"\nTraining complete.")
    print(f"Checkpoint dir: {checkpoints_dir}")
    sys.stdout.flush()
    return str(checkpoints_dir)