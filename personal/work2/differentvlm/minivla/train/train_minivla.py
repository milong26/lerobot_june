"""
MiniVLA Training Wrapper

Calls the existing lerobot-train entry point with MiniVLA policy configuration.

Experiment isolation:
- Output directory: differentvlm/minivla/experiments/{exp_name}/checkpoints/
- Log file: differentvlm/minivla/experiments/{exp_name}/logs/{exp_name}_training.log
- Training uses ONLY selected episode dataset

Resume logic:
- Checkpoints are stored under output_dir/checkpoints/000000/, 002000/, etc.
- Each checkpoint contains pretrained_model/ and training_state/
- Resume uses LeRobot standard --resume=true with --config_path pointing to train_config.json
- restart=true will delete or overwrite old experiment directory
"""

import sys
import json
import subprocess
import os
import shutil
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.minivla.configs.minivla_config import MiniVLAExperimentConfig


def _resolve_hf_checkpoint_path(hf_model_name: str) -> str | None:
    """
    Resolve HuggingFace model name to actual .pt checkpoint path.
    Looks in HF cache for checkpoints/*.pt files.
    Returns the latest checkpoint .pt path, or None if not found.
    """
    try:
        from huggingface_hub import scan_cache_dir
        hf_cache_info = scan_cache_dir()
        
        # Find the repo matching the model name
        for repo in hf_cache_info.repos:
            if repo.repo_id == hf_model_name:
                # Get the latest revision
                revisions = sorted(repo.revisions, key=lambda r: r.last_modified or "", reverse=True)
                if revisions:
                    latest_revision = revisions[0]
                    snapshot_path = Path(latest_revision.snapshot_path)
                    
                    # Look for checkpoints/*.pt files
                    checkpoints_dir = snapshot_path / "checkpoints"
                    if checkpoints_dir.exists():
                        pt_files = sorted(checkpoints_dir.glob("*.pt"))
                        if pt_files:
                            # Return the latest (highest step) checkpoint
                            return str(pt_files[-1])
                    
                    # Also check for symlinks to checkpoints
                    for item in snapshot_path.iterdir():
                        if item.name == "checkpoints" and item.is_dir():
                            pt_files = sorted(item.glob("*.pt"))
                            if pt_files:
                                return str(pt_files[-1])
                break
    except Exception as e:
        print(f"[WARNING] Failed to resolve HF checkpoint for {hf_model_name}: {e}")
    
    return None


def _find_latest_checkpoint_step(output_dir: Path) -> tuple[str, int] | None:
    """
    Find the latest checkpoint step in output_dir/checkpoints/.
    Returns (step_dir_name, step_number) or None if no checkpoints exist.
    Checkpoint directories follow LeRobot format: 000000, 002000, etc.
    """
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None

    existing_steps = sorted([
        d.name for d in checkpoints_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    ])
    if not existing_steps:
        return None

    latest = existing_steps[-1]
    return latest, int(latest)


def _resolve_to_canonical(checkpoint_path: str) -> str:
    """
    Resolve checkpoint path to canonical absolute path for comparison.
    Handles both HF model names and local .pt paths.
    """
    if not checkpoint_path:
        return ""
    
    # If it's already a .pt path, resolve to absolute
    if checkpoint_path.endswith(".pt"):
        return str(Path(checkpoint_path).resolve())
    
    # Try to resolve HF model name to actual .pt path
    resolved = _resolve_hf_checkpoint_path(checkpoint_path)
    if resolved:
        return str(Path(resolved).resolve())
    
    # Fallback: return as-is
    return checkpoint_path


def _check_resume_compatible(output_dir: Path, cfg) -> tuple[bool, str]:
    """
    Check if existing checkpoints are compatible with current training configuration.
    Returns (compatible: bool, reason: str).
    
    Checks:
    1. Action tokenizer type (e.g., 7D VQ -> 4D non-VQ is incompatible)
    2. Official init mode (random-init -> official-backbone-init is incompatible)
    3. Official pretrained checkpoint path (different checkpoints are incompatible)
    """
    latest_info = _find_latest_checkpoint_step(output_dir)
    if latest_info is None:
        return True, "No existing checkpoints found"

    step_dir, _ = latest_info
    train_config_path = output_dir / "checkpoints" / step_dir / "pretrained_model" / "train_config.json"
    if not train_config_path.exists():
        return True, "No train_config.json found in latest checkpoint"

    with open(train_config_path, "r") as f:
        train_cfg = json.load(f)

    policy_cfg = train_cfg.get("policy", {})
    
    # Check 1: Action tokenizer type
    old_tokenizer_type = policy_cfg.get("action_tokenizer_type", "")
    if old_tokenizer_type and old_tokenizer_type != cfg.action_tokenizer_type:
        return False, (
            f"Action tokenizer type changed: old='{old_tokenizer_type}', new='{cfg.action_tokenizer_type}'. "
            f"Training target has changed, cannot resume."
        )

    # Check 2: Official init mode compatibility
    old_init_mode = policy_cfg.get("official_init_mode", "none")
    if old_init_mode != cfg.official_init_mode:
        return False, (
            f"Official init mode changed: old='{old_init_mode}', new='{cfg.official_init_mode}'. "
            f"Old checkpoint was trained with '{old_init_mode}' initialization, "
            f"but current config requests '{cfg.official_init_mode}'. "
            f"Use --restart to start fresh training with new init mode."
        )

    # Check 3: Official pretrained checkpoint path (if using backbone_only mode)
    # Use canonical path comparison to handle HF model names vs resolved .pt paths
    if cfg.official_init_mode == "backbone_only":
        old_checkpoint = policy_cfg.get("official_pretrained_checkpoint", "")
        if old_checkpoint:
            # Resolve both to canonical paths for fair comparison
            old_canonical = _resolve_to_canonical(old_checkpoint)
            new_canonical = _resolve_to_canonical(cfg.official_pretrained_checkpoint)
            if old_canonical != new_canonical:
                return False, (
                    f"Official pretrained checkpoint changed: "
                    f"old='{old_checkpoint}' (resolved: {old_canonical}), "
                    f"new='{cfg.official_pretrained_checkpoint}' (resolved: {new_canonical}). "
                    f"Use --restart to start fresh training with new checkpoint."
                )

    return True, "Checkpoint is compatible with current configuration"


def _check_action_tokenizer_compatible(output_dir: Path, new_tokenizer_type: str) -> bool:
    """
    Legacy compatibility check (kept for backward compatibility).
    Use _check_resume_compatible() for full compatibility checking.
    """
    latest_info = _find_latest_checkpoint_step(output_dir)
    if latest_info is None:
        return True

    step_dir, _ = latest_info
    train_config_path = output_dir / "checkpoints" / step_dir / "pretrained_model" / "train_config.json"
    if not train_config_path.exists():
        return True

    with open(train_config_path, "r") as f:
        train_cfg = json.load(f)

    policy_cfg = train_cfg.get("policy", {})
    old_tokenizer_type = policy_cfg.get("action_tokenizer_type", "")

    if old_tokenizer_type and old_tokenizer_type != new_tokenizer_type:
        return False

    return True


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

    # Handle restart: delete old experiment directory if restart=true
    if cfg.restart and output_dir.exists():
        print(f"[RESTART] Deleting old experiment directory: {output_dir}")
        shutil.rmtree(output_dir)

    # Determine resume state
    resume = False
    training_complete = False
    latest_step_info = _find_latest_checkpoint_step(output_dir)

    if latest_step_info is not None:
        step_dir, last_step = latest_step_info
        print(f"Found existing checkpoint: {step_dir} (step {last_step})")

        # Check full resume compatibility (tokenizer + init mode + checkpoint path)
        is_compatible, compat_reason = _check_resume_compatible(output_dir, cfg)
        if not is_compatible:
            print(
                f"[WARNING] Checkpoint incompatible: {compat_reason} "
                f"Please use a new output_dir or set restart=true."
            )
            if not cfg.restart:
                raise RuntimeError(
                    f"Checkpoint incompatible: {compat_reason} "
                    f"Use restart=true to start fresh training."
                )

        if last_step >= cfg.train_steps:
            training_complete = True
            print(f"Training already complete: step {last_step} >= {cfg.train_steps}")
        else:
            resume = True
            print(f"Training will resume from step {last_step}/{cfg.train_steps}")
    else:
        print(f"No existing checkpoints found, starting fresh training")

    # If restart=true and directory exists (but no checkpoints), clean it
    if cfg.restart and output_dir.exists() and latest_step_info is None:
        print(f"[RESTART] Cleaning output directory (no checkpoints to resume)")
        shutil.rmtree(output_dir)

    train_log = Path(cfg.logs_dir) / f"{cfg.exp_name}_training.log"

    print(f"Experiment name: {cfg.exp_name}")
    print(f"Action tokenizer type: {cfg.action_tokenizer_type}")
    print(f"Selected episodes: {len(episode_indices)}")
    print(f"Episode indices: {episodes_str}")
    print(f"Dataset root: {cfg.dataset_root}")
    print(f"Output dir: {output_dir}")
    print(f"Training log: {train_log}")
    print(f"VQ model path: {cfg.vq_model_path}")
    print(f"Steps: {cfg.train_steps}")
    print(f"Batch size: {cfg.train_batch_size}")
    print(f"GPU: {cfg.gpu_id}")

    # Official VLA checkpoint initialization logging
    resolved_checkpoint_path = cfg.official_pretrained_checkpoint
    
    if cfg.official_init_mode == "backbone_only" and cfg.official_pretrained_checkpoint:
        # Resolve HF model name to actual .pt checkpoint path
        if not cfg.official_pretrained_checkpoint.endswith(".pt"):
            resolved_path = _resolve_hf_checkpoint_path(cfg.official_pretrained_checkpoint)
            if resolved_path:
                resolved_checkpoint_path = resolved_path
                print(f"Resolved HF model '{cfg.official_pretrained_checkpoint}' to: {resolved_checkpoint_path}")
            else:
                raise FileNotFoundError(
                    f"Could not resolve HuggingFace model '{cfg.official_pretrained_checkpoint}' "
                    f"to a .pt checkpoint file. Please ensure the model is downloaded in HF cache."
                )
        
        print(f"\n{'='*60}")
        print(f"[OFFICIAL CHECKPOINT INITIALIZATION]")
        print(f"  Checkpoint path: {resolved_checkpoint_path}")
        print(f"  Initialization mode: backbone_only")
        print(f"  Note: Actual module loading results will be printed by MiniVLACore during init")
        print(f"{'='*60}")
    elif cfg.official_vla_checkpoint:
        print(f"Official VLA checkpoint: {cfg.official_vla_checkpoint}")
        print(f"Initialization mode: full_policy")
    else:
        print(f"Official VLA checkpoint: None (random initialization)")
        print(f"Initialization mode: none")

    if cfg.projector_lr > 0:
        print(f"Projector LR: {cfg.projector_lr}")
    if cfg.backbone_lr > 0:
        print(f"Backbone LR: {cfg.backbone_lr}")
    if cfg.scheduler_warmup_steps > 0:
        print(f"Scheduler warmup steps: {cfg.scheduler_warmup_steps}")
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
        f"--policy.type={cfg.policy_type}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--policy.use_amp=true",
        "--dataset.repo_id=lerobot/metaworld_pick_place",
        f"--dataset.root={cfg.dataset_root}",
        f"--dataset.episodes={episodes_str}",
        "--dataset.eval_split=0.0",
        "--env.type=metaworld",
        f"--env.task={task_name}",
        f"--env.camera_name={cfg.camera_names}",
        f"--policy.vq_model_path={cfg.vq_model_path}",
        f"--policy.action_tokenizer_type={cfg.action_tokenizer_type}",
        f"--policy.primary_image_key={cfg.primary_image_key}",
        f"--policy.wrist_image_key={cfg.wrist_image_key}",
        # Official MiniVLA optimizer config (from teach_code/MiniVLA/prismatic/conf/vla.py)
        f"--policy.optimizer_lr={cfg.official_lr}",
        f"--policy.optimizer_weight_decay={cfg.official_weight_decay}",
        f"--policy.optimizer_grad_clip_norm={cfg.official_max_grad_norm}",
        # Official scheduler config
        f"--policy.scheduler_type={cfg.official_lr_scheduler_type}",
        f"--policy.scheduler_warmup_ratio={cfg.official_warmup_ratio}",
    ]

    # Add official VLA checkpoint if specified (legacy full policy loading)
    if cfg.official_vla_checkpoint:
        cmd.append(f"--policy.official_vla_checkpoint={cfg.official_vla_checkpoint}")

    # Add backbone-only pretrained initialization (new mode)
    if cfg.official_init_mode == "backbone_only":
        cmd.append(f"--policy.official_init_mode=backbone_only")
        if resolved_checkpoint_path:
            cmd.append(f"--policy.official_pretrained_checkpoint={resolved_checkpoint_path}")

    # Add scheduler warmup steps
    if cfg.scheduler_warmup_steps > 0:
        cmd.append(f"--policy.scheduler_warmup_steps={cfg.scheduler_warmup_steps}")

    cmd.extend([
        f"--save_freq={cfg.train_save_freq}",
        f"--steps={cfg.train_steps}",
        f"--batch_size={cfg.train_batch_size}",
        f"--num_workers={cfg.train_num_workers}",
        f"--eval.n_episodes={cfg.eval_n_episodes}",
        f"--eval.batch_size={cfg.eval_batch_size}",
        f"--env_eval_freq={cfg.env_eval_freq}",
        f"--env.use_self_mw={str(cfg.use_self_mw).lower()}",
        f"--seed={cfg.selection_seed}",
        f"--job_name=minivla_{cfg.exp_name}",
        f"--output_dir={output_dir}",
        '--remove_features=["observation.environment_state"]',
        "--wandb.enable=true",
    ])

    # Resume: use LeRobot standard --resume with --config_path
    if resume:
        cmd.append("--resume=true")
        step_dir, last_step = latest_step_info
        config_path = output_dir / "checkpoints" / step_dir / "pretrained_model" / "train_config.json"
        cmd.append(f"--config_path={config_path}")
        print(f"Resume enabled: adding --resume=true --config_path={config_path}")
        print(f"Will resume from step {last_step}")

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
                step_dir, last_step = latest_step_info
                log_f.write(f"Resuming from step {last_step}\n")
                log_f.write(f"Config path: {output_dir / 'checkpoints' / step_dir / 'pretrained_model' / 'train_config.json'}\n")
                log_f.write(f"{'='*60}\n\n")
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).resolve().parents[5]),
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"Training failed with exit code {result.returncode}. "
                f"Check log: {train_log}"
            )

    checkpoint_dir = output_dir / "checkpoints"
    print(f"\nTraining complete.")
    print(f"Checkpoint dir: {checkpoint_dir}")
    sys.stdout.flush()
    return str(checkpoint_dir)