"""
DemInf VAE Training

Training loop for Beta-VAE with fixed optimizer steps (no early stopping).
Matches official DemInf: fixed 50000 Adam steps, lr=1e-4, weight_decay=0.

Checkpoint validation ensures that smoke checkpoints (e.g. 50-step) cannot
be silently reused for official 50000-step experiments. Every checkpoint
carries a SHA256 fingerprint encoding all fields that affect VAE training.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from deminf.config import DemInfConfig
from deminf.models import BetaVAE, vae_loss
from deminf.utils import get_device

logger = logging.getLogger("deminf")


def build_vae_checkpoint_fingerprint(
    config: DemInfConfig,
    name: str,
    input_dim: int,
    latent_dim: int,
    normalization_stats: Optional[Dict],
) -> str:
    """
    Build a SHA256 fingerprint encoding all fields that affect VAE training.

    This fingerprint is stored inside every checkpoint and used during
    validation to detect mismatches between a saved checkpoint and the
    current experiment configuration.

    Args:
        config: DemInfConfig with training hyperparameters.
        name: 'state' or 'action'.
        input_dim: Input dimensionality for this VAE.
        latent_dim: Latent dimensionality for this VAE.
        normalization_stats: Precomputed normalization stats (used for manifest).

    Returns:
        32-char hex SHA256 fingerprint string.
    """
    beta = config.vae_beta_state if name == "state" else config.vae_beta_action

    norm_manifest = ""
    if normalization_stats is not None:
        norm_manifest = json.dumps(normalization_stats, sort_keys=True, default=str)

    key_fields = {
        "name": name,
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "hidden_dims": config.hidden_dims,
        "beta": beta,
        "vae_lr": config.vae_lr,
        "vae_steps": config.vae_steps,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "state_source": config.state_source,
        "action_key": config.action_key,
        "normalization_manifest": norm_manifest,
    }
    raw = json.dumps(key_fields, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def save_vae_checkpoint(
    model: BetaVAE,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    config: DemInfConfig,
    input_dim: int,
    latent_dim: int,
    path: str,
    normalization_stats: Optional[Dict[str, np.ndarray]] = None,
    train_log: Optional[Dict] = None,
    name: Optional[str] = None,
) -> None:
    """
    Save VAE checkpoint with comprehensive metadata for validation.

    The checkpoint includes a fingerprint computed from all fields that
    affect VAE training, so that mismatches can be detected at load time.

    Args:
        model: Trained BetaVAE model.
        optimizer: Adam optimizer state.
        global_step: Current optimizer step count.
        config: DemInfConfig with hyperparameters.
        input_dim: Input dimensionality.
        latent_dim: Latent dimensionality.
        path: File path to save checkpoint.
        normalization_stats: Optional normalization statistics.
        train_log: Optional training history.
        name: 'state' or 'action'. If None, inferred from path.
    """
    if name is None:
        name = "state" if "state" in str(path) else ("action" if "action" in str(path) else "vae")
    fingerprint = build_vae_checkpoint_fingerprint(config, name, input_dim, latent_dim, normalization_stats)

    norm_manifest = None
    if normalization_stats is not None:
        norm_manifest = json.dumps(normalization_stats, sort_keys=True, default=str)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "name": name,
        "hidden_dims": config.hidden_dims,
        "beta": config.vae_beta_state if name == "state" else config.vae_beta_action,
        "vae_lr": config.vae_lr,
        "weight_decay": config.weight_decay,
        "vae_steps": config.vae_steps,
        "batch_size": config.batch_size,
        "normalization_stats": normalization_stats,
        "normalization_manifest": norm_manifest,
        "checkpoint_fingerprint": fingerprint,
        "train_log": train_log,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_vae_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    """
    Load VAE checkpoint from file.

    This function only reads the file; it does NOT validate configuration.
    Call validate_vae_checkpoint() separately to verify compatibility.

    Args:
        path: Checkpoint file path.
        device: PyTorch device for loading tensors.

    Returns:
        Raw checkpoint dictionary.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def validate_vae_checkpoint(
    ckpt: Dict,
    config: DemInfConfig,
    name: str,
    input_dim: int,
    latent_dim: int,
    normalization_stats: Optional[Dict],
    require_target_step: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validate that a loaded checkpoint is compatible with the current experiment.

    Checks every field that affects VAE training: name, input_dim, latent_dim,
    hidden_dims, beta, vae_lr, weight_decay, batch_size, normalization fingerprint,
    checkpoint fingerprint, and global_step.

    Args:
        ckpt: Loaded checkpoint dictionary.
        config: Current DemInfConfig.
        name: Expected VAE name ('state' or 'action').
        input_dim: Expected input dimensionality.
        latent_dim: Expected latent dimensionality.
        normalization_stats: Current normalization stats for comparison.
        require_target_step: If True, global_step must be >= config.vae_steps.

    Returns:
        Tuple of (is_valid, list_of_mismatch_reasons).
        If valid, the list is empty.
    """
    reasons = []

    # Check name
    ckpt_name = ckpt.get("name", "unknown")
    if ckpt_name != name:
        reasons.append(f"name mismatch: checkpoint='{ckpt_name}', expected='{name}'")

    # Check input_dim
    ckpt_input_dim = ckpt.get("input_dim")
    if ckpt_input_dim is not None and ckpt_input_dim != input_dim:
        reasons.append(f"input_dim mismatch: checkpoint={ckpt_input_dim}, expected={input_dim}")

    # Check latent_dim
    ckpt_latent_dim = ckpt.get("latent_dim")
    if ckpt_latent_dim is not None and ckpt_latent_dim != latent_dim:
        reasons.append(f"latent_dim mismatch: checkpoint={ckpt_latent_dim}, expected={latent_dim}")

    # Check hidden_dims
    ckpt_hidden = ckpt.get("hidden_dims")
    if ckpt_hidden is not None and ckpt_hidden != config.hidden_dims:
        reasons.append(f"hidden_dims mismatch: checkpoint={ckpt_hidden}, expected={config.hidden_dims}")

    # Check beta
    expected_beta = config.vae_beta_state if name == "state" else config.vae_beta_action
    ckpt_beta = ckpt.get("beta")
    if ckpt_beta is not None and abs(ckpt_beta - expected_beta) > 1e-10:
        reasons.append(f"beta mismatch: checkpoint={ckpt_beta}, expected={expected_beta}")

    # Check vae_lr
    ckpt_lr = ckpt.get("vae_lr")
    if ckpt_lr is not None and abs(ckpt_lr - config.vae_lr) > 1e-12:
        reasons.append(f"vae_lr mismatch: checkpoint={ckpt_lr}, expected={config.vae_lr}")

    # Check weight_decay
    ckpt_wd = ckpt.get("weight_decay")
    if ckpt_wd is not None and abs(ckpt_wd - config.weight_decay) > 1e-12:
        reasons.append(f"weight_decay mismatch: checkpoint={ckpt_wd}, expected={config.weight_decay}")

    # Check batch_size
    ckpt_bs = ckpt.get("batch_size")
    if ckpt_bs is not None and ckpt_bs != config.batch_size:
        reasons.append(f"batch_size mismatch: checkpoint={ckpt_bs}, expected={config.batch_size}")

    # Check normalization fingerprint
    ckpt_norm_manifest = ckpt.get("normalization_manifest")
    if ckpt_norm_manifest is not None and normalization_stats is not None:
        current_manifest = json.dumps(normalization_stats, sort_keys=True, default=str)
        if ckpt_norm_manifest != current_manifest:
            reasons.append("normalization_manifest mismatch between checkpoint and current stats")

    # Check checkpoint fingerprint
    ckpt_fp = ckpt.get("checkpoint_fingerprint")
    if ckpt_fp is not None:
        expected_fp = build_vae_checkpoint_fingerprint(config, name, input_dim, latent_dim, normalization_stats)
        if ckpt_fp != expected_fp:
            reasons.append(
                f"checkpoint_fingerprint mismatch: "
                f"checkpoint='{ckpt_fp}', expected='{expected_fp}'"
            )
    else:
        reasons.append("legacy checkpoint lacks validation metadata (checkpoint_fingerprint missing), retraining required")

    # Check global_step
    ckpt_step = ckpt.get("global_step", 0)
    if require_target_step and ckpt_step < config.vae_steps:
        reasons.append(
            f"global_step={ckpt_step} < required vae_steps={config.vae_steps}; "
            f"smoke checkpoint cannot be used for official experiment"
        )

    return (len(reasons) == 0, reasons)


def find_checkpoint(
    checkpoint_dir: str | Path,
    name: str,
    target_step: int,
    config: Optional[DemInfConfig] = None,
    input_dim: Optional[int] = None,
    latent_dim: Optional[int] = None,
    normalization_stats: Optional[Dict] = None,
) -> Optional[str]:
    """
    Find a compatible checkpoint file for a specific target step.

    Priority:
    1. Exact match: {name}_vae_step{target_step}.pt
    2. Default: {name}_vae.pt (only if global_step >= target_step AND
       all validation checks pass via validate_vae_checkpoint)

    If the checkpoint lacks fingerprint metadata or fails validation,
    it is rejected and None is returned.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        name: 'state' or 'action'.
        target_step: Required minimum global_step (typically config.vae_steps).
        config: DemInfConfig for validation (required for full validation).
        input_dim: Expected input dimensionality.
        latent_dim: Expected latent dimensionality.
        normalization_stats: Current normalization stats.

    Returns:
        Checkpoint file path string if valid, None otherwise.
    """
    ckpt_dir = Path(checkpoint_dir)

    # Priority 1: exact step match
    target_path = ckpt_dir / f"{name}_vae_step{target_step}.pt"
    if target_path.exists():
        if config is not None and input_dim is not None and latent_dim is not None:
            ckpt = torch.load(str(target_path), map_location="cpu", weights_only=False)
            valid, reasons = validate_vae_checkpoint(
                ckpt, config, name, input_dim, latent_dim, normalization_stats,
                require_target_step=True,
            )
            if valid:
                return str(target_path)
            else:
                logger.warning(
                    f"Exact step checkpoint {target_path} rejected: {'; '.join(reasons)}"
                )
                return None
        return str(target_path)

    # Priority 2: default checkpoint
    default_path = ckpt_dir / f"{name}_vae.pt"
    if default_path.exists():
        ckpt = torch.load(str(default_path), map_location="cpu", weights_only=False)
        ckpt_step = ckpt.get("global_step", 0)

        if ckpt_step < target_step:
            logger.info(
                f"Default checkpoint {default_path} has global_step={ckpt_step} "
                f"< target_step={target_step}, not suitable"
            )
            return None

        if config is not None and input_dim is not None and latent_dim is not None:
            valid, reasons = validate_vae_checkpoint(
                ckpt, config, name, input_dim, latent_dim, normalization_stats,
                require_target_step=True,
            )
            if valid:
                return str(default_path)
            else:
                logger.warning(
                    f"Default checkpoint {default_path} rejected: {'; '.join(reasons)}"
                )
                return None

        return str(default_path)

    return None


def train_beta_vae(
    data: np.ndarray,
    input_dim: int,
    latent_dim: int,
    config: DemInfConfig,
    name: str = "vae",
    normalization_stats: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[BetaVAE, Dict[str, Any]]:
    """
    Train a Beta-VAE using fixed optimizer steps (no early stopping).

    Data stays on CPU; each batch is moved to device inside the loop.
    Training runs for exactly config.vae_steps optimizer updates.

    Checkpoint logic:
    - If resume=True and a compatible checkpoint exists, load and continue
      from its global_step until config.vae_steps is reached.
    - If skip_train_if_checkpoint_exists=True and a fully-trained checkpoint
      (global_step >= config.vae_steps) passes all validation, skip training.
    - A smoke checkpoint (e.g. 50 steps) with config.vae_steps=50000 will
      NEVER cause training to be skipped; it will either resume from 50 or
      start fresh depending on resume flag.

    Args:
        data: Input data array of shape [N, input_dim].
        input_dim: Dimension of input data.
        latent_dim: Dimension of latent space.
        config: DemInfConfig with training hyperparameters.
        name: Name for logging and checkpointing ('state' or 'action').
        normalization_stats: Optional precomputed normalization stats.

    Returns:
        model: Trained BetaVAE in eval mode.
        train_log: Dictionary with training history and metadata.
    """
    device = get_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Keep data on CPU
    data_tensor = torch.from_numpy(data).float()
    dataset = TensorDataset(data_tensor)

    # Infinite shuffle DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    # Create model
    model = BetaVAE(input_dim, latent_dim, config.hidden_dims).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.vae_lr, weight_decay=config.weight_decay
    )

    beta = config.vae_beta_state if name == "state" else config.vae_beta_action

    # Checkpoint path
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_path = ckpt_dir / f"{name}_vae.pt"

    # -----------------------------------------------------------------------
    # Resume logic: load compatible checkpoint and continue training
    # -----------------------------------------------------------------------
    global_step = 0
    if config.resume and ckpt_path.exists():
        ckpt = load_vae_checkpoint(str(ckpt_path), device)
        valid, reasons = validate_vae_checkpoint(
            ckpt, config, name, input_dim, latent_dim, normalization_stats,
            require_target_step=False,
        )
        if valid:
            logger.info(f"Resuming {name} VAE from {ckpt_path}")
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            global_step = ckpt.get("global_step", 0)
            logger.info(f"Resumed from global_step={global_step}")
        else:
            logger.warning(
                f"Checkpoint {ckpt_path} incompatible for resume: "
                f"{'; '.join(reasons)}. Starting from scratch."
            )

    # -----------------------------------------------------------------------
    # Skip logic: only skip if checkpoint is fully trained AND validated
    # -----------------------------------------------------------------------
    if config.skip_train_if_checkpoint_exists and ckpt_path.exists():
        ckpt = load_vae_checkpoint(str(ckpt_path), device)
        ckpt_step = ckpt.get("global_step", 0)
        valid, reasons = validate_vae_checkpoint(
            ckpt, config, name, input_dim, latent_dim, normalization_stats,
            require_target_step=True,
        )
        if valid:
            logger.info(
                f"Fully trained and validated {name} VAE checkpoint at step {ckpt_step}, "
                f"skipping training"
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            return model, ckpt.get("train_log", {})
        else:
            if ckpt_step < config.vae_steps:
                logger.info(
                    f"{name} VAE checkpoint at step {ckpt_step} < required {config.vae_steps}, "
                    f"will {'resume' if config.resume else 'retrain'}"
                )
                if config.resume and valid:
                    global_step = ckpt_step
                    model.load_state_dict(ckpt["model_state_dict"])
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            else:
                logger.warning(
                    f"{name} VAE checkpoint rejected: {'; '.join(reasons)}. "
                    f"Will retrain."
                )

    # -----------------------------------------------------------------------
    # Training loop with fixed steps
    # -----------------------------------------------------------------------
    history = {"train_total": [], "train_recon": [], "train_kl": []}
    total_steps = config.vae_steps

    logger.info(
        f"Training {name} VAE: input_dim={input_dim}, latent_dim={latent_dim}, "
        f"beta={beta}, steps={total_steps}, batch_size={config.batch_size}, "
        f"lr={config.vae_lr}, weight_decay={config.weight_decay}, "
        f"total_samples={len(dataset)}, starting from step={global_step}"
    )

    model.train()

    def infinite_loader():
        while True:
            for batch in train_loader:
                yield batch

    loader_iter = infinite_loader()

    while global_step < total_steps:
        (batch,) = next(loader_iter)
        batch = batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        x_recon, mu, logvar = model(batch)
        total, recon, kl = vae_loss(x_recon, batch, mu, logvar, beta=beta)
        total.backward()
        optimizer.step()

        global_step += 1

        history["train_total"].append(total.item())
        history["train_recon"].append(recon.item())
        history["train_kl"].append(kl.item())

        if global_step % 1000 == 0 or global_step == 1:
            avg_total = np.mean(history["train_total"][-1000:])
            avg_recon = np.mean(history["train_recon"][-1000:])
            avg_kl = np.mean(history["train_kl"][-1000:])
            logger.info(
                f"[{name}] Step {global_step}/{total_steps} | "
                f"Train: total={avg_total:.4f} recon={avg_recon:.4f} kl={avg_kl:.4f}"
            )

        if global_step % 10000 == 0:
            save_vae_checkpoint(
                model, optimizer, global_step, config, input_dim, latent_dim,
                str(ckpt_dir / f"{name}_vae_step{global_step}.pt"),
                normalization_stats, history, name=name,
            )
            logger.info(f"Saved {name} VAE checkpoint at step {global_step}")

    # Save final checkpoint
    final_ckpt_path = str(ckpt_dir / f"{name}_vae_step{global_step}.pt")
    save_vae_checkpoint(
        model, optimizer, global_step, config, input_dim, latent_dim,
        final_ckpt_path, normalization_stats, history, name=name,
    )
    save_vae_checkpoint(
        model, optimizer, global_step, config, input_dim, latent_dim,
        str(ckpt_path), normalization_stats, history, name=name,
    )
    logger.info(f"Saved final {name} VAE checkpoint at step {global_step}")

    model.eval()
    logger.info(f"{name} VAE training complete at step {global_step}")

    return model, {"history": history, "global_step": global_step}