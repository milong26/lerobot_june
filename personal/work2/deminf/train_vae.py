"""
DemInf VAE Training

Training loop for Beta-VAE with fixed optimizer steps (no early stopping).
Matches official DemInf: fixed 50000 Adam steps, lr=1e-4, weight_decay=0.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from deminf.config import DemInfConfig
from deminf.models import BetaVAE, vae_loss
from deminf.utils import get_device

logger = logging.getLogger("deminf")


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

    # Resume if applicable
    global_step = 0
    if config.resume and ckpt_path.exists():
        logger.info(f"Resuming {name} VAE from {ckpt_path}")
        ckpt = load_vae_checkpoint(str(ckpt_path), device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt.get("global_step", 0)
        logger.info(f"Resumed from global_step={global_step}")

    if config.skip_train_if_checkpoint_exists and ckpt_path.exists() and global_step == 0:
        logger.info(f"Checkpoint exists for {name} VAE, skipping training")
        ckpt = load_vae_checkpoint(str(ckpt_path), device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt.get("train_log", {})

    # Training loop with fixed steps
    history = {"train_total": [], "train_recon": [], "train_kl": []}
    total_steps = config.vae_steps

    logger.info(
        f"Training {name} VAE: input_dim={input_dim}, latent_dim={latent_dim}, "
        f"beta={beta}, steps={total_steps}, batch_size={config.batch_size}, "
        f"lr={config.vae_lr}, weight_decay={config.weight_decay}, "
        f"total_samples={len(dataset)}"
    )

    model.train()
    step_in_epoch = 0

    # Use a cyclic iterator over the DataLoader
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
        step_in_epoch += 1

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
                normalization_stats, history,
            )
            logger.info(f"Saved {name} VAE checkpoint at step {global_step}")

    # Save final checkpoint
    final_ckpt_path = str(ckpt_dir / f"{name}_vae_step{global_step}.pt")
    save_vae_checkpoint(
        model, optimizer, global_step, config, input_dim, latent_dim,
        final_ckpt_path, normalization_stats, history,
    )
    # Also save as the default checkpoint name
    save_vae_checkpoint(
        model, optimizer, global_step, config, input_dim, latent_dim,
        str(ckpt_path), normalization_stats, history,
    )
    logger.info(f"Saved final {name} VAE checkpoint at step {global_step}")

    model.eval()
    logger.info(f"{name} VAE training complete at step {global_step}")

    return model, {"history": history, "global_step": global_step}


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
) -> None:
    """
    Save VAE checkpoint with all metadata needed for cache validation.
    """
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "config": {
            "hidden_dims": config.hidden_dims,
            "vae_lr": config.vae_lr,
            "weight_decay": config.weight_decay,
            "vae_beta_state": config.vae_beta_state,
            "vae_beta_action": config.vae_beta_action,
            "vae_steps": config.vae_steps,
            "config_fingerprint": config.config_fingerprint(),
        },
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "normalization_stats": normalization_stats,
        "train_log": train_log,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_vae_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    """Load VAE checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def find_checkpoint(
    checkpoint_dir: str | Path,
    name: str,
    target_step: int,
) -> Optional[str]:
    """
    Find the checkpoint file for a specific step.

    Returns path if exists, None otherwise.
    """
    ckpt_dir = Path(checkpoint_dir)
    target_path = ckpt_dir / f"{name}_vae_step{target_step}.pt"
    if target_path.exists():
        return str(target_path)

    default_path = ckpt_dir / f"{name}_vae.pt"
    if default_path.exists():
        ckpt = torch.load(str(default_path), map_location="cpu", weights_only=False)
        if ckpt.get("global_step", 0) >= target_step:
            return str(default_path)

    return None