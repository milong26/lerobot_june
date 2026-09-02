"""
DemInf VAE Training

Training loop for Beta-VAE with checkpointing, validation split, and early stopping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

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
    Train a Beta-VAE on the given data.

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

    # Convert to tensor
    data_tensor = torch.from_numpy(data).float().to(device)
    dataset = TensorDataset(data_tensor)

    # Train/validation split (90/10)
    n = len(dataset)
    n_val = max(1, int(n * 0.1))
    n_train = n - n_val
    generator = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    # Create model
    model = BetaVAE(input_dim, latent_dim, config.hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.vae_lr, weight_decay=config.weight_decay)

    # Beta value
    beta = config.vae_beta_state if name == "state" else config.vae_beta_action

    # Checkpoint path
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_path = ckpt_dir / f"{name}_vae.pt"

    # Resume if applicable
    start_epoch = 0
    if config.resume and ckpt_path.exists():
        logger.info(f"Resuming {name} VAE from {ckpt_path}")
        ckpt = load_vae_checkpoint(str(ckpt_path), device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    if config.skip_train_if_checkpoint_exists and ckpt_path.exists() and start_epoch == 0:
        logger.info(f"Checkpoint exists for {name} VAE, skipping training")
        ckpt = load_vae_checkpoint(str(ckpt_path), device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt.get("train_log", {})

    # Training loop
    history = {"train_total": [], "train_recon": [], "train_kl": [],
               "val_total": [], "val_recon": [], "val_kl": []}
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    logger.info(f"Training {name} VAE: input_dim={input_dim}, latent_dim={latent_dim}, "
                f"beta={beta}, epochs={config.vae_epochs}, batch_size={config.batch_size}, "
                f"train={n_train}, val={n_val}")

    for epoch in range(start_epoch, config.vae_epochs):
        # Train
        model.train()
        train_total, train_recon, train_kl = 0.0, 0.0, 0.0
        n_batches = 0

        for (batch,) in train_loader:
            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch)
            total, recon, kl = vae_loss(x_recon, batch, mu, logvar, beta=beta)
            total.backward()
            optimizer.step()

            train_total += total.item()
            train_recon += recon.item()
            train_kl += kl.item()
            n_batches += 1

        avg_train_total = train_total / n_batches
        avg_train_recon = train_recon / n_batches
        avg_train_kl = train_kl / n_batches

        # Validate
        model.eval()
        val_total, val_recon, val_kl = 0.0, 0.0, 0.0
        n_val_batches = 0

        with torch.no_grad():
            for (batch,) in val_loader:
                x_recon, mu, logvar = model(batch)
                total, recon, kl = vae_loss(x_recon, batch, mu, logvar, beta=beta)
                val_total += total.item()
                val_recon += recon.item()
                val_kl += kl.item()
                n_val_batches += 1

        avg_val_total = val_total / n_val_batches
        avg_val_recon = val_recon / n_val_batches
        avg_val_kl = val_kl / n_val_batches

        history["train_total"].append(avg_train_total)
        history["train_recon"].append(avg_train_recon)
        history["train_kl"].append(avg_train_kl)
        history["val_total"].append(avg_val_total)
        history["val_recon"].append(avg_val_recon)
        history["val_kl"].append(avg_val_kl)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"[{name}] Epoch {epoch+1}/{config.vae_epochs} | "
                f"Train: total={avg_train_total:.4f} recon={avg_train_recon:.4f} kl={avg_train_kl:.4f} | "
                f"Val: total={avg_val_total:.4f} recon={avg_val_recon:.4f} kl={avg_val_kl:.4f}"
            )

        # Early stopping
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            patience_counter = 0
            # Save best checkpoint
            save_vae_checkpoint(
                model, optimizer, epoch, config, input_dim, latent_dim,
                str(ckpt_path), normalization_stats, history,
            )
        else:
            patience_counter += 1

    # Load best checkpoint
    if ckpt_path.exists():
        ckpt = load_vae_checkpoint(str(ckpt_path), device)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    logger.info(f"{name} VAE training complete. Best val loss: {best_val_loss:.4f}")

    return model, {"history": history, "best_val_loss": best_val_loss}


def save_vae_checkpoint(
    model: BetaVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: DemInfConfig,
    input_dim: int,
    latent_dim: int,
    path: str,
    normalization_stats: Optional[Dict[str, np.ndarray]] = None,
    train_log: Optional[Dict] = None,
) -> None:
    """
    Save VAE checkpoint.

    Args:
        model: BetaVAE model.
        optimizer: Optimizer.
        epoch: Current epoch.
        config: DemInfConfig.
        input_dim: Input dimension.
        latent_dim: Latent dimension.
        path: Checkpoint file path.
        normalization_stats: Optional normalization statistics.
        train_log: Optional training log.
    """
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": {
            "hidden_dims": config.hidden_dims,
            "vae_lr": config.vae_lr,
            "weight_decay": config.weight_decay,
            "vae_beta_state": config.vae_beta_state,
            "vae_beta_action": config.vae_beta_action,
        },
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "normalization_stats": normalization_stats,
        "train_log": train_log,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_vae_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    """
    Load VAE checkpoint.

    Args:
        path: Checkpoint file path.
        device: Target device.

    Returns:
        Checkpoint dictionary.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt