"""
DemInf VAE Models

Pure PyTorch Beta-VAE implementation for state and action latent encoding.
Architecture matches official low-dimensional DemInf VAE.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    MLP Encoder for Beta-VAE.

    Architecture: Linear(input_dim, 512) + ReLU + Linear(512, 512) + ReLU
    Then z_proj = Linear(512, 2*latent_dim), split into mu and logvar via torch.chunk.

    This matches the official lowdim VAE encoder structure.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = None, latent_dim: int = 12) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        self.shared = nn.Sequential(*layers)
        self.z_proj = nn.Linear(prev_dim, 2 * latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, input_dim].

        Returns:
            mu: Posterior mean [batch, latent_dim].
            logvar: Posterior log-variance [batch, latent_dim].
        """
        h = self.shared(x)
        z_params = self.z_proj(h)
        mu, logvar = torch.chunk(z_params, 2, dim=-1)
        return mu, logvar


class MLPDecoder(nn.Module):
    """
    MLP Decoder for Beta-VAE.

    Architecture: Linear(latent_dim, 512) + ReLU + Linear(512, 512) + ReLU + Linear(512, output_dim).
    """

    def __init__(self, latent_dim: int, hidden_dims: List[int] = None, output_dim: int = 4) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]

        layers = []
        prev_dim = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: Latent tensor of shape [batch, latent_dim].

        Returns:
            x_recon: Reconstructed output [batch, output_dim].
        """
        return self.decoder(z)


class BetaVAE(nn.Module):
    """
    Beta-VAE combining MLPEncoder and MLPDecoder.

    For scoring, use get_embedding(x, use_mean=True) to get deterministic posterior mean.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [512, 512]

        self.encoder = MLPEncoder(input_dim, self.hidden_dims, latent_dim)
        self.decoder = MLPDecoder(latent_dim, self.hidden_dims, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to posterior parameters."""
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def get_embedding(self, x: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        """
        Get latent embedding for a batch of inputs.

        For scoring, use_mean=True to get deterministic posterior mean.
        """
        mu, logvar = self.encode(x)
        if use_mean:
            return mu
        return self.reparameterize(mu, logvar)


def vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Beta-VAE loss matching official DemInf formulation.

    Reconstruction: sum over feature dimensions, then mean over batch.
        recon_per_sample = ((x - x_hat)**2).sum(dim=-1)
        recon_loss = recon_per_sample.mean()

    KL: sum over latent dimensions, then mean over batch.
        kl_per_sample = 0.5 * sum(-logvar - 1 + exp(logvar) + mu**2, dim=-1)
        kl_loss = kl_per_sample.mean()

    Total: loss = recon_loss + beta * kl_loss

    Args:
        x_recon: Reconstructed output [batch, D].
        x: Original input [batch, D].
        mu: Posterior mean [batch, latent_dim].
        logvar: Posterior log-variance [batch, latent_dim].
        beta: KL weight coefficient.

    Returns:
        total_loss: Scalar tensor.
        recon_loss: Scalar tensor.
        kl_loss: Scalar tensor.
    """
    recon_per_sample = ((x_recon - x) ** 2).sum(dim=-1)
    recon_loss = recon_per_sample.mean()

    kl_per_sample = 0.5 * torch.sum(-logvar - 1 + torch.exp(logvar) + mu ** 2, dim=-1)
    kl_loss = kl_per_sample.mean()

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss