"""
DemInf VAE Models

Pure PyTorch Beta-VAE implementation for state and action latent encoding.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """
    MLP Encoder for Beta-VAE.

    Architecture: Linear(input_dim, 512) + ReLU + Linear(512, 512) + ReLU
    Then two heads: mu and logvar, each Linear(512, latent_dim).

    Args:
        input_dim: Dimension of input vector.
        hidden_dims: List of hidden layer dimensions. Default [512, 512].
        latent_dim: Dimension of latent space.

    Forward:
        x: [batch, input_dim]
        Returns: mu [batch, latent_dim], logvar [batch, latent_dim]
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
        self.mu_head = nn.Linear(prev_dim, latent_dim)
        self.logvar_head = nn.Linear(prev_dim, latent_dim)

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
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


class MLPDecoder(nn.Module):
    """
    MLP Decoder for Beta-VAE.

    Architecture: Linear(latent_dim, 512) + ReLU + Linear(512, 512) + ReLU + Linear(512, output_dim).

    Args:
        latent_dim: Dimension of latent space.
        hidden_dims: List of hidden layer dimensions. Default [512, 512].
        output_dim: Dimension of output (reconstruction) vector.

    Forward:
        z: [batch, latent_dim]
        Returns: x_recon [batch, output_dim]
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

    Args:
        input_dim: Dimension of input data.
        latent_dim: Dimension of latent space.
        hidden_dims: Hidden layer dimensions for both encoder and decoder.

    Methods:
        encode(x): Returns mu, logvar.
        reparameterize(mu, logvar): Samples z using reparameterization trick.
        decode(z): Returns reconstruction.
        forward(x): Returns x_recon, mu, logvar.
        get_embedding(x, use_mean=True): Returns deterministic mu or sampled z.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [512, 512]

        self.encoder = MLPEncoder(input_dim, self.hidden_dims, latent_dim)
        self.decoder = MLPDecoder(latent_dim, self.hidden_dims, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to posterior parameters.

        Args:
            x: Input tensor [batch, input_dim].

        Returns:
            mu: [batch, latent_dim], logvar: [batch, latent_dim].
        """
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon.

        Args:
            mu: Posterior mean [batch, latent_dim].
            logvar: Posterior log-variance [batch, latent_dim].

        Returns:
            z: Sampled latent [batch, latent_dim].
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to reconstruction.

        Args:
            z: Latent tensor [batch, latent_dim].

        Returns:
            x_recon: Reconstruction [batch, input_dim].
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x: Input tensor [batch, input_dim].

        Returns:
            x_recon: Reconstruction [batch, input_dim].
            mu: Posterior mean [batch, latent_dim].
            logvar: Posterior log-variance [batch, latent_dim].
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def get_embedding(self, x: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        """
        Get latent embedding for a batch of inputs.

        For scoring, use_mean=True to get deterministic posterior mean.

        Args:
            x: Input tensor [batch, input_dim].
            use_mean: If True, return mu; if False, sample from posterior.

        Returns:
            z: Latent embedding [batch, latent_dim].
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
    Compute Beta-VAE loss.

    Formula:
        L = L_recon + beta * L_KL
        L_recon = mean_i sum_d (recon_{i,d} - x_{i,d})^2  (MSE)
        L_KL = -0.5 * mean_j [1 + logvar_j - mu_j^2 - exp(logvar_j)]

    Args:
        x_recon: Reconstructed output [batch, D].
        x: Original input [batch, D].
        mu: Posterior mean [batch, latent_dim].
        logvar: Posterior log-variance [batch, latent_dim].
        beta: KL weight coefficient.

    Returns:
        total_loss: Scalar tensor.
        recon_loss: Scalar tensor (MSE).
        kl_loss: Scalar tensor (KL divergence).
    """
    # Reconstruction loss: mean squared error
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")

    # KL divergence: KL(q(z|x) || N(0, I))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss