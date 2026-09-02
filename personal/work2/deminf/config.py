"""
DemInf Configuration

Unified dataclass for all DemInf hyperparameters and paths.
Faithful to the official state-based DemInf (RSS 2025) configuration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class DemInfConfig:
    """
    Unified configuration for DemInf episode selection.
    Official state-based defaults from DemInf (RSS 2025).
    """

    # Paths
    dataset_path: str = ""
    output_dir: str = ""
    target_episodes: int = 112
    seed: int = 42
    device: str = "cuda"

    # Data loading
    batch_size: int = 256
    num_workers: int = 4
    state_source: str = "observation.environment_state"
    action_key: str = "action"
    max_timesteps_per_episode: Optional[int] = None
    normalize_state: bool = True
    normalize_action: bool = True
    relative_action: Optional[bool] = None

    # VAE architecture
    state_latent_dim: int = 12
    action_latent_dim: int = 6
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512])

    # VAE training
    vae_beta_state: float = 0.05
    vae_beta_action: float = 0.05
    vae_lr: float = 1e-4
    vae_epochs: int = 100
    vae_steps: int = 50000
    weight_decay: float = 0.0

    # KSG MI estimation
    ks: Tuple[int, ...] = (5, 6, 7)
    score_batch_size: int = 1024
    ksg_chunk_size: Optional[int] = None

    # Quality inference (official DemInf scoring)
    quality_batch_size: int = 1024
    quality_repeat: int = 4
    quality_discard_fraction: float = 0.5
    quality_cache: bool = True
    quality_drop_remainder: bool = True
    score_clip_low: float = 1.0
    score_clip_high: float = 99.0

    # Checkpointing
    checkpoint_dir: Optional[str] = None
    resume: bool = False
    skip_train_if_checkpoint_exists: bool = True
    use_latent_cache: bool = True
    checkpoint_step: int = 50000

    # Advanced
    representation: str = "state"
    ksg_backend: str = "chunked"
    ksg_mode: str = "deminf_rank"

    def __post_init__(self) -> None:
        """Set derived paths after initialization."""
        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(Path(self.output_dir) / "checkpoints")

    def validate(self) -> None:
        """Validate configuration values."""
        if not self.dataset_path:
            raise ValueError("dataset_path must be specified")
        if not Path(self.dataset_path).exists():
            raise ValueError(f"dataset_path does not exist: {self.dataset_path}")
        if not self.output_dir:
            raise ValueError("output_dir must be specified")
        if self.target_episodes <= 0:
            raise ValueError(f"target_episodes must be positive, got {self.target_episodes}")
        if self.state_latent_dim <= 0:
            raise ValueError(f"state_latent_dim must be positive, got {self.state_latent_dim}")
        if self.action_latent_dim <= 0:
            raise ValueError(f"action_latent_dim must be positive, got {self.action_latent_dim}")
        if not self.hidden_dims or any(d <= 0 for d in self.hidden_dims):
            raise ValueError(f"hidden_dims must be positive integers, got {self.hidden_dims}")
        if self.vae_beta_state < 0:
            raise ValueError(f"vae_beta_state must be non-negative, got {self.vae_beta_state}")
        if self.vae_beta_action < 0:
            raise ValueError(f"vae_beta_action must be non-negative, got {self.vae_beta_action}")
        if self.vae_lr <= 0:
            raise ValueError(f"vae_lr must be positive, got {self.vae_lr}")
        if self.vae_steps <= 0:
            raise ValueError(f"vae_steps must be positive, got {self.vae_steps}")
        if not self.ks or any(k <= 0 for k in self.ks):
            raise ValueError(f"ks must be positive integers, got {self.ks}")
        if self.quality_batch_size <= max(self.ks):
            raise ValueError(
                f"quality_batch_size ({self.quality_batch_size}) must be > max(ks) ({max(self.ks)})"
            )
        if self.quality_repeat <= 0:
            raise ValueError(f"quality_repeat must be positive, got {self.quality_repeat}")
        if not (0 <= self.score_clip_low < self.score_clip_high <= 100):
            raise ValueError(
                f"score_clip_low/high must satisfy 0 <= low < high <= 100, "
                f"got low={self.score_clip_low}, high={self.score_clip_high}"
            )
        if self.score_batch_size <= 0:
            raise ValueError(f"score_batch_size must be positive, got {self.score_batch_size}")
        if self.representation not in ("state", "image"):
            raise ValueError(f"representation must be 'state' or 'image', got {self.representation}")
        if self.representation == "image":
            raise NotImplementedError("image representation is not yet implemented")
        if self.ksg_backend not in ("chunked", "full"):
            raise ValueError(f"ksg_backend must be 'chunked' or 'full', got {self.ksg_backend}")
        if self.ksg_mode not in ("deminf_rank", "full_mi"):
            raise ValueError(f"ksg_mode must be 'deminf_rank' or 'full_mi', got {self.ksg_mode}")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def effective_discard_fraction(self) -> float:
        """
        When quality_cache=True, official OpenX dataloader forces discard_fraction=0.0.
        Returns the effective discard fraction actually used.
        """
        if self.quality_cache:
            return 0.0
        return self.quality_discard_fraction

    def config_fingerprint(self) -> str:
        """Generate a fingerprint string for cache validation."""
        key_fields = {
            "state_latent_dim": self.state_latent_dim,
            "action_latent_dim": self.action_latent_dim,
            "hidden_dims": self.hidden_dims,
            "vae_beta_state": self.vae_beta_state,
            "vae_beta_action": self.vae_beta_action,
            "vae_lr": self.vae_lr,
            "vae_steps": self.vae_steps,
            "weight_decay": self.weight_decay,
            "state_source": self.state_source,
            "action_key": self.action_key,
            "representation": self.representation,
        }
        raw = json.dumps(key_fields, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]