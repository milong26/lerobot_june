"""
DemInf Configuration

Unified dataclass for all DemInf hyperparameters and paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class DemInfConfig:
    """
    Unified configuration for DemInf episode selection.

    Attributes:
        dataset_path: Path to the LeRobot dataset directory.
        output_dir: Directory to save all outputs (scores, checkpoints, subset JSON).
        target_episodes: Number of episodes to select (e.g., 112).
        seed: Random seed for reproducibility.
        device: PyTorch device string ('cuda', 'cpu', or 'cuda:0').

        # Data loading
        batch_size: Batch size for VAE training.
        num_workers: Number of DataLoader workers.
        state_keys: Keys to extract from each timestep for state vector.
                    If None, auto-detected from dataset features.
        action_key: Key to extract action vector. Default: 'action'.
        max_timesteps_per_episode: Max timesteps per episode to use. None = all.
        normalize_state: Whether to z-score normalize state vectors.
        normalize_action: Whether to z-score normalize action vectors.
        relative_action: Whether actions are relative/delta control. Auto-detected if None.

        # VAE architecture
        state_latent_dim: Latent dimension for state VAE (default 12, per paper).
        action_latent_dim: Latent dimension for action VAE (default 6, per paper).
        hidden_dims: Hidden layer dimensions for both VAEs. Default [512, 512].

        # VAE training
        vae_beta_state: Beta coefficient for state VAE KL term.
        vae_beta_action: Beta coefficient for action VAE KL term.
        vae_lr: Learning rate for VAE optimizers.
        vae_epochs: Number of training epochs for VAEs.
        weight_decay: Weight decay for Adam optimizer.

        # KSG MI estimation
        ks: Tuple of k values for KSG estimator. Default (5, 6, 7) per paper.
        score_batch_size: Batch size for KSG pairwise distance computation.
        ksg_chunk_size: Chunk size for pairwise distance computation. None = auto.

        # Checkpointing
        checkpoint_dir: Directory to save VAE checkpoints. None = output_dir/checkpoints.
        resume: Whether to resume training from existing checkpoints.
        skip_train_if_checkpoint_exists: Skip VAE training if checkpoint exists.
        use_latent_cache: Whether to load/save latent embeddings from cache.

        # Advanced
        representation: 'state' or 'image'. Only 'state' is fully implemented.
        ksg_backend: 'chunked' (default, memory-safe) or 'full' (for small data validation).
        ksg_mode: 'deminf_rank' (default, for ranking) or 'full_mi' (complete MI contribution).
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
    state_keys: Optional[List[str]] = None
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
    vae_lr: float = 1e-3
    vae_epochs: int = 100
    weight_decay: float = 1e-5

    # KSG MI estimation
    ks: Tuple[int, ...] = (5, 6, 7)
    score_batch_size: int = 1024
    ksg_chunk_size: Optional[int] = None

    # Checkpointing
    checkpoint_dir: Optional[str] = None
    resume: bool = False
    skip_train_if_checkpoint_exists: bool = True
    use_latent_cache: bool = True

    # Advanced
    representation: str = "state"
    ksg_backend: str = "chunked"
    ksg_mode: str = "deminf_rank"

    def __post_init__(self) -> None:
        """Set derived paths after initialization."""
        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(Path(self.output_dir) / "checkpoints")

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
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
        if self.vae_epochs <= 0:
            raise ValueError(f"vae_epochs must be positive, got {self.vae_epochs}")
        if not self.ks or any(k <= 0 for k in self.ks):
            raise ValueError(f"ks must be positive integers, got {self.ks}")
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