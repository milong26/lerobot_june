"""
DemInf Utilities

Seed management, I/O helpers, device selection, logging.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set random seed for Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    """
    Resolve device string to torch.device.

    Args:
        device_str: 'cuda', 'cpu', 'cuda:0', etc.

    Returns:
        torch.device object.
    """
    if "cuda" in device_str and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure directory exists, create if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """
    Save data as JSON.

    Args:
        data: JSON-serializable data.
        path: Output file path.
        indent: JSON indentation level.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_json(path: str | Path) -> Any:
    """
    Load data from JSON file.

    Args:
        path: Input file path.

    Returns:
        Parsed JSON data.
    """
    with open(path, "r") as f:
        return json.load(f)


def atomic_save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """
    Atomically save JSON by writing to temp file then renaming.

    Args:
        data: JSON-serializable data.
        path: Output file path.
        indent: JSON indentation level.
    """
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, str(path))
    except Exception:
        os.unlink(tmp_path)
        raise


def save_metadata(output_dir: str | Path, config: Any) -> Dict[str, Any]:
    """
    Save run metadata including git commit hash.

    Args:
        output_dir: Output directory.
        config: DemInfConfig instance.

    Returns:
        Metadata dictionary.
    """
    metadata = {
        "config": {
            "dataset_path": config.dataset_path,
            "target_episodes": config.target_episodes,
            "seed": config.seed,
            "state_latent_dim": config.state_latent_dim,
            "action_latent_dim": config.action_latent_dim,
            "hidden_dims": config.hidden_dims,
            "vae_beta_state": config.vae_beta_state,
            "vae_beta_action": config.vae_beta_action,
            "vae_lr": config.vae_lr,
            "vae_epochs": config.vae_epochs,
            "ks": config.ks,
            "ksg_mode": config.ksg_mode,
            "ksg_backend": config.ksg_backend,
            "representation": config.representation,
        },
    }

    # Try to get git commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(Path(__file__).parent.parent.parent.parent)
        )
        if result.returncode == 0:
            metadata["git_commit"] = result.stdout.strip()
    except Exception:
        pass  # Skip if git is not available

    metadata_path = Path(output_dir) / "deminf_metadata.json"
    atomic_save_json(metadata, metadata_path)
    return metadata


def get_git_commit(repo_root: Optional[str | Path] = None) -> str:
    """
    Get the current git commit hash using local subprocess.

    This is used for latent cache fingerprinting to ensure cache validity
    is tied to the exact code version. No network access is required.

    Args:
        repo_root: Root directory of the git repository. If None, uses
                   the directory containing this utils.py file's parent
                   (i.e., the project root).

    Returns:
        40-char git commit hash string, or 'unknown' if git is unavailable.
    """
    if repo_root is None:
        repo_root = Path(__file__).parent.parent.parent.parent

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(repo_root),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    logging.getLogger("deminf").warning(
        "git rev-parse HEAD failed or not available; using 'unknown' for cache fingerprint"
    )
    return "unknown"


def build_cache_fingerprint(
    dataset_path: str,
    state_source: str,
    action_key: str,
    state_dim: int,
    action_dim: int,
    state_latent_dim: int,
    action_latent_dim: int,
    hidden_dims: list,
    vae_beta_state: float,
    vae_beta_action: float,
    vae_lr: float,
    vae_steps: int,
    weight_decay: float,
    normalization_manifest: dict,
    state_ckpt_path: str,
    action_ckpt_path: str,
    git_commit: str,
    total_episodes: int,
    total_frames: int,
) -> str:
    """
    Build a comprehensive fingerprint for latent cache validation.

    The fingerprint includes all fields that affect latent embeddings:
    dataset identity, normalization, VAE architecture/hyperparameters,
    final checkpoint SHA256 hashes, and git commit.

    IMPORTANT: This must be called AFTER final VAE checkpoints are determined,
    because the checkpoint hashes are part of the fingerprint.

    Args:
        dataset_path: Absolute path to the dataset.
        state_source: State observation key.
        action_key: Action field key.
        state_dim: State input dimensionality.
        action_dim: Action input dimensionality.
        state_latent_dim: State VAE latent dimension.
        action_latent_dim: Action VAE latent dimension.
        hidden_dims: VAE hidden layer sizes.
        vae_beta_state: State VAE beta.
        vae_beta_action: Action VAE beta.
        vae_lr: VAE learning rate.
        vae_steps: VAE training steps.
        weight_decay: Adam weight decay.
        normalization_manifest: Normalization stats hash manifest.
        state_ckpt_path: Final state VAE checkpoint file path.
        action_ckpt_path: Final action VAE checkpoint file path.
        git_commit: Current git commit hash.
        total_episodes: Total episodes in dataset.
        total_frames: Total frames in dataset.

    Returns:
        32-char hex SHA256 fingerprint string.
    """
    import hashlib as _hashlib

    def _file_sha256(path: str) -> str:
        if path and Path(path).exists():
            with open(path, "rb") as f:
                return _hashlib.sha256(f.read()).hexdigest()[:16]
        return ""

    info_path = Path(dataset_path) / "meta" / "info.json"
    info_hash = ""
    if info_path.exists():
        with open(info_path, "rb") as f:
            info_hash = _hashlib.sha256(f.read()).hexdigest()[:16]

    state_ckpt_hash = _file_sha256(state_ckpt_path)
    action_ckpt_hash = _file_sha256(action_ckpt_path)

    key_fields = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "info_hash": info_hash,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "state_source": state_source,
        "action_key": action_key,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "state_latent_dim": state_latent_dim,
        "action_latent_dim": action_latent_dim,
        "hidden_dims": hidden_dims,
        "vae_beta_state": vae_beta_state,
        "vae_beta_action": vae_beta_action,
        "vae_lr": vae_lr,
        "vae_steps": vae_steps,
        "weight_decay": weight_decay,
        "normalization_manifest": normalization_manifest,
        "state_ckpt_hash": state_ckpt_hash,
        "action_ckpt_hash": action_ckpt_hash,
        "git_commit": git_commit,
    }
    raw = json.dumps(key_fields, sort_keys=True)
    return _hashlib.sha256(raw.encode()).hexdigest()[:32]


def init_logger(output_dir: str | Path, name: str = "deminf") -> logging.Logger:
    """
    Initialize logger that writes to both console and file.

    Args:
        output_dir: Directory for log file.
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    ensure_dir(output_dir)
    fh = logging.FileHandler(Path(output_dir) / "deminf.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def validate_latent_cache_metadata(
    metadata: dict,
    current_config: Any,
) -> Tuple[bool, str]:
    """
    Validate that latent cache metadata matches the current experiment.

    Checks:
    - state latent dimension must equal current_config.state_latent_dim
    - action latent dimension must equal current_config.action_latent_dim
    - state source must be observation.environment_state
    - representation_type must be state_action
    - dataset episode count matches
    - transition count matches
    - latent shape matches
    - normalization fingerprint matches
    - checkpoint fingerprint matches

    Args:
        metadata: Loaded latents_manifest.json dictionary.
        current_config: Current DemInfConfig instance.

    Returns:
        Tuple of (is_valid, error_reason_string). If valid, error_reason is empty.
    """
    if metadata.get("state_latent_dim") != current_config.state_latent_dim:
        return False, (
            f"state_latent_dim mismatch: cache={metadata.get('state_latent_dim')}, "
            f"expected={current_config.state_latent_dim}"
        )
    if metadata.get("action_latent_dim") != current_config.action_latent_dim:
        return False, (
            f"action_latent_dim mismatch: cache={metadata.get('action_latent_dim')}, "
            f"expected={current_config.action_latent_dim}"
        )
    if metadata.get("state_source") != current_config.state_source:
        return False, (
            f"state_source mismatch: cache={metadata.get('state_source')}, "
            f"expected={current_config.state_source}"
        )
    if current_config.representation_type != "state_action":
        return False, (
            f"representation_type mismatch: current={current_config.representation_type}, "
            f"expected=state_action"
        )

    cache_fingerprint = metadata.get("fingerprint", "")
    cached_state_dim = metadata.get("state_dim", 39)
    cached_action_dim = metadata.get("action_dim", 4)
    current_fingerprint = build_cache_fingerprint(
        dataset_path=current_config.dataset_path,
        state_source=current_config.state_source,
        action_key=current_config.action_key,
        state_dim=cached_state_dim,
        action_dim=cached_action_dim,
        state_latent_dim=current_config.state_latent_dim,
        action_latent_dim=current_config.action_latent_dim,
        hidden_dims=current_config.hidden_dims,
        vae_beta_state=current_config.vae_beta_state,
        vae_beta_action=current_config.vae_beta_action,
        vae_lr=current_config.vae_lr,
        vae_steps=current_config.vae_steps,
        weight_decay=current_config.weight_decay,
        normalization_manifest=metadata.get("normalization_manifest", {}),
        state_ckpt_path=metadata.get("state_ckpt_path", ""),
        action_ckpt_path=metadata.get("action_ckpt_path", ""),
        git_commit=get_git_commit(),
        total_episodes=metadata.get("total_episodes", 0),
        total_frames=metadata.get("total_frames", 0),
    )

    if cache_fingerprint != current_fingerprint:
        return False, (
            f"fingerprint mismatch: cache='{cache_fingerprint}', "
            f"current='{current_fingerprint}'"
        )

    return True, ""