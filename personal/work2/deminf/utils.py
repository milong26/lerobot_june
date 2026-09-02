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