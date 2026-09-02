"""
Visual Embedding Module for V3 No-Action

Loads and manages visual embeddings with strict causal access control:
embeddings can only be loaded for episodes that have been officially acquired.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Set, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_v3_no_action.config import PCA_DIM, VISUAL_NORMALIZE, VISUAL_GLOBAL_WEIGHT, VISUAL_WRIST_WEIGHT


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2 normalize a 1-D vector."""
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def validate_embedding_shape(phi_global: np.ndarray, phi_wrist: np.ndarray) -> None:
    """
    Validate that embedding arrays are well-formed 1-D finite numeric arrays.
    Checks dimensions against the configured PCA_DIM if available.
    """
    for name, arr in [("phi_global", phi_global), ("phi_wrist", phi_wrist)]:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(arr)}")
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
        if len(arr) == 0:
            raise ValueError(f"{name} is empty")

    if PCA_DIM is not None:
        for name, arr in [("phi_global", phi_global), ("phi_wrist", phi_wrist)]:
            if len(arr) != PCA_DIM:
                print(f"  [WARNING] {name} dimension {len(arr)} does not match "
                      f"configured PCA_DIM={PCA_DIM}. Using actual dimension.")


def load_acquired_visual_embedding(
    ep_idx: int,
    embedding_dir: Path,
    acquired_indices: Set[int],
) -> Dict[str, np.ndarray]:
    """
    Load visual embedding for an episode ONLY if it has been acquired.

    This is a safety interface: if ep_idx is not in acquired_indices,
    an exception is raised to prevent the algorithm from peeking at
    uncollected episode data.

    Args:
        ep_idx: episode index to load
        embedding_dir: directory containing .npy embedding cache files
        acquired_indices: set of officially acquired episode indices

    Returns:
        {"phi_global": np.ndarray, "phi_wrist": np.ndarray}

    Raises:
        RuntimeError: if ep_idx has not been acquired yet
        FileNotFoundError: if embedding file does not exist
    """
    if ep_idx not in acquired_indices:
        raise RuntimeError(
            f"CAUSAL VIOLATION: Attempted to load embedding for episode {ep_idx} "
            f"which has NOT been acquired yet. acquired_indices={sorted(acquired_indices)}"
        )

    coord_key = f"({ep_idx})"
    embedding_file = embedding_dir / f"{coord_key}.npy"

    if not embedding_file.exists():
        raise FileNotFoundError(
            f"Embedding file not found for episode {ep_idx}: {embedding_file}"
        )

    data = np.load(embedding_file, allow_pickle=True).item()
    phi_global = data["phi_global"]
    phi_wrist = data["phi_wrist"]

    validate_embedding_shape(phi_global, phi_wrist)

    if VISUAL_NORMALIZE:
        phi_global = _l2_normalize(phi_global)
        phi_wrist = _l2_normalize(phi_wrist)

    return {"phi_global": phi_global, "phi_wrist": phi_wrist}


def build_weighted_visual_embedding(
    phi_global: np.ndarray,
    phi_wrist: np.ndarray,
    global_weight: float = VISUAL_GLOBAL_WEIGHT,
    wrist_weight: float = VISUAL_WRIST_WEIGHT,
) -> np.ndarray:
    """
    Build a weighted combined visual embedding from global and wrist components.

    Processing:
    1. L2-normalize phi_global and phi_wrist independently
    2. Scale each by its respective weight
    3. Concatenate
    4. Optionally L2-normalize the combined result

    Args:
        phi_global: global visual embedding vector
        phi_wrist: wrist visual embedding vector
        global_weight: weight for the global component
        wrist_weight: weight for the wrist component

    Returns:
        Weighted combined embedding vector
    """
    phi_global_norm = _l2_normalize(phi_global)
    phi_wrist_norm = _l2_normalize(phi_wrist)

    weighted_global = phi_global_norm * global_weight
    weighted_wrist = phi_wrist_norm * wrist_weight

    combined = np.concatenate([weighted_global, weighted_wrist], axis=0)

    if VISUAL_NORMALIZE:
        combined = _l2_normalize(combined)

    return combined


def build_combined_embedding(
    phi_global: np.ndarray,
    phi_wrist: np.ndarray,
) -> np.ndarray:
    """Concatenate global and wrist embeddings, then L2 normalize."""
    combined = np.concatenate([phi_global, phi_wrist], axis=0)
    if VISUAL_NORMALIZE:
        combined = _l2_normalize(combined)
    return combined