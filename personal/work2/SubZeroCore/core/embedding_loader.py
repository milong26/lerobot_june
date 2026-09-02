"""
Embedding Loader for SubZeroCore

Loads and validates visual embeddings from existing episode cache files.
Reuses the same embedding file naming convention as our_v4: ({episode_index}).npy
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, List

from SubZeroCore.config import GLOBAL_WEIGHT, WRIST_WEIGHT, EPS


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2 normalize a 1-D numpy vector. Returns original vector if norm is too small."""
    norm = np.linalg.norm(vec)
    if norm < EPS:
        return vec
    return vec / norm


def validate_embedding(phi_global: np.ndarray, phi_wrist: np.ndarray) -> None:
    """Check that global and wrist embeddings are valid 1-D numpy arrays."""
    for name, arr in [("phi_global", phi_global), ("phi_wrist", phi_wrist)]:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(arr)}")
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
        if len(arr) == 0:
            raise ValueError(f"{name} is empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values (NaN or Inf)")


def build_visual_embedding(
    phi_global: np.ndarray,
    phi_wrist: np.ndarray,
    global_weight: float = GLOBAL_WEIGHT,
    wrist_weight: float = WRIST_WEIGHT,
) -> np.ndarray:
    """
    Build a single episode visual embedding:
    1. L2 normalize phi_global and phi_wrist independently
    2. Multiply by respective weights
    3. Concatenate
    4. L2 normalize the combined result
    """
    phi_global_norm = _l2_normalize(phi_global)
    phi_wrist_norm = _l2_normalize(phi_wrist)

    weighted_global = phi_global_norm * global_weight
    weighted_wrist = phi_wrist_norm * wrist_weight

    combined = np.concatenate([weighted_global, weighted_wrist], axis=0)
    combined = _l2_normalize(combined)

    return combined


def find_embedding_file(embedding_dir: str, episode_index: int) -> Path:
    """
    Locate the embedding file for a given episode index.
    Uses the same naming convention as our_v4: ({episode_index}).npy
    """
    embedding_path = Path(embedding_dir) / f"({episode_index}).npy"
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Embedding file not found for episode {episode_index}: {embedding_path}"
        )
    return embedding_path


def load_episode_visual_embedding(
    embedding_dir: str,
    episode_index: int,
    global_weight: float = GLOBAL_WEIGHT,
    wrist_weight: float = WRIST_WEIGHT,
) -> np.ndarray:
    """
    Load phi_global and phi_wrist for a single episode, validate and build
    the final combined visual embedding.
    """
    embedding_file = find_embedding_file(embedding_dir, episode_index)

    data = np.load(embedding_file, allow_pickle=True).item()
    phi_global = data["phi_global"]
    phi_wrist = data["phi_wrist"]

    validate_embedding(phi_global, phi_wrist)

    return build_visual_embedding(phi_global, phi_wrist, global_weight, wrist_weight)


def load_all_visual_embeddings(
    dataset_root: str,
    embedding_dir: str,
    global_weight: float = GLOBAL_WEIGHT,
    wrist_weight: float = WRIST_WEIGHT,
) -> Tuple[np.ndarray, List[int]]:
    """
    Load visual embeddings for all valid episodes in the dataset.

    Reads episode_initial_states.json from dataset_root to get all valid episode indices,
    then loads each episode's embedding from embedding_dir.

    Returns:
        embedding_matrix: shape [N, D], where N is the number of valid episodes
        episode_indices: list of episode indices, episode_indices[i] corresponds to row i
    """
    metadata_path = Path(dataset_root) / "episode_initial_states.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    episodes = metadata["episodes"]
    episode_indices = [ep["episode_index"] for ep in episodes]

    embeddings = []
    for ep_idx in episode_indices:
        emb = load_episode_visual_embedding(
            embedding_dir, ep_idx, global_weight, wrist_weight
        )
        embeddings.append(emb)

    embedding_matrix = np.stack(embeddings, axis=0)

    return embedding_matrix, episode_indices