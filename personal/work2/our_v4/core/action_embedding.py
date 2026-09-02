"""
Action Embedding Module for V4

Implements strictly causal temporal ActionEmbedding.
Action embeddings are computed ONLY AFTER an episode has been officially acquired.
No pre-computation over the full pool is allowed.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Set, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_v4.config import TEMPORAL_ACTION_STEPS, ACTION_NORMALIZE


def _detect_action_key(dataset) -> str:
    """Auto-detect the action feature key from dataset metadata."""
    features = dataset.meta.features
    for key in features.keys():
        if "action" in key.lower():
            return key
    raise ValueError(
        f"No action feature found in dataset. Available features: {list(features.keys())}"
    )


def load_acquired_action_sequence(
    dataset_root: str,
    ep_idx: int,
    acquired_indices: Set[int],
) -> np.ndarray:
    """
    Load the full action sequence for an episode ONLY if it has been officially acquired.

    This is a strict causal guard: if ep_idx is not in acquired_indices,
    a RuntimeError is raised to prevent future information leakage.

    Args:
        dataset_root: root directory of the LeRobot dataset
        ep_idx: episode index to load
        acquired_indices: set of officially acquired episode indices

    Returns:
        numpy array of shape [T, action_dim]

    Raises:
        RuntimeError: if ep_idx has not been acquired yet (CAUSAL VIOLATION)
    """
    if ep_idx not in acquired_indices:
        raise RuntimeError(
            f"CAUSAL VIOLATION: Attempted to load action sequence for episode {ep_idx} "
            f"which has NOT been acquired yet. acquired_indices={sorted(acquired_indices)}"
        )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id="1/2", root=dataset_root)
    action_key = _detect_action_key(dataset)

    if ep_idx >= dataset.num_episodes:
        raise ValueError(
            f"Episode {ep_idx} out of range (dataset has {dataset.num_episodes} episodes)"
        )

    from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
    to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]

    episode_slice = dataset.hf_dataset[from_idx:to_idx]
    action_array = np.array(episode_slice[action_key])

    if action_array.ndim != 2:
        raise ValueError(
            f"Expected 2D action array [T, D], got shape {action_array.shape}"
        )

    return action_array


def resample_action_sequence(
    action_sequence: np.ndarray,
    n_steps: int = TEMPORAL_ACTION_STEPS,
) -> np.ndarray:
    """
    Resample an action trajectory of arbitrary length T to a fixed length n_steps
    using linear interpolation along the normalized time axis.

    Args:
        action_sequence: numpy array of shape [T, action_dim]
        n_steps: target number of temporal steps (default: 16)

    Returns:
        numpy array of shape [n_steps, action_dim]
    """
    T, D = action_sequence.shape

    if T == 1:
        return np.tile(action_sequence, (n_steps, 1))

    if T == n_steps:
        return action_sequence.copy()

    original_time = np.linspace(0, 1, T)
    target_time = np.linspace(0, 1, n_steps)

    resampled = np.zeros((n_steps, D), dtype=action_sequence.dtype)
    for d in range(D):
        resampled[:, d] = np.interp(target_time, original_time, action_sequence[:, d])

    return resampled


def extract_temporal_action_embedding(
    action_sequence: np.ndarray,
    n_steps: int = TEMPORAL_ACTION_STEPS,
) -> np.ndarray:
    """
    Extract a fixed-length action embedding that preserves temporal structure.

    Core features:
    - Resampled temporal action sequence [n_steps, action_dim] flattened -> main signal
    - Auxiliary statistics: mean, std, velocity mean, velocity std, range,
      trajectory length, initial action, final action

    Args:
        action_sequence: numpy array of shape [T, action_dim]
        n_steps: number of temporal steps for resampling

    Returns:
        1-D numpy array combining temporal representation + statistics
    """
    if action_sequence.ndim != 2:
        raise ValueError(f"Expected 2D action sequence [T, D], got {action_sequence.shape}")

    T, D = action_sequence.shape
    parts = []

    # Primary signal: resampled temporal action (flattened)
    resampled = resample_action_sequence(action_sequence, n_steps=n_steps)
    parts.append(resampled.flatten())

    # Auxiliary statistics
    action_mean = np.mean(action_sequence, axis=0)
    parts.append(action_mean)

    action_std = np.std(action_sequence, axis=0)
    parts.append(action_std)

    if T > 1:
        velocity = np.diff(action_sequence, axis=0)
        vel_mean = np.mean(velocity, axis=0)
        vel_std = np.std(velocity, axis=0)
        parts.append(vel_mean)
        parts.append(vel_std)
    else:
        parts.append(np.zeros(D))
        parts.append(np.zeros(D))

    action_range = np.max(action_sequence, axis=0) - np.min(action_sequence, axis=0)
    parts.append(action_range)

    parts.append(np.array([float(T)]))

    parts.append(action_sequence[0])
    parts.append(action_sequence[-1])

    return np.concatenate(parts)


def normalize_action_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    L2-normalize an action embedding vector.

    Args:
        embedding: 1-D numpy array

    Returns:
        L2-normalized embedding
    """
    norm = np.linalg.norm(embedding)
    if norm < 1e-8:
        return embedding
    return embedding / norm


def build_action_embedding_for_acquired_episode(
    action_sequence: np.ndarray,
    normalize: bool = ACTION_NORMALIZE,
    n_steps: int = TEMPORAL_ACTION_STEPS,
) -> np.ndarray:
    """
    Build a complete action embedding for an acquired episode.

    Steps:
    1. Extract temporal + statistics embedding
    2. Optionally L2-normalize

    Args:
        action_sequence: numpy array of shape [T, action_dim]
        normalize: whether to L2-normalize the result
        n_steps: temporal resampling steps

    Returns:
        1-D numpy array (the action embedding)
    """
    emb = extract_temporal_action_embedding(action_sequence, n_steps=n_steps)
    if normalize:
        emb = normalize_action_embedding(emb)
    return emb