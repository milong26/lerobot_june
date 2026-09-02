"""
Action Embedding Module for V4

Implements strictly causal temporal ActionEmbedding.
Action embeddings are computed ONLY AFTER an episode has been officially acquired.
No pre-computation over the full pool is allowed.

Final ActionEmbedding formula:
    ActionEmb = [
        ACTION_TEMPORAL_WEIGHT * L2_norm(temporal_flat),
        ACTION_STATS_WEIGHT  * L2_norm(statistics_vector),
        ACTION_LENGTH_WEIGHT * scaled_length
    ]

Where:
    - temporal_flat: resampled 16-step action trajectory flattened (PRIMARY signal)
    - statistics_vector: action mean, std, velocity mean/std, range, initial/final action
    - scaled_length: np.log1p(T) / ACTION_LENGTH_SCALE (stable bounded scalar)

The temporal part is the PRIMARY signal; statistics and length are auxiliary.
After group-wise weighting, the combined vector may optionally undergo a final
L2-normalization (controlled by ACTION_NORMALIZE), but the group weight ratios
(1.0 : 0.25 : 0.1) are preserved through the normalization.

Two-level weight distinction:
    - ACTION_TEMPORAL_WEIGHT / ACTION_STATS_WEIGHT / ACTION_LENGTH_WEIGHT:
        Control internal composition of the ActionEmbedding vector.
    - ACTION_WEIGHT (in config.py, default 0.5):
        Controls how much the ActionDisagreement signal contributes to the
        final cell priority in the planner. These are independent.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Set, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_v4.config import (
    TEMPORAL_ACTION_STEPS, ACTION_NORMALIZE,
    ACTION_TEMPORAL_WEIGHT, ACTION_STATS_WEIGHT, ACTION_LENGTH_WEIGHT,
    ACTION_LENGTH_SCALE,
)


def _detect_action_key(dataset) -> str:
    """Auto-detect the action feature key from dataset metadata."""
    features = dataset.meta.features
    for key in features.keys():
        if "action" in key.lower():
            return key
    raise ValueError(
        f"No action feature found in dataset. Available features: {list(features.keys())}"
    )


def _l2_normalize_group(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D feature group vector. Returns zero vector if norm is near-zero."""
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


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


def extract_temporal_features(
    action_sequence: np.ndarray,
    n_steps: int = TEMPORAL_ACTION_STEPS,
) -> np.ndarray:
    """
    Extract the PRIMARY temporal feature group: resampled action trajectory flattened.

    This is the main signal in the ActionEmbedding, preserving the full temporal
    structure of the action sequence as a fixed-length vector of shape [n_steps * action_dim].

    Args:
        action_sequence: numpy array of shape [T, action_dim]
        n_steps: number of temporal steps for resampling

    Returns:
        1-D numpy array of shape [n_steps * action_dim]
    """
    if action_sequence.ndim != 2:
        raise ValueError(f"Expected 2D action sequence [T, D], got {action_sequence.shape}")

    resampled = resample_action_sequence(action_sequence, n_steps=n_steps)
    return resampled.flatten()


def extract_statistical_features(
    action_sequence: np.ndarray,
) -> np.ndarray:
    """
    Extract the auxiliary statistical feature group.

    Includes: action mean, std, velocity mean, velocity std, action range,
    initial action, final action (per dimension).

    These are secondary signals that complement the temporal features.

    Args:
        action_sequence: numpy array of shape [T, action_dim]

    Returns:
        1-D numpy array of statistical features
    """
    if action_sequence.ndim != 2:
        raise ValueError(f"Expected 2D action sequence [T, D], got {action_sequence.shape}")

    T, D = action_sequence.shape
    parts = []

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

    parts.append(action_sequence[0])
    parts.append(action_sequence[-1])

    return np.concatenate(parts)


def extract_length_feature(
    action_sequence: np.ndarray,
    length_scale: float = ACTION_LENGTH_SCALE,
) -> np.ndarray:
    """
    Extract the trajectory length feature with stable scaling.

    Uses np.log1p(T) / length_scale to avoid raw T dominating the embedding.
    This ensures that episodes with T in [50, 300] produce comparable length
    signals without overwhelming the temporal trajectory features.

    Args:
        action_sequence: numpy array of shape [T, action_dim]
        length_scale: scaling divisor for log-transformed length

    Returns:
        1-D numpy array of shape [1]
    """
    T = action_sequence.shape[0]
    return np.array([np.log1p(float(T)) / length_scale])


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
    temporal_weight: float = ACTION_TEMPORAL_WEIGHT,
    stats_weight: float = ACTION_STATS_WEIGHT,
    length_weight: float = ACTION_LENGTH_WEIGHT,
    length_scale: float = ACTION_LENGTH_SCALE,
) -> np.ndarray:
    """
    Build a complete action embedding for an acquired episode with group-wise weighting.

    ActionEmbedding composition:
        ActionEmb = [
            temporal_weight  * L2_norm(temporal_features),
            stats_weight     * L2_norm(statistical_features),
            length_weight    * scaled_length_feature
        ]

    The temporal part is the PRIMARY signal (weight=1.0).
    Statistics and length are auxiliary (weights=0.25 and 0.1).

    After concatenation, an optional final L2-normalization is applied.
    Even after this final normalization, the group weight ratios are preserved
    because L2-normalization is scale-invariant with respect to relative proportions.

    Args:
        action_sequence: numpy array of shape [T, action_dim]
        normalize: whether to apply final L2-normalization to the combined vector
        n_steps: temporal resampling steps
        temporal_weight: weight for the temporal feature group
        stats_weight: weight for the statistical feature group
        length_weight: weight for the length feature
        length_scale: scaling factor for log-transformed length

    Returns:
        1-D numpy array (the action embedding)
    """
    if action_sequence.ndim != 2:
        raise ValueError(f"Expected 2D action sequence [T, D], got {action_sequence.shape}")

    # Group 1: Temporal features (PRIMARY signal)
    temporal_flat = extract_temporal_features(action_sequence, n_steps=n_steps)
    temporal_norm = _l2_normalize_group(temporal_flat) * temporal_weight

    # Group 2: Statistical features (auxiliary)
    stats_vector = extract_statistical_features(action_sequence)
    stats_norm = _l2_normalize_group(stats_vector) * stats_weight

    # Group 3: Length feature (auxiliary, stable-scaled)
    length_feat = extract_length_feature(action_sequence, length_scale=length_scale)
    length_scaled = length_feat * length_weight

    # Concatenate all groups
    combined = np.concatenate([temporal_norm, stats_norm, length_scaled])

    # Optional final L2-normalization
    if normalize:
        combined = normalize_action_embedding(combined)

    return combined