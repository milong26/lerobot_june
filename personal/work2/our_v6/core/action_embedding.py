"""Acquired-only action representation for our_v6.

The descriptor preserves V5's information content:
  resampled trajectory + mean/std + delta mean/std + mean |delta| + path length.
Unlike V5, no unacquired episode action is read. Feature groups are normalized
and weighted before a final L2 normalization so temporal scale cannot dominate.
"""

from __future__ import annotations

from typing import Set
import numpy as np

from our_v6.config import (
    ACTION_STEPS,
    ACTION_TEMPORAL_WEIGHT,
    ACTION_STATS_WEIGHT,
    ACTION_PATH_WEIGHT,
    ACTION_FINAL_L2,
    EPS,
)


def _l2(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec if norm < EPS else vec / norm


def _find_action_key(dataset) -> str:
    for key in dataset.meta.features.keys():
        if "action" in key.lower():
            return key
    raise ValueError(f"No action feature found. Available={list(dataset.meta.features.keys())}")


def load_acquired_action_sequence(dataset, ep_idx: int, acquired_indices: Set[int]) -> np.ndarray:
    """Read an action sequence only after the episode has been acquired."""
    if ep_idx not in acquired_indices:
        raise RuntimeError(
            f"CAUSAL VIOLATION: action of episode {ep_idx} requested before acquisition"
        )
    if ep_idx < 0 or ep_idx >= dataset.num_episodes:
        raise IndexError(f"episode {ep_idx} outside [0, {dataset.num_episodes})")

    action_key = _find_action_key(dataset)
    from_idx = int(dataset.meta.episodes["dataset_from_index"][ep_idx])
    to_idx = int(dataset.meta.episodes["dataset_to_index"][ep_idx])
    actions = [np.asarray(dataset[i][action_key], dtype=np.float32) for i in range(from_idx, to_idx)]
    if not actions:
        raise ValueError(f"Episode {ep_idx} has no action frames")
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected action shape [T,D], got {array.shape} for episode {ep_idx}")
    return array


def resample_action_sequence(actions: np.ndarray, num_steps: int = ACTION_STEPS) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError(f"Expected non-empty [T,D] actions, got {actions.shape}")
    t, d = actions.shape
    if t == 1:
        return np.repeat(actions, num_steps, axis=0)
    source_t = np.linspace(0.0, 1.0, t)
    target_t = np.linspace(0.0, 1.0, num_steps)
    result = np.empty((num_steps, d), dtype=np.float32)
    for dim in range(d):
        result[:, dim] = np.interp(target_t, source_t, actions[:, dim])
    return result


def build_action_embedding_for_acquired_episode(
    actions: np.ndarray,
    num_steps: int = ACTION_STEPS,
) -> np.ndarray:
    """Build the normalized V5-content action descriptor from one acquired episode."""
    actions = np.asarray(actions, dtype=np.float32)
    temporal = resample_action_sequence(actions, num_steps=num_steps).reshape(-1)

    mean = actions.mean(axis=0)
    std = actions.std(axis=0)
    if actions.shape[0] > 1:
        delta = np.diff(actions, axis=0)
        delta_mean = delta.mean(axis=0)
        delta_std = delta.std(axis=0)
        mean_abs_delta = np.abs(delta).mean(axis=0)
        path_length = float(np.linalg.norm(delta, axis=1).sum())
    else:
        d = actions.shape[1]
        delta_mean = np.zeros(d, dtype=np.float32)
        delta_std = np.zeros(d, dtype=np.float32)
        mean_abs_delta = np.zeros(d, dtype=np.float32)
        path_length = 0.0

    stats = np.concatenate([mean, std, delta_mean, delta_std, mean_abs_delta]).astype(np.float32)
    path = np.asarray([np.log1p(path_length)], dtype=np.float32)

    combined = np.concatenate([
        ACTION_TEMPORAL_WEIGHT * _l2(temporal),
        ACTION_STATS_WEIGHT * _l2(stats),
        ACTION_PATH_WEIGHT * path,
    ]).astype(np.float32)
    return _l2(combined) if ACTION_FINAL_L2 else combined


def reveal_action_embedding(dataset, ep_idx: int, acquired_indices: Set[int]) -> np.ndarray:
    """Convenience causal interface used by the planner after acquisition commit."""
    actions = load_acquired_action_sequence(dataset, ep_idx, acquired_indices)
    return build_action_embedding_for_acquired_episode(actions)
