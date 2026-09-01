"""
Action Trajectory Embedding Extraction Module

Extracts fixed-length action embeddings from LeRobot demonstration dataset
action trajectories. Does NOT depend on trained policy internals (hidden, x_t, v_t).
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _detect_action_key(dataset) -> str:
    """
    Auto-detect the action feature key from dataset metadata.

    Args:
        dataset: LeRobotDataset instance.

    Returns:
        Action feature key string.
    """
    features = dataset.meta.features
    for key in features.keys():
        if "action" in key.lower():
            return key

    raise ValueError(
        f"No action feature found in dataset. Available features: {list(features.keys())}"
    )


def load_episode_actions(
    dataset_root: str,
    episode_indices: Optional[List[int]] = None,
) -> Dict[int, np.ndarray]:
    """
    Load action sequences from LeRobot dataset for specified episodes.

    Args:
        dataset_root: root directory of the LeRobot dataset.
        episode_indices: optional list of episode indices. If None, load all.

    Returns:
        Dict[episode_index, action_sequence]
        where action_sequence shape is [T, action_dim]
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"  Loading dataset from: {dataset_root}")
    dataset = LeRobotDataset(repo_id="1/2", root=dataset_root)

    action_key = _detect_action_key(dataset)
    print(f"  Detected action key: '{action_key}'")

    num_episodes = dataset.num_episodes
    if episode_indices is None:
        episode_indices = list(range(num_episodes))

    episode_actions = {}
    missing_count = 0

    for ep_idx in episode_indices:
        if ep_idx >= num_episodes:
            missing_count += 1
            print(f"  [WARN] Episode {ep_idx} out of range (max: {num_episodes - 1})")
            continue

        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]

        action_frames = []
        valid = True
        for idx in range(from_idx, to_idx):
            try:
                frame = dataset[idx]
                action_frames.append(frame[action_key])
            except Exception as e:
                print(f"  [WARN] Failed to load action for episode {ep_idx}, frame {idx}: {e}")
                valid = False
                break

        if valid and action_frames:
            action_array = np.array(action_frames)
            episode_actions[ep_idx] = action_array
        else:
            missing_count += 1

    print(f"Loaded action sequences for {len(episode_actions)} episodes (missing: {missing_count})")
    return episode_actions


def extract_action_embedding(action_sequence: np.ndarray) -> np.ndarray:
    """
    Convert a single episode's action trajectory to a fixed-length embedding.

    Uses only demonstration data statistics, no model internals.

    Features extracted:
    - action mean (per dimension)
    - action std (per dimension)
    - action velocity mean (per dimension)
    - action velocity std (per dimension)
    - trajectory length (scalar)
    - action range (max - min, per dimension)
    - initial action (per dimension)
    - final action (per dimension)

    Args:
        action_sequence: numpy array of shape [T, action_dim]

    Returns:
        1-D numpy array representing the action embedding.
    """
    if action_sequence.ndim != 2:
        raise ValueError(f"Expected 2D action sequence [T, D], got shape {action_sequence.shape}")

    T, D = action_sequence.shape
    parts = []

    # 1. Action mean per dimension
    action_mean = np.mean(action_sequence, axis=0)
    parts.append(action_mean)

    # 2. Action std per dimension
    action_std = np.std(action_sequence, axis=0)
    parts.append(action_std)

    # 3. Action velocity (first-order difference)
    if T > 1:
        velocity = np.diff(action_sequence, axis=0)
        vel_mean = np.mean(velocity, axis=0)
        vel_std = np.std(velocity, axis=0)
        parts.append(vel_mean)
        parts.append(vel_std)
    else:
        parts.append(np.zeros(D))
        parts.append(np.zeros(D))

    # 4. Trajectory length (normalized)
    parts.append(np.array([float(T)]))

    # 5. Action range (max - min) per dimension
    action_range = np.max(action_sequence, axis=0) - np.min(action_sequence, axis=0)
    parts.append(action_range)

    # 6. Initial and final action
    parts.append(action_sequence[0])
    parts.append(action_sequence[-1])

    return np.concatenate(parts)


def build_action_embeddings(
    dataset_root: str,
    episode_indices: Optional[List[int]] = None,
) -> Dict[int, np.ndarray]:
    """
    Batch-generate action embeddings from LeRobot demonstration dataset.

    Args:
        dataset_root: root directory of the LeRobot dataset.
        episode_indices: optional list of episode indices. If None, load all.

    Returns:
        Dict[episode_index, action_embedding_array]
    """
    episode_actions = load_episode_actions(dataset_root, episode_indices)

    action_embeddings = {}
    missing_count = 0
    action_dim = None

    for ep_idx, action_seq in episode_actions.items():
        try:
            emb = extract_action_embedding(action_seq)
            action_embeddings[ep_idx] = emb
            if action_dim is None:
                action_dim = emb.shape[0]
        except Exception as e:
            missing_count += 1
            print(f"  [ERROR] Failed to extract embedding for episode {ep_idx}: {e}")

    print(f"Loaded action embeddings for {len(action_embeddings)} episodes")
    print(f"Missing action episodes: {missing_count}")
    if action_dim is not None:
        print(f"Action embedding dimension: {action_dim}")

    return action_embeddings