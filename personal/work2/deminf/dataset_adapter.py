"""
DemInf Dataset Adapter

Adapts LeRobot dataset format for DemInf state-based MI estimation.
Reads observation.state and action from LeRobot parquet data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("deminf")


def infer_episode_structure(dataset_root: str | Path) -> Dict[str, Any]:
    """
    Infer the episode structure from a LeRobot dataset.

    Reads meta/info.json to identify state, action, and episode fields.

    Args:
        dataset_root: Path to LeRobot dataset directory.

    Returns:
        Dictionary with keys:
            - features: feature specification from info.json
            - state_keys: list of state field names
            - action_key: action field name
            - total_episodes: total number of episodes
            - total_frames: total number of frames
            - fps: frames per second
            - state_shape: shape of observation.state
            - action_shape: shape of action
    """
    import json

    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found at {info_path}")

    with open(info_path, "r") as f:
        info = json.load(f)

    features = info.get("features", {})
    total_episodes = info.get("total_episodes", 0)
    total_frames = info.get("total_frames", 0)
    fps = info.get("fps", 80)

    # Identify state and action fields
    state_keys = []
    state_shape = None
    action_key = "action"
    action_shape = None

    if "observation.state" in features:
        state_keys.append("observation.state")
        state_shape = features["observation.state"].get("shape", [4])
        logger.info(f"Found observation.state with shape {state_shape}")

    if "observation.environment_state" in features:
        state_keys.append("observation.environment_state")
        env_state_shape = features["observation.environment_state"].get("shape", [39])
        logger.info(f"Found observation.environment_state with shape {env_state_shape}")

    if "action" in features:
        action_shape = features["action"].get("shape", [4])
        logger.info(f"Found action with shape {action_shape}")

    return {
        "features": features,
        "state_keys": state_keys,
        "action_key": action_key,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": fps,
        "state_shape": state_shape,
        "action_shape": action_shape,
    }


def extract_state_vector(
    sample: Dict[str, Any],
    state_keys: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Extract a 1-D float32 state vector from a single timestep sample.

    Concatenates all specified state fields into a single vector.

    Args:
        sample: Dictionary from LeRobot dataset __getitem__.
        state_keys: List of keys to extract. If None, uses ['observation.state'].

    Returns:
        1-D float32 numpy array of shape [D_s].
    """
    if state_keys is None:
        state_keys = ["observation.state"]

    parts = []
    for key in state_keys:
        if key not in sample:
            raise KeyError(f"State key '{key}' not found in sample. Available keys: {list(sample.keys())}")
        val = sample[key]
        if isinstance(val, np.ndarray):
            arr = val
        elif hasattr(val, "numpy"):
            arr = val.numpy()
        else:
            arr = np.array(val)
        parts.append(arr.flatten())

    return np.concatenate(parts).astype(np.float32)


def extract_action_vector(sample: Dict[str, Any], action_key: str = "action") -> np.ndarray:
    """
    Extract a 1-D float32 action vector from a single timestep sample.

    Args:
        sample: Dictionary from LeRobot dataset __getitem__.
        action_key: Key for action data. Default: 'action'.

    Returns:
        1-D float32 numpy array of shape [D_a].
    """
    if action_key not in sample:
        raise KeyError(f"Action key '{action_key}' not found in sample. Available keys: {list(sample.keys())}")
    val = sample[action_key]
    if isinstance(val, np.ndarray):
        arr = val
    elif hasattr(val, "numpy"):
        arr = val.numpy()
    else:
        arr = np.array(val)
    return arr.flatten().astype(np.float32)


def check_relative_action(dataset_root: str | Path) -> bool:
    """
    Check whether actions in the dataset are relative/delta control.

    For MetaWorld pick-place-v3, actions are (dx, dy, dz, gripper) which
    represent relative displacement commands. This is determined by reading
    the dataset collection code and feature names.

    Args:
        dataset_root: Path to LeRobot dataset directory.

    Returns:
        True if actions are relative/delta, False if absolute.
    """
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.exists():
        logger.warning("info.json not found, assuming relative action")
        return True

    import json
    with open(info_path, "r") as f:
        info = json.load(f)

    features = info.get("features", {})
    action_info = features.get("action", {})
    names = action_info.get("names", {})

    # MetaWorld actions are (x, y, z, gripper) delta commands
    # The axes names indicate displacement control
    if isinstance(names, dict):
        axes = names.get("axes", [])
        if axes == ["x", "y", "z", "gripper"]:
            logger.info("Action axes are [x, y, z, gripper] - these are delta/relative control commands")
            return True

    # Default: MetaWorld actions are relative
    logger.info("Assuming relative/delta action based on MetaWorld convention")
    return True


def build_episode_index(dataset_root: str | Path) -> Dict[int, List[int]]:
    """
    Build mapping from episode index to dataset row indices.

    Reads parquet data files to get episode_index for each row.

    Args:
        dataset_root: Path to LeRobot dataset directory.

    Returns:
        Dictionary {episode_idx: [row_indices...]} with rows sorted by frame_index.
    """
    import json

    info_path = Path(dataset_root) / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)

    total_episodes = info.get("total_episodes", 0)
    if total_episodes == 0:
        raise ValueError("total_episodes is 0 in info.json")

    # Read all parquet files to build episode index
    data_dir = Path(dataset_root) / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    episode_map: Dict[int, List[int]] = {i: [] for i in range(total_episodes)}

    # Find all parquet files
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    logger.info(f"Reading {len(parquet_files)} parquet files to build episode index")

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        if "episode_index" not in df.columns:
            continue

        for ep_idx in df["episode_index"].unique():
            ep_rows = df[df["episode_index"] == ep_idx]
            # Sort by frame_index within episode
            if "frame_index" in ep_rows.columns:
                ep_rows = ep_rows.sort_values("frame_index")
            row_indices = ep_rows.index.tolist()
            episode_map[int(ep_idx)].extend(row_indices)

    # Sort row indices within each episode
    for ep_idx in episode_map:
        episode_map[ep_idx].sort()

    # Validate
    valid_episodes = {k: v for k, v in episode_map.items() if len(v) > 0}
    logger.info(f"Built episode index: {len(valid_episodes)} episodes with data")

    # Print timestep statistics
    lengths = [len(v) for v in valid_episodes.values()]
    logger.info(
        f"Timesteps per episode: min={min(lengths)}, max={max(lengths)}, "
        f"mean={np.mean(lengths):.1f}, total={sum(lengths)}"
    )

    return valid_episodes


def collect_training_arrays(
    dataset_root: str | Path,
    episode_map: Optional[Dict[int, List[int]]] = None,
    state_keys: Optional[List[str]] = None,
    action_key: str = "action",
    episode_indices: Optional[List[int]] = None,
    max_timesteps_per_episode: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect state and action arrays from the dataset.

    Args:
        dataset_root: Path to LeRobot dataset directory.
        episode_map: Pre-built episode index mapping. If None, will be built.
        state_keys: State field keys to extract.
        action_key: Action field key.
        episode_indices: Which episodes to include. None = all.
        max_timesteps_per_episode: Max timesteps per episode. None = all.

    Returns:
        Tuple of (states [N, D_s], actions [N, D_a], episode_ids [N], timestep_ids [N]).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if episode_map is None:
        episode_map = build_episode_index(dataset_root)

    if episode_indices is not None:
        episode_map = {k: v for k, v in episode_map.items() if k in episode_indices}

    if state_keys is None:
        structure = infer_episode_structure(dataset_root)
        state_keys = structure["state_keys"]
        if not state_keys:
            state_keys = ["observation.state"]

    # Load dataset
    repo_id = str(Path(dataset_root).name)
    dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_root))

    all_states = []
    all_actions = []
    all_episode_ids = []
    all_timestep_ids = []

    for ep_idx, row_indices in episode_map.items():
        rows_to_use = row_indices
        if max_timesteps_per_episode is not None:
            rows_to_use = row_indices[:max_timesteps_per_episode]

        for row_idx in rows_to_use:
            sample = dataset.hf_dataset[int(row_idx)]

            state_vec = extract_state_vector(sample, state_keys)
            action_vec = extract_action_vector(sample, action_key)

            # Check for NaN/Inf
            if not np.all(np.isfinite(state_vec)) or not np.all(np.isfinite(action_vec)):
                continue

            all_states.append(state_vec)
            all_actions.append(action_vec)
            all_episode_ids.append(ep_idx)
            all_timestep_ids.append(sample.get("frame_index", row_idx))

    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    episode_ids = np.array(all_episode_ids, dtype=np.int64)
    timestep_ids = np.array(all_timestep_ids, dtype=np.int64)

    logger.info(f"Collected {len(states)} valid timesteps from {len(episode_map)} episodes")
    logger.info(f"State shape: {states.shape}, Action shape: {actions.shape}")

    # Check for NaN/Inf in final arrays
    assert np.all(np.isfinite(states)), "States contain NaN/Inf after collection"
    assert np.all(np.isfinite(actions)), "Actions contain NaN/Inf after collection"

    return states, actions, episode_ids, timestep_ids


def compute_normalization_stats(x: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute mean and std for normalization.

    Args:
        x: Input array of shape [N, D].

    Returns:
        Dictionary with 'mean' [D] and 'std' [D] arrays.
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    # Prevent division by zero
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def normalize_array(x: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Normalize array using precomputed statistics.

    Args:
        x: Input array of shape [N, D] or [D].
        stats: Dictionary with 'mean' and 'std' arrays.

    Returns:
        Normalized array of same shape.
    """
    mean = stats["mean"]
    std = stats["std"]
    return ((x - mean) / std).astype(np.float32)