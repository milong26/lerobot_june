"""
DemInf Dataset Adapter

Adapts LeRobot dataset format for DemInf state-based MI estimation.
Uses LeRobotDataset with proper global row indexing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("deminf")

ENV_STATE_LAYOUT = {
    "hand_pos": slice(0, 3),
    "gripper": slice(3, 4),
    "obj1_pos": slice(4, 7),
    "obj1_quat": slice(7, 11),
    "obj2_pos": slice(11, 14),
    "obj2_quat": slice(14, 18),
    "prev_frame_stack": slice(18, 36),
    "prev_gripper": slice(21, 22),
    "goal_pos": slice(36, 39),
}


def infer_episode_structure(dataset_root: str | Path) -> Dict[str, Any]:
    """
    Infer the episode structure from a LeRobot dataset.
    """
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found at {info_path}")

    with open(info_path, "r") as f:
        info = json.load(f)

    features = info.get("features", {})
    total_episodes = info.get("total_episodes", 0)
    total_frames = info.get("total_frames", 0)
    fps = info.get("fps", 80)

    state_shape = None
    action_shape = None

    if "observation.state" in features:
        state_shape = features["observation.state"].get("shape", [4])
        logger.info(f"Found observation.state with shape {state_shape}")

    if "observation.environment_state" in features:
        env_state_shape = features["observation.environment_state"].get("shape", [39])
        logger.info(f"Found observation.environment_state with shape {env_state_shape}")

    if "action" in features:
        action_shape = features["action"].get("shape", [4])
        logger.info(f"Found action with shape {action_shape}")

    return {
        "features": features,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": fps,
        "state_shape": state_shape,
        "action_shape": action_shape,
    }


def build_episode_index_from_lerobot(
    dataset_root: str | Path,
    repo_id: str,
) -> Dict[int, List[int]]:
    """
    Build mapping from episode index to global row indices using LeRobotDataset.

    Traverses global row indices in dataset.hf_dataset, reads episode_index
    and frame_index for each row, and builds {episode_idx: [global_row_idx...]}
    sorted by frame_index.

    Args:
        dataset_root: Path to LeRobot dataset directory.
        repo_id: Repository ID for LeRobotDataset.

    Returns:
        Dictionary {episode_idx: [global_row_idx...]} with rows sorted by frame_index.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_root))
    hf_dataset = dataset.hf_dataset

    total_episodes = dataset.num_episodes
    total_frames = dataset.num_frames

    logger.info(f"LeRobotDataset: {total_episodes} episodes, {total_frames} frames")

    episode_map: Dict[int, List[int]] = {i: [] for i in range(total_episodes)}

    for global_idx in range(len(hf_dataset)):
        row = hf_dataset[int(global_idx)]
        ep_idx = int(row["episode_index"])
        frame_idx = int(row["frame_index"])
        episode_map[ep_idx].append((frame_idx, global_idx))

    for ep_idx in episode_map:
        episode_map[ep_idx].sort(key=lambda x: x[0])
        episode_map[ep_idx] = [gidx for _, gidx in episode_map[ep_idx]]

    valid_episodes = {k: v for k, v in episode_map.items() if len(v) > 0}
    logger.info(f"Built episode index: {len(valid_episodes)} episodes with data")

    lengths = [len(v) for v in valid_episodes.values()]
    logger.info(
        f"Timesteps per episode: min={min(lengths)}, max={max(lengths)}, "
        f"mean={np.mean(lengths):.1f}, total={sum(lengths)}"
    )

    return valid_episodes


def validate_episode_index(
    dataset_root: str | Path,
    repo_id: str,
    episode_map: Dict[int, List[int]],
    max_checks: int = 100,
) -> None:
    """
    Validate that episode_map global row indices have correct episode_index.

    Samples up to max_checks row indices per episode and asserts
    dataset.hf_dataset[row_idx]["episode_index"] == ep_idx.

    Raises:
        AssertionError: If any row index has wrong episode_index.
    """
    import random
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_root))
    hf_dataset = dataset.hf_dataset

    for ep_idx, row_indices in episode_map.items():
        if not row_indices:
            continue
        check_indices = row_indices
        if len(row_indices) > max_checks:
            rng = random.Random(42)
            check_indices = rng.sample(row_indices, max_checks)

        for global_idx in check_indices:
            actual_ep = int(hf_dataset[int(global_idx)]["episode_index"])
            assert actual_ep == ep_idx, (
                f"Row {global_idx} has episode_index={actual_ep}, expected {ep_idx}"
            )

    logger.info(f"Episode index validation passed: {len(episode_map)} episodes verified")


def drop_terminal_transitions(episode_map: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """
    Drop the last terminal transition from each episode.

    Official DemInf _stepify behavior: after chunking, all transitions use x[:-1].
    This removes the final timestep from each episode's row list.

    Args:
        episode_map: {episode_idx: [global_row_idx...]} sorted by frame_index.

    Returns:
        Episode map with last row removed from each episode.
    """
    result = {}
    dropped = 0
    for ep_idx, rows in episode_map.items():
        if len(rows) > 1:
            result[ep_idx] = rows[:-1]
            dropped += 1
        else:
            result[ep_idx] = rows
            logger.warning(f"Episode {ep_idx} has only 1 row, cannot drop terminal")

    logger.info(f"Dropped terminal transitions from {dropped} episodes")
    return result


def collect_training_arrays(
    dataset_root: str | Path,
    repo_id: str,
    episode_map: Optional[Dict[int, List[int]]] = None,
    state_source: str = "observation.environment_state",
    action_key: str = "action",
    episode_indices: Optional[List[int]] = None,
    max_timesteps_per_episode: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect state and action arrays from the dataset using global row indices.

    Args:
        dataset_root: Path to LeRobot dataset directory.
        repo_id: Repository ID for LeRobotDataset.
        episode_map: Pre-built episode index mapping. If None, will be built.
        state_source: Which observation key to use for state.
        action_key: Action field key.
        episode_indices: Which episodes to include. None = all.
        max_timesteps_per_episode: Max timesteps per episode. None = all.

    Returns:
        Tuple of (states [N, D_s], actions [N, D_a], episode_ids [N],
                  timestep_ids [N], global_row_ids [N]).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if episode_map is None:
        episode_map = build_episode_index_from_lerobot(dataset_root, repo_id)

    if episode_indices is not None:
        episode_map = {k: v for k, v in episode_map.items() if k in episode_indices}

    dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_root))
    hf_dataset = dataset.hf_dataset

    all_states = []
    all_actions = []
    all_episode_ids = []
    all_timestep_ids = []
    all_global_row_ids = []

    for ep_idx, row_indices in episode_map.items():
        rows_to_use = row_indices
        if max_timesteps_per_episode is not None:
            rows_to_use = row_indices[:max_timesteps_per_episode]

        for global_idx in rows_to_use:
            sample = hf_dataset[int(global_idx)]

            state_val = sample[state_source]
            if isinstance(state_val, np.ndarray):
                state_vec = state_val.flatten().astype(np.float32)
            elif hasattr(state_val, "numpy"):
                state_vec = state_val.numpy().flatten().astype(np.float32)
            else:
                state_vec = np.array(state_val).flatten().astype(np.float32)

            action_val = sample[action_key]
            if isinstance(action_val, np.ndarray):
                action_vec = action_val.flatten().astype(np.float32)
            elif hasattr(action_val, "numpy"):
                action_vec = action_val.numpy().flatten().astype(np.float32)
            else:
                action_vec = np.array(action_val).flatten().astype(np.float32)

            if not np.all(np.isfinite(state_vec)) or not np.all(np.isfinite(action_vec)):
                continue

            all_states.append(state_vec)
            all_actions.append(action_vec)
            all_episode_ids.append(ep_idx)
            all_timestep_ids.append(int(sample.get("frame_index", global_idx)))
            all_global_row_ids.append(int(global_idx))

    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    episode_ids = np.array(all_episode_ids, dtype=np.int64)
    timestep_ids = np.array(all_timestep_ids, dtype=np.int64)
    global_row_ids = np.array(all_global_row_ids, dtype=np.int64)

    logger.info(f"Collected {len(states)} valid timesteps from {len(episode_map)} episodes")
    logger.info(f"State shape: {states.shape}, Action shape: {actions.shape}")

    assert np.all(np.isfinite(states)), "States contain NaN/Inf after collection"
    assert np.all(np.isfinite(actions)), "Actions contain NaN/Inf after collection"

    return states, actions, episode_ids, timestep_ids, global_row_ids


def check_relative_action(
    dataset_root: str | Path,
    repo_id: str,
) -> Tuple[bool, str]:
    """
    Check whether actions in the dataset are relative/delta control.

    Uses feature axes names and MetaWorld collection code to determine.
    Returns (is_relative, evidence_string).

    MetaWorld actions are (x, y, z, gripper) delta commands.
    """
    info_path = Path(dataset_root) / "meta" / "info.json"
    evidence_parts = []

    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)

        features = info.get("features", {})
        action_info = features.get("action", {})
        names = action_info.get("names", {})

        if isinstance(names, dict):
            axes = names.get("axes", [])
            if axes == ["x", "y", "z", "gripper"]:
                evidence_parts.append(f"Action axes are {axes} - delta/relative control commands")
                logger.info("Action axes are [x, y, z, gripper] - these are delta/relative control commands")
                return True, "; ".join(evidence_parts)

    evidence_parts.append("MetaWorld expert action uses (x,y,z,gripper) incremental/relative control")
    logger.info("Assuming relative/delta action based on MetaWorld convention")
    return True, "; ".join(evidence_parts)


class DemInfNormalizer:
    """
    Normalization following DemInf official conventions.

    For MetaWorld environment_state (39-dim):
    - Continuous non-gripper dims: Gaussian normalization (x-mean)/std
    - Gripper dims (index 3 and index 21): no normalization (NONE, per official)
    - Action xyz (0:3): Gaussian normalization
    - Action gripper (3): ALWAYS bounds normalization to [-1,1] using
      the dataset's actual min/max, matching official NormalizationType.BOUNDS.
      Even if the gripper values already fall in [-1,1], the bounds transform
      is still applied (if min=-1, max=1, the result is identical).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        state_gripper_indices: Optional[List[int]] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Only use gripper indices that are within the actual state dimension
        all_gripper = state_gripper_indices or [3, 21]
        self.state_gripper_indices = [i for i in all_gripper if i < state_dim]

        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        self.action_gripper_min = None
        self.action_gripper_max = None
        self.action_gripper_bounds_applied = True

    def fit(self, states: np.ndarray, actions: np.ndarray) -> None:
        """Compute normalization statistics from data."""
        state_non_gripper = np.ones(self.state_dim, dtype=bool)
        state_non_gripper[self.state_gripper_indices] = False

        self.state_mean = np.zeros(self.state_dim, dtype=np.float32)
        self.state_std = np.ones(self.state_dim, dtype=np.float32)

        mean_all = np.mean(states, axis=0)
        std_all = np.std(states, axis=0)

        self.state_mean[state_non_gripper] = mean_all[state_non_gripper]
        self.state_std[state_non_gripper] = np.where(
            std_all[state_non_gripper] < 1e-6, 1.0, std_all[state_non_gripper]
        )

        self.action_mean = np.zeros(self.action_dim, dtype=np.float32)
        self.action_std = np.ones(self.action_dim, dtype=np.float32)

        action_xyz_mean = np.mean(actions[:, :3], axis=0)
        action_xyz_std = np.std(actions[:, :3], axis=0)
        self.action_mean[:3] = action_xyz_mean
        self.action_std[:3] = np.where(action_xyz_std < 1e-6, 1.0, action_xyz_std)

        # ALWAYS record actual gripper min/max for bounds normalization
        gripper_vals = actions[:, 3]
        g_min, g_max = float(np.min(gripper_vals)), float(np.max(gripper_vals))
        self.action_gripper_min = g_min
        self.action_gripper_max = g_max

        logger.info(
            f"Action gripper range [{g_min:.4f}, {g_max:.4f}]; "
            f"bounds normalization will always be applied (official NormalizationType.BOUNDS)"
        )

    def normalize_state(self, states: np.ndarray) -> np.ndarray:
        """Normalize state array."""
        result = ((states - self.state_mean) / self.state_std).astype(np.float32)
        for idx in self.state_gripper_indices:
            if idx < states.shape[-1]:
                result[..., idx] = states[..., idx]
        return result

    def normalize_action(self, actions: np.ndarray) -> np.ndarray:
        """
        Normalize action array.

        Action xyz: Gaussian normalization.
        Action gripper: ALWAYS bounds normalization using dataset min/max,
        matching official NormalizationType.BOUNDS. If g_max - g_min <= epsilon,
        outputs 0 (simulate divide_no_nan semantics).
        """
        result = actions.copy()
        result[:, :3] = ((actions[:, :3] - self.action_mean[:3]) / self.action_std[:3]).astype(np.float32)

        # ALWAYS apply bounds normalization (official BOUNDS semantics)
        g_min, g_max = self.action_gripper_min, self.action_gripper_max
        g = np.clip(actions[:, 3], g_min, g_max)
        if g_max - g_min > 1e-6:
            result[:, 3] = (2.0 * (g - g_min) / (g_max - g_min) - 1.0).astype(np.float32)
        else:
            # Constant dimension: output 0 (divide_no_nan semantics)
            result[:, 3] = 0.0

        return result

    def save(self, path: str | Path) -> None:
        """Save normalization statistics to both npz and json."""
        np.savez(
            str(path),
            state_mean=self.state_mean,
            state_std=self.state_std,
            action_mean=self.action_mean,
            action_std=self.action_std,
            state_gripper_indices=np.array(self.state_gripper_indices),
            action_gripper_min=np.array([self.action_gripper_min]),
            action_gripper_max=np.array([self.action_gripper_max]),
            action_gripper_bounds_applied=np.array([self.action_gripper_bounds_applied]),
        )
        logger.info(f"Saved normalization stats to {path}")

        json_path = str(path).replace(".npz", ".json")
        stats_json = {
            "state": {
                "mean": self.state_mean.tolist(),
                "std": self.state_std.tolist(),
                "dimension": self.state_dim,
                "source_key": "observation.environment_state",
            },
            "action": {
                "xyz_mean": self.action_mean[:3].tolist(),
                "xyz_std": self.action_std[:3].tolist(),
                "gripper_min": self.action_gripper_min,
                "gripper_max": self.action_gripper_max,
            },
            "normalization_type": "gaussian_state_bounds_action",
        }
        with open(json_path, "w") as f:
            json.dump(stats_json, f, indent=2)
        logger.info(f"Saved normalization stats JSON to {json_path}")

    @classmethod
    def load(cls, path: str | Path) -> "DemInfNormalizer":
        """Load normalization statistics."""
        data = np.load(str(path), allow_pickle=True)
        normalizer = cls(
            state_dim=len(data["state_mean"]),
            action_dim=len(data["action_mean"]),
            state_gripper_indices=data["state_gripper_indices"].tolist(),
        )
        normalizer.state_mean = data["state_mean"]
        normalizer.state_std = data["state_std"]
        normalizer.action_mean = data["action_mean"]
        normalizer.action_std = data["action_std"]
        normalizer.action_gripper_min = float(data["action_gripper_min"][0])
        normalizer.action_gripper_max = float(data["action_gripper_max"][0])
        # Support both old and new field names
        if "action_gripper_bounds_applied" in data:
            normalizer.action_gripper_bounds_applied = bool(data["action_gripper_bounds_applied"][0])
        else:
            normalizer.action_gripper_bounds_applied = bool(data.get("action_gripper_normalized", [True])[0])
        return normalizer

    def get_manifest(self) -> Dict[str, Any]:
        """Get normalization manifest for cache fingerprinting."""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "state_gripper_indices": self.state_gripper_indices,
            "state_mean_hash": _array_hash(self.state_mean),
            "state_std_hash": _array_hash(self.state_std),
            "action_mean_hash": _array_hash(self.action_mean),
            "action_std_hash": _array_hash(self.action_std),
            "action_gripper_min": self.action_gripper_min,
            "action_gripper_max": self.action_gripper_max,
            "action_gripper_bounds_applied": self.action_gripper_bounds_applied,
        }


def _array_hash(arr: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def get_deminf_data_manifest(
    dataset_root: str | Path,
    repo_id: str,
    state_source: str = "observation.environment_state",
    action_key: str = "action",
    episode_map: Optional[Dict[int, List[int]]] = None,
) -> Dict[str, Any]:
    """
    Return a data manifest for DemInf pipeline.

    Returns:
        Dictionary with state_key, action_key, state_dim, action_dim,
        episode_count, transition_count, normalization_type.
    """
    if episode_map is None:
        episode_map = build_episode_index_from_lerobot(dataset_root, repo_id)

    episode_count = len(episode_map)
    transition_count = sum(len(rows) for rows in episode_map.values())

    state_dim = 39
    action_dim = 4
    info_path = Path(dataset_root) / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)
        features = info.get("features", {})
        if "observation.environment_state" in features:
            env_shape = features["observation.environment_state"].get("shape", [39])
            state_dim = env_shape[0] if env_shape else 39
        elif "observation.state" in features:
            state_shape = features["observation.state"].get("shape", [4])
            state_dim = state_shape[0] if state_shape else 4
        if "action" in features:
            action_shape = features["action"].get("shape", [4])
            action_dim = action_shape[0] if action_shape else 4

    return {
        "state_key": state_source,
        "action_key": action_key,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "episode_count": episode_count,
        "transition_count": transition_count,
        "normalization_type": "gaussian_state_bounds_action",
    }