#!/usr/bin/env python
"""
Robomme initial-configuration feature extraction.

Reads episode_initial_states.json and extracts deterministic configuration
features from each episode's initial_configuration. These features describe
ONLY the environment configuration at episode start (object positions, target
locations, task parameters), never trajectory results, rewards, or success.

Each task has its own feature schema but all episodes within a task share
the same dimensionality.

Cache is saved to: personal/work2/duibi_robomme/cache/<task>/configuration/
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Pose flattening helpers
# ---------------------------------------------------------------------------

def _flatten_pose(pose: dict) -> List[float]:
    """Flatten a pose dict (position + quaternion) to a list of floats."""
    values = []
    pos = pose.get("position", pose.get("1", None))
    if pos is not None:
        if isinstance(pos, list) and len(pos) > 0 and isinstance(pos[0], list):
            pos = pos[0]
        values.extend(float(x) for x in pos)
    else:
        values.extend([0.0, 0.0, 0.0])

    quat = pose.get("quaternion", None)
    if quat is not None:
        if isinstance(quat, list) and len(quat) > 0 and isinstance(quat[0], list):
            quat = quat[0]
        values.extend(float(x) for x in quat)
    else:
        values.extend([1.0, 0.0, 0.0, 0.0])
    return values


# ---------------------------------------------------------------------------
# Per-task configuration feature builders
# ---------------------------------------------------------------------------

def _build_movecube_features(config: dict) -> np.ndarray:
    """
    MoveCube configuration features based on actual adapter output:
    - cube position (3) + quaternion (4) = 7
    - cube_2 position (3) + quaternion (4) = 7  (if present)
    - goal_site position (3) + quaternion (4) = 7
    - goal_site_2 position (3) + quaternion (4) = 7  (if present)
    - pegs articulation qpos (variable, use max 2 pegs * 1 qpos = 2)
    - task_config: direction, direction2, obj_flag, length, radius = 5
    Total: 7+7+7+7+2+5 = 35
    """
    features = []

    # Movable objects: cube
    movable = config.get("movable_objects", [])
    cube_found = False
    cube2_found = False
    for obj in movable:
        name = obj.get("name", "")
        if name == "cube" and not cube_found:
            features.extend(_flatten_pose(obj.get("pose", {})))
            cube_found = True
        elif name == "cube_2" and not cube2_found:
            features.extend(_flatten_pose(obj.get("pose", {})))
            cube2_found = True

    if not cube_found:
        features.extend([0.0] * 7)
    if not cube2_found:
        features.extend([0.0] * 7)

    # Randomized targets: goal_site, goal_site_2
    targets = config.get("randomized_targets", [])
    goal_found = False
    goal2_found = False
    for tgt in targets:
        name = tgt.get("name", "")
        if name == "goal_site" and not goal_found:
            features.extend(_flatten_pose(tgt.get("pose", {})))
            goal_found = True
        elif name == "goal_site_2" and not goal2_found:
            features.extend(_flatten_pose(tgt.get("pose", {})))
            goal2_found = True

    if not goal_found:
        features.extend([0.0] * 7)
    if not goal2_found:
        features.extend([0.0] * 7)

    # Articulations: pegs qpos
    arts = config.get("articulations", [])
    max_pegs = 2
    peg_qpos_list = []
    for art in arts:
        qpos = art.get("qpos", None)
        if qpos is not None:
            # Recursively flatten nested lists
            def _flatten(val):
                if isinstance(val, list):
                    for item in val:
                        yield from _flatten(item)
                else:
                    yield float(val)
            peg_qpos_list.extend(_flatten(qpos))

    while len(peg_qpos_list) < max_pegs:
        peg_qpos_list.append(0.0)
    features.extend(peg_qpos_list[:max_pegs])

    # Task config
    tc = config.get("task_config", {})
    features.append(float(tc.get("direction", 1)))
    features.append(float(tc.get("direction2", 1)))
    features.append(float(tc.get("obj_flag", 1)))
    features.append(float(tc.get("length", 0.1)))
    features.append(float(tc.get("radius", 0.01)))

    return np.array(features, dtype=np.float32)


def _build_patternlock_features(config: dict) -> np.ndarray:
    """
    PatternLock configuration features:
    - selected_buttons positions: each button has position (3) + quaternion (4) = 7
      We use up to 4 selected buttons (max for medium difficulty)
    - task_config: grid_size, num_selected = 2
    Total: 4*7 + 2 = 30
    """
    features = []

    # Selected buttons (the pattern to press)
    targets = config.get("randomized_targets", [])
    selected_buttons = []
    for tgt in targets:
        name = tgt.get("name", "")
        if name.startswith("selected_"):
            selected_buttons.append(tgt)

    selected_buttons.sort(
        key=lambda x: int(x["name"].split("_")[1]) if x["name"].split("_")[-1].isdigit() else 99
    )

    max_buttons = 4
    for i in range(max_buttons):
        if i < len(selected_buttons):
            features.extend(_flatten_pose(selected_buttons[i].get("pose", {})))
        else:
            features.extend([0.0] * 7)

    # Task config
    tc = config.get("task_config", {})
    features.append(float(tc.get("grid_size", 3)))
    features.append(float(tc.get("num_selected", 0)))

    return np.array(features, dtype=np.float32)


def _build_routestick_features(config: dict) -> np.ndarray:
    """
    RouteStick configuration features:
    - waypoint positions (selected_buttons): each has position (3) + quaternion (4) = 7
      We use up to 4 waypoints
    - task_config: grid_size, num_selected, direction = 3
    Total: 4*7 + 3 = 31
    """
    features = []

    # Waypoints
    targets = config.get("randomized_targets", [])
    waypoints = []
    for tgt in targets:
        name = tgt.get("name", "")
        if name.startswith("waypoint_"):
            waypoints.append(tgt)

    waypoints.sort(
        key=lambda x: int(x["name"].split("_")[1]) if x["name"].split("_")[-1].isdigit() else 99
    )

    max_waypoints = 4
    for i in range(max_waypoints):
        if i < len(waypoints):
            features.extend(_flatten_pose(waypoints[i].get("pose", {})))
        else:
            features.extend([0.0] * 7)

    # Task config
    tc = config.get("task_config", {})
    features.append(float(tc.get("grid_size", 3)))
    features.append(float(tc.get("num_selected", 0)))
    features.append(float(tc.get("direction", 1)))

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_FEATURE_BUILDERS = {
    "MoveCube_easy": _build_movecube_features,
    "MoveCube": _build_movecube_features,
    "PatternLock_medium": _build_patternlock_features,
    "PatternLock": _build_patternlock_features,
    "RouteStick_hard": _build_routestick_features,
    "RouteStick": _build_routestick_features,
}

TASK_FEATURE_DIMS = {
    "MoveCube_easy": 35,
    "MoveCube": 35,
    "PatternLock_medium": 30,
    "PatternLock": 30,
    "RouteStick_hard": 31,
    "RouteStick": 31,
}


def extract_configuration_features(
    dataset_root: str,
    task_name: str,
    cache_dir: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """
    Extract configuration features for all episodes in a Robomme dataset.

    Args:
        dataset_root: path to LeRobotDataset root
        task_name: one of MoveCube_easy, PatternLock_medium, RouteStick_hard
        cache_dir: optional cache directory; if None, uses default

    Returns:
        dict mapping episode_index -> configuration feature array
    """
    builder = TASK_FEATURE_BUILDERS.get(task_name)
    if builder is None:
        raise ValueError(f"No configuration feature builder for task: {task_name}")

    expected_dim = TASK_FEATURE_DIMS[task_name]

    metadata_file = Path(dataset_root) / "episode_initial_states.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"episode_initial_states.json not found: {metadata_file}")

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    episodes = metadata.get("episodes", [])
    features = {}

    for ep_info in episodes:
        ep_idx = ep_info.get("episode_index")
        if ep_idx is None:
            continue

        initial_config = ep_info.get("initial_configuration")
        if initial_config is None:
            continue

        feat = builder(initial_config)
        if feat.shape[0] != expected_dim:
            raise ValueError(
                f"Episode {ep_idx}: feature dim {feat.shape[0]} != expected {expected_dim}"
            )
        features[int(ep_idx)] = feat

    # Save to cache if cache_dir provided
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Save features as npz
        indices = sorted(features.keys())
        feat_array = np.stack([features[i] for i in indices], axis=0)
        np.savez(cache_path / "config_features.npz",
                 episode_indices=np.array(indices, dtype=np.int64),
                 features=feat_array)

        # Save schema
        schema = {
            "task": task_name,
            "feature_dim": expected_dim,
            "num_episodes": len(features),
            "episode_indices": indices,
        }
        with open(cache_path / "config_schema.json", "w") as f:
            json.dump(schema, f, indent=2)

    return features


def load_configuration_features(
    dataset_root: str,
    task_name: str,
    cache_dir: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """Load configuration features from cache, or extract and cache if not available."""
    if cache_dir is None:
        # Default cache location
        repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        cache_dir = str(repo_root / "personal" / "work2" / "duibi_robomme" / "cache" / task_name / "configuration")

    cache_path = Path(cache_dir)
    features_file = cache_path / "config_features.npz"

    if features_file.exists():
        data = np.load(str(features_file))
        indices = data["episode_indices"].tolist()
        feat_array = data["features"]
        features = {int(idx): feat_array[i] for i, idx in enumerate(indices)}
        print(f"Loaded configuration features from cache: {cache_path}")
        return features

    print(f"Cache not found, extracting configuration features...")
    features = extract_configuration_features(dataset_root, task_name, cache_dir)
    print(f"Extracted features for {len(features)} episodes, cached to {cache_path}")
    return features


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract Robomme configuration features")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--task-name", type=str, required=True,
                       choices=["MoveCube_easy", "PatternLock_medium", "RouteStick_hard"])
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    features = extract_configuration_features(
        args.dataset_root, args.task_name, args.cache_dir
    )
    print(f"Extracted configuration features for {len(features)} episodes")
    for ep_idx in sorted(features.keys())[:3]:
        print(f"  Episode {ep_idx}: dim={features[ep_idx].shape}, "
              f"first5={features[ep_idx][:5].tolist()}")