#!/usr/bin/env python
"""
Episode action descriptor extraction for V5 selection.

Reads action sequences from LeRobotDataset, resamples to fixed length,
computes statistics, and caches episode-level action descriptors.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WORK2_ROOT = Path(__file__).resolve().parent.parent
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

from embedding_utils.config_v5 import (
    V5_ACTION_STEPS,
    V5_ACTION_FEATURES,
    V5_ACTION_DESCRIPTOR_VERSION,
    V5_ACTION_DESCRIPTOR_OUTPUT_DIR,
    build_v5_action_descriptor_name,
)
from embedding_utils.cache import (
    normalize_dataset_name,
    get_dataset_episode_indices,
    write_cache_metadata,
)


def load_lerobot_dataset_for_action_descriptor(dataset_path: str):
    """
    Load LeRobotDataset using the same pattern as extract_embeddings.py.

    Args:
        dataset_path: path to the LeRobotDataset root directory

    Returns:
        LeRobotDataset object
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id="work2/metaworld_pick_place",
        root=str(dataset_path)
    )
    return dataset


def _find_action_key(dataset) -> str:
    """Find the action feature key in dataset metadata."""
    features = dataset.meta.features
    for key in features.keys():
        if "action" in key.lower():
            return key
    raise ValueError(f"No action feature found in dataset. Available: {list(features.keys())}")


def extract_episode_actions(dataset, episode_index: int) -> np.ndarray:
    """
    Read the full action sequence for a single episode.

    Args:
        dataset: LeRobotDataset object
        episode_index: episode index within the dataset

    Returns:
        np.ndarray of shape (T, action_dim) with the full action sequence
    """
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]
    frame_indices = list(range(from_idx, to_idx))

    action_key = _find_action_key(dataset)
    actions = []
    for idx in frame_indices:
        frame = dataset[idx]
        actions.append(frame[action_key])

    return np.array(actions)


def resample_action_sequence(actions: np.ndarray, num_steps: int) -> np.ndarray:
    """
    Resample an action sequence to a fixed number of steps.

    Args:
        actions: np.ndarray of shape (T, action_dim)
        num_steps: target number of steps

    Returns:
        np.ndarray of shape (num_steps, action_dim)
    """
    T = actions.shape[0]

    if T == 0:
        raise ValueError("Empty action sequence")

    if T == 1:
        return np.tile(actions[0], (num_steps, 1))

    if T == num_steps:
        return actions.copy()

    from scipy.interpolate import interp1d

    original_steps = np.arange(T)
    target_steps = np.linspace(0, T - 1, num_steps)

    resampled_actions = []
    for dim_idx in range(actions.shape[1]):
        f = interp1d(original_steps, actions[:, dim_idx], kind="linear")
        resampled_actions.append(f(target_steps))

    return np.array(resampled_actions).T


def compute_action_statistics(actions: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute action sequence statistics.

    Args:
        actions: np.ndarray of shape (T, action_dim)

    Returns:
        Dict with keys:
            - mean_action: shape (action_dim,)
            - std_action: shape (action_dim,)
            - delta_action_mean: shape (action_dim,)
            - delta_action_std: shape (action_dim,)
            - velocity_mean: shape (action_dim,)
            - trajectory_length: scalar
    """
    mean_action = np.mean(actions, axis=0)
    std_action = np.std(actions, axis=0)

    delta_actions = np.diff(actions, axis=0)
    delta_action_mean = np.mean(delta_actions, axis=0)
    delta_action_std = np.std(delta_actions, axis=0)

    velocity_mean = np.mean(np.abs(delta_actions), axis=0)

    trajectory_length = float(np.sum(np.linalg.norm(delta_actions, axis=1)))

    return {
        "mean_action": mean_action,
        "std_action": std_action,
        "delta_action_mean": delta_action_mean,
        "delta_action_std": delta_action_std,
        "velocity_mean": velocity_mean,
        "trajectory_length": np.array([trajectory_length]),
    }


def extract_episode_action_descriptor(
    dataset,
    episode_index: int,
    num_steps: int = V5_ACTION_STEPS,
) -> np.ndarray:
    """
    Extract a fixed-dimension action descriptor for a single episode.

    Args:
        dataset: LeRobotDataset object
        episode_index: episode index
        num_steps: resampled action sequence length

    Returns:
        1D numpy array representing the episode action descriptor
    """
    actions = extract_episode_actions(dataset, episode_index)
    resampled = resample_action_sequence(actions, num_steps)
    stats = compute_action_statistics(actions)

    descriptor = np.concatenate([
        resampled.flatten(),
        stats["mean_action"],
        stats["std_action"],
        stats["delta_action_mean"],
        stats["delta_action_std"],
        stats["velocity_mean"],
        stats["trajectory_length"],
    ])

    return descriptor


def extract_all_action_descriptors(
    dataset_path: str,
    output_dir: Path,
) -> Dict[int, np.ndarray]:
    """
    Extract action descriptors for all episodes in a dataset.

    Args:
        dataset_path: path to LeRobotDataset root
        output_dir: output directory for cached descriptors

    Returns:
        Dict mapping episode_index -> action descriptor array
    """
    dataset = load_lerobot_dataset_for_action_descriptor(dataset_path)
    episode_indices = get_dataset_episode_indices(dataset_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    action_descriptors = {}

    for ep_idx in episode_indices:
        ep_start = time.time()
        descriptor = extract_episode_action_descriptor(dataset, ep_idx)
        action_descriptors[ep_idx] = descriptor

        ep_file = output_dir / f"({ep_idx}).npy"
        np.save(ep_file, {
            "action_descriptor": descriptor,
            "episode_index": ep_idx,
        }, allow_pickle=True)

        elapsed = time.time() - ep_start
        completed = len(action_descriptors)
        progress = completed / len(episode_indices) * 100
        print(f"  Episode {ep_idx}: descriptor shape={descriptor.shape}, time={elapsed:.2f}s ({completed}/{len(episode_indices)}, {progress:.1f}%)")

    return action_descriptors


def save_action_descriptors(
    action_descriptors: Dict[int, np.ndarray],
    output_dir: Path,
    dataset_path: str = "",
    dataset_name: str = "",
) -> None:
    """
    Save action descriptors to cache directory.

    Saves:
        - episode_id.npy files with action_descriptor and episode_index
        - action_descriptor.npy (combined array)
        - action_descriptor_metadata.json

    Args:
        action_descriptors: Dict[episode_index, descriptor_array]
        output_dir: output directory
        dataset_path: original dataset path for metadata
        dataset_name: dataset name for metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx, descriptor in action_descriptors.items():
        ep_file = output_dir / f"({ep_idx}).npy"
        np.save(ep_file, {
            "action_descriptor": descriptor,
            "episode_index": ep_idx,
        }, allow_pickle=True)

    if action_descriptors:
        sorted_indices = sorted(action_descriptors.keys())
        combined = np.array([action_descriptors[i] for i in sorted_indices])
        combined_file = output_dir / "action_descriptor.npy"
        np.save(str(combined_file), {
            "descriptors": combined,
            "episode_indices": sorted_indices,
        }, allow_pickle=True)

    if dataset_name:
        episode_indices = sorted(action_descriptors.keys())
        metadata = {
            "dataset_name": normalize_dataset_name(dataset_name),
            "dataset_realpath": str(Path(dataset_path).resolve()) if dataset_path else "",
            "num_episodes": len(episode_indices),
            "episode_indices": episode_indices,
            "descriptor_version": V5_ACTION_DESCRIPTOR_VERSION,
            "action_steps": V5_ACTION_STEPS,
            "features": V5_ACTION_FEATURES,
            "extraction_method_name": build_v5_action_descriptor_name(),
            "source": "generated",
        }
        write_cache_metadata(output_dir, metadata)


def load_action_descriptors(output_dir: Path) -> Dict[int, np.ndarray]:
    """
    Load previously generated action descriptor cache.

    Tries to load from combined action_descriptor.npy first,
    falls back to loading individual episode files.

    Args:
        output_dir: cache directory path

    Returns:
        Dict[episode_index, descriptor_array]
    """
    combined_file = output_dir / "action_descriptor.npy"
    if combined_file.exists():
        data = np.load(str(combined_file), allow_pickle=True).item()
        descriptors = data["descriptors"]
        indices = data["episode_indices"]
        return {int(idx): descriptors[i] for i, idx in enumerate(indices)}

    action_descriptors = {}
    for f in sorted(output_dir.glob("*.npy")):
        if f.name == "action_descriptor.npy":
            continue
        try:
            data = np.load(str(f), allow_pickle=True).item()
            ep_idx = data.get("episode_index")
            if ep_idx is not None and "action_descriptor" in data:
                action_descriptors[int(ep_idx)] = data["action_descriptor"]
        except Exception:
            continue

    return action_descriptors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract action descriptors for V5 selection")
    parser.add_argument("--dataset-path", type=str, required=True, help="LeRobotDataset root directory")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name (e.g., corner)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: auto)")
    parser.add_argument("--num-steps", type=int, default=V5_ACTION_STEPS, help="Resampled action steps")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset_name = args.dataset_name

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        method_name = build_v5_action_descriptor_name(num_steps=args.num_steps)
        normalized = normalize_dataset_name(dataset_name)
        output_dir = V5_ACTION_DESCRIPTOR_OUTPUT_DIR / normalized / method_name

    print(f"Dataset: {dataset_name}")
    print(f"Output dir: {output_dir}")
    print(f"Action steps: {args.num_steps}")

    start = time.time()
    descriptors = extract_all_action_descriptors(dataset_path, output_dir)
    save_action_descriptors(descriptors, output_dir, dataset_path, dataset_name)

    elapsed = time.time() - start
    print(f"\nAction descriptor extraction complete: {len(descriptors)} episodes in {elapsed:.2f}s")
    print(f"Output: {output_dir}")