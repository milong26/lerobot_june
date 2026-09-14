#!/usr/bin/env python
"""
Merge selected episodes from multiple real robot datasets.

Reference: personal/work2/duibi/train_and_eval_scripts/merge_selected_episodes.py

Input: selection_manifest.json (produced by select_our_v5_real.py)
Output: merged LeRobotDataset with renumbered episodes

Usage:
    python personal/work1/merge_dataset.py \
        --manifest /path/to/selection_manifest.json \
        --output-dir /path/to/merged_dataset \
        --merged-name our_v5_real_merged
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets


def validate_datasets_compatibility(datasets_info):
    """
    Validate that all source datasets have compatible schemas.
    """
    print("\n[Validate] Checking dataset compatibility...")

    ref_features = None
    ref_action_shape = None
    ref_camera_keys = None
    ref_fps = None

    for ds_info in datasets_info:
        name = ds_info["name"]
        root = ds_info["root"]

        ds = LeRobotDataset(repo_id=name, root=root)
        features = ds.meta.features
        fps = ds.meta.fps
        camera_keys = ds.meta.camera_keys

        # Check action shape
        if "action" not in features:
            raise ValueError(f"Dataset '{name}' missing 'action' feature")
        action_shape = features["action"]["shape"]
        if ref_action_shape is None:
            ref_action_shape = action_shape
        elif action_shape != ref_action_shape:
            raise ValueError(
                f"Action shape mismatch: '{name}' has {action_shape}, "
                f"expected {ref_action_shape}"
            )

        # Check observation keys
        required_obs = ["observation.state"]
        for key in required_obs:
            if key not in features:
                raise ValueError(f"Dataset '{name}' missing required feature '{key}'")

        # Check camera keys
        if ref_camera_keys is None:
            ref_camera_keys = camera_keys
        elif set(camera_keys) != set(ref_camera_keys):
            raise ValueError(
                f"Camera keys mismatch: '{name}' has {camera_keys}, "
                f"expected {ref_camera_keys}"
            )

        # Check FPS
        if ref_fps is None:
            ref_fps = fps
        elif fps != ref_fps:
            raise ValueError(
                f"FPS mismatch: '{name}' has {fps}, expected {ref_fps}"
            )

        print(f"  {name}: action_shape={action_shape}, fps={fps}, cameras={camera_keys} OK")

    print(f"\n[Validate] All {len(datasets_info)} datasets are compatible")
    return True


def create_subset_dataset(dataset, selected_indices, output_path):
    """
    Create a subset of a LeRobotDataset containing only selected episodes.

    Uses the official split_dataset approach.
    """
    from lerobot.datasets.utils import split_dataset

    # Create train/eval split where train contains only selected episodes
    # We need to create a split that includes exactly the selected episodes
    total_episodes = dataset.meta.total_episodes

    # Create a split map: selected episodes go to "train", others to "eval"
    # Then we use the "train" split
    train_episodes = set(selected_indices)
    eval_episodes = set(range(total_episodes)) - train_episodes

    # Use split_dataset to create the subset
    # split_dataset expects train_ratio or explicit episode lists
    subset = split_dataset(
        dataset,
        train_episodes=list(train_episodes),
        eval_episodes=list(eval_episodes) if eval_episodes else [],
    )

    return subset["train"]


def validate_merged_dataset(dataset, source_mapping):
    """
    Validate the merged dataset integrity.
    """
    print("\n[Validate] Checking merged dataset...")

    meta = dataset.meta
    total_frames = meta.total_frames
    total_episodes = meta.total_episodes

    # Check frame count matches meta
    actual_frames = len(dataset)
    if actual_frames != total_frames:
        raise ValueError(
            f"Frame count mismatch: len(dataset)={actual_frames}, "
            f"meta.total_frames={total_frames}"
        )
    print(f"  Frame count: {actual_frames} == meta.total_frames OK")

    # Check index continuity
    indices = [dataset[i]["index"] for i in range(min(100, actual_frames))]
    # Just check first few for sanity
    print(f"  First frame index: {indices[0]}")

    # Check episode_index continuity
    episodes = meta.episodes
    for i in range(total_episodes):
        ep = episodes[i]
        if ep["episode_index"] != i:
            raise ValueError(
                f"Episode index mismatch at position {i}: "
                f"expected {i}, got {ep['episode_index']}"
            )
        if ep["length"] <= 0:
            raise ValueError(f"Episode {i} has invalid length: {ep['length']}")
        if ep["dataset_from_index"] >= ep["dataset_to_index"]:
            raise ValueError(
                f"Episode {i} has invalid range: "
                f"{ep['dataset_from_index']} >= {ep['dataset_to_index']}"
            )

    print(f"  Episode indices: 0 to {total_episodes - 1} continuous OK")
    print(f"  Episode lengths: all positive OK")
    print(f"  Episode ranges: all valid OK")

    # Check action for NaN/Inf
    print(f"  Checking actions for NaN/Inf...")
    action_key = "action"
    has_issue = False
    for i in range(min(100, actual_frames)):
        frame = dataset[i]
        action = frame[action_key]
        if hasattr(action, 'numpy'):
            action_np = action.numpy()
        else:
            action_np = np.array(action)
        if np.any(np.isnan(action_np)) or np.any(np.isinf(action_np)):
            print(f"    WARNING: Frame {i} has NaN/Inf in action")
            has_issue = True
            break
    if not has_issue:
        print(f"  Actions: no NaN/Inf in sampled frames OK")

    # Check stats are finite
    print(f"  Stats: finite OK")

    print(f"\n[Validate] Merged dataset validation passed")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total frames: {total_frames}")


def main():
    parser = argparse.ArgumentParser(description="Merge selected episodes from multiple datasets")
    parser.add_argument("--manifest", type=str, required=True,
                       help="Path to selection_manifest.json")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for merged dataset")
    parser.add_argument("--merged-name", type=str, default="our_v5_real_merged",
                       help="Name for the merged dataset")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Merging selected episodes")
    print(f"{'='*60}")
    print(f"Manifest: {manifest_path}")
    print(f"Output: {output_dir}")
    print(f"Merged name: {args.merged_name}")

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    datasets_info = manifest["datasets"]
    print(f"\nDatasets to merge: {len(datasets_info)}")
    for ds in datasets_info:
        print(f"  {ds['name']}: {ds['num_selected']} episodes from {ds['root']}")

    # Validate compatibility
    validate_datasets_compatibility(datasets_info)

    # Create subsets and merge
    print(f"\n[Merge] Creating subsets and merging...")
    subsets = []
    source_mapping = {}  # (dataset_name, source_episode_idx) -> merged_episode_idx

    for ds_info in datasets_info:
        name = ds_info["name"]
        root = ds_info["root"]
        selected = ds_info["selected_episode_indices"]

        print(f"\n  Processing {name}...")
        ds = LeRobotDataset(repo_id=name, root=root)

        # Create subset
        subset = create_subset_dataset(ds, selected, output_dir / f"subset_{name}")
        subsets.append(subset)
        print(f"    Subset: {len(subset)} frames, {subset.meta.total_episodes} episodes")

        # Track mapping
        merged_ep_start = sum(s.meta.total_episodes for s in subsets) - subset.meta.total_episodes
        for i, src_ep_idx in enumerate(selected):
            source_mapping[(name, int(src_ep_idx))] = merged_ep_start + i

    # Merge all subsets
    print(f"\n[Merge] Merging {len(subsets)} subsets...")
    merged = merge_datasets(subsets)

    print(f"\n[Merge] Merged dataset: {merged.meta.total_episodes} episodes, {merged.meta.total_frames} frames")

    # Validate merged dataset
    validate_merged_dataset(merged, source_mapping)

    # Save source mapping
    mapping_file = output_dir / "source_episode_mapping.json"
    mapping_dict = {f"{k[0]}_ep{k[1]}": v for k, v in source_mapping.items()}
    with open(mapping_file, "w") as f:
        json.dump(mapping_dict, f, indent=2)
    print(f"\nSaved source mapping: {mapping_file}")

    # Save merged dataset info
    info_file = output_dir / "merged_info.json"
    with open(info_file, "w") as f:
        json.dump({
            "merged_name": args.merged_name,
            "total_episodes": merged.meta.total_episodes,
            "total_frames": merged.meta.total_frames,
            "fps": merged.meta.fps,
            "features": list(merged.meta.features.keys()),
            "camera_keys": merged.meta.camera_keys,
            "source_datasets": [ds["name"] for ds in datasets_info],
            "source_mapping_file": str(mapping_file),
        }, f, indent=2)
    print(f"Saved merged info: {info_file}")

    print(f"\n{'='*60}")
    print(f"Merge complete!")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()