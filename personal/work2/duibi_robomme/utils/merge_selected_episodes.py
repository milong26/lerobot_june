#!/usr/bin/env python3
"""
Merge selected episodes from multiple Robomme LeRobot datasets into a single dataset.

This script:
1. Loads three selected LeRobotDataset subsets (one per Robomme task)
2. Validates feature schema compatibility (observation images, state, action shape/dtype)
3. Merges them using LeRobot's official merge_datasets
4. Recomputes stats
5. Generates a provenance manifest (JSON) tracking source of each episode
6. Validates merged dataset integrity

Usage:
    python merge_selected_episodes.py \
        --source-datasets MoveCube_easy=/path/to/movecube PatternLock_medium=/path/to/patternlock RouteStick_hard=/path/to/routestick \
        --method ours_v5 \
        --seed 42 \
        --k 28 \
        --output-dir /path/to/merged_dataset
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import (
    split_dataset,
    merge_datasets,
    recompute_stats,
)

ROBOMME_TASKS = ["MoveCube_easy", "PatternLock_medium", "RouteStick_hard"]


def validate_schema_compatibility(datasets, task_names):
    """Validate that all datasets have compatible feature schemas."""
    print(f"\n{'='*60}")
    print(f"Validating schema compatibility across {len(datasets)} datasets")
    print(f"{'='*60}")

    ref_ds = datasets[0]
    ref_name = task_names[0]

    required_keys = ["observation.state", "action"]
    image_keys = [k for k in ref_ds.hf_dataset.column_names if k.startswith("observation.images.")]

    for key in required_keys:
        if key not in ref_ds.hf_dataset.column_names:
            raise ValueError(f"Reference dataset {ref_name} missing required key: {key}")

    ref_state_shape = ref_ds.hf_dataset[0]["observation.state"].shape if hasattr(ref_ds.hf_dataset[0]["observation.state"], 'shape') else (len(ref_ds.hf_dataset[0]["observation.state"]),)
    ref_action_shape = ref_ds.hf_dataset[0]["action"].shape if hasattr(ref_ds.hf_dataset[0]["action"], 'shape') else (len(ref_ds.hf_dataset[0]["action"]),)

    print(f"  Reference ({ref_name}):")
    print(f"    observation.state shape: {ref_state_shape}")
    print(f"    action shape: {ref_action_shape}")
    print(f"    Image keys: {image_keys}")

    for i, (ds, name) in enumerate(zip(datasets, task_names)):
        if i == 0:
            continue

        for key in required_keys:
            if key not in ds.hf_dataset.column_names:
                raise ValueError(f"Dataset {name} missing required key: {key}")

        state_shape = ds.hf_dataset[0]["observation.state"].shape if hasattr(ds.hf_dataset[0]["observation.state"], 'shape') else (len(ds.hf_dataset[0]["observation.state"]),)
        action_shape = ds.hf_dataset[0]["action"].shape if hasattr(ds.hf_dataset[0]["action"], 'shape') else (len(ds.hf_dataset[0]["action"]),)

        if state_shape != ref_state_shape:
            raise ValueError(
                f"Schema mismatch: {name} observation.state shape {state_shape} != "
                f"reference {ref_state_shape}"
            )
        if action_shape != ref_action_shape:
            raise ValueError(
                f"Schema mismatch: {name} action shape {action_shape} != "
                f"reference {ref_action_shape}"
            )

        ds_image_keys = [k for k in ds.hf_dataset.column_names if k.startswith("observation.images.")]
        if set(ds_image_keys) != set(image_keys):
            raise ValueError(
                f"Schema mismatch: {name} image keys {ds_image_keys} != "
                f"reference {image_keys}"
            )

        print(f"  {name}: schema OK")

    print(f"  All datasets schema-compatible!")


def merge_selected_datasets(source_datasets_dict, output_dir, method, seed, k):
    """
    Merge selected episode subsets from multiple Robomme tasks.

    Args:
        source_datasets_dict: dict mapping task_name -> (dataset_path, selected_episode_indices)
        output_dir: output directory for merged dataset
        method: selection method name
        seed: selection seed
        k: episodes per task
    """
    output_dir = Path(output_dir)
    task_names = list(source_datasets_dict.keys())

    if len(task_names) != 3:
        raise ValueError(f"Expected 3 tasks, got {len(task_names)}: {task_names}")

    print(f"\n{'='*60}")
    print(f"Merging {len(task_names)} selected Robomme datasets")
    print(f"Method: {method}, Seed: {seed}, K per task: {k}")
    print(f"{'='*60}")

    subset_datasets = []
    provenance = []
    temp_root = output_dir.parent / f"{output_dir.name}_subsets"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    total_selected = 0
    for task_name in task_names:
        dataset_path, selected_episodes = source_datasets_dict[task_name]
        n_selected = len(selected_episodes)
        total_selected += n_selected

        print(f"\n--- Processing {task_name} ---")
        print(f"  Dataset path: {dataset_path}")
        print(f"  Selected episodes: {n_selected}")

        if n_selected == 0:
            raise ValueError(f"No selected episodes for task {task_name}")

        src_dataset = LeRobotDataset(repo_id=f"local/{task_name}", root=dataset_path)
        print(f"  Source dataset: {src_dataset.meta.total_episodes} total episodes")

        split_output = temp_root / task_name
        result = split_dataset(
            dataset=src_dataset,
            splits={"selected": selected_episodes},
            output_dir=split_output,
        )

        subset_dataset = result["selected"]
        subset_datasets.append(subset_dataset)
        print(f"  Subset: {subset_dataset.meta.total_episodes} episodes, {subset_dataset.meta.total_frames} frames")

        for new_ep_idx, old_ep_idx in enumerate(selected_episodes):
            provenance.append({
                "new_episode_index": new_ep_idx + sum(len(source_datasets_dict[t][1]) for t in task_names[:task_names.index(task_name)]),
                "source_task": task_name,
                "source_repo_id": f"local/{task_name}",
                "source_episode_index": old_ep_idx,
                "method": method,
                "seed": seed,
                "k": k,
            })

    expected_total = k * len(task_names)
    if total_selected != expected_total:
        raise ValueError(
            f"Total selected episodes ({total_selected}) != expected ({expected_total} = {len(task_names)} * {k})"
        )

    print(f"\n{'='*60}")
    print(f"Merging {len(subset_datasets)} subsets")
    print(f"{'='*60}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    merged_dataset = merge_datasets(
        datasets=subset_datasets,
        output_repo_id=output_dir.name,
        output_dir=output_dir,
        concatenate_videos=False,
        concatenate_data=False,
    )

    print(f"  Merged: {merged_dataset.meta.total_episodes} episodes, {merged_dataset.meta.total_frames} frames")

    print(f"\n{'='*60}")
    print(f"Recomputing stats")
    print(f"{'='*60}")

    recompute_stats(merged_dataset, skip_image_video=True)
    print(f"  Stats recomputed")

    print(f"\n{'='*60}")
    print(f"Validating merged dataset")
    print(f"{'='*60}")

    assert len(merged_dataset) == merged_dataset.meta.total_frames, "Total frames mismatch"
    print(f"  Total frames: {len(merged_dataset)} OK")

    indices = merged_dataset.hf_dataset["index"]
    assert indices == list(range(len(indices))), "Index not continuous"
    print(f"  Index continuous: OK")

    episode_indices = merged_dataset.hf_dataset["episode_index"]
    unique_ep = sorted(set(int(x) if hasattr(x, 'item') else x for x in episode_indices))
    expected_ep = list(range(merged_dataset.meta.total_episodes))
    assert unique_ep == expected_ep, f"episode_index not continuous: {unique_ep[:10]}... vs {expected_ep[:10]}..."
    print(f"  episode_index continuous: OK ({merged_dataset.meta.total_episodes} episodes)")

    for i in range(merged_dataset.meta.total_episodes):
        ep = merged_dataset.meta.episodes[i]
        ep_length = ep["length"]
        from_idx = ep["dataset_from_index"]
        to_idx = ep["dataset_to_index"]
        assert to_idx - from_idx == ep_length, f"Episode {i}: to_idx - from_idx != length"
    print(f"  Episode boundaries: OK")

    for i in range(len(merged_dataset)):
        item = merged_dataset[i]
        if 'action' in item:
            action = item['action'].numpy() if hasattr(item['action'], 'numpy') else np.array(item['action'])
            assert not np.isnan(action).any(), f"Frame {i}: action contains NaN"
            assert not np.isinf(action).any(), f"Frame {i}: action contains Inf"
    print(f"  Action values: no NaN/Inf OK")

    if merged_dataset.meta.total_episodes != expected_total:
        raise ValueError(
            f"Merged dataset has {merged_dataset.meta.total_episodes} episodes, expected {expected_total}"
        )
    print(f"  Total episodes: {merged_dataset.meta.total_episodes} == expected {expected_total} OK")

    manifest_file = output_dir / "provenance_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump({
            "method": method,
            "seed": seed,
            "k_per_task": k,
            "num_tasks": len(task_names),
            "tasks": task_names,
            "total_episodes": merged_dataset.meta.total_episodes,
            "total_frames": merged_dataset.meta.total_frames,
            "episodes": provenance,
        }, f, indent=2)
    print(f"\n  Provenance manifest saved to: {manifest_file}")

    per_task_counts = {}
    for p in provenance:
        task = p["source_task"]
        per_task_counts[task] = per_task_counts.get(task, 0) + 1

    for task in task_names:
        count = per_task_counts.get(task, 0)
        if count != k:
            raise ValueError(f"Task {task} has {count} episodes in manifest, expected {k}")
        print(f"  Task {task}: {count} episodes in manifest OK")

    if temp_root.exists():
        shutil.rmtree(temp_root)
        print(f"  Temp directory cleaned")

    print(f"\n{'='*60}")
    print(f"Merge complete!")
    print(f"  Output: {output_dir}")
    print(f"  Episodes: {merged_dataset.meta.total_episodes}")
    print(f"  Frames: {merged_dataset.meta.total_frames}")
    print(f"{'='*60}")

    return str(output_dir)


def parse_source_datasets(source_datasets_str):
    """Parse 'Task1=/path1 Task2=/path2 Task3=/path3' format."""
    result = {}
    for item in source_datasets_str:
        if "=" not in item:
            raise ValueError(f"Invalid source dataset format: {item}. Expected 'task_name=/path'")
        task_name, path = item.split("=", 1)
        task_name = task_name.strip()
        path = path.strip()
        if not Path(path).exists():
            raise FileNotFoundError(f"Source dataset path does not exist: {path}")
        result[task_name] = path
    return result


def main():
    parser = argparse.ArgumentParser(description="Merge selected Robomme episodes into a single dataset")
    parser.add_argument("--source-datasets", type=str, nargs="+", required=True,
                       help="Source datasets in format: TaskName=/path/to/dataset")
    parser.add_argument("--selected-episodes", type=str, nargs="+", required=True,
                       help="Selected episode JSON files, one per task, in same order as --source-datasets")
    parser.add_argument("--method", type=str, required=True,
                       help="Selection method name (ours_v5, random, grid_uniform, deminf, fps)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, required=True,
                       help="Episodes selected per task")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    source_paths = parse_source_datasets(args.source_datasets)
    task_names = list(source_paths.keys())

    if len(task_names) != 3:
        raise ValueError(f"Expected exactly 3 source datasets, got {len(task_names)}")

    if len(args.selected_episodes) != 3:
        raise ValueError(f"Expected exactly 3 selected episode files, got {len(args.selected_episodes)}")

    source_datasets_dict = {}
    for task_name, episodes_file in zip(task_names, args.selected_episodes):
        episodes_file = Path(episodes_file)
        if not episodes_file.exists():
            raise FileNotFoundError(f"Selected episodes file not found: {episodes_file}")
        with open(episodes_file, "r") as f:
            data = json.load(f)
        selected_indices = data["selected_episode_indices"]
        source_datasets_dict[task_name] = (source_paths[task_name], selected_indices)
        print(f"Task {task_name}: {len(selected_indices)} episodes selected from {source_paths[task_name]}")

    for task_name, (_, selected) in source_datasets_dict.items():
        if len(selected) != args.k:
            raise ValueError(
                f"Task {task_name} has {len(selected)} selected episodes, expected K={args.k}"
            )

    validate_schema_compatibility(
        [LeRobotDataset(repo_id=f"local/{t}", root=source_datasets_dict[t][0]) for t in task_names],
        task_names
    )

    merge_selected_datasets(
        source_datasets_dict=source_datasets_dict,
        output_dir=args.output_dir,
        method=args.method,
        seed=args.seed,
        k=args.k,
    )


if __name__ == "__main__":
    main()