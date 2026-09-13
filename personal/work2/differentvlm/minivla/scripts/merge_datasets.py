#!/usr/bin/env python3
"""
Merge multiple LeRobot datasets into a single dataset.

This script:
1. Selects N episodes from each source dataset (random or uniform selection)
2. Re-indexes episodes to create a continuous episode index
3. Copies and merges all data files (parquet, videos, meta)
4. Updates stats.json with combined statistics

Usage:
    python merge_datasets.py \
        --datasets disassemble-v3_corner pick_place-v3_corner coffee-button-v3_corner \
        --episodes-per-dataset 112 \
        --output-dir personal/work2/dataset_view/merged_dataset \
        --selection-mode random \
        --seed 42
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset names to merge (e.g., disassemble-v3_corner pick_place-v3_corner)"
    )
    parser.add_argument(
        "--episodes-per-dataset",
        type=int,
        default=112,
        help="Number of episodes to select from each dataset (default: 112)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for merged dataset"
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=["random", "uniform"],
        default="random",
        help="Episode selection mode (default: random)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selection (default: 42)"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="personal/work2/dataset_view",
        help="Root directory containing source datasets (default: personal/work2/dataset_view)"
    )
    return parser.parse_args()


def get_episode_count(dataset_path: Path) -> int:
    """Get the number of episodes in a dataset from info.json."""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found at {info_path}")
    
    with open(info_path, "r") as f:
        info = json.load(f)
    
    return info.get("total_episodes", 0)


def select_episodes(dataset_path: Path, n_episodes: int, mode: str, seed: int) -> list[int]:
    """Select episode indices from a dataset."""
    total_episodes = get_episode_count(dataset_path)
    if n_episodes > total_episodes:
        print(f"  Warning: Requested {n_episodes} episodes but dataset only has {total_episodes}")
        n_episodes = total_episodes
    
    if mode == "random":
        rng = np.random.RandomState(seed)
        selected = rng.choice(total_episodes, size=n_episodes, replace=False)
        selected = sorted(selected.tolist())
    elif mode == "uniform":
        indices = np.linspace(0, total_episodes - 1, n_episodes, dtype=int)
        selected = indices.tolist()
    else:
        raise ValueError(f"Unknown selection mode: {mode}")
    
    return selected


def copy_parquet_data(
    src_dataset: Path,
    selected_episodes: list[int],
    episode_offset: int,
    chunk_id: int = 0,
    task_name_to_index: dict = None,
    src_episodes_meta: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Copy parquet data files with re-indexed episodes.
    Returns the filtered DataFrame (not writing to disk yet).
    """
    src_chunk_dir = src_dataset / "data" / f"chunk-{chunk_id:03d}"
    
    # Read all parquet files from source
    src_files = sorted(src_chunk_dir.glob("file-*.parquet"))
    if not src_files:
        print(f"  Warning: No parquet files found in {src_chunk_dir}")
        return pd.DataFrame()
    
    # Read and concatenate all source data
    all_data = []
    for src_file in src_files:
        df = pd.read_parquet(src_file)
        all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Filter to selected episodes
    mask = combined_df["episode_index"].isin(selected_episodes)
    filtered_df = combined_df[mask].copy()
    
    if len(filtered_df) == 0:
        print(f"  Warning: No data found for selected episodes")
        return pd.DataFrame()
    
    # Re-index episodes
    episode_mapping = {old_idx: new_idx + episode_offset for new_idx, old_idx in enumerate(selected_episodes)}
    filtered_df["episode_index"] = filtered_df["episode_index"].map(episode_mapping)
    
    # Sort by episode_index and frame_index to ensure consistent order
    if "frame_index" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    else:
        filtered_df = filtered_df.sort_values(["episode_index"]).reset_index(drop=True)
    
    # Update task_index to global mapping
    if task_name_to_index is not None and src_episodes_meta is not None and "task_index" in filtered_df.columns:
        # Build a mapping from old episode_index to task_name
        episode_to_task = {}
        if "tasks" in src_episodes_meta.columns:
            for _, row in src_episodes_meta.iterrows():
                ep_idx = row["episode_index"]
                task_val = row["tasks"]
                if hasattr(task_val, '__len__') and len(task_val) > 0:
                    task_name = str(task_val[0])
                elif isinstance(task_val, str):
                    task_name = task_val
                else:
                    task_name = str(task_val)
                episode_to_task[ep_idx] = task_name
        
        # Update task_index for each frame based on its episode
        updated_count = 0
        for old_ep_idx, new_ep_idx in episode_mapping.items():
            if old_ep_idx in episode_to_task:
                task_name = episode_to_task[old_ep_idx]
                if task_name in task_name_to_index:
                    global_task_idx = task_name_to_index[task_name]
                    # Update all frames belonging to this episode
                    ep_mask = filtered_df["episode_index"] == new_ep_idx
                    filtered_df.loc[ep_mask, "task_index"] = global_task_idx
                    updated_count += 1
        
        print(f"  Updated task_index for {updated_count} episodes")
    
    print(f"  Loaded {len(filtered_df)} rows from {len(selected_episodes)} episodes")
    
    return filtered_df


def copy_episode_meta(
    src_dataset: Path,
    selected_episodes: list[int],
    episode_offset: int,
    chunk_id: int = 0,
    video_file_index_mapping: dict = None,
    global_frame_offset: int = 0,
    src_episodes_meta: pd.DataFrame = None
) -> pd.DataFrame:
    """Copy episode metadata with re-indexed episodes. Returns DataFrame."""
    src_meta_dir = src_dataset / "meta" / "episodes" / f"chunk-{chunk_id:03d}"
    
    src_files = sorted(src_meta_dir.glob("file-*.parquet"))
    if not src_files:
        print(f"  Warning: No episode meta files found in {src_meta_dir}")
        return pd.DataFrame()
    
    # Read all episode meta
    all_meta = []
    for src_file in src_files:
        df = pd.read_parquet(src_file)
        all_meta.append(df)
    
    if not all_meta:
        return pd.DataFrame()
    
    combined_meta = pd.concat(all_meta, ignore_index=True)
    
    # Filter to selected episodes
    mask = combined_meta["episode_index"].isin(selected_episodes)
    filtered_meta = combined_meta[mask].copy()
    
    if len(filtered_meta) == 0:
        return pd.DataFrame()
    
    # Re-index episodes
    episode_mapping = {old_idx: new_idx + episode_offset for new_idx, old_idx in enumerate(selected_episodes)}
    filtered_meta["episode_index"] = filtered_meta["episode_index"].map(episode_mapping)
    
    # Update video file_index based on mapping
    if video_file_index_mapping:
        for video_key, offset in video_file_index_mapping.items():
            file_index_col = f"videos/{video_key}/file_index"
            if file_index_col in filtered_meta.columns:
                filtered_meta[file_index_col] = filtered_meta[file_index_col] + offset
    
    # Update dataset_from_index and dataset_to_index to global frame indices
    # LeRobot uses EXCLUSIVE boundary: to_index = from_index + length
    # Episode 0: frames [0, 1, ..., length-1], to_index = length (start of next episode)
    if "dataset_from_index" in filtered_meta.columns and "dataset_to_index" in filtered_meta.columns:
        # Sort by episode_index to ensure correct order
        filtered_meta = filtered_meta.sort_values("episode_index").reset_index(drop=True)
        
        # Build new from/to indices based on global_frame_offset and cumulative lengths
        new_from_indices = []
        new_to_indices = []
        current_offset = global_frame_offset
        for _, row in filtered_meta.iterrows():
            ep_length = int(row["length"])
            from_idx = current_offset
            to_idx = current_offset + ep_length  # EXCLUSIVE boundary
            new_from_indices.append(from_idx)
            new_to_indices.append(to_idx)
            current_offset = to_idx  # Next episode starts where this one ends
        
        filtered_meta["dataset_from_index"] = new_from_indices
        filtered_meta["dataset_to_index"] = new_to_indices
    print(f"  Loaded episode meta for {len(filtered_meta)} episodes")
    
    return filtered_meta


def copy_videos(
    src_dataset: Path,
    dst_dataset: Path,
    selected_episodes: list[int],
    episode_offset: int,
    video_keys: list[str],
    chunk_id: int = 0,
    video_file_offset: dict = None
) -> dict:
    """
    Copy video files with re-indexed to avoid conflicts.
    Returns a mapping from old file_index to new file_index for each video key.
    """
    file_index_mapping = {}
    
    for video_key in video_keys:
        src_video_dir = src_dataset / "videos" / video_key / f"chunk-{chunk_id:03d}"
        dst_video_dir = dst_dataset / "videos" / video_key / f"chunk-{chunk_id:03d}"
        dst_video_dir.mkdir(parents=True, exist_ok=True)
        
        if not src_video_dir.exists():
            print(f"  Warning: Video directory not found: {src_video_dir}")
            continue
        
        # Get current max file_index in destination
        dst_files = sorted(dst_video_dir.glob("file-*.mp4"))
        current_offset = len(dst_files)
        file_index_mapping[video_key] = current_offset
        
        # Copy video files with new indices
        src_files = sorted(src_video_dir.glob("file-*.mp4"))
        for src_file in src_files:
            old_idx = int(src_file.stem.split("-")[1])
            new_idx = current_offset + old_idx
            dst_file = dst_video_dir / f"file-{new_idx:03d}.mp4"
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
        
        print(f"  Copied {len(src_files)} video files for {video_key} (offset: {current_offset})")
    
    return file_index_mapping


def copy_tasks(src_datasets: list[Path], dst_dataset: Path):
    """Merge tasks.parquet from all datasets."""
    all_tasks = []
    task_name_to_index = {}
    task_index = 0
    
    for src_dataset in src_datasets:
        src_tasks = src_dataset / "meta" / "tasks.parquet"
        if src_tasks.exists():
            tasks_df = pd.read_parquet(src_tasks)
            for task_name in tasks_df.index:
                if task_name not in task_name_to_index:
                    task_name_to_index[task_name] = task_index
                    all_tasks.append({"task": task_name, "task_index": task_index})
                    task_index += 1
    
    if all_tasks:
        merged_tasks_df = pd.DataFrame(all_tasks)
        merged_tasks_df = merged_tasks_df.set_index("task")
        dst_tasks = dst_dataset / "meta" / "tasks.parquet"
        merged_tasks_df.to_parquet(dst_tasks)
        print(f"  Merged {len(all_tasks)} tasks: {list(task_name_to_index.keys())}")
    
    return task_name_to_index


def copy_episode_initial_states(
    src_dataset: Path,
    dst_dataset: Path,
    selected_episodes: list[int],
    episode_offset: int
):
    """Copy and re-index episode initial states."""
    src_states = src_dataset / "episode_initial_states.json"
    dst_states = dst_dataset / "episode_initial_states.json"
    
    if not src_states.exists():
        return
    
    with open(src_states, "r") as f:
        states_data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(states_data, list):
        states = states_data
    elif isinstance(states_data, dict):
        states = states_data.get("episodes", [])
    else:
        print(f"  Warning: Unexpected format in episode_initial_states.json")
        return
    
    # Filter and re-index
    new_states = []
    for state in states:
        if isinstance(state, dict):
            old_idx = state.get("episode_index", -1)
        else:
            # If state is not a dict, skip or handle differently
            continue
        
        if old_idx in selected_episodes:
            new_state = state.copy()
            new_state["episode_index"] = episode_offset + selected_episodes.index(old_idx)
            new_states.append(new_state)
    
    # Append to existing states if file exists
    if dst_states.exists():
        with open(dst_states, "r") as f:
            existing_data = json.load(f)
        
        # Handle both formats
        if isinstance(existing_data, dict):
            existing_states = existing_data.get("episodes", [])
            existing_states.extend(new_states)
            existing_data["episodes"] = existing_states
            existing_data["num_episodes"] = len(existing_states)
            new_data = existing_data
        elif isinstance(existing_data, list):
            existing_data.extend(new_states)
            new_data = existing_data
        else:
            new_data = {"episodes": new_states}
    else:
        # Create new file with same structure as source
        if isinstance(states_data, dict):
            new_data = states_data.copy()
            new_data["episodes"] = new_states
            new_data["num_episodes"] = len(new_states)
        else:
            new_data = new_states
    
    with open(dst_states, "w") as f:
        json.dump(new_data, f, indent=2)
    
    print(f"  Copied {len(new_states)} episode initial states")


def merge_info_json(
    src_datasets: list[Path],
    dst_dataset: Path,
    total_episodes: int,
    total_frames: int
):
    """Create merged info.json."""
    # Use first dataset's info as template
    src_info = src_datasets[0] / "meta" / "info.json"
    with open(src_info, "r") as f:
        info = json.load(f)
    
    # Update episode and frame counts
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_chunks"] = 1
    info["total_videos"] = total_episodes  # Approximate
    
    dst_info = dst_dataset / "meta" / "info.json"
    with open(dst_info, "w") as f:
        json.dump(info, f, indent=4)
    
    print(f"  Created info.json with {total_episodes} episodes, {total_frames} frames")


def merge_stats(src_datasets: list[Path], dst_dataset: Path, episodes_per_dataset: list[int]):
    """
    Merge stats.json from multiple datasets.
    For video features, use first dataset's stats.
    For action/state features, compute weighted average.
    """
    # Start with first dataset's stats
    src_stats = src_datasets[0] / "meta" / "stats.json"
    with open(src_stats, "r") as f:
        stats = json.load(f)
    
    # For subsequent datasets, merge statistics
    total_episodes = sum(episodes_per_dataset)
    
    for i, (src_dataset, n_eps) in enumerate(zip(src_datasets[1:], episodes_per_dataset[1:]), start=1):
        src_stats_file = src_dataset / "meta" / "stats.json"
        if not src_stats_file.exists():
            continue
        
        with open(src_stats_file, "r") as f:
            other_stats = json.load(f)
        
        # Weight for this dataset
        weight = n_eps / total_episodes
        first_weight = (total_episodes - n_eps) / total_episodes
        
        # Merge statistics for each feature
        for feature_name, feature_stats in other_stats.items():
            if feature_name not in stats:
                # New feature, add with weight
                stats[feature_name] = feature_stats
                continue
            
            # Merge q01, q99, mean, std for numeric features
            for stat_key in ["q01", "q99", "q10", "q50", "q90", "mean", "std", "min", "max"]:
                if stat_key in feature_stats and stat_key in stats[feature_name]:
                    old_val = stats[feature_name][stat_key]
                    new_val = feature_stats[stat_key]
                    
                    # Handle both list and scalar values
                    if isinstance(old_val, list) and isinstance(new_val, list):
                        if len(old_val) == len(new_val):
                            merged = []
                            for o, n in zip(old_val, new_val):
                                # Handle nested lists recursively
                                while isinstance(o, list):
                                    o = o[0] if o else 0.0
                                while isinstance(n, list):
                                    n = n[0] if n else 0.0
                                merged.append(first_weight * float(o) + weight * float(n))
                            stats[feature_name][stat_key] = merged
                    elif isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                        stats[feature_name][stat_key] = first_weight * old_val + weight * new_val
    
    dst_stats = dst_dataset / "meta" / "stats.json"
    with open(dst_stats, "w") as f:
        json.dump(stats, f, indent=4)
    
    print(f"  Merged stats.json from {len(src_datasets)} datasets")


def get_video_keys(dataset_path: Path) -> list[str]:
    """Get video feature keys from info.json."""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    
    video_keys = []
    for feature_name, feature_info in info.get("features", {}).items():
        if feature_info.get("dtype") == "video":
            video_keys.append(feature_name)
    
    return video_keys


def main():
    args = parse_args()
    
    # Validate datasets
    dataset_root = Path(args.dataset_root)
    src_datasets = []
    
    for dataset_name in args.datasets:
        dataset_path = dataset_root / dataset_name
        if not dataset_path.exists():
            print(f"Error: Dataset not found: {dataset_path}")
            sys.exit(1)
        
        episode_count = get_episode_count(dataset_path)
        print(f"Dataset: {dataset_name}")
        print(f"  Path: {dataset_path}")
        print(f"  Total episodes: {episode_count}")
        src_datasets.append(dataset_path)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "meta").mkdir(exist_ok=True)
    (output_dir / "videos").mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Selection mode: {args.selection_mode}")
    print(f"Episodes per dataset: {args.episodes_per_dataset}")
    print(f"Random seed: {args.seed}")
    
    # Select episodes from each dataset
    print("\n--- Selecting Episodes ---")
    all_selected_episodes = []
    episodes_per_dataset = []
    
    for i, src_dataset in enumerate(src_datasets):
        dataset_name = src_dataset.name
        print(f"\nDataset {i+1}: {dataset_name}")
        
        selected = select_episodes(
            src_dataset,
            args.episodes_per_dataset,
            args.selection_mode,
            args.seed + i  # Different seed for each dataset
        )
        
        all_selected_episodes.append(selected)
        episodes_per_dataset.append(len(selected))
        print(f"  Selected {len(selected)} episodes")
    
    # Calculate total episodes
    total_episodes = sum(episodes_per_dataset)
    print(f"\nTotal episodes in merged dataset: {total_episodes}")
    
    # Build global task mapping before copying data
    print("\n--- Building Global Task Mapping ---")
    task_name_to_index = {}
    task_index = 0
    for src_dataset in src_datasets:
        src_tasks = src_dataset / "meta" / "tasks.parquet"
        if src_tasks.exists():
            tasks_df = pd.read_parquet(src_tasks)
            for task_name in tasks_df.index:
                if task_name not in task_name_to_index:
                    task_name_to_index[task_name] = task_index
                    task_index += 1
    print(f"Global tasks: {task_name_to_index}")
    
    # Load episodes metadata for each dataset (needed for task_index mapping)
    dataset_episodes_meta = {}
    for src_dataset in src_datasets:
        src_meta_file = src_dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        if src_meta_file.exists():
            dataset_episodes_meta[src_dataset.name] = pd.read_parquet(src_meta_file)
    
    # Copy and merge data
    print("\n--- Merging Data ---")
    episode_offset = 0
    total_frames = 0
    global_frame_offset = 0  # Track global frame index for dataset_from/to_index
    all_data_frames = []  # Collect all data frames to write once at the end
    all_episode_meta_frames = []  # Collect all episode meta frames to write once at the end
    
    for i, (src_dataset, selected_episodes) in enumerate(zip(src_datasets, all_selected_episodes)):
        dataset_name = src_dataset.name
        print(f"\nProcessing dataset {i+1}: {dataset_name}")
        print(f"  Episodes: {selected_episodes[:5]}... ({len(selected_episodes)} total)")
        print(f"  Episode offset: {episode_offset}")
        print(f"  Global frame offset: {global_frame_offset}")
        
        # Get video keys
        video_keys = get_video_keys(src_dataset)
        print(f"  Video keys: {video_keys}")
        
        # Copy videos first (to get file_index mapping)
        video_file_index_mapping = copy_videos(
            src_dataset,
            output_dir,
            selected_episodes,
            episode_offset,
            video_keys
        )
        
        # Load parquet data with task_index mapping (don't write yet)
        df = copy_parquet_data(
            src_dataset,
            selected_episodes,
            episode_offset,
            task_name_to_index=task_name_to_index,
            src_episodes_meta=dataset_episodes_meta.get(dataset_name)
        )
        if not df.empty:
            all_data_frames.append(df)
            total_frames += len(df)
        
        # Load episode metadata with video file_index mapping (don't write yet)
        ep_meta = copy_episode_meta(
            src_dataset,
            selected_episodes,
            episode_offset,
            video_file_index_mapping=video_file_index_mapping,
            global_frame_offset=global_frame_offset
        )
        if not ep_meta.empty:
            all_episode_meta_frames.append(ep_meta)
            # Update global_frame_offset for next dataset
            global_frame_offset += len(df)
        
        
        # Copy tasks (merge all datasets at the end)
        # This will be done after the loop
        
        # Copy episode initial states
        copy_episode_initial_states(
            src_dataset,
            output_dir,
            selected_episodes,
            episode_offset
        )
        
        episode_offset += len(selected_episodes)
    
    # Write all merged data to parquet file (once, after collecting all datasets)
    print("\n--- Writing Merged Data ---")
    if all_data_frames:
        dst_chunk_dir = output_dir / "data" / "chunk-000"
        dst_chunk_dir.mkdir(parents=True, exist_ok=True)
        merged_df = pd.concat(all_data_frames, ignore_index=True)
        dst_file = dst_chunk_dir / "file-000.parquet"
        merged_df.to_parquet(dst_file, index=False)
        print(f"  Saved {len(merged_df)} total rows to {dst_file}")
        
        # Verify task_index distribution
        print(f"  Task index distribution:")
        print(f"    {merged_df['task_index'].value_counts().sort_index().to_dict()}")
    
    # Write all merged episode metadata
    if all_episode_meta_frames:
        dst_meta_dir = output_dir / "meta" / "episodes" / "chunk-000"
        dst_meta_dir.mkdir(parents=True, exist_ok=True)
        merged_ep_meta = pd.concat(all_episode_meta_frames, ignore_index=True)
        dst_meta_file = dst_meta_dir / "file-000.parquet"
        merged_ep_meta.to_parquet(dst_meta_file, index=False)
        print(f"  Saved {len(merged_ep_meta)} episode metadata rows to {dst_meta_file}")
    
    # Create merged tasks.parquet from all datasets
    print("\n--- Creating Metadata ---")
    copy_tasks(src_datasets, output_dir)
    
    # Create merged info.json
    merge_info_json(src_datasets, output_dir, total_episodes, total_frames)
    
    # Merge stats
    merge_stats(src_datasets, output_dir, episodes_per_dataset)
    
    # Create subset file for training
    subset_file = output_dir / "subset_file.json"
    subset_data = {
        "episodes": list(range(total_episodes)),
        "total_episodes": total_episodes,
        "source_datasets": [ds.name for ds in src_datasets],
        "episodes_per_dataset": episodes_per_dataset,
        "selection_mode": args.selection_mode,
        "seed": args.seed
    }
    with open(subset_file, "w") as f:
        json.dump(subset_data, f, indent=2)
    
    print(f"\n--- Merge Complete ---")
    print(f"Output: {output_dir}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames: {total_frames}")
    print(f"Source datasets: {[ds.name for ds in src_datasets]}")
    print(f"Episodes per dataset: {episodes_per_dataset}")
    print(f"\nTo use this merged dataset:")
    print(f"  --dataset merged_dataset")
    print(f"  --num-episodes {total_episodes}")
    print(f"  (add to minivla_config.py if needed)")


if __name__ == "__main__":
    main()