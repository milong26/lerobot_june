#!/usr/bin/env python3
"""
Merge multiple LeRobot datasets into a single dataset.
This script copies selected episodes from multiple datasets into a new combined dataset.
"""

import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def load_dataset_info(dataset_dir: Path) -> dict:
    """Load dataset metadata."""
    with open(dataset_dir / "meta" / "info.json", "r") as f:
        return json.load(f)


def load_episodes_meta(dataset_dir: Path) -> pd.DataFrame:
    """Load episodes metadata."""
    episodes_dir = dataset_dir / "meta" / "episodes"
    all_eps = []
    for chunk_dir in sorted(episodes_dir.glob("chunk-*")):
        for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
            df = pd.read_parquet(parquet_file)
            all_eps.append(df)
    return pd.concat(all_eps, ignore_index=True) if all_eps else pd.DataFrame()


def get_episode_ranges(episodes_meta: pd.DataFrame) -> dict:
    """Get episode index to data range mapping."""
    episode_ranges = {}
    for _, row in episodes_meta.iterrows():
        ep_idx = row["episode_index"]
        episode_ranges[ep_idx] = {
            "data_start": row["data_start_index"],
            "data_end": row["data_end_index"],
        }
    return episode_ranges


def copy_episode_data(
    src_dataset_dir: Path,
    dst_dataset_dir: Path,
    episode_idx: int,
    episode_ranges: dict,
    data_offset: int,
    video_offset: int,
    camera_names: list[str],
) -> tuple[int, int, int]:
    """Copy a single episode's data and videos to the new dataset."""
    
    # Get data range for this episode
    ep_range = episode_ranges[episode_idx]
    data_start = ep_range["data_start"]
    data_end = ep_range["data_end"]
    num_frames = data_end - data_start + 1
    
    # Copy data parquet files
    src_data_dir = src_dataset_dir / "data"
    dst_data_dir = dst_dataset_dir / "data"
    
    # Find which chunk files contain this episode's data
    for chunk_dir in sorted(src_data_dir.glob("chunk-*")):
        for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
            df = pd.read_parquet(parquet_file)
            # Filter rows for this episode
            episode_mask = (df["frame_index"] >= data_start) & (df["frame_index"] <= data_end)
            if episode_mask.any():
                episode_data = df[episode_mask].copy()
                # Update frame indices
                episode_data["frame_index"] = episode_data["frame_index"] - data_start + data_offset
                episode_data["episode_index"] = 0  # Will be updated later
                
                # Save to destination
                dst_chunk_dir = dst_data_dir / "chunk-000"
                dst_chunk_dir.mkdir(parents=True, exist_ok=True)
                dst_parquet = dst_chunk_dir / f"file-{0:05d}.parquet"
                
                # Append or create
                if dst_parquet.exists():
                    existing_df = pd.read_parquet(dst_parquet)
                    combined_df = pd.concat([existing_df, episode_data], ignore_index=True)
                else:
                    combined_df = episode_data
                
                combined_df.to_parquet(dst_parquet, index=False)
    
    # Copy video files
    src_videos_dir = src_dataset_dir / "videos"
    dst_videos_dir = dst_dataset_dir / "videos"
    
    for camera_name in camera_names:
        src_video_dir = src_videos_dir / f"observation.images.{camera_name}"
        dst_video_dir = dst_videos_dir / f"observation.images.{camera_name}"
        dst_video_dir.mkdir(parents=True, exist_ok=True)
        
        # Find and copy video files for this episode
        for chunk_dir in sorted(src_video_dir.glob("chunk-*")):
            for video_file in sorted(chunk_dir.glob("file-*.mp4")):
                # Extract episode index from filename or use sequential numbering
                dst_video_file = dst_video_dir / "chunk-000" / f"file-{video_offset:05d}.mp4"
                dst_video_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video_file, dst_video_file)
                video_offset += 1
    
    new_data_offset = data_offset + num_frames
    return new_data_offset, video_offset, num_frames


def merge_datasets(
    dataset_configs: list[dict],
    output_dir: Path,
    camera_names: list[str] = ["top", "wrist"],
):
    """
    Merge multiple datasets.
    
    Args:
        dataset_configs: List of dicts with keys:
            - dataset_dir: Path to source dataset
            - episode_indices: List of episode indices to include
        output_dir: Path to output merged dataset
        camera_names: List of camera names to copy videos for
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all source datasets info
    all_episodes_data = []
    all_episodes_meta = []
    camera_names_all = set()
    
    for config in dataset_configs:
        dataset_dir = Path(config["dataset_dir"])
        episode_indices = config["episode_indices"]
        
        print(f"\nLoading dataset: {dataset_dir.name}")
        print(f"  Episodes to select: {len(episode_indices)}")
        
        # Load info
        info = load_dataset_info(dataset_dir)
        if not camera_names_all:
            camera_names_all = set(info.get("camera_keys", []))
            # Extract camera names from keys like "observation.images.top"
            camera_names_all = {k.replace("observation.images.", "") for k in camera_names_all if "observation.images." in k}
        
        # Load episodes metadata
        episodes_meta = load_episodes_meta(dataset_dir)
        episode_ranges = get_episode_ranges(episodes_meta)
        
        # Collect episode data
        for ep_idx in episode_indices:
            if ep_idx not in episode_ranges:
                print(f"  WARNING: Episode {ep_idx} not found in {dataset_dir.name}, skipping")
                continue
            
            all_episodes_data.append({
                "dataset_dir": dataset_dir,
                "episode_idx": ep_idx,
                "episode_range": episode_ranges[ep_idx],
            })
    
    print(f"\nTotal episodes to merge: {len(all_episodes_data)}")
    
    # Create output directory structure
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos").mkdir(parents=True, exist_ok=True)
    
    # Copy info.json from first dataset and update
    first_dataset = Path(dataset_configs[0]["dataset_dir"])
    info = load_dataset_info(first_dataset)
    
    # Update info for merged dataset
    total_episodes = len(all_episodes_data)
    info["total_episodes"] = total_episodes
    info["total_frames"] = 0  # Will be updated
    info["total_videos"] = 0  # Will be updated
    
    # Merge data
    all_data_frames = []
    episode_meta_rows = []
    data_offset = 0
    video_offset = 0
    
    print("\nMerging episodes...")
    for ep_global_idx, ep_data in enumerate(tqdm(all_episodes_data)):
        src_dataset_dir = ep_data["dataset_dir"]
        ep_idx = ep_data["episode_idx"]
        ep_range = ep_data["episode_range"]
        
        # Read source data
        src_data_dir = src_dataset_dir / "data"
        ep_frames = []
        
        for chunk_dir in sorted(src_data_dir.glob("chunk-*")):
            for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
                df = pd.read_parquet(parquet_file)
                data_start = ep_range["data_start"]
                data_end = ep_range["data_end"]
                episode_mask = (df["frame_index"] >= data_start) & (df["frame_index"] <= data_end)
                if episode_mask.any():
                    episode_data = df[episode_mask].copy()
                    ep_frames.append(episode_data)
        
        if not ep_frames:
            print(f"  WARNING: No data found for episode {ep_idx} in {src_dataset_dir.name}")
            continue
        
        # Concatenate and update indices
        episode_df = pd.concat(ep_frames, ignore_index=True)
        num_frames = len(episode_df)
        
        # Update frame indices
        episode_df["frame_index"] = episode_df["frame_index"] - ep_range["data_start"] + data_offset
        episode_df["episode_index"] = ep_global_idx
        
        all_data_frames.append(episode_df)
        
        # Create episode metadata
        episode_meta_rows.append({
            "episode_index": ep_global_idx,
            "data_start_index": data_offset,
            "data_end_index": data_offset + num_frames - 1,
            "tasks": episode_df["task_index"].iloc[0] if "task_index" in episode_df.columns else 0,
        })
        
        data_offset += num_frames
        video_offset += 1
    
    # Save merged data
    print("\nSaving merged data...")
    if all_data_frames:
        merged_data = pd.concat(all_data_frames, ignore_index=True)
        merged_data.to_parquet(output_dir / "data" / "chunk-000" / "file-000.parquet", index=False)
        print(f"  Total frames: {len(merged_data)}")
    
    # Save episodes metadata
    if episode_meta_rows:
        episodes_df = pd.DataFrame(episode_meta_rows)
        episodes_df.to_parquet(output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    
    # Copy videos (simplified: just copy all video files and rename)
    print("\nCopying videos...")
    video_file_idx = 0
    for ep_data in tqdm(all_episodes_data):
        src_dataset_dir = ep_data["dataset_dir"]
        src_videos_dir = src_dataset_dir / "videos"
        
        for camera_name in camera_names_all:
            src_video_dir = src_videos_dir / f"observation.images.{camera_name}"
            dst_video_dir = output_dir / "videos" / f"observation.images.{camera_name}" / "chunk-000"
            dst_video_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy video file
            src_video_file = src_video_dir / "chunk-000" / f"file-{video_file_idx:05d}.mp4"
            if src_video_file.exists():
                dst_video_file = dst_video_dir / f"file-{video_file_idx:05d}.mp4"
                shutil.copy2(src_video_file, dst_video_file)
        
        video_file_idx += 1
    
    # Update and save info.json
    info["total_frames"] = data_offset
    info["total_videos"] = video_file_idx
    info["total_episodes"] = len(episode_meta_rows)
    
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    # Copy stats.json and tasks.parquet
    if (first_dataset / "meta" / "stats.json").exists():
        shutil.copy2(first_dataset / "meta" / "stats.json", output_dir / "meta" / "stats.json")
    
    if (first_dataset / "meta" / "tasks.parquet").exists():
        shutil.copy2(first_dataset / "meta" / "tasks.parquet", output_dir / "meta" / "tasks.parquet")
    
    print(f"\nMerged dataset saved to: {output_dir}")
    print(f"  Total episodes: {len(episode_meta_rows)}")
    print(f"  Total frames: {data_offset}")
    print(f"  Total videos: {video_file_idx}")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot datasets")
    parser.add_argument("--subset-file", type=str, required=True,
                        help="Path to merged subset JSON file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to output merged dataset directory")
    parser.add_argument("--dataset-base-dir", type=str,
                        default="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view",
                        help="Base directory containing all datasets")
    args = parser.parse_args()
    
    # Load subset file
    with open(args.subset_file, "r") as f:
        subset_data = json.load(f)
    
    datasets = subset_data["datasets"]
    episodes_per_dataset = subset_data["episodes_per_dataset"]
    all_indices = subset_data["selected_episode_indices"]
    
    # Split indices by dataset
    dataset_configs = []
    for i, dataset_name in enumerate(datasets):
        start_idx = i * episodes_per_dataset
        end_idx = start_idx + episodes_per_dataset
        episode_indices = all_indices[start_idx:end_idx]
        
        dataset_configs.append({
            "dataset_dir": f"{args.dataset_base_dir}/{dataset_name}",
            "episode_indices": episode_indices,
        })
    
    # Merge datasets
    merge_datasets(
        dataset_configs=dataset_configs,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()