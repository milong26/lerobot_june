#!/usr/bin/env python3
"""
Merge selected episodes from multiple LeRobot datasets into a single dataset.
This creates a new dataset directory containing only the selected episodes.
Uses symlinks for videos to save disk space and time.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


def merge_datasets_for_training(
    subset_file: str,
    output_dir: str,
    dataset_base_dir: str = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view",
):
    """
    Merge selected episodes from multiple datasets into a single training dataset.
    
    Args:
        subset_file: Path to the merged subset JSON file
        output_dir: Path to output merged dataset
        dataset_base_dir: Base directory containing source datasets
    """
    
    # Load subset data
    with open(subset_file, "r") as f:
        subset_data = json.load(f)
    
    datasets = subset_data["datasets"]
    episodes_per_dataset = subset_data["episodes_per_dataset"]
    all_indices = subset_data["selected_episode_indices"]
    
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Merging {len(datasets)} datasets")
    print(f"Episodes per dataset: {episodes_per_dataset}")
    print(f"Total episodes: {len(all_indices)}")
    
    # Collect all episode data
    all_episode_data = []  # List of (dataset_name, episode_idx, dataframe)
    
    # Load episodes from each dataset
    for i, dataset_name in enumerate(datasets):
        dataset_dir = Path(dataset_base_dir) / dataset_name
        episode_indices = all_indices[i * episodes_per_dataset : (i + 1) * episodes_per_dataset]
        
        print(f"\nLoading {len(episode_indices)} episodes from {dataset_name}...")
        
        # Load episodes metadata
        episodes_meta_file = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        if not episodes_meta_file.exists():
            print(f"  ERROR: Episodes metadata file not found: {episodes_meta_file}")
            continue
        episodes_meta = pd.read_parquet(episodes_meta_file)
        
        # Load data
        data_file = dataset_dir / "data" / "chunk-000" / "file-000.parquet"
        if not data_file.exists():
            print(f"  ERROR: Data file not found: {data_file}")
            continue
        data_df = pd.read_parquet(data_file)
        
        for ep_idx in tqdm(episode_indices, desc=f"  {dataset_name}"):
            # Extract episode data by filtering on episode_index
            ep_data = data_df[data_df["episode_index"] == ep_idx].copy()
            
            if len(ep_data) == 0:
                print(f"  WARNING: Episode {ep_idx} not found in {dataset_name}")
                continue
            
            all_episode_data.append((dataset_name, ep_idx, ep_data))
    
    print(f"\nTotal episodes loaded: {len(all_episode_data)}")
    if not all_episode_data:
        print("ERROR: No episodes loaded!")
        return
    
    total_frames = sum(len(ep_data) for _, _, ep_data in all_episode_data)
    print(f"Total frames: {total_frames}")
    
    # Create merged dataset
    print("\nCreating merged dataset...")
    
    # Create directory structure
    (output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_path / "videos").mkdir(parents=True, exist_ok=True)
    
    # Merge all data with updated indices
    merged_frames = []
    episode_meta_rows = []
    current_frame_idx = 0
    current_video_idx = 0
    
    print("\nMerging episode data...")
    for global_ep_idx, (dataset_name, orig_ep_idx, ep_data) in enumerate(
        tqdm(all_episode_data, desc="Merging episodes")
    ):
        num_frames = len(ep_data)
        
        # Update indices
        ep_data["frame_index"] = list(range(current_frame_idx, current_frame_idx + num_frames))
        ep_data["episode_index"] = global_ep_idx
        
        merged_frames.append(ep_data)
        
        # Create episode metadata matching original format
        # Get task string from episodes_meta if available
        task_str = ""
        if "tasks" in episodes_meta.columns:
            task_row = episodes_meta[episodes_meta["episode_index"] == ep_idx]
            if len(task_row) > 0 and "tasks" in task_row.columns:
                task_val = task_row["tasks"].iloc[0]
                if isinstance(task_val, list):
                    task_str = task_val[0] if len(task_val) > 0 else ""
                else:
                    task_str = str(task_val)
        
        episode_meta_rows.append({
            "episode_index": global_ep_idx,
            "tasks": [task_str] if task_str else [],
            "length": num_frames,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 0,
            "dataset_to_index": num_frames - 1,
            "videos/observation.images.top/chunk_index": 0,
            "videos/observation.images.top/file_index": 0,
            "videos/observation.images.top/from_timestamp": 0.0,
            "videos/observation.images.top/to_timestamp": float(num_frames - 1) / 80.0,
            "videos/observation.images.wrist/chunk_index": 0,
            "videos/observation.images.wrist/file_index": 0,
            "videos/observation.images.wrist/from_timestamp": 0.0,
            "videos/observation.images.wrist/to_timestamp": float(num_frames - 1) / 80.0,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        })
        
        current_frame_idx += num_frames
        current_video_idx += 1
    
    # Save merged data
    print("\nSaving merged data...")
    if merged_frames:
        merged_df = pd.concat(merged_frames, ignore_index=True)
        merged_df.to_parquet(
            output_path / "data" / "chunk-000" / "file-000.parquet",
            index=False
        )
        print(f"  Saved {len(merged_df)} frames")
    
    # Save episode metadata
    if episode_meta_rows:
        episodes_df = pd.DataFrame(episode_meta_rows)
        episodes_df.to_parquet(
            output_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
            index=False
        )
        print(f"  Saved {len(episodes_df)} episode metadata")
    
    # Create symlinks for videos
    print("\nCreating video symlinks...")
    video_idx = 0
    processed_cameras = set()
    
    for dataset_name, orig_ep_idx, ep_data in tqdm(all_episode_data, desc="Linking videos"):
        dataset_dir = Path(dataset_base_dir) / dataset_name
        
        # Find video files for this episode
        videos_dir = dataset_dir / "videos"
        if videos_dir.exists():
            for camera_dir in sorted(videos_dir.glob("observation.images.*")):
                camera_name = camera_dir.name
                dst_camera_dir = output_path / "videos" / camera_name / "chunk-000"
                dst_camera_dir.mkdir(parents=True, exist_ok=True)
                
                # Create symlink to video file (only once per camera)
                if camera_name not in processed_cameras:
                    src_video = camera_dir / "chunk-000" / "file-000.mp4"
                    if src_video.exists():
                        dst_video = dst_camera_dir / "file-000.mp4"
                        # Remove existing file/symlink if exists
                        if dst_video.exists() or dst_video.is_symlink():
                            dst_video.unlink()
                        # Create absolute symlink
                        os.symlink(src_video.resolve(), dst_video)
                        print(f"  Linked video: {camera_name}")
                        processed_cameras.add(camera_name)
        
        video_idx += 1
    
    # Copy metadata files from first dataset
    first_dataset = Path(dataset_base_dir) / datasets[0]
    print("\nCopying metadata...")
    for meta_file in ["info.json", "stats.json", "tasks.parquet"]:
        src_file = first_dataset / "meta" / meta_file
        if src_file.exists():
            dst_file = output_path / "meta" / meta_file
            if meta_file == "info.json":
                # Update info.json
                with open(src_file, "r") as f:
                    info = json.load(f)
                info["total_episodes"] = len(all_episode_data)
                info["total_frames"] = current_frame_idx
                info["total_videos"] = video_idx
                with open(dst_file, "w") as f:
                    json.dump(info, f, indent=2)
            else:
                shutil.copy2(src_file, dst_file)
            print(f"  Copied {meta_file}")
    
    # Copy episode_initial_states.json if exists
    src_states = first_dataset / "episode_initial_states.json"
    if src_states.exists():
        # Merge initial states from all datasets
        merged_states = {"initial_states": []}
        for dataset_name, orig_ep_idx, ep_data in all_episode_data:
            dataset_dir = Path(dataset_base_dir) / dataset_name
            states_file = dataset_dir / "episode_initial_states.json"
            if states_file.exists():
                with open(states_file, "r") as f:
                    states = json.load(f)
                if "initial_states" in states and orig_ep_idx < len(states["initial_states"]):
                    merged_states["initial_states"].append(states["initial_states"][orig_ep_idx])
        
        with open(output_path / "episode_initial_states.json", "w") as f:
            json.dump(merged_states, f, indent=2)
        print("  Copied episode_initial_states.json")
    
    print(f"\n✓ Merged dataset saved to: {output_path}")
    print(f"  Episodes: {len(all_episode_data)}")
    print(f"  Frames: {current_frame_idx}")
    print(f"  Videos: {video_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge selected episodes from multiple datasets")
    parser.add_argument("--subset-file", type=str, required=True,
                        help="Path to merged subset JSON file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to output merged dataset directory")
    parser.add_argument("--dataset-base-dir", type=str,
                        default="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view",
                        help="Base directory containing source datasets")
    args = parser.parse_args()
    
    merge_datasets_for_training(
        subset_file=args.subset_file,
        output_dir=args.output_dir,
        dataset_base_dir=args.dataset_base_dir,
    )