#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fix episode numbering issues in merged LeRobot datasets.

After merging datasets, episode indices and file mappings may become inconsistent.
This script rebuilds the episode metadata to ensure correct sequential numbering
and proper file references.

Usage:
    lerobot-fix-merged-dataset \
        --repo_id ep10/cylinder_lay \
        --root /home/qwe/.cache/huggingface/lerobot/ep10_old_mixed/cylinder_lay \
        --output_repo_id ep10/cylinder_lay_fixed \
        --output_root /home/qwe/.cache/huggingface/lerobot/ep10_old_mixed/cylinder_lay_fixed
"""

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.io_utils import load_episodes
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    EPISODES_DIR,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


def get_all_parquet_files(root: Path, pattern: str) -> list[tuple[int, int, Path]]:
    """Get all parquet files matching the pattern and extract chunk/file indices.
    
    Returns:
        List of (chunk_index, file_index, path) tuples sorted by path.
    """
    files = sorted(root.glob(pattern))
    result = []
    for f in files:
        parts = f.parts
        chunk_str = parts[-2]
        file_str = parts[-1]
        chunk_idx = int(chunk_str.replace("chunk-", "").replace(".parquet", ""))
        file_idx = int(file_str.replace("file-", "").replace(".parquet", ""))
        result.append((chunk_idx, file_idx, f))
    return result


def scan_data_files_for_episodes(root: Path) -> dict[int, dict]:
    """Scan all data parquet files to build episode-to-file mapping.
    
    Returns:
        Dict mapping episode_index to {data/chunk_index, data/file_index, length}
    """
    data_files = get_all_parquet_files(root, "data/*/*.parquet")
    
    if not data_files:
        raise FileNotFoundError(f"No data parquet files found in {root}/data/")
    
    logging.info(f"Found {len(data_files)} data parquet files")
    
    episode_mapping = {}
    
    for chunk_idx, file_idx, data_path in data_files:
        df = pd.read_parquet(data_path)
        
        unique_eps = df["episode_index"].unique()
        for ep_idx in unique_eps:
            ep_data = df[df["episode_index"] == ep_idx]
            episode_mapping[int(ep_idx)] = {
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "length": len(ep_data),
            }
    
    logging.info(f"Found {len(episode_mapping)} unique episodes in data files")
    return episode_mapping


def scan_video_files_for_episodes(root: Path, video_keys: list[str]) -> dict[int, dict]:
    """Scan video files to build episode-to-video-file mapping.
    
    Returns:
        Dict mapping episode_index to video file indices for each video key
    """
    video_mapping = {}
    
    for vid_key in video_keys:
        video_files = get_all_parquet_files(root, f"videos/{vid_key}/*/*.mp4")
        
        if not video_files:
            logging.warning(f"No video files found for {vid_key}")
            continue
        
        logging.info(f"Found {len(video_files)} video files for {vid_key}")
        
        for chunk_idx, file_idx, video_path in video_files:
            if video_path.name.endswith(".mp4"):
                episode_mapping_key = f"videos/{vid_key}"
                if episode_mapping_key not in video_mapping:
                    video_mapping[episode_mapping_key] = {}
                
                video_mapping[episode_mapping_key][(chunk_idx, file_idx)] = {
                    "chunk_index": chunk_idx,
                    "file_index": file_idx,
                }
    
    return video_mapping


def rebuild_episode_metadata(
    dataset: LeRobotDataset,
    output_dir: Path,
    output_repo_id: str,
) -> LeRobotDataset:
    """Rebuild episode metadata with correct sequential numbering.
    
    This function:
    1. Scans actual data files to find all episodes
    2. Rebuilds episode metadata with sequential episode_index (0, 1, 2, ...)
    3. Updates all file references (data, videos, meta/episodes)
    4. Creates a new dataset with corrected metadata
    
    Args:
        dataset: Source dataset with potentially incorrect metadata
        output_dir: Directory to save the fixed dataset
        output_repo_id: Repository ID for the fixed dataset
    
    Returns:
        New LeRobotDataset with fixed metadata
    """
    root = dataset.root
    
    logging.info("=" * 60)
    logging.info("SCANNING ACTUAL DATA FILES")
    logging.info("=" * 60)
    
    episode_mapping = scan_data_files_for_episodes(root)
    
    if not episode_mapping:
        raise ValueError("No episodes found in data files")
    
    sorted_episode_indices = sorted(episode_mapping.keys())
    logging.info(f"Episodes found: {sorted_episode_indices}")
    
    video_keys = dataset.video_keys
    video_mapping = scan_video_files_for_episodes(root, video_keys) if video_keys else {}
    
    existing_episodes = load_episodes(root)
    logging.info(f"Loaded {len(existing_episodes)} existing episode records")
    
    logging.info("=" * 60)
    logging.info("REBUILDING EPISODE METADATA")
    logging.info("=" * 60)
    
    rebuilt_episodes = []
    
    for new_idx, old_idx in enumerate(sorted_episode_indices):
        try:
            old_episode = existing_episodes[old_idx]
        except (IndexError, KeyError):
            logging.warning(f"Episode {old_idx} not found in existing metadata, skipping")
            continue
        
        episode_dict = {
            "episode_index": new_idx,
        }
        
        for key in existing_episodes.column_names:
            if key == "episode_index":
                continue
            episode_dict[key] = old_episode[key]
        
        episode_dict["data/chunk_index"] = episode_mapping[old_idx]["data/chunk_index"]
        episode_dict["data/file_index"] = episode_mapping[old_idx]["data/file_index"]
        
        episode_chunk = new_idx // DEFAULT_CHUNK_SIZE
        episode_file = new_idx % DEFAULT_CHUNK_SIZE
        episode_dict["meta/episodes/chunk_index"] = episode_chunk
        episode_dict["meta/episodes/file_index"] = episode_file
        
        for vid_key in video_keys:
            vid_chunk_key = f"videos/{vid_key}/chunk_index"
            vid_file_key = f"videos/{vid_key}/file_index"
            
            if vid_chunk_key in old_episode and vid_file_key in old_episode:
                episode_dict[vid_chunk_key] = old_episode[vid_chunk_key]
                episode_dict[vid_file_key] = old_episode[vid_file_key]
        
        rebuilt_episodes.append(episode_dict)
        logging.info(f"Episode {old_idx} -> {new_idx}: "
                    f"data/chunk={episode_dict['data/chunk_index']}, "
                    f"data/file={episode_dict['data/file_index']}")
    
    logging.info(f"Rebuilt metadata for {len(rebuilt_episodes)} episodes")
    
    if not rebuilt_episodes:
        raise ValueError("No episodes could be rebuilt from data files")
    
    logging.info("=" * 60)
    logging.info("COPYING DATASET AND WRITING FIXED METADATA")
    logging.info("=" * 60)
    
    if output_dir.exists():
        backup_path = output_dir.with_name(output_dir.name + "_old")
        logging.warning(f"Output directory exists. Moving to {backup_path}")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.move(output_dir, backup_path)
    
    logging.info(f"Copying dataset from {root} to {output_dir}")
    shutil.copytree(root, output_dir)
    
    episodes_dir = output_dir / EPISODES_DIR
    if episodes_dir.exists():
        shutil.rmtree(episodes_dir)
    
    episodes_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_size = DEFAULT_CHUNK_SIZE
    current_chunk = None
    current_records = []
    
    for episode_dict in rebuilt_episodes:
        ep_chunk = episode_dict["meta/episodes/chunk_index"]
        ep_file = episode_dict["meta/episodes/file_index"]
        
        if ep_chunk != current_chunk:
            if current_records:
                chunk_dir = episodes_dir / f"chunk-{current_chunk:03d}"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                file_path = chunk_dir / f"file-{0:03d}.parquet"
                
                df = pd.DataFrame(current_records)
                df.to_parquet(file_path, index=False)
                logging.info(f"Wrote {len(current_records)} episodes to {file_path}")
            
            current_chunk = ep_chunk
            current_records = []
        
        current_records.append(episode_dict)
    
    if current_records:
        chunk_dir = episodes_dir / f"chunk-{current_chunk:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        file_path = chunk_dir / f"file-{0:03d}.parquet"
        
        df = pd.DataFrame(current_records)
        df.to_parquet(file_path, index=False)
        logging.info(f"Wrote {len(current_records)} episodes to {file_path}")
    
    logging.info(f"Fixed dataset saved to {output_dir}")
    
    new_dataset = LeRobotDataset(
        repo_id=output_repo_id,
        root=output_dir,
    )
    
    return new_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Fix episode numbering issues in merged LeRobot datasets"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Dataset repository identifier",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory of the dataset (defaults to $HF_LEROBOT_HOME/repo_id)",
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        default=None,
        help="Output dataset repository ID (defaults to {repo_id}_fixed)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output directory for the fixed dataset",
    )
    
    args = parser.parse_args()
    init_logging()
    
    root_path = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
    root_path = root_path.resolve()
    
    logging.info(f"Loading dataset from {root_path}")
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    
    logging.info("=" * 60)
    logging.info("FIXING MERGED DATASET")
    logging.info("=" * 60)
    logging.info(f"Source dataset: {dataset.repo_id}")
    logging.info(f"Source location: {dataset.root}")
    logging.info(f"Episodes: {dataset.meta.total_episodes}")
    logging.info(f"Frames: {dataset.meta.total_frames}")
    
    output_repo_id = args.output_repo_id or f"{args.repo_id}_fixed"
    output_dir = Path(args.output_root) if args.output_root else HF_LEROBOT_HOME / output_repo_id
    output_dir = output_dir.resolve()
    
    logging.info(f"Saving fixed dataset to {output_dir}")
    
    fixed_dataset = rebuild_episode_metadata(
        dataset,
        output_dir=output_dir,
        output_repo_id=output_repo_id,
    )
    
    logging.info("=" * 60)
    logging.info("FIX COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Fixed dataset: {fixed_dataset.repo_id}")
    logging.info(f"Location: {fixed_dataset.root}")
    logging.info(f"Episodes: {fixed_dataset.meta.total_episodes}")
    logging.info(f"Frames: {fixed_dataset.meta.total_frames}")
    
    logging.info("\nVerifying fixed dataset...")
    for ep_idx in range(min(3, fixed_dataset.meta.total_episodes)):
        try:
            sample = fixed_dataset[ep_idx]
            logging.info(f"Episode {ep_idx}: OK (frame_index={sample.get('frame_index', 'N/A')}, "
                        f"episode_index={sample.get('episode_index', 'N/A')})")
        except Exception as e:
            logging.error(f"Episode {ep_idx}: FAILED - {e}")
    
    logging.info("\nFix completed successfully!")


if __name__ == "__main__":
    main()