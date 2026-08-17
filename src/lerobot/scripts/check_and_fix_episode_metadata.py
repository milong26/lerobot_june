#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All include. All rights reserved.
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
Check and fix episode metadata inconsistencies in LeRobot datasets.

This tool validates that episode metadata correctly references actual parquet files
and can automatically fix mismatches by rebuilding the correct file mappings.

Usage Examples:

Check dataset without fixing:
    lerobot-check-episodes \
        --repo_id ep10/cylinder_lay \
        --root /path/to/dataset

Check and fix with output to new dataset:
    lerobot-check-episodes \
        --repo_id ep10/cylinder_lay \
        --root /path/to/dataset \
        --new_repo_id ep10/cylinder_lay_fixed \
        --new_root /path/to/fixed_dataset \
        --fix

Check and fix in-place (creates backup):
    lerobot-check-episodes \
        --repo_id ep10/cylinder_lay \
        --root /path/to/dataset \
        --fix \
        --in_place
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


def check_episode_metadata(dataset: LeRobotDataset) -> dict:
    """Check if episode metadata correctly references actual files.
    
    Returns:
        Dictionary with validation results.
    """
    root = dataset.root
    episodes_meta = dataset.meta.episodes
    
    issues = {
        "total_episodes": len(episodes_meta),
        "missing_episode_files": [],
        "wrong_episode_mapping": [],
        "missing_data_files": [],
        "wrong_data_mapping": [],
        "is_valid": True,
    }
    
    logging.info(f"Checking {issues['total_episodes']} episodes...")
    
    for ep_idx in range(issues['total_episodes']):
        ep_meta = episodes_meta[ep_idx]
        
        episode_chunk = ep_meta.get("meta/episodes/chunk_index")
        episode_file = ep_meta.get("meta/episodes/file_index")
        data_chunk = ep_meta.get("data/chunk_index")
        data_file = ep_meta.get("data/file_index")
        
        if episode_chunk is not None and episode_file is not None:
            episode_path = root / DEFAULT_EPISODES_PATH.format(
                chunk_index=episode_chunk, file_index=episode_file
            )
            if not episode_path.exists():
                issues["missing_episode_files"].append({
                    "episode_index": ep_idx,
                    "expected_path": str(episode_path),
                    "chunk_index": episode_chunk,
                    "file_index": episode_file,
                })
                issues["is_valid"] = False
                logging.warning(f"Episode {ep_idx}: Missing episode metadata file {episode_path}")
        
        if data_chunk is not None and data_file is not None:
            data_path = root / DEFAULT_DATA_PATH.format(
                chunk_index=data_chunk, file_index=data_file
            )
            if not data_path.exists():
                issues["missing_data_files"].append({
                    "episode_index": ep_idx,
                    "expected_path": str(data_path),
                    "chunk_index": data_chunk,
                    "file_index": data_file,
                })
                issues["is_valid"] = False
                logging.warning(f"Episode {ep_idx}: Missing data file {data_path}")
    
    return issues


def rebuild_episode_metadata(dataset: LeRobotDataset) -> list[dict]:
    """Rebuild episode metadata by scanning actual parquet files.
    
    This function:
    1. Loads existing episode metadata from available parquet files
    2. Scans all data/*.parquet files to get real episode indices
    3. Rebuilds the correct mapping with all required fields
    
    Returns:
        List of episode metadata dictionaries with all required fields
    """
    root = dataset.root
    
    existing_episodes = load_episodes(root)
    logging.info(f"Loaded {len(existing_episodes)} existing episode records")
    
    data_files = get_all_parquet_files(root, "data/*/*.parquet")
    
    if not data_files:
        raise FileNotFoundError(f"No data parquet files found in {root}/data/")
    
    logging.info(f"Found {len(data_files)} data parquet files")
    
    episode_data_mapping = {}
    
    for chunk_idx, file_idx, data_path in data_files:
        df = pd.read_parquet(data_path)
        
        unique_eps = df["episode_index"].unique()
        for ep_idx in unique_eps:
            episode_data_mapping[int(ep_idx)] = {
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
            }
    
    logging.info(f"Found {len(episode_data_mapping)} unique episodes in data files")
    
    sorted_episode_indices = sorted(episode_data_mapping.keys())
    
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
        
        episode_dict["data/chunk_index"] = episode_data_mapping[old_idx]["data/chunk_index"]
        episode_dict["data/file_index"] = episode_data_mapping[old_idx]["data/file_index"]
        
        # All episode metadata will be written to a single file (chunk-000/file-000.parquet)
        # So all episodes should point to the same chunk/file index
        episode_dict["meta/episodes/chunk_index"] = 0
        episode_dict["meta/episodes/file_index"] = 0
        
        video_keys = [key for key in existing_episodes.column_names if key.startswith("videos/")]
        for vid_key in video_keys:
            if "chunk_index" in vid_key or "file_index" in vid_key:
                episode_dict[vid_key] = episode_data_mapping[old_idx].get(
                    vid_key.replace("videos/", "data/"),
                    episode_dict.get(vid_key, 0)
                )
        
        rebuilt_episodes.append(episode_dict)
    
    logging.info(f"Rebuilt metadata for {len(rebuilt_episodes)} episodes")
    
    return rebuilt_episodes


def fix_episode_metadata(
    dataset: LeRobotDataset,
    output_dir: Path,
    output_repo_id: str,
) -> LeRobotDataset:
    """Fix episode metadata and create a new dataset with corrected mappings.
    
    Args:
        dataset: Source dataset with potentially incorrect metadata
        output_dir: Directory to save the fixed dataset
        output_repo_id: Repository ID for the fixed dataset
    
    Returns:
        New LeRobotDataset with fixed metadata
    """
    root = dataset.root
    
    logging.info("Rebuilding episode metadata from actual data files...")
    rebuilt_episodes = rebuild_episode_metadata(dataset)
    
    if not rebuilt_episodes:
        raise ValueError("No episodes could be rebuilt from data files")
    
    logging.info(f"Copying dataset from {root} to {output_dir}")
    if output_dir.exists():
        backup_path = output_dir.with_name(output_dir.name + "_old")
        logging.warning(f"Output directory exists. Moving to {backup_path}")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.move(output_dir, backup_path)
    
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
        description="Check and fix episode metadata inconsistencies in LeRobot datasets"
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
        "--new_repo_id",
        type=str,
        default=None,
        help="Output dataset repository identifier",
    )
    parser.add_argument(
        "--new_root",
        type=str,
        default=None,
        help="Output directory for the fixed dataset",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix metadata issues",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Fix metadata in-place (creates backup of original)",
    )
    
    args = parser.parse_args()
    init_logging()
    
    root_path = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
    root_path = root_path.resolve()
    
    logging.info(f"Loading dataset from {root_path}")
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    
    logging.info("=" * 60)
    logging.info("CHECKING EPISODE METADATA")
    logging.info("=" * 60)
    
    issues = check_episode_metadata(dataset)
    
    logging.info("=" * 60)
    logging.info("VALIDATION RESULTS")
    logging.info("=" * 60)
    logging.info(f"Total episodes: {issues['total_episodes']}")
    logging.info(f"Missing episode metadata files: {len(issues['missing_episode_files'])}")
    logging.info(f"Missing data files: {len(issues['missing_data_files'])}")
    logging.info(f"Dataset is valid: {issues['is_valid']}")
    
    if issues['missing_episode_files']:
        logging.warning("\nMissing episode metadata files:")
        for issue in issues['missing_episode_files']:
            logging.warning(f"  Episode {issue['episode_index']}: {issue['expected_path']}")
    
    if issues['missing_data_files']:
        logging.warning("\nMissing data files:")
        for issue in issues['missing_data_files']:
            logging.warning(f"  Episode {issue['episode_index']}: {issue['expected_path']}")
    
    if not issues['is_valid'] and not args.fix:
        logging.error("\nDataset has metadata issues. Use --fix to automatically repair.")
        return
    
    if args.fix:
        logging.info("=" * 60)
        logging.info("FIXING METADATA")
        logging.info("=" * 60)
        
        if args.in_place:
            output_repo_id = args.repo_id
            output_dir = root_path
            backup_path = root_path.with_name(root_path.name + "_old")
            
            logging.info(f"Creating backup at {backup_path}")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.move(root_path, backup_path)
            
            logging.info("Fixing metadata in-place...")
            fixed_dataset = fix_episode_metadata(
                LeRobotDataset(args.repo_id, root=str(backup_path)),
                output_dir=root_path,
                output_repo_id=output_repo_id,
            )
        else:
            output_repo_id = args.new_repo_id or f"{args.repo_id}_fixed"
            output_dir = Path(args.new_root) if args.new_root else HF_LEROBOT_HOME / output_repo_id
            output_dir = output_dir.resolve()
            
            logging.info(f"Saving fixed dataset to {output_dir}")
            fixed_dataset = fix_episode_metadata(
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
        new_issues = check_episode_metadata(fixed_dataset)
        if new_issues['is_valid']:
            logging.info("✓ Fixed dataset passed validation!")
        else:
            logging.error("✗ Fixed dataset still has issues. Please review manually.")


if __name__ == "__main__":
    main()