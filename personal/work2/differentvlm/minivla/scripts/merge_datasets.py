#!/usr/bin/env python3
"""
Merge multiple LeRobot datasets into a single dataset using official LeRobot tools.

This script:
1. Selects N episodes from each source dataset (random or uniform selection)
2. Uses official split_dataset() to create valid subset datasets
3. Uses official merge_datasets() to merge subsets
4. Uses official recompute_stats() to ensure correct statistics
5. Handles custom episode_initial_states.json for MetaWorld datasets

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

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import (
    split_dataset,
    merge_datasets,
    recompute_stats,
)


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


def load_episode_initial_states(dataset_path: Path) -> list[dict]:
    """Load episode_initial_states.json if it exists."""
    states_file = dataset_path / "episode_initial_states.json"
    if not states_file.exists():
        return []
    
    with open(states_file, "r") as f:
        states_data = json.load(f)
    
    if isinstance(states_data, list):
        return states_data
    elif isinstance(states_data, dict):
        return states_data.get("episodes", [])
    else:
        print(f"  Warning: Unexpected format in episode_initial_states.json")
        return []


def save_merged_episode_initial_states(
    src_datasets: list[Path],
    all_selected_episodes: list[list[int]],
    output_dir: Path,
):
    """
    Merge episode_initial_states.json from all source datasets with re-indexed episodes.
    
    This handles the custom MetaWorld episode_initial_states.json that official
    LeRobot tools don't know about.
    """
    merged_states = []
    episode_offset = 0
    
    for src_dataset, selected_episodes in zip(src_datasets, all_selected_episodes):
        states = load_episode_initial_states(src_dataset)
        if not states:
            episode_offset += len(selected_episodes)
            continue
        
        for state in states:
            if not isinstance(state, dict):
                continue
            
            old_idx = state.get("episode_index", -1)
            if old_idx in selected_episodes:
                new_state = state.copy()
                new_state["episode_index"] = episode_offset + selected_episodes.index(old_idx)
                merged_states.append(new_state)
        
        episode_offset += len(selected_episodes)
    
    if merged_states:
        dst_states = output_dir / "episode_initial_states.json"
        output_data = {
            "episodes": merged_states,
            "num_episodes": len(merged_states)
        }
        with open(dst_states, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Saved {len(merged_states)} episode initial states")


def run_acceptance_test(dataset: LeRobotDataset):
    """Run comprehensive acceptance tests on the merged dataset."""
    print("\n" + "=" * 70)
    print("RUNNING ACCEPTANCE TESTS")
    print("=" * 70)
    
    errors = []
    
    # Test 1: Total frames consistency
    print("\n1. Checking total frames consistency...")
    if len(dataset) != dataset.meta.total_frames:
        errors.append(f"len(dataset)={len(dataset)} != meta.total_frames={dataset.meta.total_frames}")
        print(f"   ❌ FAIL: len(dataset)={len(dataset)} != meta.total_frames={dataset.meta.total_frames}")
    else:
        print(f"   ✅ PASS: {len(dataset)} frames")
    
    # Test 2: Index continuity
    print("\n2. Checking index continuity...")
    indices = dataset.hf_dataset["index"]
    expected_indices = list(range(len(indices)))
    if indices != expected_indices:
        errors.append(f"Index not continuous from 0 to {len(indices)-1}")
        print(f"   ❌ FAIL: Index not continuous")
    else:
        print(f"   ✅ PASS: Index continuous from 0 to {len(indices)-1}")
    
    # Test 3: Episode index continuity
    print("\n3. Checking episode index continuity...")
    ep_indices = dataset.hf_dataset["episode_index"]
    unique_ep_indices = sorted(set(ep_indices))
    expected_ep_indices = list(range(dataset.meta.total_episodes))
    if unique_ep_indices != expected_ep_indices:
        errors.append(f"Episode index not continuous from 0 to {dataset.meta.total_episodes-1}")
        print(f"   ❌ FAIL: Episode index not continuous")
    else:
        print(f"   ✅ PASS: Episode index continuous from 0 to {dataset.meta.total_episodes-1}")
    
    # Test 4: Episode from/to index consistency
    print("\n4. Checking episode from/to index consistency...")
    ep_meta_valid = True
    for i in range(dataset.meta.total_episodes):
        ep = dataset.meta.episodes[i]
        calc_length = ep["dataset_to_index"] - ep["dataset_from_index"]
        if calc_length != ep["length"]:
            errors.append(f"Episode {i}: to-from={calc_length} != length={ep['length']}")
            ep_meta_valid = False
            if len(errors) <= 5:
                print(f"   ❌ Episode {i}: to-from={calc_length} != length={ep['length']}")
    
    if ep_meta_valid:
        print(f"   ✅ PASS: All {dataset.meta.total_episodes} episodes have correct from/to indices")
    
    # Test 5: Check for NaN/Inf in action
    print("\n5. Checking for NaN/Inf in action...")
    has_nan = False
    has_inf = False
    sample_size = min(100, dataset.num_frames)
    for i in range(sample_size):
        frame = dataset[i]
        if "action" in frame:
            action = frame["action"].numpy()
            if np.isnan(action).any():
                has_nan = True
                break
            if np.isinf(action).any():
                has_inf = True
                break
    
    if has_nan:
        errors.append("Found NaN in action")
        print(f"   ❌ FAIL: Found NaN in action")
    elif has_inf:
        errors.append("Found Inf in action")
        print(f"   ❌ FAIL: Found Inf in action")
    else:
        print(f"   ✅ PASS: No NaN/Inf in action (sampled {sample_size} frames)")
    
    # Test 6: Check for NaN/Inf in observation.state
    print("\n6. Checking for NaN/Inf in observation.state...")
    has_nan = False
    has_inf = False
    for i in range(sample_size):
        frame = dataset[i]
        if "observation.state" in frame:
            state = frame["observation.state"].numpy()
            if np.isnan(state).any():
                has_nan = True
                break
            if np.isinf(state).any():
                has_inf = True
                break
    
    if has_nan:
        errors.append("Found NaN in observation.state")
        print(f"   ❌ FAIL: Found NaN in observation.state")
    elif has_inf:
        errors.append("Found Inf in observation.state")
        print(f"   ❌ FAIL: Found Inf in observation.state")
    else:
        print(f"   ✅ PASS: No NaN/Inf in observation.state (sampled {sample_size} frames)")
    
    # Test 7: Check stats are finite
    print("\n7. Checking stats are finite...")
    stats_valid = True
    for feature_name, feature_stats in dataset.meta.stats.items():
        for stat_key, val in feature_stats.items():
            if isinstance(val, (int, float)):
                if not np.isfinite(val):
                    errors.append(f"Stats {feature_name}/{stat_key} is not finite: {val}")
                    stats_valid = False
            elif isinstance(val, list):
                if not all(np.isfinite(v) for v in val):
                    errors.append(f"Stats {feature_name}/{stat_key} contains non-finite values")
                    stats_valid = False
    
    if stats_valid:
        print(f"   ✅ PASS: All stats are finite")
    
    # Test 8: Task index distribution
    print("\n8. Checking task index distribution...")
    task_counts = {}
    for i in range(min(1000, dataset.num_frames)):
        frame = dataset[i]
        if "task_index" in frame:
            task_idx = int(frame["task_index"])
            task_counts[task_idx] = task_counts.get(task_idx, 0) + 1
    print(f"   Task distribution (first 1000 frames): {task_counts}")
    
    # Summary
    print("\n" + "=" * 70)
    if errors:
        print(f"❌ ACCEPTANCE TEST FAILED: {len(errors)} errors found")
        for i, error in enumerate(errors[:10], 1):
            print(f"   {i}. {error}")
    else:
        print("✅ ALL ACCEPTANCE TESTS PASSED")
    print("=" * 70)
    
    return len(errors) == 0


def fix_episode_timestamps(dataset: LeRobotDataset) -> bool:
    """Fix incorrect to_timestamp in episode metadata.
    
    Some datasets have incorrect to_timestamp for the first episode,
    causing length mismatch when split_dataset() calculates frame ranges.
    """
    from lerobot.datasets.io_utils import load_episodes, write_episodes
    
    if dataset.meta.episodes is None:
        episodes_ds = load_episodes(dataset.meta.root)
    else:
        episodes_ds = dataset.meta.episodes
    
    # Convert to pandas for modification
    episodes_df = episodes_ds.to_pandas()
    
    fps = dataset.meta.fps
    fixed = False
    
    for i in range(len(episodes_df)):
        row = episodes_df.iloc[i]
        length = row['length']
        
        # Check all video features
        for col in episodes_df.columns:
            if col.startswith('videos/') and col.endswith('/from_timestamp'):
                video_key = col.replace('videos/', '').replace('/from_timestamp', '')
                to_col = col.replace('from_timestamp', 'to_timestamp')
                
                from_ts = row[col]
                to_ts = row[to_col]
                
                # Calculate expected frame count
                from_frame = round(from_ts * fps)
                to_frame = round(to_ts * fps)
                calc_length = to_frame - from_frame
                
                # Fix if mismatch
                if calc_length != length:
                    expected_to_ts = (from_frame + length) / fps
                    episodes_df.at[i, to_col] = expected_to_ts
                    fixed = True
                    if i < 3:  # Only print first few
                        print(f"    Fixed {video_key} episode {i}: to_ts {to_ts:.4f} -> {expected_to_ts:.4f}")
    
    if fixed:
        # Write back to disk
        write_episodes(episodes_df, dataset.meta.root)
        # Reload to ensure consistency
        dataset.meta.episodes = load_episodes(dataset.meta.root)
        print(f"  ✅ Fixed timestamps in episode metadata")
    
    return fixed


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
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Selection mode: {args.selection_mode}")
    print(f"Episodes per dataset: {args.episodes_per_dataset}")
    print(f"Random seed: {args.seed}")
    
    # Step 1: Select episodes from each dataset
    print("\n" + "=" * 70)
    print("STEP 1: SELECTING EPISODES")
    print("=" * 70)
    all_selected_episodes = []
    
    for i, src_dataset in enumerate(src_datasets):
        dataset_name = src_dataset.name
        print(f"\nDataset {i+1}: {dataset_name}")
        
        selected = select_episodes(
            src_dataset,
            args.episodes_per_dataset,
            args.selection_mode,
            args.seed + i
        )
        
        all_selected_episodes.append(selected)
        print(f"  Selected {len(selected)} episodes: {selected[:5]}...")
    
    # Step 2: Create subset datasets using official split_dataset()
    print("\n" + "=" * 70)
    print("STEP 2: CREATING SUBSET DATASETS (official split_dataset)")
    print("=" * 70)
    temp_root = output_dir.parent / f"{output_dir.name}_subsets"
    subset_datasets = []
    
    for i, (src_dataset, selected_episodes) in enumerate(zip(src_datasets, all_selected_episodes)):
        dataset_name = src_dataset.name
        print(f"\nProcessing dataset {i+1}: {dataset_name}")
        
        # Load source dataset
        src_dataset_obj = LeRobotDataset(
            repo_id=f"local/{dataset_name}",
            root=src_dataset,
        )
        print(f"  Loaded source dataset: {src_dataset_obj.num_episodes} episodes, {src_dataset_obj.num_frames} frames")
        
        # Fix timestamps if needed (before split)
        print(f"  Checking episode timestamps...")
        fix_episode_timestamps(src_dataset_obj)
        
        # Split to create subset
        split_output = temp_root / dataset_name
        result = split_dataset(
            dataset=src_dataset_obj,
            splits={"selected": selected_episodes},
            output_dir=split_output,
        )
        
        subset_dataset = result["selected"]
        subset_datasets.append(subset_dataset)
        print(f"  Created subset: {subset_dataset.num_episodes} episodes, {subset_dataset.num_frames} frames")
    
    # Step 3: Merge subset datasets using official merge_datasets()
    print("\n" + "=" * 70)
    print("STEP 3: MERGING SUBSET DATASETS (official merge_datasets)")
    print("=" * 70)
    
    merged_dataset = merge_datasets(
        datasets=subset_datasets,
        output_repo_id="joint_metaworld",
        output_dir=output_dir,
        concatenate_videos=False,
        concatenate_data=False,
    )
    
    print(f"\nMerged dataset: {merged_dataset.num_episodes} episodes, {merged_dataset.num_frames} frames")
    
    # Step 4: Recompute stats from actual merged data
    print("\n" + "=" * 70)
    print("STEP 4: RECOMPUTING STATS (official recompute_stats)")
    print("=" * 70)
    
    recompute_stats(
        merged_dataset,
        skip_image_video=True,
    )
    print("  Stats recomputed successfully")
    
    # Step 5: Handle custom episode_initial_states.json
    print("\n" + "=" * 70)
    print("STEP 5: MERGING EPISODE INITIAL STATES (custom MetaWorld)")
    print("=" * 70)
    
    save_merged_episode_initial_states(
        src_datasets,
        all_selected_episodes,
        output_dir,
    )
    
    # Step 6: Create subset_file.json for training pipeline
    print("\n" + "=" * 70)
    print("STEP 6: CREATING SUBSET FILE FOR TRAINING")
    print("=" * 70)
    
    subset_file = output_dir / "subset_file.json"
    subset_data = {
        "selected_episode_indices": list(range(merged_dataset.meta.total_episodes)),
        "total_episodes": merged_dataset.meta.total_episodes,
        "source_datasets": [ds.name for ds in src_datasets],
        "episodes_per_dataset": [len(ep) for ep in all_selected_episodes],
        "selection_mode": args.selection_mode,
        "seed": args.seed
    }
    with open(subset_file, "w") as f:
        json.dump(subset_data, f, indent=2)
    print(f"  Created subset_file.json with {len(subset_data['selected_episode_indices'])} episodes")
    
    # Step 7: Run acceptance tests
    print("\n" + "=" * 70)
    print("STEP 7: RUNNING ACCEPTANCE TESTS")
    print("=" * 70)
    
    test_passed = run_acceptance_test(merged_dataset)
    
    # Summary
    print("\n" + "=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Total episodes: {merged_dataset.meta.total_episodes}")
    print(f"Total frames: {merged_dataset.meta.total_frames}")
    print(f"Source datasets: {[ds.name for ds in src_datasets]}")
    print(f"Episodes per dataset: {[len(ep) for ep in all_selected_episodes]}")
    
    if test_passed:
        print(f"\n✅ Dataset is ready for training!")
        print(f"  --dataset {output_dir}")
        print(f"  --num-episodes {merged_dataset.meta.total_episodes}")
    else:
        print(f"\n❌ Dataset has issues. Please review acceptance test errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()