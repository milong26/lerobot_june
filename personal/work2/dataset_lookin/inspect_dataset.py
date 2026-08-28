#!/usr/bin/env python
"""
Inspect LeRobotDataset structure to understand available attributes and metadata.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATASET_DIR = Path(__file__).parent.parent / "dataset"
REPO_ID = "lerobot/metaworld_pick_place"


def main():
    print("Loading dataset...")
    dataset = LeRobotDataset(REPO_ID, root=str(DATASET_DIR))

    print("\n" + "=" * 60)
    print("Dataset Overview")
    print("=" * 60)
    print(f"Repo ID: {dataset.repo_id}")
    print(f"Num episodes: {dataset.num_episodes}")
    print(f"Num frames: {dataset.num_frames}")
    print(f"FPS: {dataset.fps}")
    print(f"Features: {list(dataset.features.keys())}")

    print("\n" + "=" * 60)
    print("Episode Metadata (first 3 episodes)")
    print("=" * 60)
    print(f"Type of dataset.meta.episodes: {type(dataset.meta.episodes)}")

    if hasattr(dataset.meta.episodes, 'columns'):
        # It's a DataFrame
        print(f"Columns: {list(dataset.meta.episodes.columns)}")
        print(f"\nFirst 3 episodes:")
        print(dataset.meta.episodes.head(3).to_string())
    elif isinstance(dataset.meta.episodes, list):
        print(f"Length: {len(dataset.meta.episodes)}")
        print(f"\nFirst episode keys: {dataset.meta.episodes[0].keys()}")
        print(f"\nFirst episode:")
        for k, v in dataset.meta.episodes[0].items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("First Frame Structure")
    print("=" * 60)
    frame0 = dataset[0]
    print(f"Frame keys: {list(frame0.keys())}")
    for key, value in frame0.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        elif hasattr(value, '__len__'):
            print(f"  {key}: len={len(value)}, type={type(value)}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Episode 0 Frame (episode_index check)")
    print("=" * 60)
    print(f"  episode_index: {frame0.get('episode_index', 'N/A')}")
    print(f"  frame_index: {frame0.get('frame_index', 'N/A')}")
    print(f"  index: {frame0.get('index', 'N/A')}")

    print("\n" + "=" * 60)
    print("Episode Meta Details")
    print("=" * 60)
    if hasattr(dataset.meta.episodes, 'iloc'):
        ep0 = dataset.meta.episodes.iloc[0]
        for k, v in ep0.items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)


if __name__ == "__main__":
    main()