#!/usr/bin/env python
"""
Check if actual images are different between frames and episodes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import numpy as np
from PIL import Image

DATASET_DIR = Path(__file__).parent.parent / "dataset"
repo_id = "lerobot/metaworld_pick_place"

print("Loading dataset...")
dataset = LeRobotDataset(repo_id, root=DATASET_DIR)

# Find frames for episode 230 and 459
print("Finding episode frame ranges...")
ep_frames = {}
for idx in range(min(dataset.num_frames, 35000)):
    frame = dataset[idx]
    ep_idx = int(frame["episode_index"])
    if ep_idx not in ep_frames:
        ep_frames[ep_idx] = []
    ep_frames[ep_idx].append(idx)
    if len(ep_frames) >= 460:
        break

# Check episode 230
ep230_frames = ep_frames.get(230, [])
ep459_frames = ep_frames.get(459, [])

print(f"\nEpisode 230: {len(ep230_frames)} frames")
print(f"Episode 459: {len(ep459_frames)} frames")

if len(ep230_frames) >= 2:
    # Check first 2 frames of episode 230
    f0 = dataset[ep230_frames[0]]
    f1 = dataset[ep230_frames[1]]
    
    top0 = f0["observation.images.top"]
    top1 = f1["observation.images.top"]
    
    print(f"\n--- Episode 230 Frame 0 vs Frame 1 ---")
    print(f"Frame 0 top shape: {top0.shape}")
    print(f"Frame 1 top shape: {top1.shape}")
    
    # Check if images are different
    if isinstance(top0, np.ndarray) and isinstance(top1, np.ndarray):
        diff = np.abs(top0.astype(float) - top1.astype(float)).mean()
        print(f"Mean pixel difference: {diff:.4f}")
        print(f"Images identical? {np.allclose(top0, top1)}")
    else:
        # Convert to numpy
        if hasattr(top0, 'numpy'):
            top0_np = top0.numpy()
        else:
            top0_np = np.array(top0)
        if hasattr(top1, 'numpy'):
            top1_np = top1.numpy()
        else:
            top1_np = np.array(top1)
        diff = np.abs(top0_np.astype(float) - top1_np.astype(float)).mean()
        print(f"Mean pixel difference: {diff:.4f}")
        print(f"Images identical? {np.allclose(top0_np, top1_np)}")

if len(ep230_frames) > 0 and len(ep459_frames) > 0:
    # Check first frame of each episode
    f230 = dataset[ep230_frames[0]]
    f459 = dataset[ep459_frames[0]]
    
    top230 = f230["observation.images.top"]
    top459 = f459["observation.images.top"]
    
    print(f"\n--- Episode 230 Frame 0 vs Episode 459 Frame 0 ---")
    
    if isinstance(top230, np.ndarray) and isinstance(top459, np.ndarray):
        diff = np.abs(top230.astype(float) - top459.astype(float)).mean()
        print(f"Mean pixel difference: {diff:.4f}")
        print(f"Images identical? {np.allclose(top230, top459)}")
    else:
        if hasattr(top230, 'numpy'):
            top230_np = top230.numpy()
        else:
            top230_np = np.array(top230)
        if hasattr(top459, 'numpy'):
            top459_np = top459.numpy()
        else:
            top459_np = np.array(top459)
        diff = np.abs(top230_np.astype(float) - top459_np.astype(float)).mean()
        print(f"Mean pixel difference: {diff:.4f}")
        print(f"Images identical? {np.allclose(top230_np, top459_np)}")