#!/usr/bin/env python
"""
Compare raw images between episode 230 and 459 - direct access.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATASET_DIR = Path(__file__).parent.parent.parent / "dataset"

EP_A = 230
EP_B = 459

print("Loading dataset...")
repo_id = "lerobot/metaworld_pick_place"
dataset = LeRobotDataset(repo_id, root=DATASET_DIR)

# Direct access: use episode_index to find first frame
# Each episode has ~60-70 frames, so episode N starts around frame N*65
print(f"\nDirect access to episodes...")

# Find first frame of each episode by sampling
def find_episode_start(ep_target):
    # Estimate start frame
    est_start = int(ep_target * dataset.num_frames / dataset.num_episodes)
    # Scan forward to find the episode
    for idx in range(est_start, min(est_start + 200, dataset.num_frames)):
        frame = dataset[idx]
        ep_idx = int(frame["episode_index"])
        if ep_idx == ep_target:
            return idx
    return None

start_a = find_episode_start(EP_A)
start_b = find_episode_start(EP_B)

print(f"Episode {EP_A} starts at frame: {start_a}")
print(f"Episode {EP_B} starts at frame: {start_b}")

if start_a is None or start_b is None:
    print("ERROR: Could not find episodes")
    sys.exit(1)

# Compare first 5 frames
print(f"\n{'='*60}")
print(f"IMAGE COMPARISON")
print(f"{'='*60}")

for i in range(5):
    print(f"\n--- Frame {i} ---")
    
    # Episode A
    frame_a = dataset[start_a + i]
    img_a_top = frame_a["observation.images.top"]
    img_a_wrist = frame_a["observation.images.wrist"]
    
    if isinstance(img_a_top, np.ndarray):
        img_a_top_np = img_a_top
    else:
        img_a_top_np = img_a_top.numpy()
    
    if isinstance(img_a_wrist, np.ndarray):
        img_a_wrist_np = img_a_wrist
    else:
        img_a_wrist_np = img_a_wrist.numpy()
    
    # Episode B
    frame_b = dataset[start_b + i]
    img_b_top = frame_b["observation.images.top"]
    img_b_wrist = frame_b["observation.images.wrist"]
    
    if isinstance(img_b_top, np.ndarray):
        img_b_top_np = img_b_top
    else:
        img_b_top_np = img_b_top.numpy()
    
    if isinstance(img_b_wrist, np.ndarray):
        img_b_wrist_np = img_b_wrist
    else:
        img_b_wrist_np = img_b_wrist.numpy()
    
    # Image comparison
    img_diff_top = np.abs(img_a_top_np.astype(float) - img_b_top_np.astype(float)).mean()
    img_diff_wrist = np.abs(img_a_wrist_np.astype(float) - img_b_wrist_np.astype(float)).mean()
    
    print(f"Top image pixel diff: {img_diff_top:.4f}")
    print(f"Wrist image pixel diff: {img_diff_wrist:.4f}")
    print(f"Top images identical? {np.allclose(img_a_top_np, img_b_top_np)}")
    print(f"Wrist images identical? {np.allclose(img_a_wrist_np, img_b_wrist_np)}")

print(f"\n{'='*60}")
print(f"CONCLUSION")
print(f"{'='*60}")
if img_diff_top > 0 or img_diff_wrist > 0:
    print("Images are DIFFERENT between episodes.")
else:
    print("Images are IDENTICAL between episodes.")