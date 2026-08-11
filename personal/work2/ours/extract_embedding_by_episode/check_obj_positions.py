#!/usr/bin/env python
"""
Find 2 episodes with maximum position difference in obj_x and obj_y.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

POOL_DIR = Path(__file__).parent.parent / "pool"
df_meta = pd.read_csv(POOL_DIR / "episode_metadata.csv")

positions = df_meta[["obj_x", "obj_y"]].values

# Find pair with maximum distance
dist_matrix = squareform(pdist(positions, metric="euclidean"))
max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)

ep1_idx = max_idx[0]
ep2_idx = max_idx[1]

print("=" * 60)
print("2 EPISODES WITH MAXIMUM POSITION DIFFERENCE")
print("=" * 60)

ep1 = df_meta.iloc[ep1_idx]
ep2 = df_meta.iloc[ep2_idx]

print(f"Episode {int(ep1['episode_index'])}:")
print(f"  obj_x = {ep1['obj_x']:.4f}")
print(f"  obj_y = {ep1['obj_y']:.4f}")
print()

print(f"Episode {int(ep2['episode_index'])}:")
print(f"  obj_x = {ep2['obj_x']:.4f}")
print(f"  obj_y = {ep2['obj_y']:.4f}")
print()

diff_x = abs(ep1['obj_x'] - ep2['obj_x'])
diff_y = abs(ep1['obj_y'] - ep2['obj_y'])
print(f"Position difference:")
print(f"  delta_x = {diff_x:.4f}")
print(f"  delta_y = {diff_y:.4f}")
print(f"  Euclidean distance = {np.sqrt(diff_x**2 + diff_y**2):.4f}")