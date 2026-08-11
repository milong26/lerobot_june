#!/usr/bin/env python
"""
Check if 6 episodes with very different initial positions have distinguishable embeddings.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

BASE_DIR = Path(__file__).parent.parent
ALL_FRAMES_DIR = BASE_DIR / "all_frames"
POOL_DIR = BASE_DIR / "pool"

# Load metadata
df_meta = pd.read_csv(POOL_DIR / "episode_metadata.csv")
episode_indices = df_meta["episode_index"].tolist()
obj_positions = df_meta[["obj_x", "obj_y", "obj_z"]].values

# Find 6 episodes with maximum position spread
# Use greedy farthest-point sampling
selected = [0]
for _ in range(5):
    dists = np.min(squareform(pdist(obj_positions[selected])), axis=1)
    dists[selected] = -1
    selected.append(np.argmax(dists))

print("=" * 60)
print("6 EPISODES WITH MAXIMUM POSITION SPREAD")
print("=" * 60)
for i, idx in enumerate(selected):
    print(f"Episode {idx}: obj_x={obj_positions[idx][0]:.3f}, obj_y={obj_positions[idx][1]:.3f}, obj_z={obj_positions[idx][2]:.3f}")

# Load embeddings
print("\n" + "=" * 60)
print("LOADING EMBEDDINGS")
print("=" * 60)
frames = []
for ep_idx in selected:
    ep_path = ALL_FRAMES_DIR / f"episode_{ep_idx:04d}.npy"
    emb = np.load(ep_path)
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    frames.append(emb)
    print(f"Episode {ep_idx}: shape={emb.shape}")

# Uniform full pooling
episode_embs = [f.mean(axis=0) for f in frames]
episode_embs = np.array(episode_embs)
print(f"\nEpisode embeddings shape: {episode_embs.shape}")

# Pairwise distances
print("\n" + "=" * 60)
print("PAIRWISE DISTANCES")
print("=" * 60)
emb_dists = squareform(pdist(episode_embs, metric="cosine"))
phys_dists = squareform(pdist(obj_positions[selected], metric="euclidean"))

print("\nEmbedding cosine distances:")
print(np.round(emb_dists, 4))

print("\nPhysical distances (XYZ):")
print(np.round(phys_dists, 4))

# Correlation
from scipy.stats import spearmanr
mask = np.triu_indices(len(selected), k=1)
corr, p_value = spearmanr(emb_dists[mask], phys_dists[mask])
print(f"\nSpearman correlation: {corr:.4f} (p={p_value:.4f})")

# PCA visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(episode_embs)

print("\n" + "=" * 60)
print("PCA 2D PROJECTION")
print("=" * 60)
for i, idx in enumerate(selected):
    print(f"Episode {idx}: PCA1={emb_2d[i][0]:.3f}, PCA2={emb_2d[i][1]:.3f}")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
if corr > 0.5:
    print("GOOD: Embeddings can distinguish different positions")
elif corr > 0.2:
    print("MODERATE: Some correlation but weak")
else:
    print("POOR: Embeddings cannot distinguish positions")