#!/usr/bin/env python
"""
Check frame-by-frame embeddings between 2 episodes WITHOUT any pooling.
"""
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

ALL_FRAMES_DIR = Path(__file__).parent.parent / "all_frames"

EP_A = 230
EP_B = 459

emb_a = np.load(ALL_FRAMES_DIR / f"episode_{EP_A:04d}.npy")
emb_b = np.load(ALL_FRAMES_DIR / f"episode_{EP_B:04d}.npy")

print("=" * 60)
print("FRAME-BY-FRAME EMBEDDING COMPARISON")
print("=" * 60)

print(f"Episode {EP_A}: shape={emb_a.shape}")
print(f"Episode {EP_B}: shape={emb_b.shape}")

# Check if all frames within each episode are the same
print(f"\n--- Within Episode {EP_A} ---")
for i in range(min(5, len(emb_a))):
    print(f"Frame {i}: mean={emb_a[i].mean():.6f}, std={emb_a[i].std():.6f}")
    if i > 0:
        sim = cosine_similarity(emb_a[0:1], emb_a[i:i+1])[0, 0]
        diff = np.abs(emb_a[0] - emb_a[i]).max()
        print(f"  vs Frame 0: cos_sim={sim:.6f}, max_diff={diff:.10f}")

print(f"\n--- Within Episode {EP_B} ---")
for i in range(min(5, len(emb_b))):
    print(f"Frame {i}: mean={emb_b[i].mean():.6f}, std={emb_b[i].std():.6f}")
    if i > 0:
        sim = cosine_similarity(emb_b[0:1], emb_b[i:i+1])[0, 0]
        diff = np.abs(emb_b[0] - emb_b[i]).max()
        print(f"  vs Frame 0: cos_sim={sim:.6f}, max_diff={diff:.10f}")

# Compare frame-by-frame between episodes
print(f"\n--- Between Episode {EP_A} and {EP_B} ---")
min_len = min(len(emb_a), len(emb_b))
print(f"Comparing first {min_len} frames")

for i in range(min(5, min_len)):
    sim = cosine_similarity(emb_a[i:i+1], emb_b[i:i+1])[0, 0]
    diff = np.abs(emb_a[i] - emb_b[i]).max()
    print(f"Frame {i}: cos_sim={sim:.6f}, max_diff={diff:.10f}")

# Check if there's a pattern: maybe only top camera was saved?
print(f"\n--- Checking embedding structure ---")
print(f"Episode {EP_A} frame 0 first 10 values: {emb_a[0][:10]}")
print(f"Episode {EP_B} frame 0 first 10 values: {emb_b[0][:10]}")

# Check if the embedding is 1920 dim (top+wrist) or 960 dim (top only)
print(f"\nEmbedding dimension: {emb_a.shape[1]}")
if emb_a.shape[1] == 960:
    print("WARNING: Only top camera embeddings (960 dim)")
elif emb_a.shape[1] == 1920:
    print("OK: Both top+wrist embeddings (1920 dim)")
    # Check if top and wrist parts are different
    top_part = emb_a[0, :960]
    wrist_part = emb_a[0, 960:]
    print(f"Top part mean: {top_part.mean():.6f}")
    print(f"Wrist part mean: {wrist_part.mean():.6f}")
    print(f"Top vs Wrist cos_sim: {cosine_similarity(top_part.reshape(1,-1), wrist_part.reshape(1,-1))[0,0]:.6f}")