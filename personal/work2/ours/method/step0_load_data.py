#!/usr/bin/env python
"""
Step 0: Load all 500 episodes, extract initial positions (environment_state[:3]) and episode_index.
Also re-extract episode embeddings for ALL 500 episodes using the strategy from step 3.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
EMBEDDING_DIR = Path(__file__).parent.parent / "extract_embedding_by_episode"
ALL_FRAMES_DIR = Path(__file__).parent.parent / "all_frames"
POOL_DIR = Path(__file__).parent.parent / "pool"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def temporal_multi_window_pool(frame_embeddings, start_pct=0.15, end_pct=0.30):
    K = frame_embeddings.shape[0]
    s = max(0, int(start_pct * K))
    e = min(K, int(end_pct * K))
    mid = (s + e) / 2
    half_w = (e - s) / 4
    s1, e1 = max(s, int(mid - half_w)), int(mid)
    s2, e2 = int(mid), min(e, int(mid + half_w))
    if e1 <= s1:
        e1 = s1 + 1
    if e2 <= s2:
        e2 = s2 + 1
    emb1 = frame_embeddings[s1:e1].mean(axis=0)
    emb2 = frame_embeddings[s2:e2].mean(axis=0)
    return 0.5 * emb1 + 0.5 * emb2


def main():
    print("=" * 60)
    print("Step 0: Load all episodes and extract initial positions")
    print("=" * 60)

    meta_path = POOL_DIR / "episode_metadata.csv"
    df_meta = pd.read_csv(meta_path)
    episode_indices = df_meta["episode_index"].tolist()
    obj_positions = df_meta[["obj_x", "obj_y", "obj_z"]].values

    print(f"Total episodes from metadata: {len(episode_indices)}")
    print(f"Object position range (x): {obj_positions[:, 0].min():.4f} ~ {obj_positions[:, 0].max():.4f}")
    print(f"Object position range (y): {obj_positions[:, 1].min():.4f} ~ {obj_positions[:, 1].max():.4f}")
    print(f"Object position range (z): {obj_positions[:, 2].min():.4f} ~ {obj_positions[:, 2].max():.4f}")

    print("\nLoading frame embeddings for all episodes...")
    all_frame_embeddings = []
    valid_ep_indices = []
    for ep_idx in tqdm(episode_indices, desc="Loading frames"):
        ep_path = ALL_FRAMES_DIR / f"episode_{ep_idx:04d}.npy"
        if not ep_path.exists():
            print(f"  WARNING: {ep_path} not found, skipping")
            continue
        emb = np.load(ep_path)
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        all_frame_embeddings.append(emb)
        valid_ep_indices.append(ep_idx)

    n_episodes = len(all_frame_embeddings)
    print(f"Loaded {n_episodes} episodes, frame dim = {all_frame_embeddings[0].shape[1]}")

    print("\nExtracting episode embeddings (temporal_multi_window 15%-30%, PCA-32)...")
    raw_embs = []
    for frames in tqdm(all_frame_embeddings, desc="Pooling"):
        emb = temporal_multi_window_pool(frames, 0.15, 0.30)
        raw_embs.append(emb)
    raw_embs = np.array(raw_embs)
    raw_embs = np.nan_to_num(raw_embs, nan=0.0, posinf=0.0, neginf=0.0)

    pca_path = EMBEDDING_DIR / "pca_model.pkl"
    if pca_path.exists():
        with open(pca_path, "rb") as f:
            pca_model = pickle.load(f)
        print(f"Loaded PCA model from {pca_path}")
        episode_embs = pca_model.transform(raw_embs)
    else:
        print("PCA model not found, fitting new PCA...")
        pca_model = PCA(n_components=32, random_state=RANDOM_SEED)
        episode_embs = pca_model.fit_transform(raw_embs)

    print(f"Episode embeddings shape: {episode_embs.shape}, dtype: {episode_embs.dtype}")

    output = {
        "episode_indices": valid_ep_indices,
        "obj_positions": obj_positions[np.array(valid_ep_indices)],
        "episode_embs": episode_embs,
        "pca_model": pca_model,
        "raw_embs": raw_embs,
    }

    out_path = RESULTS_DIR / "step0_all_data.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(output, f)
    print(f"\nSaved all data to {out_path}")

    meta_out = []
    for i, ep_idx in enumerate(valid_ep_indices):
        meta_out.append({
            "episode_index": ep_idx,
            "obj_x": obj_positions[ep_idx, 0],
            "obj_y": obj_positions[ep_idx, 1],
            "obj_z": obj_positions[ep_idx, 2],
        })
    meta_df = pd.DataFrame(meta_out)
    meta_df.to_csv(RESULTS_DIR / "step0_episode_meta.csv", index=False)
    print(f"Saved episode metadata to {RESULTS_DIR / 'step0_episode_meta.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()