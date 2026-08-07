#!/usr/bin/env python
"""
Run STEP 2 only: Extract frozen VLM embeddings from existing candidate_pool dataset.
This script assumes step1 has already been run and the candidate_pool dataset exists.
"""

import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import CFG
from sic.embeddings import load_frozen_vlm, extract_all_episode_embeddings, get_episode_mean_embedding


def run_step2():
    """Extract frozen VLM embeddings for all demos in candidate pool."""
    print("\n" + "=" * 80)
    print("STEP 2: Extracting frozen VLM embeddings for all demos")
    print("=" * 80)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    pool_path = CFG.datasets_dir / 'candidate_pool'
    if not pool_path.exists():
        print(f"ERROR: Candidate pool dataset not found at {pool_path}")
        print("Please run step1 first to generate the candidate pool.")
        sys.exit(1)

    print(f"Loading candidate pool from: {pool_path}")
    pool_dataset = LeRobotDataset("work5/candidate_pool", root=str(pool_path))
    print(f"Dataset loaded: {pool_dataset.num_episodes} episodes, {pool_dataset.num_frames} frames")

    print(f"\nLoading VLM model: {CFG.vlm_model_id}")
    model, processor = load_frozen_vlm(CFG.vlm_model_id, CFG.device)
    print("Model loaded successfully")

    cache_dir = CFG.work_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = str(cache_dir / "pool_embeddings.pkl")

    all_embeddings = extract_all_episode_embeddings(
        model, processor, pool_dataset, CFG.global_cam_key, CFG.device,
        cache_path=cache_path
    )

    # Compute mean embeddings for each episode
    all_mean_embeddings = {}
    for ep_idx, ep_embs in all_embeddings.items():
        all_mean_embeddings[ep_idx] = get_episode_mean_embedding(ep_embs)

    all_mean_array = np.array([all_mean_embeddings[k] for k in sorted(all_mean_embeddings.keys())])
    print(f"\nAll mean embeddings shape: {all_mean_array.shape}")

    # Save embeddings for later use
    embeddings_data = {
        'all_embeddings': all_embeddings,
        'all_mean_array': all_mean_array,
        'num_episodes': pool_dataset.num_episodes,
    }
    
    save_path = str(cache_dir / "step2_embeddings.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(embeddings_data, f)
    print(f"Embeddings saved to: {save_path}")

    print("\n" + "=" * 80)
    print("STEP 2 COMPLETE!")
    print("=" * 80)
    print(f"Processed {pool_dataset.num_episodes} episodes")
    print(f"Embedding dimension: {all_mean_array.shape[1]}")
    print(f"Cache saved at: {cache_path}")
    print(f"Step2 output saved at: {save_path}")
    print("\nNext step: Run step3 to apply selection strategies")
    print("Command: python personal/work5/run_step3.py")


if __name__ == "__main__":
    run_step2()