#!/usr/bin/env python
"""
Quick test script: Process only 3 episodes to verify the pipeline works.
Takes ~1-2 minutes instead of hours.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import CFG
from sic.embeddings import load_frozen_vlm, extract_visual_embedding


def quick_test():
    """Test the pipeline with only 3 episodes."""
    print("\n" + "=" * 80)
    print("QUICK TEST: Processing 3 episodes only")
    print("=" * 80)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    pool_path = CFG.datasets_dir / 'candidate_pool'
    if not pool_path.exists():
        print(f"ERROR: Candidate pool not found at {pool_path}")
        print("Please run step1 first.")
        sys.exit(1)

    print(f"Loading dataset from: {pool_path}")
    dataset = LeRobotDataset("work5/candidate_pool", root=str(pool_path))
    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    # Test only first 3 episodes
    n_test = min(3, dataset.num_episodes)
    print(f"\nTesting with {n_test} episodes...")

    # Load model
    print(f"Loading VLM: {CFG.vlm_model_id}")
    model, processor = load_frozen_vlm(CFG.vlm_model_id, CFG.device)
    print("Model loaded ✓")

    # Test embedding extraction
    embeddings = []
    for ep_idx in range(n_test):
        print(f"\nEpisode {ep_idx}:")
        
        # Find first frame of this episode
        first_frame_idx = None
        for idx in range(len(dataset)):
            frame = dataset[idx]
            if frame.get('episode_index') == ep_idx:
                first_frame_idx = idx
                break
        
        if first_frame_idx is None:
            print(f"  Episode {ep_idx} not found!")
            continue
        
        # Get image from first frame
        frame = dataset[first_frame_idx]
        img_tensor = frame[CFG.global_cam_key]
        
        if isinstance(img_tensor, torch.Tensor):
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            img_np = img_tensor.astype(np.uint8)
        
        print(f"  Image shape: {img_np.shape}")
        
        # Extract embedding
        img_pil = Image.fromarray(img_np)
        emb = extract_visual_embedding(model, processor, img_pil, CFG.device)
        embeddings.append(emb)
        
        print(f"  Embedding shape: {emb.shape}")
        print(f"  Embedding norm: {np.linalg.norm(emb):.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("QUICK TEST RESULTS")
    print("=" * 80)
    print(f"Processed {len(embeddings)} episodes successfully")
    if len(embeddings) > 0:
        print(f"Embedding dimension: {embeddings[0].shape[0]}")
        print(f"Mean embedding norm: {np.mean([np.linalg.norm(e) for e in embeddings]):.4f}")
        print("\n✓ Pipeline is working correctly!")
        print("You can now run the full pipeline with confidence.")
    else:
        print("\n✗ No embeddings extracted. Check errors above.")


if __name__ == "__main__":
    quick_test()