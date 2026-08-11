#!/usr/bin/env python
"""
Extract wrist camera embeddings and append to existing episode embeddings.

This script:
1. Loads existing top camera embeddings from all_frames/episode_XXXX.npy
2. Extracts wrist camera embeddings
3. Concatenates top + wrist embeddings
4. Saves back to all_frames/episode_XXXX.npy

Usage:
    python extract_wrist_embeddings.py --force
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

EMBEDDINGS_DIR = Path(__file__).parent / "all_frames"
DATASET_DIR = Path(__file__).parent.parent / "dataset"

VLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def load_vlm_model(model_id=VLM_MODEL_ID, device=None):
    """Load SmolVLM2 as a frozen visual feature extractor."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    for param in model.parameters():
        param.requires_grad = False

    print(f"Loaded VLM model: {model_id} on {device}")
    return model, processor


def extract_frame_embedding(model, processor, frame, device):
    """Extract visual embedding from a single frame."""
    if isinstance(frame, torch.Tensor):
        img_np = (frame.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
    elif isinstance(frame, np.ndarray):
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        pil_img = Image.fromarray(frame)
    elif isinstance(frame, Image.Image):
        pil_img = frame
    else:
        raise ValueError(f"Unknown frame type: {type(frame)}")

    inputs = processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    if pixel_values.ndim == 5:
        pixel_values = pixel_values[:, 0]

    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=pixel_values)
        hidden_states = vision_out.last_hidden_state
        embedding = model.connector(hidden_states)
        embedding = embedding.mean(dim=1)

    return embedding.cpu().numpy()


def extract_wrist_and_concatenate(dataset_dir, episode_indices, model_id=VLM_MODEL_ID, device=None, force=False):
    """Extract wrist embeddings and concatenate with existing top embeddings."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = load_vlm_model(model_id, device)

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = "lerobot/metaworld_pick_place"
    dataset = LeRobotDataset(repo_id, root=dataset_dir)

    print("Building episode -> frame index mapping...")
    episode_frame_ranges = {}
    scan_limit = min(dataset.num_frames, 35000)
    for idx in tqdm(range(scan_limit), desc="Scanning episodes"):
        frame = dataset[idx]
        ep_idx = int(frame["episode_index"])
        if ep_idx not in episode_frame_ranges:
            episode_frame_ranges[ep_idx] = {"start": idx, "end": idx}
        else:
            episode_frame_ranges[ep_idx]["end"] = idx
        if len(episode_frame_ranges) >= len(episode_indices):
            print(f"  Found all {len(episode_indices)} episodes after scanning {idx+1} frames")
            break

    missing = [ep for ep in episode_indices if ep not in episode_frame_ranges]
    if missing:
        print(f"  WARNING: Missing {len(missing)} episodes, using fallback")
        avg_frames_per_ep = dataset.num_frames // dataset.num_episodes
        for ep_idx in episode_indices:
            if ep_idx not in episode_frame_ranges:
                start = int(ep_idx * avg_frames_per_ep)
                episode_frame_ranges[ep_idx] = {"start": start, "end": start + avg_frames_per_ep - 1}

    for ep_idx in tqdm(episode_indices, desc="Extracting wrist + concatenating"):
        ep_path = EMBEDDINGS_DIR / f"episode_{ep_idx:04d}.npy"
        
        if not ep_path.exists():
            print(f"  WARNING: {ep_path} not found, skipping")
            continue

        if ep_path.exists() and not force:
            existing = np.load(ep_path)
            if existing.shape[1] == 1920:
                continue

        frame_range = episode_frame_ranges[ep_idx]
        start_idx = frame_range["start"]
        end_idx = frame_range["end"]

        wrist_embeddings = []
        for frame_idx in range(start_idx, end_idx + 1):
            frame = dataset[frame_idx]
            emb = extract_frame_embedding(model, processor, frame["observation.images.wrist"], device)
            wrist_embeddings.append(emb)

        wrist_emb = np.concatenate(wrist_embeddings, axis=0)

        existing_top = np.load(ep_path)
        combined = np.concatenate([existing_top, wrist_emb], axis=1)

        np.save(ep_path, combined)

    print(f"Updated episode embeddings in {EMBEDDINGS_DIR}/")
    print("New shape: (num_frames, 1920) = top(960) + wrist(960)")


def main():
    parser = argparse.ArgumentParser(description="Extract wrist embeddings and concatenate")
    parser.add_argument("--force", action="store_true", help="Force re-extraction")
    args = parser.parse_args()

    pool_dir = Path(__file__).parent / "pool"
    meta_path = pool_dir / "episode_metadata.csv"
    import pandas as pd
    df = pd.read_csv(meta_path)
    episode_indices = df["episode_index"].tolist()

    extract_wrist_and_concatenate(
        dataset_dir=DATASET_DIR,
        episode_indices=episode_indices,
        force=args.force,
    )

    print("\nDone! Wrist embeddings extracted and concatenated.")


if __name__ == "__main__":
    main()