"""
Step 2: Extract VLM embeddings from dataset videos using temporal keyframe sampling.

Supports K=3/5/7/9 keyframes per episode.
Each K is saved separately in work2/ours/pca/k{K}/

Usage:
    python extract_embeddings.py --num-keyframes 5
    python extract_embeddings.py --num-keyframes 5 --force
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

PCA_DIR = Path(__file__).parent / "pca"
DATASET_DIR = Path(__file__).parent.parent / "dataset"

VLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def load_vlm_model(model_id=VLM_MODEL_ID, device=None):
    """Load SmolVLM2 as a frozen visual feature extractor."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    # Freeze
    for param in model.parameters():
        param.requires_grad = False

    print(f"Loaded VLM model: {model_id} on {device}")
    return model, processor


def sample_keyframe_indices(total_frames, num_keyframes):
    """Sample keyframe indices by normalized time.
    
    For K=5: 0%, 25%, 50%, 75%, 100%
    For K=3: 0%, 50%, 100%
    
    Returns:
        list of exactly num_keyframes frame indices
    """
    indices = []
    for i in range(num_keyframes):
        t = i / (num_keyframes - 1)  # normalized time [0, 1]
        idx = round(t * (total_frames - 1))
        idx = min(idx, total_frames - 1)
        indices.append(idx)
    
    return indices


def extract_vlm_embedding(model, processor, frames, device):
    """Extract visual embedding from a list of frames using SmolVLM2.
    
    Args:
        model: SmolVLM model
        processor: SmolVLM processor  
        frames: list of frame tensors (C, H, W)
        device: torch device
    
    Returns:
        numpy array of shape (num_frames, hidden_dim)
    """
    frame_embeddings = []
    
    for frame in frames:
        # Convert to PIL
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
        
        # Process through processor
        inputs = processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        
        # SmolVLM2 processor returns 5D: (batch, num_frames, C, H, W)
        # vision_model expects 4D: (batch, C, H, W)
        if pixel_values.ndim == 5:
            pixel_values = pixel_values[:, 0]  # squeeze frame dim
        
        pixel_values = pixel_values.to(device)
        
        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=pixel_values)
            hidden_states = vision_out.last_hidden_state
            embedding = model.connector(hidden_states)
            embedding = embedding.mean(dim=1)  # (1, hidden_dim)
            frame_embeddings.append(embedding)
    
    # Stack: (num_keyframes, hidden_dim)
    return torch.cat(frame_embeddings, dim=0).cpu().numpy()


def extract_embeddings_from_dataset(dataset_dir, episode_indices, num_keyframes, model_id=VLM_MODEL_ID, device=None, force=False):
    """Extract keyframe embeddings from the dataset using SmolVLM2."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Per-K cache directory
    k_dir = PCA_DIR / f"k{num_keyframes}"
    k_dir.mkdir(parents=True, exist_ok=True)
    
    cache_path = k_dir / "raw_embeddings.npy"
    cache_meta_path = k_dir / "embedding_metadata.json"

    if cache_path.exists() and cache_meta_path.exists() and not force:
        with open(cache_meta_path) as f:
            meta = json.load(f)
        if meta.get("model_name") == model_id and meta.get("num_keyframes") == num_keyframes and meta.get("num_episodes") == len(episode_indices):
            print(f"Loading cached embeddings from {cache_path}")
            raw_embeddings = np.load(cache_path)
            print(f"Loaded embeddings shape: {raw_embeddings.shape}")
            return raw_embeddings, list(range(len(episode_indices))), None

    model, processor = load_vlm_model(model_id, device)

    # Load dataset
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = "lerobot/metaworld_pick_place"
    dataset = LeRobotDataset(repo_id, root=dataset_dir)

    # Build mapping: episode_index -> (start_frame_idx, end_frame_idx)
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

    # Verify we found all episodes
    missing = [ep for ep in episode_indices if ep not in episode_frame_ranges]
    if missing:
        print(f"  WARNING: Missing {len(missing)} episodes, using fallback")
        avg_frames_per_ep = dataset.num_frames // dataset.num_episodes
        for ep_idx in episode_indices:
            if ep_idx not in episode_frame_ranges:
                start = int(ep_idx * avg_frames_per_ep)
                episode_frame_ranges[ep_idx] = {"start": start, "end": start + avg_frames_per_ep - 1}

    # Extract embeddings
    embeddings_list = []

    for ep_idx in tqdm(episode_indices, desc=f"Extracting VLM embeddings (K={num_keyframes})"):
        frame_range = episode_frame_ranges[ep_idx]
        start_idx = frame_range["start"]
        end_idx = frame_range["end"]
        total_frames = end_idx - start_idx + 1
        
        # Sample keyframe indices
        keyframe_offsets = sample_keyframe_indices(total_frames, num_keyframes)
        
        # Collect keyframe images
        keyframes = []
        for offset in keyframe_offsets:
            frame = dataset[start_idx + offset]
            keyframes.append(frame["observation.images.top"])

        # Extract embeddings for all keyframes: (K, hidden_dim)
        embedding = extract_vlm_embedding(model, processor, keyframes, device)
        embeddings_list.append(embedding)

    # Shape: (N, K, hidden_dim)
    embeddings = np.stack(embeddings_list, axis=0)
    np.save(cache_path, embeddings)

    meta = {
        "model_name": model_id,
        "num_keyframes": num_keyframes,
        "num_episodes": len(episode_indices),
        "embedding_dim": int(embeddings.shape[2]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(cache_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved embeddings: {embeddings.shape} to {cache_path}")
    return embeddings, list(range(len(episode_indices))), None


def apply_pca(embeddings, n_components=32, cache=True, num_keyframes=5):
    """Apply PCA to embeddings. Supports caching."""
    from sklearn.decomposition import PCA
    import pickle

    k_dir = PCA_DIR / f"k{num_keyframes}"
    k_dir.mkdir(parents=True, exist_ok=True)
    
    pca_path = k_dir / "pca_embeddings.npy"
    pca_model_path = k_dir / "pca_model.pkl"
    pca_meta_path = k_dir / "pca_metadata.json"

    if cache and pca_path.exists() and pca_model_path.exists():
        with open(pca_meta_path) as f:
            meta = json.load(f)
        if meta.get("n_components") == n_components:
            print(f"Loading cached PCA embeddings from {pca_path}")
            pca_embeddings = np.load(pca_path)
            with open(pca_model_path, "rb") as f:
                pca_model = pickle.load(f)
            return pca_embeddings, pca_model

    # Flatten for PCA: (N, K*D) -> (N, n_components)
    N, K, D = embeddings.shape
    flat = embeddings.reshape(N, K * D)

    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(flat)

    if cache:
        np.save(pca_path, pca_embeddings)
        with open(pca_model_path, "wb") as f:
            pickle.dump(pca, f)
        pca_meta = {
            "n_components": n_components,
            "original_dim": K * D,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "num_episodes": N,
            "num_keyframes": num_keyframes,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(pca_meta_path, "w") as f:
            json.dump(pca_meta, f, indent=2)

    print(f"PCA: {K*D} -> {n_components} dimensions")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return pca_embeddings, pca


def load_embeddings(num_keyframes=5, n_components=32, force=False, num_episodes=None):
    """Load or compute embeddings and PCA."""
    # Load pool metadata
    pool_dir = Path(__file__).parent / "pool"
    meta_path = pool_dir / "episode_metadata.csv"
    
    import pandas as pd
    df = pd.read_csv(meta_path)
    episode_indices = df["episode_index"].tolist()
    
    # Limit episodes for quick testing
    if num_episodes is not None:
        episode_indices = episode_indices[:num_episodes]
        print(f"Using first {num_episodes} episodes for testing")

    raw_embeddings, indices, _ = extract_embeddings_from_dataset(
        dataset_dir=DATASET_DIR,
        episode_indices=episode_indices,
        num_keyframes=num_keyframes,
        force=force,
    )

    # Check if PCA cache matches raw embeddings
    k_dir = PCA_DIR / f"k{num_keyframes}"
    pca_path = k_dir / "pca_embeddings.npy"
    force_pca = force
    if pca_path.exists() and not force_pca:
        cached_pca = np.load(pca_path)
        if cached_pca.shape[0] != len(episode_indices):
            print(f"  PCA cache mismatch: cached {cached_pca.shape[0]} episodes, need {len(episode_indices)}")
            print(f"  Forcing PCA recomputation...")
            force_pca = True

    pca_embeddings, pca_model = apply_pca(raw_embeddings, n_components=n_components, num_keyframes=num_keyframes, cache=not force_pca)

    return raw_embeddings, pca_embeddings, indices, pca_model


def main():
    parser = argparse.ArgumentParser(description="Extract VLM embeddings with temporal keyframe sampling")
    parser.add_argument("--num-keyframes", type=int, default=5, choices=[3, 5, 7, 9],
                        help="Number of temporal keyframes per episode (default: 5)")
    parser.add_argument("--pca-dim", type=int, default=32, help="PCA dimensionality (default: 32)")
    parser.add_argument("--num-episodes", type=int, default=None, help="Number of episodes to process (default: all)")
    parser.add_argument("--force", action="store_true", help="Force re-extraction even if cache exists")
    args = parser.parse_args()

    print(f"Extracting embeddings using {VLM_MODEL_ID}...")
    print(f"Keyframes: K={args.num_keyframes}")
    print(f"PCA dim: {args.pca_dim}")
    if args.num_episodes:
        print(f"Episodes: {args.num_episodes} (testing mode)")
    else:
        print("Episodes: all 500")

    raw_emb, pca_emb, indices, pca_model = load_embeddings(
        num_keyframes=args.num_keyframes,
        n_components=args.pca_dim,
        force=args.force,
        num_episodes=args.num_episodes,
    )

    print(f"\nDone!")
    print(f"Raw embeddings shape: {raw_emb.shape}")
    print(f"PCA embeddings shape: {pca_emb.shape}")


if __name__ == "__main__":
    main()