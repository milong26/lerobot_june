"""
Step 2: Extract VLM embeddings from ALL frames of each episode.

Saves per-episode embeddings so that K5/K7 analysis can sample from these
embeddings directly without re-reading raw images.

Usage:
    python extract_embeddings.py --force
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

# Save directory: all_frames/episode_XXX.npy
EMBEDDINGS_DIR = Path(__file__).parent / "all_frames"
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

    return embedding.cpu().numpy()  # (1, D)


def extract_all_episodes(dataset_dir, episode_indices, model_id=VLM_MODEL_ID, device=None, force=False):
    """Extract VLM embeddings for ALL frames of each episode."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Check cache
    all_done = True
    for ep_idx in episode_indices:
        ep_path = EMBEDDINGS_DIR / f"episode_{ep_idx:04d}.npy"
        if not ep_path.exists():
            all_done = False
            break

    if all_done and not force:
        print("All episode embeddings already cached.")
        return

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

    # Extract embeddings for ALL frames of each episode
    for ep_idx in tqdm(episode_indices, desc="Extracting VLM embeddings (all frames)"):
        ep_path = EMBEDDINGS_DIR / f"episode_{ep_idx:04d}.npy"
        if ep_path.exists() and not force:
            continue

        frame_range = episode_frame_ranges[ep_idx]
        start_idx = frame_range["start"]
        end_idx = frame_range["end"]
        total_frames = end_idx - start_idx + 1

        # Extract ALL frames
        embeddings_list = []
        for frame_idx in range(start_idx, end_idx + 1):
            frame = dataset[frame_idx]
            emb = extract_frame_embedding(model, processor, frame["observation.images.top"], device)
            embeddings_list.append(emb)

        # Shape: (num_frames, D)
        episode_embeddings = np.concatenate(embeddings_list, axis=0)
        np.save(ep_path, episode_embeddings)

    print(f"Saved episode embeddings to {EMBEDDINGS_DIR}/")


def sample_keyframe_indices(total_frames, num_keyframes):
    """Sample keyframe indices by normalized time."""
    indices = []
    for i in range(num_keyframes):
        t = i / (num_keyframes - 1)
        idx = round(t * (total_frames - 1))
        idx = min(idx, total_frames - 1)
        indices.append(idx)
    return indices


def load_episode_embeddings(episode_indices):
    """Load pre-extracted episode embeddings from disk."""
    embeddings_list = []
    frame_counts = []
    for ep_idx in episode_indices:
        ep_path = EMBEDDINGS_DIR / f"episode_{ep_idx:04d}.npy"
        if not ep_path.exists():
            raise FileNotFoundError(f"Episode {ep_idx} not found at {ep_path}")
        emb = np.load(ep_path)
        embeddings_list.append(emb)
        frame_counts.append(len(emb))
    return embeddings_list, frame_counts


def build_episode_features(episode_indices, num_keyframes=5, pooling="concat"):
    """
    Build per-episode features from pre-extracted frame embeddings.
    
    Args:
        episode_indices: list of episode indices
        num_keyframes: K value for keyframe sampling
        pooling: 'max', 'mean', 'concat' - how to pool K frames
    
    Returns:
        (N, D') array
    """
    embeddings_list, frame_counts = load_episode_embeddings(episode_indices)
    
    features = []
    for emb in embeddings_list:
        total_frames = len(emb)
        keyframe_offsets = sample_keyframe_indices(total_frames, num_keyframes)
        keyframe_embs = emb[keyframe_offsets]  # (K, D)
        
        if pooling == "max":
            features.append(keyframe_embs.max(axis=0))
        elif pooling == "mean":
            features.append(keyframe_embs.mean(axis=0))
        elif pooling == "concat":
            features.append(keyframe_embs.reshape(-1))
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
    
    return np.array(features)


def apply_pca(features, n_components=32, cache=True, num_keyframes=5, pooling="concat"):
    """Apply PCA to features."""
    from sklearn.decomposition import PCA
    import pickle

    k_dir = PCA_DIR / f"k{num_keyframes}_{pooling}"
    k_dir.mkdir(parents=True, exist_ok=True)
    
    pca_path = k_dir / "pca_embeddings.npy"
    pca_model_path = k_dir / "pca_model.pkl"
    pca_meta_path = k_dir / "pca_metadata.json"

    if cache and pca_path.exists() and pca_model_path.exists():
        with open(pca_meta_path) as f:
            meta = json.load(f)
        if meta.get("n_components") == n_components and meta.get("num_episodes") == features.shape[0]:
            print(f"Loading cached PCA embeddings from {pca_path}")
            pca_embeddings = np.load(pca_path)
            with open(pca_model_path, "rb") as f:
                pca_model = pickle.load(f)
            return pca_embeddings, pca_model

    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(features)

    if cache:
        np.save(pca_path, pca_embeddings)
        with open(pca_model_path, "wb") as f:
            pickle.dump(pca, f)
        pca_meta = {
            "n_components": n_components,
            "original_dim": features.shape[-1],
            "num_keyframes": num_keyframes,
            "pooling": pooling,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "num_episodes": features.shape[0],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(pca_meta_path, "w") as f:
            json.dump(pca_meta, f, indent=2)

    print(f"PCA: {features.shape[-1]} -> {n_components} dimensions")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return pca_embeddings, pca


def load_embeddings(num_keyframes=5, n_components=32, pooling="concat", force=False, num_episodes=None):
    """Load or compute embeddings and PCA."""
    # Load pool metadata
    pool_dir = Path(__file__).parent / "pool"
    meta_path = pool_dir / "episode_metadata.csv"
    
    import pandas as pd
    df = pd.read_csv(meta_path)
    episode_indices = df["episode_index"].tolist()
    
    if num_episodes is not None:
        episode_indices = episode_indices[:num_episodes]
        print(f"Using first {num_episodes} episodes for testing")

    # Build features from pre-extracted frame embeddings
    features = build_episode_features(episode_indices, num_keyframes, pooling)
    print(f"Features shape: {features.shape}")

    # PCA
    pca_embeddings, pca_model = apply_pca(features, n_components, num_keyframes=num_keyframes, pooling=pooling)

    return features, pca_embeddings, episode_indices, pca_model


def main():
    parser = argparse.ArgumentParser(description="Extract VLM embeddings for all frames")
    parser.add_argument("--force", action="store_true", help="Force re-extraction")
    args = parser.parse_args()

    print(f"Extracting embeddings using {VLM_MODEL_ID}...")

    # Load pool metadata
    pool_dir = Path(__file__).parent / "pool"
    meta_path = pool_dir / "episode_metadata.csv"
    import pandas as pd
    df = pd.read_csv(meta_path)
    episode_indices = df["episode_index"].tolist()

    extract_all_episodes(
        dataset_dir=DATASET_DIR,
        episode_indices=episode_indices,
        force=args.force,
    )

    print("\nDone! All episode embeddings saved.")


if __name__ == "__main__":
    main()