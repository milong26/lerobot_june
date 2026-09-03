"""
Our V5 Embedding Extraction

Episode-level vision-action representation that fuses:
- observation.images.top (top camera visual features)
- observation.images.wrist (wrist camera visual features)
- action trajectory (action sequence features)

Uses HuggingFaceTB/SmolVLM2-500M-Video-Instruct for visual feature extraction,
then concatenates with action features and applies PCA for dimensionality reduction.

Output: single episode-level embedding vector per episode.
"""

import sys
import os
import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List

sys.stdout.reconfigure(line_buffering=True)

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WORK2_ROOT = Path(__file__).resolve().parents[2]
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

import torch
from sklearn.decomposition import PCA

from embedding_utils.config_v5 import (
    V5_MODEL_NAME, V5_PROMPT_TEXT, V5_PCA_DIM, V5_EMBEDDING_VERSION,
    V5_TEMPORAL_ACTION_STEPS, V5_FRAME_SAMPLE_COUNT,
)


def load_v5_encoder(device: str = "cuda"):
    """
    Load the VLM model for V5 embedding extraction.
    
    Args:
        device: compute device
    
    Returns:
        (model, processor) tuple
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor
    
    print(f"Loading V5 model: {V5_MODEL_NAME}")
    
    processor = AutoProcessor.from_pretrained(V5_MODEL_NAME)
    print("Processor loaded.")
    sys.stdout.flush()
    
    model = AutoModelForImageTextToText.from_pretrained(
        V5_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    print("Model weights loaded, freezing parameters...")
    sys.stdout.flush()
    
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    print(f"Model ready on device: {device}")
    sys.stdout.flush()
    return model, processor


def extract_frame_features(
    model,
    processor,
    frames: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 256
) -> np.ndarray:
    """
    Extract visual features from a sequence of frames.
    
    Args:
        model: VLM model (frozen)
        processor: VLM processor
        frames: torch.Tensor, shape=(n_frames, C, H, W)
        device: compute device
        batch_size: batch size for processing
    
    Returns:
        np.ndarray, shape=(n_frames, embedding_dim)
    """
    n_frames = len(frames)
    n_batches = (n_frames + batch_size - 1) // batch_size
    
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    
    if frames.ndim == 4 and frames.shape[-1] in [1, 3, 4]:
        frames = frames.permute(0, 3, 1, 2)
    
    frames = frames.cpu()
    
    embeddings = []
    
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_frames)
            batch_frames = frames[start_idx:end_idx]
            
            batch_embs_list = []
            for frame in batch_frames:
                inputs = processor(
                    text=V5_PROMPT_TEXT,
                    images=frame,
                    return_tensors="pt"
                ).to(device)
                
                outputs = model(
                    **inputs,
                    output_hidden_states=True
                )
                
                hidden_states = outputs.hidden_states[-1]
                batch_embs_list.append(hidden_states.mean(dim=1).cpu().numpy())
            
            embeddings.append(np.concatenate(batch_embs_list, axis=0))
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                processed = end_idx
                print(f"    Frame batch: {batch_idx + 1}/{n_batches} ({processed}/{n_frames} frames)")
                sys.stdout.flush()
    
    return np.concatenate(embeddings, axis=0).squeeze()


def extract_action_features(
    action_sequence: np.ndarray,
    temporal_steps: int = V5_TEMPORAL_ACTION_STEPS
) -> np.ndarray:
    """
    Extract action trajectory features from an episode.
    
    Resamples the action trajectory to a fixed number of steps,
    then computes statistics to create a compact action representation.
    
    Args:
        action_sequence: np.ndarray, shape=(T, action_dim)
        temporal_steps: number of steps to resample to
    
    Returns:
        np.ndarray, 1D action feature vector
    """
    T = action_sequence.shape[0]
    
    if T == 0:
        raise ValueError("Empty action sequence")
    
    if T == 1:
        resampled = np.tile(action_sequence[0], temporal_steps)
    else:
        from scipy.interpolate import interp1d
        
        original_steps = np.arange(T)
        target_steps = np.linspace(0, T - 1, temporal_steps)
        
        resampled_actions = []
        for dim_idx in range(action_sequence.shape[1]):
            f = interp1d(original_steps, action_sequence[:, dim_idx], kind="linear")
            resampled_actions.append(f(target_steps))
        
        resampled = np.array(resampled_actions).T
    
    action_mean = np.mean(action_sequence, axis=0)
    action_std = np.std(action_sequence, axis=0)
    action_vel = np.diff(action_sequence, axis=0)
    vel_mean = np.mean(action_vel, axis=0)
    vel_std = np.std(action_vel, axis=0)
    action_range = np.max(action_sequence, axis=0) - np.min(action_sequence, axis=0)
    
    action_feature = np.concatenate([
        resampled.flatten(),
        action_mean,
        action_std,
        vel_mean,
        vel_std,
        action_range,
        action_sequence[0],
        action_sequence[-1],
    ])
    
    return action_feature


def extract_episode_embedding(
    model,
    processor,
    episode_data: Dict,
    device: str = "cuda"
) -> np.ndarray:
    """
    Extract a single episode-level embedding that fuses visual and action features.
    
    Visual features:
    - Top camera: sample frames uniformly across the episode
    - Wrist camera: sample frames uniformly across the episode
    
    Action features:
    - Resampled action trajectory + statistics
    
    Args:
        model: VLM model
        processor: VLM processor
        episode_data: Dict with keys:
            - "observation.images.top": torch.Tensor
            - "observation.images.wrist": torch.Tensor
            - "action": np.ndarray
        device: compute device
    
    Returns:
        np.ndarray, 1D episode embedding vector (before PCA)
    """
    top_frames = episode_data["observation.images.top"]
    wrist_frames = episode_data["observation.images.wrist"]
    action_seq = episode_data["action"]
    
    n_frames = len(top_frames)
    
    n_sample = min(V5_FRAME_SAMPLE_COUNT, n_frames)
    if n_sample == n_frames:
        sample_indices = list(range(n_frames))
    else:
        sample_indices = np.linspace(0, n_frames - 1, n_sample, dtype=int).tolist()
    
    print(f"  Sampling {n_sample}/{n_frames} frames for visual features")
    sys.stdout.flush()
    
    top_sampled = top_frames[sample_indices]
    wrist_sampled = wrist_frames[sample_indices]
    
    print(f"  Extracting top camera features...")
    sys.stdout.flush()
    top_features = extract_frame_features(model, processor, top_sampled, device)
    
    print(f"  Extracting wrist camera features...")
    sys.stdout.flush()
    wrist_features = extract_frame_features(model, processor, wrist_sampled, device)
    
    top_pooled = top_features.mean(axis=0)
    wrist_pooled = wrist_features.mean(axis=0)
    
    print(f"  Extracting action features...")
    sys.stdout.flush()
    action_features = extract_action_features(action_seq)
    
    episode_embedding = np.concatenate([top_pooled, wrist_pooled, action_features])
    
    print(f"  Episode embedding shape: {episode_embedding.shape}")
    print(f"    Top pooled: {top_pooled.shape}, Wrist pooled: {wrist_pooled.shape}, Action: {action_features.shape}")
    sys.stdout.flush()
    
    return episode_embedding


def fit_v5_pca(
    embeddings_list: List[np.ndarray],
    n_components: int = V5_PCA_DIM
) -> PCA:
    """
    Fit PCA on all episode embeddings for dimensionality reduction.
    
    Args:
        embeddings_list: list of 1D episode embedding arrays
        n_components: target dimensionality
    
    Returns:
        fitted PCA object
    """
    embeddings_array = np.array(embeddings_list)
    
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(embeddings_array)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  PCA fitted: {embeddings_array.shape} -> {n_components}D, explained variance: {explained_var:.4f}")
    sys.stdout.flush()
    
    return pca


def process_v5_embeddings(
    dataset_dir: Path,
    output_dir: Path,
    n_components: int = V5_PCA_DIM,
    device: str = "cuda",
) -> Dict:
    """
    Process entire LeRobotDataset: extract episode embeddings, fit PCA, save cache.
    
    Cache format:
    - ({episode_index}).npy files with {"episode_embedding": ..., "episode_index": ...}
    - pca_models/pca_v5_{n_components}.joblib
    
    Args:
        dataset_dir: LeRobotDataset root directory
        output_dir: output cache directory
        n_components: PCA dimensionality
        device: compute device
    
    Returns:
        Dict with processing info
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from embedding_utils.cache import get_dataset_episode_indices
    
    process_start_time = time.time()
    print(f"\nProcessing dataset: {dataset_dir}")
    
    print("Loading LeRobotDataset...")
    dataset = LeRobotDataset(
        repo_id="work2/metaworld_pick_place",
        root=str(dataset_dir)
    )
    print(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    
    expected_episode_indices = get_dataset_episode_indices(str(dataset_dir))
    num_episodes = len(expected_episode_indices)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    existing_count = 0
    for ep_idx in expected_episode_indices:
        ep_file = output_dir / f"({ep_idx}).npy"
        if ep_file.exists():
            existing_count += 1
    
    pca_dir = output_dir / "pca_models"
    pca_v5_file = pca_dir / f"pca_v5_{n_components}.joblib"
    pca_complete = pca_v5_file.exists()
    
    if existing_count == 0:
        print(f"\nNo cache found, will extract all {num_episodes} episodes")
    elif existing_count == num_episodes and pca_complete:
        print(f"\nCache complete ({existing_count} episodes + PCA), skipping extraction")
        return {
            "episode_indices": expected_episode_indices,
            "num_episodes": num_episodes,
            "output_dir": str(output_dir),
            "skipped": True,
        }
    else:
        print(f"\nError: PARTIAL_OR_INCONSISTENT_CACHE")
        print(f"  Existing: {existing_count}/{num_episodes} episode files")
        print(f"  PCA complete: {pca_complete}")
        print(f"  Use a fresh output directory or the shared ensure flow.")
        raise RuntimeError(
            f"PARTIAL_OR_INCONSISTENT_CACHE: {existing_count}/{num_episodes} episodes, "
            f"PCA complete: {pca_complete}."
        )
    
    print("\nLoading V5 encoder model...")
    model, processor = load_v5_encoder(device)
    print("Model loaded, starting extraction...\n")
    
    episode_embeddings = {}
    raw_embeddings_list = []
    episode_coords = []
    
    print("\nBuilding episode frame index mapping...")
    sys.stdout.flush()
    ep_indices_map = {}
    for ep_idx in range(dataset.num_episodes):
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]
        ep_indices_map[ep_idx] = list(range(from_idx, to_idx))
    print(f"Index mapping built: {len(ep_indices_map)} episodes")
    sys.stdout.flush()
    
    action_key = None
    features = dataset.meta.features
    for key in features.keys():
        if "action" in key.lower():
            action_key = key
            break
    if action_key is None:
        raise ValueError(f"No action feature found in dataset. Available: {list(features.keys())}")
    print(f"Action key: {action_key}")
    sys.stdout.flush()
    
    for ep_idx in range(dataset.num_episodes):
        ep_start_time = time.time()
        
        print(f"\nProcessing episode {ep_idx + 1}/{dataset.num_episodes} (index {ep_idx})")
        sys.stdout.flush()
        
        episode_indices = ep_indices_map.get(ep_idx, [])
        
        if not episode_indices:
            print(f"  Skipping episode {ep_idx}: no valid frames")
            continue
        
        print(f"  Found {len(episode_indices)} frames (index {episode_indices[0]}-{episode_indices[-1]}), loading...")
        sys.stdout.flush()
        
        episode_frames_top = []
        episode_frames_wrist = []
        episode_actions = []
        for idx in episode_indices:
            frame = dataset[idx]
            episode_frames_top.append(frame["observation.images.top"])
            episode_frames_wrist.append(frame["observation.images.wrist"])
            episode_actions.append(frame[action_key])
        
        top_tensor = torch.stack(episode_frames_top)
        wrist_tensor = torch.stack(episode_frames_wrist)
        action_array = np.array(episode_actions)
        
        load_time = time.time() - ep_start_time
        print(f"  Frame load: top={top_tensor.shape}, wrist={wrist_tensor.shape}, action={action_array.shape}, time={load_time:.2f}s")
        sys.stdout.flush()
        
        episode_data = {
            "observation.images.top": top_tensor,
            "observation.images.wrist": wrist_tensor,
            "action": action_array,
        }
        
        extract_start = time.time()
        ep_embedding = extract_episode_embedding(model, processor, episode_data, device)
        extract_time = time.time() - extract_start
        
        episode_embeddings[ep_idx] = ep_embedding
        raw_embeddings_list.append(ep_embedding)
        episode_coords.append(ep_idx)
        
        completed = len(episode_coords)
        progress = completed / dataset.num_episodes * 100
        avg_time = extract_time / completed if completed > 0 else 0
        eta = avg_time * (dataset.num_episodes - completed)
        
        ep_total_time = time.time() - ep_start_time
        print(f"  Episode {ep_idx} complete!")
        print(f"    Embedding dim: {ep_embedding.shape}")
        print(f"    Time: load={load_time:.1f}s, extract={extract_time:.1f}s, total={ep_total_time:.1f}s")
        print(f"    Progress: {completed}/{dataset.num_episodes} ({progress:.1f}%), ETA: {eta/60:.1f}min")
        sys.stdout.flush()
    
    if not raw_embeddings_list:
        print("No valid episode data found")
        return {
            "episode_indices": [],
            "num_episodes": 0,
            "output_dir": str(output_dir),
            "skipped": False,
        }
    
    print(f"\n{'='*60}")
    print(f"All episode embeddings extracted!")
    print(f"Processed {len(raw_embeddings_list)} episodes")
    print(f"{'='*60}")
    
    total_extract_time = time.time() - process_start_time
    print(f"Total extraction time: {total_extract_time:.2f}s ({total_extract_time/60:.1f} min)")
    if len(raw_embeddings_list) > 0:
        print(f"Average per episode: {total_extract_time/len(raw_embeddings_list):.2f}s")
    sys.stdout.flush()
    
    print(f"\nFitting PCA ({n_components}D)...")
    pca_v5 = fit_v5_pca(raw_embeddings_list, n_components)
    print("PCA fitting complete")
    
    print(f"\nSaving embeddings to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_array = np.array(raw_embeddings_list)
    for ep_idx, coord in enumerate(episode_coords):
        ep_pca = pca_v5.transform(raw_array[ep_idx:ep_idx+1])[0]
        
        coord_key = f"({coord})"
        output_file = output_dir / f"{coord_key}.npy"
        
        np.save(output_file, {
            "episode_embedding": ep_pca,
            "episode_index": coord,
        }, allow_pickle=True)
        
        if (ep_idx + 1) % 10 == 0 or ep_idx == len(episode_coords) - 1:
            print(f"  Saved {ep_idx + 1}/{len(episode_coords)} embedding files")
    
    print(f"\nSaving PCA model...")
    pca_output_dir = output_dir / "pca_models"
    pca_output_dir.mkdir(parents=True, exist_ok=True)
    
    import joblib
    joblib.dump(pca_v5, pca_output_dir / f"pca_v5_{n_components}.joblib")
    
    print(f"\n{'='*60}")
    print(f"V5 embedding extraction complete!")
    print(f"Embedding files: {output_dir}")
    print(f"PCA model: {pca_output_dir}")
    print(f"{'='*60}")
    
    return {
        "episode_indices": episode_coords,
        "num_episodes": len(episode_coords),
        "output_dir": str(output_dir),
        "skipped": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract V5 vision-action episode embeddings")
    parser.add_argument("--dataset-dir", type=str, required=True,
                       help="Dataset directory path")
    parser.add_argument("--output-dir", type=str,
                       default="personal/work2/our_v5/cache",
                       help="Embedding cache output directory")
    parser.add_argument("--n-components", type=int, default=V5_PCA_DIM,
                       help=f"PCA target dimension (default: {V5_PCA_DIM})")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Compute device")
    parser.add_argument("--metadata-output", type=str, default=None,
                       help="Optional: output metadata.json path for ensure_embeddings.py")
    args = parser.parse_args()
    
    result = process_v5_embeddings(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        n_components=args.n_components,
        device=args.device,
    )
    
    if args.metadata_output and not result.get("skipped", False):
        from embedding_utils.cache import build_expected_metadata
        meta = build_expected_metadata(
            dataset_root=str(args.dataset_dir),
            dataset_name=Path(args.dataset_dir).name.replace("pick_place_", ""),
            pca_dim=args.n_components,
            episode_indices=result["episode_indices"],
            source="v5_cli_extract",
        )
        meta_path = Path(args.metadata_output)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata written to: {meta_path}")