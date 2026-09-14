#!/usr/bin/env python
"""
Robomme visual embedding and action descriptor extraction.

Extracts phi_global, phi_wrist visual embeddings and action descriptors
for Robomme datasets, using the same extraction pipeline as the original
Ours-v5 (personal/work2/embedding_utils/ensure_embeddings_v5.py).

Robomme camera keys:
  - observation.images.image -> global camera (mapped to camera1 for training)
  - observation.images.wrist_image -> wrist camera (mapped to camera2 for training)

Cache is saved to: personal/work2/duibi_robomme/cache/<task>/embeddings/
The same cache is shared between Ours-v5 and FPS selectors.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project roots to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
WORK2_ROOT = REPO_ROOT / "personal" / "work2"
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embedding_utils.config import (
    DEFAULT_PCA_DIM,
    build_extraction_method_name,
)
from embedding_utils.config_v5 import (
    V5_SHARED_EMBEDDING_ROOT,
    V5_EMBEDDING_VERSION,
    V5_ACTION_DESCRIPTOR_VERSION,
    V5_ACTION_STEPS,
    V5_ACTION_FEATURES,
    build_v5_action_descriptor_name,
)
from embedding_utils.cache import (
    normalize_dataset_name,
    get_dataset_episode_indices,
    write_cache_metadata,
)


# ---------------------------------------------------------------------------
# Visual embedding extraction (adapted for Robomme camera keys)
# ---------------------------------------------------------------------------

ROBOMME_CAMERA_KEYS = {
    "global": "observation.images.image",
    "wrist": "observation.images.wrist_image",
}


def extract_visual_embeddings_for_episode(
    dataset,
    episode_idx: int,
    model,
    processor,
    device: str,
    pca_model_global=None,
    pca_model_wrist=None,
    global_frame_indices: Optional[List[int]] = None,
    wrist_frame_range: Optional[Tuple[int, int]] = None,
) -> Optional[Dict]:
    """
    Extract phi_global and phi_wrist for a single episode.

    Video-encoded images are accessed via dataset[idx][cam_key], not via
    hf_dataset.filter().column_names.
    """
    global_cam_key = ROBOMME_CAMERA_KEYS["global"]
    wrist_cam_key = ROBOMME_CAMERA_KEYS["wrist"]

    camera_keys = dataset.meta.camera_keys
    if global_cam_key not in camera_keys:
        print(f"  WARNING: {global_cam_key} not in camera_keys {camera_keys}")
        return None
    if wrist_cam_key not in camera_keys:
        print(f"  WARNING: {wrist_cam_key} not in camera_keys {camera_keys}")
        return None

    # Get frame range for this episode from metadata
    ep_meta = dataset.meta.episodes[episode_idx]
    ep_from = ep_meta["dataset_from_index"]
    ep_to = ep_meta["dataset_to_index"]
    n_frames = ep_to - ep_from

    if n_frames == 0:
        return None

    if global_frame_indices is None:
        global_frame_indices = list(range(min(5, n_frames)))

    if wrist_frame_range is None:
        wrist_frame_range = (0.2, 0.7)

    wrist_start = int(n_frames * wrist_frame_range[0])
    wrist_end = int(n_frames * wrist_frame_range[1])
    wrist_indices = list(range(wrist_start, wrist_end))

    # Extract frames via dataset[idx] (handles video decoding)
    global_images = []
    for i in global_frame_indices:
        if i < n_frames:
            frame_idx = ep_from + i
            img = dataset[frame_idx][global_cam_key]
            if hasattr(img, "numpy"):
                img = img.numpy()
            global_images.append(np.array(img))

    wrist_images = []
    for i in wrist_indices:
        if i < n_frames:
            frame_idx = ep_from + i
            img = dataset[frame_idx][wrist_cam_key]
            if hasattr(img, "numpy"):
                img = img.numpy()
            wrist_images.append(np.array(img))

    if len(global_images) == 0 or len(wrist_images) == 0:
        return None

    with torch.no_grad():
        inputs_global = processor(images=[global_images], return_tensors="pt")
        inputs_global = {k: v.to(device) for k, v in inputs_global.items()}
        outputs_global = model(**inputs_global, output_hidden_states=True)
        hidden_global = outputs_global.hidden_states[-1]
        phi_global = hidden_global.mean(dim=1).cpu().numpy().squeeze(0)

        inputs_wrist = processor(images=[wrist_images], return_tensors="pt")
        inputs_wrist = {k: v.to(device) for k, v in inputs_wrist.items()}
        outputs_wrist = model(**inputs_wrist, output_hidden_states=True)
        hidden_wrist = outputs_wrist.hidden_states[-1]
        phi_wrist = hidden_wrist.mean(dim=1).cpu().numpy().squeeze(0)

    if pca_model_global is not None:
        phi_global = pca_model_global.transform(phi_global.reshape(1, -1)).squeeze(0)
    if pca_model_wrist is not None:
        phi_wrist = pca_model_wrist.transform(phi_wrist.reshape(1, -1)).squeeze(0)

    return {
        "episode_index": episode_idx,
        "phi_global": phi_global.astype(np.float32),
        "phi_wrist": phi_wrist.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Action descriptor extraction
# ---------------------------------------------------------------------------

def extract_action_descriptor_for_episode(
    dataset,
    episode_idx: int,
    n_steps: int = V5_ACTION_STEPS,
    feature_types: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    if feature_types is None:
        feature_types = V5_ACTION_FEATURES

    ep_meta = dataset.meta.episodes[episode_idx]
    ep_from = ep_meta["dataset_from_index"]
    ep_to = ep_meta["dataset_to_index"]
    n_frames = ep_to - ep_from

    if n_frames == 0:
        return None

    if n_frames <= n_steps:
        indices = list(range(n_frames))
    else:
        indices = np.linspace(0, n_frames - 1, n_steps, dtype=int).tolist()

    features = []
    prev_action = None
    for i in indices:
        frame_idx = ep_from + i
        action = np.array(dataset[frame_idx]["action"], dtype=np.float32)
        for feat_type in feature_types:
            if feat_type == "raw":
                features.extend(action.tolist())
            elif feat_type == "diff":
                if prev_action is not None:
                    diff = action - prev_action
                else:
                    diff = np.zeros_like(action)
                features.extend(diff.tolist())
            elif feat_type == "norm":
                features.append(float(np.linalg.norm(action)))
        prev_action = action.copy()

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def get_robomme_embedding_cache_dir(task_name: str, pca_dim: int = DEFAULT_PCA_DIM) -> Path:
    return Path(V5_SHARED_EMBEDDING_ROOT) / f"robomme_{task_name}" / build_extraction_method_name(pca_dim)


def get_robomme_action_cache_dir(task_name: str) -> Path:
    return Path(V5_SHARED_EMBEDDING_ROOT) / f"robomme_{task_name}" / build_v5_action_descriptor_name()


def ensure_visual_embeddings(
    dataset_root: str,
    task_name: str,
    model,
    processor,
    device: str,
    pca_dim: int = DEFAULT_PCA_DIM,
    force_reextract: bool = False,
) -> Dict[int, np.ndarray]:
    cache_dir = get_robomme_embedding_cache_dir(task_name, pca_dim)
    cache_dir.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(repo_id="1/2", root=dataset_root)
    total_episodes = dataset.num_episodes
    expected_indices = list(range(total_episodes))

    if not force_reextract:
        embeddings = {}
        for ep_idx in expected_indices:
            cache_file = cache_dir / f"({ep_idx}).npy"
            if cache_file.exists():
                try:
                    data = np.load(str(cache_file), allow_pickle=True).item()
                    if "phi_global" in data and "phi_wrist" in data:
                        embeddings[int(ep_idx)] = np.concatenate([
                            data["phi_global"], data["phi_wrist"]
                        ])
                except Exception:
                    pass

        if len(embeddings) == len(expected_indices):
            print(f"Loaded {len(embeddings)} visual embeddings from cache: {cache_dir}")
            return embeddings

    print(f"Extracting visual embeddings for {len(expected_indices)} episodes...")

    pca_global_file = cache_dir / "pca_global.npy"
    pca_wrist_file = cache_dir / "pca_wrist.npy"
    pca_global = None
    pca_wrist = None

    if pca_global_file.exists():
        from sklearn.decomposition import PCA
        data = np.load(str(pca_global_file), allow_pickle=True).item()
        pca_global = PCA(n_components=pca_dim)
        pca_global.mean_ = data["mean"]
        pca_global.components_ = data["components"]
        pca_global.explained_variance_ = data["explained_variance"]
        pca_global.explained_variance_ratio_ = data["explained_variance_ratio"]
        pca_global.n_components_ = pca_dim

    if pca_wrist_file.exists():
        from sklearn.decomposition import PCA
        data = np.load(str(pca_wrist_file), allow_pickle=True).item()
        pca_wrist = PCA(n_components=pca_dim)
        pca_wrist.mean_ = data["mean"]
        pca_wrist.components_ = data["components"]
        pca_wrist.explained_variance_ = data["explained_variance"]
        pca_wrist.explained_variance_ratio_ = data["explained_variance_ratio"]
        pca_wrist.n_components_ = pca_dim

    embeddings = {}
    for ep_idx in expected_indices:
        result = extract_visual_embeddings_for_episode(
            dataset, ep_idx, model, processor, device,
            pca_model_global=pca_global,
            pca_model_wrist=pca_wrist,
        )
        if result is not None:
            cache_file = cache_dir / f"({ep_idx}).npy"
            np.save(str(cache_file), result)
            embeddings[int(ep_idx)] = np.concatenate([
                result["phi_global"], result["phi_wrist"]
            ])

    write_cache_metadata(cache_dir, len(expected_indices), pca_dim)
    print(f"Extracted {len(embeddings)} visual embeddings, cached to {cache_dir}")
    return embeddings


def ensure_action_descriptors(
    dataset_root: str,
    task_name: str,
    force_reextract: bool = False,
) -> Dict[int, np.ndarray]:
    cache_dir = get_robomme_action_cache_dir(task_name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(repo_id="1/2", root=dataset_root)
    total_episodes = dataset.num_episodes
    expected_indices = list(range(total_episodes))

    if not force_reextract:
        descriptors = {}
        for ep_idx in expected_indices:
            cache_file = cache_dir / f"({ep_idx}).npy"
            if cache_file.exists():
                try:
                    data = np.load(str(cache_file), allow_pickle=True).item()
                    descriptors[int(ep_idx)] = data["action_descriptor"]
                except Exception:
                    pass

        if len(descriptors) == len(expected_indices):
            print(f"Loaded {len(descriptors)} action descriptors from cache: {cache_dir}")
            return descriptors

    print(f"Extracting action descriptors for {len(expected_indices)} episodes...")

    descriptors = {}
    for ep_idx in expected_indices:
        desc = extract_action_descriptor_for_episode(dataset, ep_idx)
        if desc is not None:
            cache_file = cache_dir / f"({ep_idx}).npy"
            np.save(str(cache_file), {
                "episode_index": ep_idx,
                "action_descriptor": desc,
            })
            descriptors[int(ep_idx)] = desc

    metadata = {
        "num_episodes": len(expected_indices),
        "version": V5_ACTION_DESCRIPTOR_VERSION,
        "action_steps": V5_ACTION_STEPS,
        "action_features": V5_ACTION_FEATURES,
    }
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Extracted {len(descriptors)} action descriptors, cached to {cache_dir}")
    return descriptors


def load_cached_visual_embeddings(
    dataset_root: str,
    task_name: str,
    pca_dim: int = DEFAULT_PCA_DIM,
) -> Optional[Dict[int, np.ndarray]]:
    cache_dir = get_robomme_embedding_cache_dir(task_name, pca_dim)
    if not cache_dir.exists():
        return None

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(repo_id="1/2", root=dataset_root)
    total_episodes = dataset.num_episodes
    expected_indices = list(range(total_episodes))

    embeddings = {}
    for ep_idx in expected_indices:
        cache_file = cache_dir / f"({ep_idx}).npy"
        if cache_file.exists():
            try:
                data = np.load(str(cache_file), allow_pickle=True).item()
                if "phi_global" in data and "phi_wrist" in data:
                    embeddings[int(ep_idx)] = np.concatenate([
                        data["phi_global"], data["phi_wrist"]
                    ])
            except Exception:
                pass

    if len(embeddings) == len(expected_indices):
        return embeddings
    return None


def load_cached_action_descriptors(
    dataset_root: str,
    task_name: str,
) -> Optional[Dict[int, np.ndarray]]:
    cache_dir = get_robomme_action_cache_dir(task_name)
    if not cache_dir.exists():
        return None

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(repo_id="1/2", root=dataset_root)
    total_episodes = dataset.num_episodes
    expected_indices = list(range(total_episodes))

    descriptors = {}
    for ep_idx in expected_indices:
        cache_file = cache_dir / f"({ep_idx}).npy"
        if cache_file.exists():
            try:
                data = np.load(str(cache_file), allow_pickle=True).item()
                descriptors[int(ep_idx)] = data["action_descriptor"]
            except Exception:
                pass

    if len(descriptors) == len(expected_indices):
        return descriptors
    return None


# ---------------------------------------------------------------------------
# SmolVLM feature extractor loading
# ---------------------------------------------------------------------------

VLM_MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def load_smolvla_feature_extractor(device: str = "cuda"):
    from transformers import AutoModel, AutoProcessor

    print(f"Loading SmolVLM feature extractor: {VLM_MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME)
    model = AutoModel.from_pretrained(
        VLM_MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    print(f"SmolVLM loaded on {device}")
    return model, processor