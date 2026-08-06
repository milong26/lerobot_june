"""SIC Framework - Embedding extraction with caching."""

import os
import pickle
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm


def load_frozen_vlm(model_id: str, device: str):
    """Load SmolVLM2 as a frozen visual feature extractor."""
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, processor


def extract_visual_embedding(model, processor, image: Image.Image, device: str) -> np.ndarray:
    """Extract visual embedding from a single PIL image."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        if hasattr(model, 'vision_model'):
            visual_out = model.vision_model(**{k: v for k, v in inputs.items()
                                               if 'pixel' in k})
            embedding = visual_out.last_hidden_state.mean(dim=1)
        else:
            visual_out = model(**inputs, output_hidden_states=True)
            embedding = visual_out.last_hidden_state[:, 0, :]
    return embedding.squeeze().float().cpu().numpy()


def extract_trajectory_embeddings(
    model, processor,
    dataset,
    episode_index: int,
    cam_key: str,
    device: str,
    batch_size: int = 4
) -> np.ndarray:
    """Extract embeddings for all frames of one episode."""
    ep_from = dataset.episode_data_index['from'][episode_index]
    ep_to = dataset.episode_data_index['to'][episode_index]

    embeddings = []
    for i in range(ep_from, ep_to, batch_size):
        batch_end = min(i + batch_size, ep_to)
        for idx in range(i, batch_end):
            frame = dataset[idx]
            img_tensor = frame[cam_key]
            img_pil = Image.fromarray(
                (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            emb = extract_visual_embedding(model, processor, img_pil, device)
            embeddings.append(emb)

    return np.array(embeddings)


def extract_and_cache_all_embeddings(
    model, processor,
    dataset,
    config_map: dict,
    cam_key: str,
    device: str,
    cache_path: str,
    batch_size: int = 4
) -> dict:
    """Extract embeddings for all episodes in config_map with caching."""
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"Extracting embeddings for {len(config_map)} episodes...")
    embeddings = {}
    for ep_idx in tqdm(config_map.keys()):
        embeddings[ep_idx] = extract_trajectory_embeddings(
            model, processor, dataset, ep_idx, cam_key, device, batch_size
        )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Cached embeddings saved to {cache_path}")
    return embeddings