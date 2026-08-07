"""Extract frozen VLM embeddings with caching."""

import os
import pickle
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm


def load_frozen_vlm(model_id, device):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, processor


def extract_visual_embedding(model, processor, image, device):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        if hasattr(model, 'vision_model'):
            # SmolVLM uses pixel_values as input to vision_model
            pixel_values = inputs.get('pixel_values')
            if pixel_values is not None:
                # Ensure pixel_values is 4D: (batch, channels, height, width)
                if pixel_values.dim() == 5:
                    # Video format: (batch, frames, channels, height, width)
                    # Take first frame
                    pixel_values = pixel_values[:, 0]
                # Match model dtype (model is loaded in float16)
                pixel_values = pixel_values.to(model.dtype)
                visual_out = model.vision_model(pixel_values=pixel_values)
            else:
                visual_out = model.vision_model(**inputs)
            embedding = visual_out.last_hidden_state.mean(dim=1)
        else:
            visual_out = model(**inputs, output_hidden_states=True)
            embedding = visual_out.last_hidden_state[:, 0, :]
    return embedding.squeeze().float().cpu().numpy()


def extract_episode_embeddings(model, processor, dataset, episode_index, cam_key, device):
    # Compute episode frame ranges manually since episode_data_index doesn't exist
    ep_from = None
    ep_to = None
    for idx in range(len(dataset)):
        frame = dataset[idx]
        if frame.get('episode_index') == episode_index:
            if ep_from is None:
                ep_from = idx
            ep_to = idx + 1
        elif ep_from is not None:
            break
    
    if ep_from is None:
        raise ValueError(f"Episode {episode_index} not found in dataset")

    embeddings = []
    for idx in range(ep_from, ep_to):
        frame = dataset[idx]
        img_tensor = frame[cam_key]
        if isinstance(img_tensor, torch.Tensor):
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            img_np = img_tensor.astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        emb = extract_visual_embedding(model, processor, img_pil, device)
        embeddings.append(emb)

    return np.array(embeddings)


def extract_all_episode_embeddings(model, processor, dataset, cam_key, device, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    n_episodes = dataset.num_episodes
    print(f"Extracting embeddings for {n_episodes} episodes...")
    embeddings = {}
    for ep_idx in tqdm(range(n_episodes), desc="Extracting VLM embeddings"):
        embeddings[ep_idx] = extract_episode_embeddings(
            model, processor, dataset, ep_idx, cam_key, device)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Cached embeddings saved to {cache_path}")

    return embeddings


def get_episode_mean_embedding(episode_embeddings):
    if len(episode_embeddings) == 0:
        return np.zeros(1)
    return episode_embeddings.mean(axis=0)