#!/usr/bin/env python
"""
Re-extract wrist embeddings for 2 episodes to verify they're different from top.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from lerobot.datasets.lerobot_dataset import LeRobotDataset

VLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DATASET_DIR = Path(__file__).parent.parent / "dataset"

print("Loading dataset...")
repo_id = "lerobot/metaworld_pick_place"
dataset = LeRobotDataset(repo_id, root=DATASET_DIR)

print("Loading VLM model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
model = AutoModel.from_pretrained(VLM_MODEL_ID).to(device).eval()

for param in model.parameters():
    param.requires_grad = False

# Find frames for episode 230
print("Finding episode 230 frames...")
ep230_frames = []
for idx in range(min(dataset.num_frames, 35000)):
    frame = dataset[idx]
    ep_idx = int(frame["episode_index"])
    if ep_idx == 230:
        ep230_frames.append(idx)
    if len(ep230_frames) > 0 and ep_idx > 230:
        break

print(f"Episode 230 has {len(ep230_frames)} frames")

if len(ep230_frames) > 0:
    # Extract first frame
    frame_data = dataset[ep230_frames[0]]
    
    # Top camera
    top_frame = frame_data["observation.images.top"]
    if isinstance(top_frame, torch.Tensor):
        top_img = Image.fromarray((top_frame.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    else:
        top_img = Image.fromarray(top_frame)
    
    # Wrist camera
    wrist_frame = frame_data["observation.images.wrist"]
    if isinstance(wrist_frame, torch.Tensor):
        wrist_img = Image.fromarray((wrist_frame.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    else:
        wrist_img = Image.fromarray(wrist_frame)
    
    print(f"\nTop image size: {top_img.size}")
    print(f"Wrist image size: {wrist_img.size}")
    
    # Extract embeddings
    def extract_emb(img):
        inputs = processor(images=img, return_tensors="pt")
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
    
    top_emb = extract_emb(top_img)
    wrist_emb = extract_emb(wrist_img)
    
    print(f"\nTop embedding: shape={top_emb.shape}, mean={top_emb.mean():.6f}")
    print(f"Wrist embedding: shape={wrist_emb.shape}, mean={wrist_emb.mean():.6f}")
    
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(top_emb, wrist_emb)[0, 0]
    diff = np.abs(top_emb - wrist_emb).max()
    
    print(f"\nTop vs Wrist:")
    print(f"  Cosine similarity: {sim:.6f}")
    print(f"  Max diff: {diff:.6f}")
    print(f"  Identical? {np.allclose(top_emb, wrist_emb)}")
    
    # Compare with saved file
    print(f"\n--- Comparing with saved file ---")
    saved_emb = np.load(Path(__file__).parent.parent / "all_frames" / "episode_0230.npy")
    print(f"Saved embedding shape: {saved_emb.shape}")
    print(f"Saved top part mean: {saved_emb[0, :960].mean():.6f}")
    print(f"Saved wrist part mean: {saved_emb[0, 960:].mean():.6f}")
    
    saved_top_sim = cosine_similarity(top_emb, saved_emb[0, :960].reshape(1, -1))[0, 0]
    saved_wrist_sim = cosine_similarity(wrist_emb, saved_emb[0, 960:].reshape(1, -1))[0, 0]
    
    print(f"Extracted top vs saved top: cos_sim={saved_top_sim:.6f}")
    print(f"Extracted wrist vs saved wrist: cos_sim={saved_wrist_sim:.6f}")