#!/usr/bin/env python
"""
Test VLM model with different images to see if it produces different embeddings.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

VLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

print("Loading VLM model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
model = AutoModel.from_pretrained(VLM_MODEL_ID).to(device).eval()

for param in model.parameters():
    param.requires_grad = False

print(f"Model loaded on {device}")

# Create test images
# Image 1: all red
img_red = Image.new('RGB', (224, 224), color='red')
# Image 2: all blue
img_blue = Image.new('RGB', (224, 224), color='blue')
# Image 3: random noise
img_noise = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

test_images = [
    ("Red", img_red),
    ("Blue", img_blue),
    ("Noise", img_noise),
]

print("\n" + "=" * 60)
print("TESTING VLM WITH DIFFERENT IMAGES")
print("=" * 60)

embeddings = {}
for name, img in test_images:
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
    
    embeddings[name] = embedding.cpu().numpy()
    print(f"{name}: shape={embedding.shape}, mean={embedding.mean():.6f}")

# Compare embeddings
print("\n" + "-" * 60)
print("COMPARISON")
print("-" * 60)

for name1 in embeddings:
    for name2 in embeddings:
        if name1 < name2:
            emb1 = embeddings[name1]
            emb2 = embeddings[name2]
            
            # Cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(emb1, emb2)[0, 0]
            
            # Mean absolute difference
            diff = np.abs(emb1 - emb2).mean()
            
            print(f"{name1} vs {name2}:")
            print(f"  Cosine similarity: {sim:.6f}")
            print(f"  Mean abs diff: {diff:.6f}")
            print(f"  Identical? {np.allclose(emb1, emb2)}")