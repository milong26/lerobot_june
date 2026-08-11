#!/usr/bin/env python
"""
Attention Visualization for SmolVLA Models

Extracts and visualizes attention weights from trained SmolVLA models.
Supports multiple models in one run.

Usage:
    python plot_attention.py --model-paths /path/to/model1 /path/to/model2
    python plot_attention.py  (uses default MODELS list)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TEST_IMAGE = Path(__file__).parent.parent / "dataset_lookin" / "wrist_sample.png"
DEFAULT_MODELS = []


def extract_wrist_image_from_dataset():
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset_dir = Path(__file__).parent.parent / "dataset"
        dataset = LeRobotDataset("lerobot/metaworld_pick_place", root=str(dataset_dir))
        frame = dataset[0]
        img = frame["observation.images.wrist"]
        if hasattr(img, "numpy"):
            img = img.numpy()
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img
        save_path = OUTPUT_DIR / "test_wrist_image.png"
        Image.fromarray(img).save(save_path)
        print(f"Extracted test image from dataset: {save_path}")
        return str(save_path)
    except Exception as e:
        print(f"Could not extract image from dataset: {e}")
        return None


def load_model_and_get_attention(model_path, image_path, device="cuda"):
    print(f"\nLoading model from: {model_path}")
    print(f"Using test image: {image_path}")

    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        return None

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    image = Image.open(image_path).convert("RGB")

    prompt = "Pick up the object and place it to the target position."
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions

    image_np = np.array(image)

    return {
        "attentions": attentions,
        "image": image_np,
        "model_name": Path(model_path).name,
    }


def visualize_attention(attn_data, output_path):
    image = attn_data["image"]
    attentions = attn_data["attentions"]
    model_name = attn_data["model_name"]

    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]

    fig_cols = min(4, n_heads)
    fig_rows = min(4, n_layers)

    fig, axes = plt.subplots(fig_rows + 1, fig_cols, figsize=(fig_cols * 4, (fig_rows + 1) * 3.5))
    if fig_rows == 1 and fig_cols == 1:
        axes = np.array([[axes]])
    elif fig_rows == 1:
        axes = axes[np.newaxis, :]
    elif fig_cols == 1:
        axes = axes[:, np.newaxis]

    ax_orig = axes[0, 0]
    ax_orig.imshow(image)
    ax_orig.set_title("Original Image")
    ax_orig.axis("off")

    for c in range(1, fig_cols):
        axes[0, c].axis("off")

    attn_to_viz = []
    for layer_idx in range(min(fig_rows, n_layers)):
        for head_idx in range(min(fig_cols, n_heads)):
            attn_to_viz.append((layer_idx, head_idx))

    for viz_idx, (layer_idx, head_idx) in enumerate(attn_to_viz):
        row = viz_idx // fig_cols + 1
        col = viz_idx % fig_cols
        ax = axes[row, col]

        attn = attentions[layer_idx][0, head_idx].cpu().float().numpy()

        if attn.ndim == 3:
            attn_map = attn[-1].mean(axis=0)
        elif attn.ndim == 2:
            attn_map = attn
        else:
            attn_map = attn.flatten()

        h_img, w_img = image.shape[:2]

        if attn_map.ndim == 2:
            attn_2d = attn_map
        else:
            n_tokens = len(attn_map)
            grid_size = int(np.ceil(np.sqrt(n_tokens)))
            attn_2d = np.zeros((grid_size, grid_size))
            attn_2d.flat[:n_tokens] = attn_map

        attn_resized = cv2.resize(attn_2d, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
        attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

        heatmap = cv2.applyColorMap((attn_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

        ax.imshow(overlay)
        ax.set_title(f"Layer {layer_idx} / Head {head_idx}")
        ax.axis("off")

    for viz_idx in range(len(attn_to_viz), fig_rows * fig_cols):
        row = viz_idx // fig_cols + 1
        col = viz_idx % fig_cols
        axes[row, col].axis("off")

    plt.suptitle(f"Attention Visualization - {model_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved attention visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Attention Visualization for SmolVLA Models")
    parser.add_argument("--model-paths", nargs="+", default=None, help="List of model paths")
    parser.add_argument("--test-image", type=str, default=None, help="Path to test image")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    args = parser.parse_args()

    model_paths = args.model_paths if args.model_paths else DEFAULT_MODELS
    if not model_paths:
        print("No model paths specified. Use --model-paths or edit DEFAULT_MODELS in the script.")
        print("Example: python plot_attention.py --model-paths /path/to/model1 /path/to/model2")
        return

    test_image = args.test_image
    if test_image is None:
        if DEFAULT_TEST_IMAGE.exists():
            test_image = str(DEFAULT_TEST_IMAGE)
        else:
            extracted = extract_wrist_image_from_dataset()
            if extracted:
                test_image = extracted
            else:
                print("ERROR: No test image found. Please specify --test-image.")
                return

    print(f"Using test image: {test_image}")
    print(f"Models to process: {len(model_paths)}")

    for model_path in model_paths:
        model_name = Path(model_path).name
        output_path = OUTPUT_DIR / f"{model_name}.png"

        attn_data = load_model_and_get_attention(model_path, test_image, args.device)
        if attn_data is not None:
            visualize_attention(attn_data, output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()