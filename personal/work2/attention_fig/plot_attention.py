#!/usr/bin/env python
"""
Attention Visualization for SmolVLA Models

Extracts and visualizes attention weights from trained SmolVLA models.
Supports multiple models and two camera views (wrist + top).

Usage:
    python plot_attention.py
    (edit MODEL_PATHS below before running)
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# ============================================================
# Configuration - Edit these paths before running
# ============================================================
MODEL_PATHS = [
    "/data/zhonglinye/jun/lerobot/personal/work2/ours/method/training_output/N150/seed42/checkpoints/last/pretrained_model",
    "/data/zhonglinye/jun/lerobot/personal/work2/duibi/random/random_200_seed42/checkpoints/last/pretrained_model",
    "/data/zhonglinye/jun/lerobot/personal/work2/duibi/random/random_100_seed42/checkpoints/last/pretrained_model",
]
DEVICE = "cpu"
# ============================================================

OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DIR = Path(__file__).parent.parent / "dataset"


def extract_images_from_dataset():
    """Extract wrist and top images from episode 0, frame 0."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset("lerobot/metaworld_pick_place", root=str(DATASET_DIR))
        frame = dataset[0]

        results = {}
        for cam_key, save_name in [
            ("observation.images.wrist", "test_wrist_image.png"),
            ("observation.images.top", "test_top_image.png"),
        ]:
            img = frame[cam_key]
            if hasattr(img, "numpy"):
                img = img.numpy()
            if img.ndim == 4:
                img = img[0]
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[-1] == 1:
                img = np.concatenate([img] * 3, axis=-1)
            elif img.shape[-1] == 4:
                img = img[..., :3]
            save_path = OUTPUT_DIR / save_name
            Image.fromarray(img).save(save_path)
            print(f"Extracted: {save_path}")
            results[cam_key] = str(save_path)
        return results
    except Exception as e:
        print(f"Could not extract images from dataset: {e}")
        import traceback
        traceback.print_exc()
        return {}


def load_model_and_get_attention(model_path, wrist_image_path, top_image_path, device="cpu"):
    """Load lerobot SmolVLA policy, run inference with two images, return attentions."""
    print(f"\nLoading model from: {model_path}")
    print(f"Wrist image: {wrist_image_path}")
    print(f"Top image: {top_image_path}")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError:
        print("ERROR: Cannot import SmolVLAPolicy from lerobot")
        return None

    policy = SmolVLAPolicy.from_pretrained(
        model_path,
        strict=False,
    )
    policy.eval()
    policy.to(device)

    vlm_model = policy.model.vlm_with_expert.get_vlm_model()
    processor = policy.model.vlm_with_expert.processor

    # Switch to eager attention to support output_attentions=True
    vlm_model.config._attn_implementation = "eager"
    for layer in vlm_model.text_model.layers:
        layer.self_attn._attn_implementation = "eager"

    wrist_img = Image.open(wrist_image_path).convert("RGB")
    top_img = Image.open(top_image_path).convert("RGB")

    prompt = "Pick up the object and place it to the target position."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[wrist_img, top_img],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = vlm_model(**inputs, output_attentions=True)

    attentions = outputs.attentions
    if attentions is None:
        print("ERROR: No attentions returned. Make sure attention mode is 'eager'.")
        return None

    print(f"Number of layers with attention: {len(attentions)}")
    print(f"Attention[0] shape: {attentions[0].shape}")

    return {
        "attentions": attentions,
        "wrist_image": np.array(wrist_img),
        "top_image": np.array(top_img),
        "model_name": Path(model_path).name,
    }


def visualize_attention(attn_data, output_path):
    """Create a figure with both wrist and top views, each showing attention maps."""
    wrist_image = attn_data["wrist_image"]
    top_image = attn_data["top_image"]
    attentions = attn_data["attentions"]
    model_name = attn_data["model_name"]

    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]

    viz_layers = min(4, n_layers)
    viz_heads = min(4, n_heads)

    fig_cols = viz_heads
    fig_rows = viz_layers * 2 + 1

    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 4, fig_rows * 3.5))
    if fig_rows == 1 and fig_cols == 1:
        axes = np.array([[axes]])
    elif fig_rows == 1:
        axes = axes[np.newaxis, :]
    elif fig_cols == 1:
        axes = axes[:, np.newaxis]

    ax_wrist = axes[0, 0]
    ax_wrist.imshow(wrist_image)
    ax_wrist.set_title("Wrist Camera (Original)")
    ax_wrist.axis("off")

    if fig_cols > 1:
        ax_top = axes[0, 1]
        ax_top.imshow(top_image)
        ax_top.set_title("Top Camera (Original)")
        ax_top.axis("off")
    else:
        fig2, ax_top2 = plt.subplots(figsize=(4, 4))
        ax_top2.imshow(top_image)
        ax_top2.set_title("Top Camera (Original)")
        ax_top2.axis("off")
        plt.savefig(str(output_path).replace(".png", "_top_orig.png"), dpi=150, bbox_inches="tight")
        plt.close(fig2)

    for c in range(fig_cols):
        if c not in (0, 1):
            axes[0, c].axis("off")

    attn_to_viz = []
    for layer_idx in range(viz_layers):
        for head_idx in range(viz_heads):
            attn_to_viz.append((layer_idx, head_idx))

    h_wrist, w_wrist = wrist_image.shape[:2]
    h_top, w_top = top_image.shape[:2]

    for viz_idx, (layer_idx, head_idx) in enumerate(attn_to_viz):
        attn = attentions[layer_idx][0, head_idx].cpu().float().numpy()

        if attn.ndim == 3:
            attn_map = attn[-1].mean(axis=0)
        elif attn.ndim == 2:
            attn_map = attn
        else:
            attn_map = attn.flatten()

        if attn_map.ndim == 2:
            attn_2d = attn_map
        else:
            n_tokens = len(attn_map)
            grid_size = int(np.ceil(np.sqrt(n_tokens)))
            attn_2d = np.zeros((grid_size, grid_size))
            attn_2d.flat[:n_tokens] = attn_map

        attn_wrist = cv2.resize(attn_2d, (w_wrist, h_wrist), interpolation=cv2.INTER_CUBIC)
        attn_wrist_norm = (attn_wrist - attn_wrist.min()) / (attn_wrist.max() - attn_wrist.min() + 1e-8)
        heatmap_wrist = cv2.applyColorMap((attn_wrist_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_wrist = cv2.cvtColor(heatmap_wrist, cv2.COLOR_BGR2RGB)
        overlay_wrist = cv2.addWeighted(wrist_image, 0.5, heatmap_wrist, 0.5, 0)

        row_wrist = viz_idx // fig_cols + 1
        col = viz_idx % fig_cols
        ax = axes[row_wrist, col]
        ax.imshow(overlay_wrist)
        ax.set_title(f"Layer {layer_idx} / Head {head_idx} (Wrist)")
        ax.axis("off")

        attn_top = cv2.resize(attn_2d, (w_top, h_top), interpolation=cv2.INTER_CUBIC)
        attn_top_norm = (attn_top - attn_top.min()) / (attn_top.max() - attn_top.min() + 1e-8)
        heatmap_top = cv2.applyColorMap((attn_top_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_top = cv2.cvtColor(heatmap_top, cv2.COLOR_BGR2RGB)
        overlay_top = cv2.addWeighted(top_image, 0.5, heatmap_top, 0.5, 0)

        row_top = row_wrist + viz_layers
        ax2 = axes[row_top, col]
        ax2.imshow(overlay_top)
        ax2.set_title(f"Layer {layer_idx} / Head {head_idx} (Top)")
        ax2.axis("off")

    plt.suptitle(f"Attention Visualization - {model_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved attention visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Attention Visualization for SmolVLA Models")
    parser.add_argument("--model-paths", nargs="+", default=None, help="Override MODEL_PATHS")
    parser.add_argument("--wrist-image", type=str, default=None, help="Path to wrist test image")
    parser.add_argument("--top-image", type=str, default=None, help="Path to top test image")
    parser.add_argument("--device", type=str, default=None, help="Override DEVICE")
    args = parser.parse_args()

    model_paths = args.model_paths if args.model_paths else MODEL_PATHS
    device = args.device if args.device else DEVICE
    if not model_paths:
        print("No model paths specified. Edit MODEL_PATHS in the script or use --model-paths.")
        return

    wrist_image = args.wrist_image
    top_image = args.top_image

    if wrist_image is None or top_image is None:
        extracted = extract_images_from_dataset()
        if wrist_image is None:
            wrist_image = extracted.get("observation.images.wrist")
        if top_image is None:
            top_image = extracted.get("observation.images.top")
        if wrist_image is None or top_image is None:
            print("ERROR: Could not extract test images. Please specify --wrist-image and --top-image.")
            return

    print(f"Wrist image: {wrist_image}")
    print(f"Top image: {top_image}")
    print(f"Models to process: {len(model_paths)}")

    for idx, model_path in enumerate(model_paths):
        model_name = Path(model_path).name
        parent_name = Path(model_path).parent.name
        grandparent_name = Path(model_path).parent.parent.name
        unique_name = f"{grandparent_name}_{parent_name}_{model_name}_model{idx}"
        output_path = OUTPUT_DIR / f"{unique_name}.png"

        attn_data = load_model_and_get_attention(model_path, wrist_image, top_image, device)
        if attn_data is not None:
            attn_data["model_name"] = unique_name
            visualize_attention(attn_data, output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()