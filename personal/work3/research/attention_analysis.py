"""SIC Framework - Attention map analysis (optional, requires trained models)."""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def extract_attention_maps(
    model,
    processor,
    image: Image.Image,
    instruction: str,
    device: str,
    layer_indices: list = [14, 15, 16, 17, 18]
) -> dict:
    """Extract attention maps from specified transformer layers."""
    attention_maps = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
                if attn_weights is not None:
                    attention_maps[layer_idx] = attn_weights.mean(dim=1).cpu().numpy()
        return hook_fn

    for idx in layer_indices:
        try:
            layer = model.language_model.model.layers[idx]
            hook = layer.self_attn.register_forward_hook(make_hook(idx))
            hooks.append(hook)
        except (AttributeError, IndexError):
            pass

    inputs = processor(images=image, text=instruction, return_tensors="pt").to(device)

    with torch.no_grad():
        model(**inputs, output_attentions=True)

    for hook in hooks:
        hook.remove()

    return attention_maps


def compare_attention_across_datasets(
    model_paths: dict,
    test_images: list,
    processor,
    device: str,
    save_dir: str
):
    """Compare attention maps of models trained on different dataset configurations."""
    if not model_paths:
        print("[SKIP] attention_analysis: no model paths provided.")
        return None

    from transformers import AutoModelForCausalLM
    results = {}

    for dataset_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"[SKIP] Model not found: {model_path}")
            continue

        print(f"Loading model: {dataset_name}")
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()

        model_attns = []
        for img, instruction in test_images:
            attns = extract_attention_maps(model, processor, img, instruction, device)
            model_attns.append(attns)

        results[dataset_name] = model_attns
        del model
        torch.cuda.empty_cache()

    if results:
        _plot_attention_comparison(results, test_images, save_dir)

    return results


def _plot_attention_comparison(results, test_images, save_dir):
    """Generate attention map comparison grid figure."""
    model_names = list(results.keys())
    n_models = len(model_names)
    n_images = len(test_images)
    layer_to_show = list(list(results.values())[0][0].keys())[-1]

    fig, axes = plt.subplots(n_models + 1, n_images,
                              figsize=(4 * n_images, 4 * (n_models + 1)))
    fig.suptitle("Attention Map Comparison Across Dataset Configurations\n"
                 f"(Layer {layer_to_show}, Mean over Attention Heads)",
                 fontsize=14, fontweight='bold')

    for j, (img, instruction) in enumerate(test_images):
        ax = axes[0, j] if n_images > 1 else axes[0]
        ax.imshow(img)
        ax.set_title(f"Test image {j+1}\n'{instruction[:30]}...'", fontsize=9)
        ax.axis('off')
        if j == 0:
            ax.set_ylabel("Original", fontsize=10, fontweight='bold')

    for i, model_name in enumerate(model_names):
        for j in range(n_images):
            ax = axes[i+1, j] if n_images > 1 else axes[i+1]

            attn_data = results[model_name][j]
            if layer_to_show in attn_data:
                attn = attn_data[layer_to_show][0]
                img_attn = attn[-1, :int(attn.shape[1] * 0.8)]
                side = int(np.sqrt(len(img_attn)))
                if side * side <= len(img_attn):
                    img_attn_grid = img_attn[:side*side].reshape(side, side)
                    ax.imshow(img_attn_grid, cmap='hot', interpolation='bilinear')
                    ax.set_title(f"Max: {img_attn.max():.3f}", fontsize=8)
            ax.axis('off')
            if j == 0:
                short_name = model_name.replace('_', ' ').replace('demo', 'd')
                ax.set_ylabel(short_name[:20], fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, "fig_attention_map_comparison")
    plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_path + ".png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}.pdf")