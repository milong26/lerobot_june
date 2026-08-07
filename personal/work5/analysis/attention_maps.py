"""Extract and visualize attention maps during training."""

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class AttentionExtractor:
    def __init__(self, model, layer_indices=None):
        self.model = model
        self.layer_indices = layer_indices or list(range(14, 19))
        self.attentions = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        if hasattr(self.model, 'vision_model'):
            vision = self.model.vision_model
            if hasattr(vision, 'encoder') and hasattr(vision.encoder, 'layers'):
                layers = vision.encoder.layers
                for idx in self.layer_indices:
                    if idx < len(layers):
                        hook = layers[idx].register_forward_hook(self._hook_fn(idx))
                        self.hooks.append(hook)

    def _hook_fn(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            if hasattr(hidden_states, 'attn_weights'):
                self.attentions[layer_idx] = hidden_states.attn_weights
            elif isinstance(output, tuple) and len(output) > 1:
                if hasattr(output[1], 'attn_weights'):
                    self.attentions[layer_idx] = output[1].attn_weights
        return hook

    def extract_attention(self, model, processor, image, device):
        self.attentions = {}
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            _ = model(**inputs)

        return self._process_attentions()

    def _process_attentions(self):
        if not self.attentions:
            return None

        all_attn = []
        for layer_idx in self.layer_indices:
            if layer_idx in self.attentions:
                attn = self.attentions[layer_idx]
                if isinstance(attn, torch.Tensor):
                    attn = attn.cpu().numpy()
                all_attn.append(attn)

        if not all_attn:
            return None

        avg_attn = np.mean(all_attn, axis=0)
        if avg_attn.ndim == 4:
            avg_attn = np.mean(avg_attn, axis=1)

        return avg_attn

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def visualize_attention(image, attention_weights, save_path, title="Attention Map"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    if attention_weights is not None and attention_weights.size > 0:
        attn_2d = attention_weights
        if attn_2d.ndim > 2:
            attn_2d = attn_2d.mean(axis=-1)
        if attn_2d.ndim > 2:
            attn_2d = attn_2d[0]

        target_size = (image.shape[1], image.shape[0])
        from scipy.ndimage import zoom
        if attn_2d.shape != (target_size[1], target_size[0]):
            zoom_factors = (target_size[1] / attn_2d.shape[0], target_size[0] / attn_2d.shape[1])
            attn_2d = zoom(attn_2d, zoom_factors, order=1)

        attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)

        ax2.imshow(image)
        ax2.imshow(attn_2d, alpha=0.5, cmap='jet')
        ax2.set_title("Attention Overlay")
        ax2.axis('off')
    else:
        ax2.imshow(image)
        ax2.set_title("No Attention Data")
        ax2.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention visualization: {save_path}")