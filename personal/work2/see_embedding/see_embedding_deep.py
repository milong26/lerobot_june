"""
Deep Embedding Analysis - Fine-grained comparison methods

Compare multiple fine-grained methods:
1. Per-token L2 distance (find spatial difference)
2. Attention weight difference (where model focuses)
3. Feature map visualization (spatial feature difference)
4. Layer-wise comparison (shallow vs deep layers)
5. Local patch comparison (sliding window)
6. Manifold distance (trajectory distance)

Results saved to result_deep/ folder
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lerobot.datasets import LeRobotDataset

# Configuration
DATASET_ROOT = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pickplacev3"
EPISODE_STATES_PATH = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pickplacev3/episode_initial_states.json"
MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TASK_PROMPT = "pick and place"
OUTPUT_DIR = str(Path(__file__).parent / "result_deep")
# 指定空闲gpu，单应该就够了
GPU_ID = 0  # Specify GPU ID here
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"

# Set CUDA device
if torch.cuda.is_available():
    torch.cuda.set_device(GPU_ID)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
EMBEDDING_DIR = str(Path(__file__).parent)


def save_embeddings_deep(embs_dict, ep_a, ep_b):
    path = f"{EMBEDDING_DIR}/embedding_deep_ep{ep_a}_ep{ep_b}.pt"
    torch.save(embs_dict, path)
    print(f"Deep embeddings saved: {path}")


def load_embeddings_deep(ep_a, ep_b):
    path = f"{EMBEDDING_DIR}/embedding_deep_ep{ep_a}_ep{ep_b}.pt"
    if Path(path).exists():
        embs_dict = torch.load(path, weights_only=False)
        print(f"Loaded cached deep embeddings: {path}")
        return embs_dict
    return None


def find_most_different_episodes(min_distance=0.5):
    print("Loading episode initial states...")
    with open(EPISODE_STATES_PATH, "r") as f:
        states_data = json.load(f)

    episodes = states_data["episodes"]
    print(f"Total episodes: {len(episodes)}")

    # Find all pairs with distance >= min_distance
    all_pairs = []
    for i in range(len(episodes)):
        for j in range(i + 1, len(episodes)):
            xi, yi = episodes[i]["obj_init_pos"][0], episodes[i]["obj_init_pos"][1]
            xj, yj = episodes[j]["obj_init_pos"][0], episodes[j]["obj_init_pos"][1]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if dist >= min_distance:
                all_pairs.append((episodes[i]["episode_index"], episodes[j]["episode_index"], dist))

    if not all_pairs:
        print(f"No pairs with distance >= {min_distance}, finding max distance pair...")
        max_dist = 0
        best_pair = (0, 1)
        for i in range(len(episodes)):
            for j in range(i + 1, len(episodes)):
                xi, yi = episodes[i]["obj_init_pos"][0], episodes[i]["obj_init_pos"][1]
                xj, yj = episodes[j]["obj_init_pos"][0], episodes[j]["obj_init_pos"][1]
                dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (episodes[i]["episode_index"], episodes[j]["episode_index"])
        ep_a, ep_b = best_pair
        pos_a = episodes[ep_a]["obj_init_pos"][:2]
        pos_b = episodes[ep_b]["obj_init_pos"][:2]
        print(f"  Episode {ep_a}: obj_init_pos=({pos_a[0]:.4f}, {pos_a[1]:.4f})")
        print(f"  Episode {ep_b}: obj_init_pos=({pos_b[0]:.4f}, {pos_b[1]:.4f})")
        print(f"  XY distance: {max_dist:.4f}")
        return ep_a, ep_b

    # Sort by distance descending, pick the pair with largest distance
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    ep_a, ep_b, max_dist = all_pairs[0]
    pos_a = episodes[ep_a]["obj_init_pos"][:2]
    pos_b = episodes[ep_b]["obj_init_pos"][:2]

    print(f"Found {len(all_pairs)} pairs with distance >= {min_distance}")
    print(f"Selected the most different pair:")
    print(f"  Episode {ep_a}: obj_init_pos=({pos_a[0]:.4f}, {pos_a[1]:.4f})")
    print(f"  Episode {ep_b}: obj_init_pos=({pos_b[0]:.4f}, {pos_b[1]:.4f})")
    print(f"  XY distance: {max_dist:.4f}")

    return ep_a, ep_b


def load_episode_frames(dataset, ep_idx):
    from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
    to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]

    frames = []
    for idx in range(int(from_idx), int(to_idx)):
        frame = dataset[int(idx)]
        img = frame["observation.images.top"]
        frames.append(img)

    return torch.stack(frames)


def extract_all_embeddings(frames, model, processor):
    """
    Extract multiple levels of embeddings and intermediate features:
    1. vision_encoder: vision_model last_hidden_state (before connector)
    2. connector: connector output (after connector, before VLM)
    3. vlm_final: VLM final hidden state mean pooling
    4. image_tokens_only: only image tokens from VLM output
    5. per_token_features: per-token features for fine-grained comparison
    """
    results = {
        "vision_encoder": [],
        "connector": [],
        "vlm_final": [],
        "image_tokens_only": [],
        "vision_per_token": [],  # per-token for method 1
        "connector_per_token": [],  # per-token for method 1
    }

    image_token = processor.tokenizer.image_token if hasattr(processor.tokenizer, 'image_token') else "<image>"

    with torch.no_grad():
        for i, frame in enumerate(frames):
            text = f"{image_token}\n{TASK_PROMPT}"
            inputs = processor(
                text=text,
                images=[frame],
                return_tensors="pt",
            ).to(DEVICE)

            vlm_model = model.model
            pixel_values = inputs["pixel_values"]
            if pixel_values.ndim == 5:
                pixel_values = pixel_values[:, 0]
            pixel_values = pixel_values.to(dtype=vlm_model.vision_model.dtype)

            # 1. Vision encoder output
            vision_output = vlm_model.vision_model(pixel_values=pixel_values)
            vision_hidden = vision_output.last_hidden_state  # (1, num_patches, vision_dim)

            # 2. Connector output
            connector_output = vlm_model.connector(vision_hidden)  # (1, num_image_tokens, hidden_dim)

            # 3. VLM final output
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
            attention_mask = inputs.get("attention_mask", None)

            # 4. Image tokens only
            num_image_tokens = connector_output.shape[1]
            image_only_emb = last_hidden[:, :num_image_tokens, :]

            # Mean pooling
            vision_mean = vision_hidden.mean(dim=1).cpu().squeeze(0)
            connector_mean = connector_output.mean(dim=1).cpu().squeeze(0)

            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                vlm_mean = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                vlm_mean = last_hidden.mean(dim=1)
            vlm_mean = vlm_mean.cpu().squeeze(0)

            image_only_mean = image_only_emb.mean(dim=1).cpu().squeeze(0)

            results["vision_encoder"].append(vision_mean)
            results["connector"].append(connector_mean)
            results["vlm_final"].append(vlm_mean)
            results["image_tokens_only"].append(image_only_mean)

            # Save per-token features for fine-grained analysis
            results["vision_per_token"].append(vision_hidden.cpu().squeeze(0))  # (num_patches, dim)
            results["connector_per_token"].append(connector_output.cpu().squeeze(0))  # (num_tokens, dim)

            if i % 10 == 0:
                print(f"  Progress: {i}/{len(frames)} frames")

    for key in results:
        if key in ["vision_per_token", "connector_per_token"]:
            results[key] = results[key]  # list of tensors
        else:
            results[key] = torch.stack(results[key], dim=0)

    return results


def method1_per_token_l2_distance(embs_a, embs_b, method_name, ep_a, ep_b):
    """Method 1: Per-token L2 distance analysis"""
    print(f"\n{'='*60}")
    print(f"Method 1: Per-token L2 Distance ({method_name})")
    print(f"{'='*60}")

    min_len = min(len(embs_a), len(embs_b))

    # Compute per-token L2 distance for each frame pair
    all_distances = []
    for i in range(min_len):
        tokens_a = embs_a[i]  # (num_tokens, dim)
        tokens_b = embs_b[i]  # (num_tokens, dim)

        # Compute pairwise L2 distance between tokens
        dists = torch.cdist(tokens_a, tokens_b, p=2)  # (num_tokens_a, num_tokens_b)
        mean_dist = dists.mean().item()
        max_dist = dists.max().item()
        min_dist = dists.min().item()

        all_distances.append({
            "mean": mean_dist,
            "max": max_dist,
            "min": min_dist,
            "std": dists.std().item(),
        })

    # Analyze by phase
    n = min_len
    early_end = n // 3
    late_start = 2 * n // 3

    early_dists = [d["mean"] for d in all_distances[:early_end]]
    mid_dists = [d["mean"] for d in all_distances[early_end:late_start]]
    late_dists = [d["mean"] for d in all_distances[late_start:]]

    print(f"  Mean L2 Distance - Early: {np.mean(early_dists):.4f}, Mid: {np.mean(mid_dists):.4f}, Late: {np.mean(late_dists):.4f}")
    print(f"  Max L2 Distance - Early: {np.max(early_dists):.4f}, Late: {np.max(late_dists):.4f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mean distance over time
    ax = axes[0, 0]
    ax.plot(range(min_len), [d["mean"] for d in all_distances], label="Mean L2", color="steelblue")
    ax.axvline(x=early_end, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=late_start, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Mean L2 Distance")
    ax.set_title("Per-token L2 Distance Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Max distance over time
    ax = axes[0, 1]
    ax.plot(range(min_len), [d["max"] for d in all_distances], label="Max L2", color="coral")
    ax.axvline(x=early_end, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=late_start, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Max L2 Distance")
    ax.set_title("Max Per-token L2 Distance Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Std over time
    ax = axes[1, 0]
    ax.plot(range(min_len), [d["std"] for d in all_distances], label="Std L2", color="green")
    ax.axvline(x=early_end, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=late_start, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Std L2 Distance")
    ax.set_title("Std of Per-token L2 Distance Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Phase comparison
    ax = axes[1, 1]
    phases = ["Early", "Mid", "Late"]
    means = [np.mean(early_dists), np.mean(mid_dists), np.mean(late_dists)]
    ax.bar(phases, means, color=["steelblue", "goldenrod", "coral"], alpha=0.8, edgecolor="black")
    ax.set_ylabel("Mean L2 Distance")
    ax.set_title("Phase Comparison")
    for i, v in enumerate(means):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom")

    plt.suptitle(f"Method 1: Per-token L2 Distance ({method_name})", fontsize=14)
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/method1_per_token_l2_{method_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    return {
        "early_mean": np.mean(early_dists),
        "mid_mean": np.mean(mid_dists),
        "late_mean": np.mean(late_dists),
        "early_max": np.max(early_dists),
        "late_max": np.max(late_dists),
    }


def method2_attention_difference(embs_dict_a, embs_dict_b, model, processor, frames_a, frames_b, ep_a, ep_b):
    """Method 2: Attention weight difference analysis"""
    print(f"\n{'='*60}")
    print(f"Method 2: Attention Weight Difference")
    print(f"{'='*60}")

    min_len = min(len(frames_a), len(frames_b))
    sample_indices = [0, min_len // 4, min_len // 2, 3 * min_len // 4, min_len - 1]

    all_attention_diffs = []

    image_token = processor.tokenizer.image_token if hasattr(processor.tokenizer, 'image_token') else "<image>"

    with torch.no_grad():
        for idx in sample_indices:
            frame_a = frames_a[idx]
            frame_b = frames_b[idx]

            text = f"{image_token}\n{TASK_PROMPT}"

            # Process frame A
            inputs_a = processor(text=text, images=[frame_a], return_tensors="pt").to(DEVICE)
            outputs_a = model(**inputs_a, output_attentions=True)
            attentions_a = outputs_a.attentions  # tuple of (batch, heads, seq, seq) or None

            # Process frame B
            inputs_b = processor(text=text, images=[frame_b], return_tensors="pt").to(DEVICE)
            outputs_b = model(**inputs_b, output_attentions=True)
            attentions_b = outputs_b.attentions

            # Check if attentions are available
            if attentions_a is None or len(attentions_a) == 0:
                print(f"  Frame {idx}: Attention weights not available (None or empty)")
                all_attention_diffs.append(0.0)
                continue

            # Compare attention weights (last layer)
            last_attn_a = attentions_a[-1].cpu().squeeze(0).mean(dim=0).numpy()  # (seq, seq)
            last_attn_b = attentions_b[-1].cpu().squeeze(0).mean(dim=0).numpy()

            # Compute difference
            attn_diff = np.abs(last_attn_a - last_attn_b).mean()
            all_attention_diffs.append(attn_diff)

            print(f"  Frame {idx}: Attention diff = {attn_diff:.6f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_indices, all_attention_diffs, marker="o", linewidth=2, markersize=8, color="steelblue")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Mean Attention Weight Difference")
    ax.set_title("Attention Weight Difference Between Episodes")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/method2_attention_diff.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    return {
        "attention_diffs": all_attention_diffs,
        "mean_diff": np.mean(all_attention_diffs),
    }


def method3_feature_map_visualization(embs_dict_a, embs_dict_b, ep_a, ep_b):
    """Method 3: Feature map spatial difference visualization"""
    print(f"\n{'='*60}")
    print(f"Method 3: Feature Map Spatial Difference")
    print(f"{'='*60}")

    # Use vision_per_token features
    vision_a = embs_dict_a["vision_per_token"]
    vision_b = embs_dict_b["vision_per_token"]

    min_len = min(len(vision_a), len(vision_b))

    # Compare first few frames
    num_frames_to_compare = min(5, min_len)

    fig, axes = plt.subplots(num_frames_to_compare, 3, figsize=(15, 5 * num_frames_to_compare))
    if num_frames_to_compare == 1:
        axes = axes.reshape(1, -1)

    for frame_idx in range(num_frames_to_compare):
        tokens_a = vision_a[frame_idx]  # (num_patches, dim)
        tokens_b = vision_b[frame_idx]

        # Compute per-token L2 distance
        dists = torch.cdist(tokens_a, tokens_b, p=2)  # (num_patches, num_patches)

        # Take diagonal (corresponding tokens)
        diag_dists = torch.diag(dists).numpy()

        # Reshape to spatial grid (assuming square)
        num_patches = len(diag_dists)
        grid_size = int(np.sqrt(num_patches))
        if grid_size * grid_size == num_patches:
            spatial_map = diag_dists.reshape(grid_size, grid_size)
        else:
            # Pad to nearest square
            next_square = (grid_size + 1) ** 2
            padded = np.pad(diag_dists, (0, next_square - num_patches), mode="constant")
            spatial_map = padded.reshape(grid_size + 1, grid_size + 1)

        # Plot
        ax = axes[frame_idx, 0]
        im = ax.imshow(spatial_map, cmap="hot", interpolation="nearest")
        ax.set_title(f"Frame {frame_idx}: Token Distance Map")
        plt.colorbar(im, ax=ax)

        # Histogram of distances
        ax = axes[frame_idx, 1]
        ax.hist(diag_dists, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
        ax.set_xlabel("L2 Distance")
        ax.set_ylabel("Count")
        ax.set_title(f"Distance Distribution")

        # Cumulative distribution
        ax = axes[frame_idx, 2]
        sorted_dists = np.sort(diag_dists)
        cumulative = np.arange(1, len(sorted_dists) + 1) / len(sorted_dists)
        ax.plot(sorted_dists, cumulative, linewidth=2, color="coral")
        ax.set_xlabel("L2 Distance")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(f"Cumulative Distribution")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Method 3: Feature Map Spatial Difference (First {num_frames_to_compare} Frames)", fontsize=14)
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/method3_feature_map_diff.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    return {"num_frames_compared": num_frames_to_compare}


def method4_layerwise_comparison(model, processor, frames_a, frames_b, ep_a, ep_b):
    """Method 4: Layer-wise feature comparison"""
    print(f"\n{'='*60}")
    print(f"Method 4: Layer-wise Feature Comparison")
    print(f"{'='*60}")

    # Compare first frame of each episode across all layers
    frame_a = frames_a[0]
    frame_b = frames_b[0]

    image_token = processor.tokenizer.image_token if hasattr(processor.tokenizer, 'image_token') else "<image>"
    text = f"{image_token}\n{TASK_PROMPT}"

    layer_sims = []

    with torch.no_grad():
        inputs_a = processor(text=text, images=[frame_a], return_tensors="pt").to(DEVICE)
        outputs_a = model(**inputs_a, output_hidden_states=True)
        hidden_states_a = outputs_a.hidden_states  # tuple of (batch, seq, dim) for each layer

        inputs_b = processor(text=text, images=[frame_b], return_tensors="pt").to(DEVICE)
        outputs_b = model(**inputs_b, output_hidden_states=True)
        hidden_states_b = outputs_b.hidden_states

        for layer_idx, (h_a, h_b) in enumerate(zip(hidden_states_a, hidden_states_b)):
            # Mean pooling
            emb_a = h_a.mean(dim=1).cpu().squeeze(0)
            emb_b = h_b.mean(dim=1).cpu().squeeze(0)

            cos_sim = torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=0).item()
            l2_dist = torch.norm(emb_a - emb_b, p=2).item()

            layer_sims.append({
                "layer": layer_idx,
                "cos_sim": cos_sim,
                "l2_dist": l2_dist,
            })

            if layer_idx % 5 == 0:
                print(f"  Layer {layer_idx}: cos_sim={cos_sim:.4f}, l2_dist={l2_dist:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    layers = [d["layer"] for d in layer_sims]
    cos_sims = [d["cos_sim"] for d in layer_sims]
    l2_dists = [d["l2_dist"] for d in layer_sims]

    ax = axes[0]
    ax.plot(layers, cos_sims, marker="o", linewidth=2, markersize=6, color="steelblue")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity Across Layers")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(layers, l2_dists, marker="s", linewidth=2, markersize=6, color="coral")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("L2 Distance")
    ax.set_title("L2 Distance Across Layers")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Method 4: Layer-wise Comparison (First Frame)", fontsize=14)
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/method4_layerwise.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    return {
        "layer_sims": layer_sims,
        "shallow_sim": np.mean([d["cos_sim"] for d in layer_sims[:4]]),
        "deep_sim": np.mean([d["cos_sim"] for d in layer_sims[-4:]]),
    }


def method5_local_patch_comparison(embs_dict_a, embs_dict_b, ep_a, ep_b):
    """Method 5: Local patch comparison with sliding window"""
    print(f"\n{'='*60}")
    print(f"Method 5: Local Patch Comparison")
    print(f"{'='*60}")

    vision_a = embs_dict_a["vision_per_token"]
    vision_b = embs_dict_b["vision_per_token"]

    min_len = min(len(vision_a), len(vision_b))

    # Compare first frame
    tokens_a = vision_a[0]  # (num_patches, dim)
    tokens_b = vision_b[0]

    num_patches = tokens_a.shape[0]
    grid_size = int(np.sqrt(num_patches))

    if grid_size * grid_size != num_patches:
        print(f"  Cannot reshape {num_patches} patches to square grid, skipping")
        return {}

    # Reshape to grid
    grid_a = tokens_a.reshape(grid_size, grid_size, -1)
    grid_b = tokens_b.reshape(grid_size, grid_size, -1)

    # Sliding window comparison
    window_size = 3
    stride = 1

    local_diffs = []
    for i in range(0, grid_size - window_size + 1, stride):
        for j in range(0, grid_size - window_size + 1, stride):
            patch_a = grid_a[i:i+window_size, j:j+window_size, :].reshape(-1)
            patch_b = grid_b[i:i+window_size, j:j+window_size, :].reshape(-1)

            dist = torch.norm(patch_a - patch_b, p=2).item()
            local_diffs.append({
                "i": i,
                "j": j,
                "dist": dist,
            })

    # Find most different regions
    local_diffs_sorted = sorted(local_diffs, key=lambda x: x["dist"], reverse=True)

    print(f"  Top 5 most different local patches:")
    for i in range(min(5, len(local_diffs_sorted))):
        d = local_diffs_sorted[i]
        print(f"    Position ({d['i']}, {d['j']}): dist={d['dist']:.4f}")

    # Create heatmap
    heatmap = np.zeros((grid_size - window_size + 1, grid_size - window_size + 1))
    for d in local_diffs:
        heatmap[d["i"], d["j"]] = d["dist"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(heatmap, cmap="hot", interpolation="nearest")
    ax.set_title("Local Patch Difference Heatmap")
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    dists = [d["dist"] for d in local_diffs]
    ax.hist(dists, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_xlabel("L2 Distance")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Local Patch Differences")

    plt.suptitle(f"Method 5: Local Patch Comparison (First Frame)", fontsize=14)
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/method5_local_patch.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    return {
        "top_diffs": local_diffs_sorted[:5],
        "mean_diff": np.mean(dists),
        "max_diff": np.max(dists),
    }


def method6_manifold_distance(embs_dict_a, embs_dict_b, ep_a, ep_b):
    """Method 6: Manifold distance (trajectory comparison using per-token max distance)"""
    print(f"\n{'='*60}")
    print(f"Method 6: Manifold Distance (Trajectory Comparison)")
    print(f"{'='*60}")

    # Use per-token features directly (no mean pooling), prefer vision_per_token (more stable)
    tokens_a_list = embs_dict_a.get("vision_per_token", embs_dict_a.get("connector_per_token"))
    tokens_b_list = embs_dict_b.get("vision_per_token", embs_dict_b.get("connector_per_token"))

    if tokens_a_list is None or tokens_b_list is None:
        print("  No per-token features available, using vlm_final as fallback")
        emb_a = embs_dict_a["vlm_final"].cpu().numpy()
        emb_b = embs_dict_b["vlm_final"].cpu().numpy()
        dist_matrix = cdist(emb_a, emb_b, metric="euclidean")
        min_len = min(len(emb_a), len(emb_b))
        diag_dists = [dist_matrix[i, i] for i in range(min_len)]
        frame_dists = diag_dists
    else:
        # For each frame pair, compute max per-token distance (sensitive to local changes)
        min_len = min(len(tokens_a_list), len(tokens_b_list))
        frame_dists = []
        has_inf = False
        for i in range(min_len):
            tokens_a = tokens_a_list[i]  # (num_tokens, dim)
            tokens_b = tokens_b_list[i]  # (num_tokens, dim)
            dist_matrix_frame = torch.cdist(tokens_a, tokens_b, p=2)
            max_dist = dist_matrix_frame.max().item()
            if not torch.isfinite(dist_matrix_frame).all():
                has_inf = True
                # Filter out inf values, use 95th percentile instead
                finite_mask = torch.isfinite(dist_matrix_frame)
                if finite_mask.any():
                    max_dist = torch.quantile(dist_matrix_frame[finite_mask], 0.95).item()
                else:
                    max_dist = 0.0
            frame_dists.append(max_dist)
        
        if has_inf:
            print("  Warning: connector contains inf values, using 95th percentile instead of max")
        
        diag_dists = frame_dists

    # Simple DTW on frame distances
    n = len(frame_dists)
    dtw_dist = sum(frame_dists) / n if n > 0 else 0

    mean_dist = np.mean(frame_dists)
    max_dist = np.max(frame_dists)

    print(f"  DTW Distance: {dtw_dist:.4f}")
    print(f"  Mean Distance: {mean_dist:.4f}")
    print(f"  Max Distance: {max_dist:.4f}")

    # Plot frame distances over time
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(range(len(frame_dists)), frame_dists, linewidth=2, color="steelblue")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Max Per-token L2 Distance")
    ax.set_title("Per-frame Max Token Distance")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(frame_dists, bins=20, color="coral", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Max Per-token L2 Distance")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Frame Distances")

    plt.suptitle(f"Method 6: Manifold Distance", fontsize=14)
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/method6_manifold.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    return {
        "dtw_dist": dtw_dist,
        "mean_dist": mean_dist,
        "diag_mean": mean_dist,
        "max_dist": max_dist,
    }


def summarize_all_methods(results, ep_a, ep_b):
    """Summarize all methods and rank them by discriminative power"""
    print(f"\n{'='*60}")
    print(f"SUMMARY - All Fine-grained Methods")
    print(f"{'='*60}")

    # Collect all method scores for ranking
    method_scores = {}

    print("\nMethod 1: Per-token L2 Distance")
    if results["method1"]:
        for method_name, res in results["method1"].items():
            print(f"  {method_name}: Early={res['early_mean']:.4f}, Late={res['late_mean']:.4f}, "
                  f"MaxEarly={res['early_max']:.4f}")
            method_scores[f"Per-token L2 ({method_name})"] = res["early_mean"]
    else:
        print("  Skipped: Cache missing per-token features")

    print("\nMethod 2: Attention Weight Difference")
    print(f"  Mean attention diff: {results['method2']['mean_diff']:.6f}")
    method_scores["Attention Diff"] = results["method2"]["mean_diff"]

    print("\nMethod 4: Layer-wise Comparison")
    print(f"  Shallow layers avg cos_sim: {results['method4']['shallow_sim']:.4f}")
    print(f"  Deep layers avg cos_sim: {results['method4']['deep_sim']:.4f}")
    # Use 1 - cos_sim as distance metric
    method_scores["Layer-wise (1-cos_sim)"] = 1 - results["method4"]["shallow_sim"]

    print("\nMethod 5: Local Patch Comparison")
    if "skipped" in results["method5"]:
        print("  Skipped: vision_per_token not found in cached embeddings")
    elif "mean_diff" in results["method5"]:
        print(f"  Mean local diff: {results['method5']['mean_diff']:.4f}")
        print(f"  Max local diff: {results['method5']['max_diff']:.4f}")
        method_scores["Local Patch"] = results["method5"]["mean_diff"]

    print("\nMethod 6: Manifold Distance")
    print(f"  DTW distance: {results['method6']['dtw_dist']:.4f}")
    print(f"  Mean distance: {results['method6']['mean_dist']:.4f}")
    print(f"  Max distance: {results['method6']['max_dist']:.4f}")
    method_scores["Manifold (DTW)"] = results["method6"]["dtw_dist"]

    # Rank methods by discriminative power
    print(f"\n{'='*60}")
    print("METHOD RANKING - Which method shows the biggest difference?")
    print(f"{'='*60}")

    # Normalize scores for comparison (z-score normalization)
    scores_list = list(method_scores.items())
    if len(scores_list) > 1:
        values = np.array([s[1] for s in scores_list])
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val > 0:
            normalized = [(name, (val - mean_val) / std_val) for name, val in scores_list]
        else:
            normalized = [(name, 0.0) for name, val in scores_list]

        normalized.sort(key=lambda x: x[1], reverse=True)

        print("\nRanking (by normalized discriminative power):")
        for rank, (name, score) in enumerate(normalized, 1):
            marker = "★" if rank == 1 else " "
            print(f"  {marker} {rank}. {name}: {score:.4f}")

        best_method = normalized[0][0]
        print(f"\n>>> Best discriminative method: {best_method}")
    else:
        print("\nNot enough methods to rank.")

    # Create summary plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Method 1 summary
    ax = axes[0, 0]
    if results["method1"]:
        methods = list(results["method1"].keys())
        early_vals = [results["method1"][m]["early_mean"] for m in methods]
        late_vals = [results["method1"][m]["late_mean"] for m in methods]
        x = np.arange(len(methods))
        width = 0.35
        ax.bar(x - width/2, early_vals, width, label="Early", color="lightblue", alpha=0.8)
        ax.bar(x + width/2, late_vals, width, label="Late", color="lightcoral", alpha=0.8)
        ax.set_ylabel("Mean L2 Distance")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Skipped:\nCache missing\nper-token features",
                ha="center", va="center", fontsize=12)
    ax.set_title("Method 1: Per-token L2")

    # Method 2
    ax = axes[0, 1]
    ax.text(0.5, 0.5, f"Attention Diff\n{results['method2']['mean_diff']:.6f}",
            ha="center", va="center", fontsize=20)
    ax.set_title("Method 2: Attention")
    ax.axis("off")

    # Method 4
    ax = axes[0, 2]
    ax.text(0.5, 0.5, f"Shallow: {results['method4']['shallow_sim']:.4f}\n"
                       f"Deep: {results['method4']['deep_sim']:.4f}",
            ha="center", va="center", fontsize=16)
    ax.set_title("Method 4: Layer-wise")
    ax.axis("off")

    # Method 5
    ax = axes[1, 0]
    if "skipped" in results["method5"]:
        ax.text(0.5, 0.5, "Skipped:\nCache missing\nper-token features",
                ha="center", va="center", fontsize=12)
    elif "mean_diff" in results["method5"]:
        ax.text(0.5, 0.5, f"Mean: {results['method5']['mean_diff']:.4f}\n"
                           f"Max: {results['method5']['max_diff']:.4f}",
                ha="center", va="center", fontsize=16)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=20)
    ax.set_title("Method 5: Local Patch")
    ax.axis("off")

    # Method 6
    ax = axes[1, 1]
    ax.text(0.5, 0.5, f"DTW: {results['method6']['dtw_dist']:.4f}\n"
                       f"Mean: {results['method6']['mean_dist']:.4f}\n"
                       f"Max: {results['method6']['max_dist']:.4f}",
            ha="center", va="center", fontsize=14)
    ax.set_title("Method 6: Manifold")
    ax.axis("off")

    # Overall assessment with ranking
    ax = axes[1, 2]
    assessment = "Method Ranking:\n\n"
    if len(scores_list) > 1:
        for rank, (name, score) in enumerate(normalized, 1):
            marker = "★" if rank == 1 else " "
            assessment += f"{marker} {rank}. {name}\n"
        assessment += f"\nBest: {best_method}"
    else:
        assessment += "Not enough methods"

    ax.text(0.5, 0.5, assessment, ha="center", va="center", fontsize=12)
    ax.set_title("Overall Assessment")
    ax.axis("off")

    plt.suptitle(f"Episode {ep_a} vs {ep_b}: Fine-grained Method Summary", fontsize=14)
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/00_all_methods_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSummary chart saved: {out_path}")


def main():
    print("=" * 60)
    print("Deep Embedding Analysis - Fine-grained Methods")
    print("=" * 60)

    # 1. Find episodes
    ep_a, ep_b = find_most_different_episodes()

    # 2. Load dataset
    print(f"\nLoading dataset: {DATASET_ROOT}")
    dataset = LeRobotDataset(
        repo_id="work2/metaworld_pick_place",
        root=DATASET_ROOT
    )
    print(f"Total episodes: {dataset.num_episodes}")

    # 3. Load frames
    print(f"\nLoading Episode {ep_a} frames...")
    frames_a = load_episode_frames(dataset, ep_a)
    print(f"  Loaded {len(frames_a)} frames")

    print(f"\nLoading Episode {ep_b} frames...")
    frames_b = load_episode_frames(dataset, ep_b)
    print(f"  Loaded {len(frames_b)} frames")

    # 4. Load or extract embeddings
    embs_dict = load_embeddings_deep(ep_a, ep_b)

    if embs_dict is None:
        print(f"\nLoading model: {MODEL_ID}")
        print(f"Device: {DEVICE}")

        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            dtype=torch.float16 if "cuda" in DEVICE else torch.float32,
            device_map=None,
        )
        model = model.to(DEVICE)
        model.eval()

        print(f"\nExtracting Episode {ep_a} embeddings...")
        embs_a = extract_all_embeddings(frames_a, model, processor)

        print(f"\nExtracting Episode {ep_b} embeddings...")
        embs_b = extract_all_embeddings(frames_b, model, processor)

        embs_dict = {"a": embs_a, "b": embs_b}
        save_embeddings_deep(embs_dict, ep_a, ep_b)
    else:
        embs_a = embs_dict["a"]
        embs_b = embs_dict["b"]
        print(f"\nUsing cached embeddings")

    # 5. Run all fine-grained methods
    all_results = {}

    # Method 1: Per-token L2 distance
    all_results["method1"] = {}
    for method_name, key_name in [("vision_encoder", "vision_per_token"), ("connector", "connector_per_token")]:
        if key_name in embs_a:
            res = method1_per_token_l2_distance(
                embs_a[key_name],
                embs_b[key_name],
                method_name, ep_a, ep_b
            )
            all_results["method1"][method_name] = res

    # Method 2: Attention difference (need model with eager attention)
    print(f"\nLoading model for attention analysis...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if "cuda" in DEVICE else torch.float32,
        device_map=None,
        attn_implementation="eager",
    )
    model = model.to(DEVICE)
    model.eval()

    all_results["method2"] = method2_attention_difference(
        embs_a, embs_b, model, processor, frames_a, frames_b, ep_a, ep_b
    )

    # Method 3: Feature map visualization
    if "vision_per_token" in embs_a and "vision_per_token" in embs_b:
        all_results["method3"] = method3_feature_map_visualization(
            embs_a, embs_b, ep_a, ep_b
        )
    else:
        print(f"\n{'='*60}")
        print(f"Method 3: Feature Map Spatial Difference")
        print(f"{'='*60}")
        print("  Skipped: vision_per_token not found in cached embeddings")
        print("  Please delete the cache file and re-run to generate per-token features")
        all_results["method3"] = {"skipped": True}

    # Method 4: Layer-wise comparison
    all_results["method4"] = method4_layerwise_comparison(
        model, processor, frames_a, frames_b, ep_a, ep_b
    )

    # Method 5: Local patch comparison
    if "vision_per_token" in embs_a and "vision_per_token" in embs_b:
        all_results["method5"] = method5_local_patch_comparison(
            embs_a, embs_b, ep_a, ep_b
        )
    else:
        print(f"\n{'='*60}")
        print(f"Method 5: Local Patch Comparison")
        print(f"{'='*60}")
        print("  Skipped: vision_per_token not found in cached embeddings")
        all_results["method5"] = {"skipped": True}

    # Method 6: Manifold distance
    all_results["method6"] = method6_manifold_distance(
        embs_a, embs_b, ep_a, ep_b
    )

    # 6. Summarize
    summarize_all_methods(all_results, ep_a, ep_b)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()