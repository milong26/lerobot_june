#!/usr/bin/env python
"""
Our V5 Real Episode Selection - 基于像素特征替代 rand_vec 的 our_v5 选择流程

核心算法 = 原版 our_v5，只把 rand_vec 替换为真实 episode 初始图像像素特征。
- episode_data 仍然包含三种信息: {pixel_vec, visual_embedding, action_descriptor}
- 区域划分用 pixel_vec 做 KMeans（替代原版 rand_vec）
- region priority 仍然用 coverage_gap + visual_uncertainty + action_uncertainty
- region 内候选评分仍然用 visual diversity + action diversity
- visual embedding 和 action descriptor 复用 work2 embedding_utils V5 流程

Usage (single dataset):
    python select_our_v5_real.py \
        --dataset-path /data/zhonglinye/jun/ep10_30episodes_standing40/ep10_30episodes_standing40 \
        --dataset-name ep10_30episodes_standing40 \
        --output-dir /data/zhonglinye/jun/lerobot/personal/work1/our_v5_real_results \
        --num-selected 20 \
        --seed 42

Usage (multi-dataset, called by launcher):
    python select_our_v5_real.py \
        --dataset-path /data/.../bushing \
        --dataset-name bushing \
        --output-dir /data/.../our_v5_real_results \
        --num-selected 20 \
        --seed 42 \
        --pixel-camera-key observation.images.top \
        --pixel-size 64
"""

import sys
import os
import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WORK2_ROOT = PROJECT_ROOT / "personal" / "work2"
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Reuse V5 embedding utilities
from embedding_utils.ensure_embeddings_v5 import ensure_v5_embeddings_exist
from embedding_utils.action_descriptor import (
    extract_all_action_descriptors,
    load_action_descriptors,
)
from embedding_utils.config_v5 import V5_ACTION_STEPS
from embedding_utils.cache import normalize_dataset_name, get_dataset_episode_indices

V5_ACTION_DESCRIPTOR_OUTPUT_DIR = WORK2_ROOT / "cache" / "action_descriptors_v5"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_pixel_vecs(
    dataset: LeRobotDataset,
    episode_indices: List[int],
    camera_key: str = "observation.images.top",
    pixel_size: int = 64,
) -> Dict[int, np.ndarray]:
    """
    从每个 episode 的初始帧读取固定相机图像，resize 后 flatten 为 pixel vector。

    确定性处理：
    - 兼容 torch Tensor / numpy、CHW / HWC、uint8 / float
    - float 且 max <= 1 时保持 [0,1]
    - uint8 或 max > 1 时除以 255
    - 统一 RGB，固定 resize 到 pixel_size x pixel_size
    - 直接 flatten 为 float32
    - 禁止逐 episode mean/std 或 L2 normalization
    """
    import torch
    import torch.nn.functional as F

    pixel_vecs = {}

    for ep_idx in episode_indices:
        ep_meta = dataset.meta.episodes[ep_idx]
        from_idx = ep_meta["dataset_from_index"]

        frame = dataset[int(from_idx)]
        if camera_key not in frame:
            print(f"  WARNING: episode {ep_idx} missing camera key '{camera_key}'")
            continue

        img = frame[camera_key]

        # Convert to numpy HWC
        if isinstance(img, torch.Tensor):
            img_np = img.detach().cpu().numpy()
        else:
            img_np = np.array(img)

        if img_np.ndim == 3 and img_np.shape[0] in (1, 3, 4):
            img_np = np.transpose(img_np, (1, 2, 0))

        # Normalize to [0, 1] float32
        img_np = img_np.astype(np.float32)
        if img_np.dtype == np.float32 and img_np.max() <= 1.0:
            pass  # already [0, 1]
        else:
            img_np = img_np / 255.0

        # Ensure RGB (3 channels)
        if img_np.shape[-1] == 1:
            img_np = np.repeat(img_np, 3, axis=-1)
        elif img_np.shape[-1] == 4:
            img_np = img_np[..., :3]

        # Resize to pixel_size x pixel_size
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        resized = F.interpolate(
            img_tensor, size=(pixel_size, pixel_size), mode="bilinear", align_corners=False
        )
        resized_np = resized.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, C)

        # Flatten to float32 vector
        vec = resized_np.flatten().astype(np.float32)
        pixel_vecs[int(ep_idx)] = vec

    return pixel_vecs


def log_pixel_diagnostics(
    pixel_vecs: Dict[int, np.ndarray],
    episode_indices: List[int],
):
    """打印 pixel feature diagnostic 信息。"""
    valid = [pixel_vecs[ep] for ep in episode_indices if ep in pixel_vecs]
    if not valid:
        print("  WARNING: no valid pixel vectors for diagnostics")
        return

    mat = np.array(valid)
    print(f"\n  Pixel diagnostics:")
    print(f"    shape: {mat.shape}")
    print(f"    all finite: {np.all(np.isfinite(mat))}")
    print(f"    per-dim min: {mat.min(axis=0).min():.6f}, max: {mat.max(axis=0).max():.6f}")
    print(f"    overall min: {mat.min():.6f}, mean: {mat.mean():.6f}, max: {mat.max():.6f}")

    # Pairwise distances (sample if too many)
    n = len(valid)
    if n <= 200:
        from scipy.spatial.distance import pdist
        dists = pdist(mat, metric="euclidean")
        print(f"    pairwise distance min: {dists.min():.4f}, median: {np.median(dists):.4f}, max: {dists.max():.4f}")
        # Count duplicates
        dup_count = int(np.sum(dists < 1e-6))
        print(f"    near-duplicate pairs (dist < 1e-6): {dup_count}")
    else:
        rng = np.random.RandomState(0)
        sample_idx = rng.choice(n, 200, replace=False)
        sample_mat = mat[sample_idx]
        from scipy.spatial.distance import pdist
        dists = pdist(sample_mat, metric="euclidean")
        print(f"    pairwise distance (sampled 200) min: {dists.min():.4f}, median: {np.median(dists):.4f}, max: {dists.max():.4f}")


def build_pixel_regions(
    pixel_vecs: Dict[int, np.ndarray],
    episode_indices: List[int],
    region_ratio: float = 0.1,
    min_regions: int = 4,
    max_regions: int = 16,
    seed: int = 42,
    use_pca: bool = False,
    pca_dim: int = 64,
) -> Tuple[Dict[int, int], int]:
    """
    基于 pixel_vec 做 KMeans 区域划分（等价于原版 build_rand_vec_regions）。

    除输入从 rand_vec 改为 pixel_vec 外，region 数计算、KMeans random_state、
    region mapping 等语义保持原版。
    """
    print(f"\n{'='*60}")
    print(f"Building pixel regions (KMeans)...")
    print(f"{'='*60}")

    valid_episodes = [ep for ep in episode_indices if ep in pixel_vecs]
    n_episodes = len(valid_episodes)
    if n_episodes == 0:
        raise ValueError("No valid pixel vectors")

    num_regions = max(min_regions, min(max_regions, int(n_episodes * region_ratio)))
    num_regions = min(num_regions, n_episodes)

    print(f"  Episodes: {n_episodes}")
    print(f"  Region ratio: {region_ratio}")
    print(f"  Num regions: {num_regions}")

    feature_matrix = np.array([pixel_vecs[ep] for ep in valid_episodes])
    print(f"  Feature matrix shape: {feature_matrix.shape}")

    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(pca_dim, feature_matrix.shape[1]), random_state=seed)
        feature_matrix = pca.fit_transform(feature_matrix).astype(np.float32)
        print(f"  After PCA: {feature_matrix.shape}")

    kmeans = KMeans(n_clusters=num_regions, random_state=seed, n_init=10)
    region_ids = kmeans.fit_predict(feature_matrix)

    episode_to_region = {}
    for i, ep_idx in enumerate(valid_episodes):
        episode_to_region[ep_idx] = int(region_ids[i])

    region_counts = {}
    for rid in region_ids:
        rid = int(rid)
        region_counts[rid] = region_counts.get(rid, 0) + 1

    print(f"  Created regions: {num_regions}")
    print(f"  Region sizes - min: {min(region_counts.values())}, max: {max(region_counts.values())}, "
          f"avg: {np.mean(list(region_counts.values())):.1f}")

    return episode_to_region, num_regions


def select_initial_b0_by_pixel_regions(
    episode_to_region: Dict[int, int],
    num_regions: int,
    b0_region_ratio: float = 0.5,
    b0_size: int = 4,
    seed: int = 42,
) -> List[int]:
    """
    初始 B0 选择：从不同区域均匀采样（保持原版动态 b0_region_ratio 逻辑）。
    """
    rng = np.random.RandomState(seed)

    region_to_episodes = {}
    for ep_idx, rid in episode_to_region.items():
        region_to_episodes.setdefault(rid, []).append(ep_idx)

    n_b0_regions = max(1, int(num_regions * b0_region_ratio))
    n_b0_regions = min(n_b0_regions, len(region_to_episodes))

    region_ids = sorted(region_to_episodes.keys())
    selected_regions = rng.choice(region_ids, size=n_b0_regions, replace=False).tolist()

    selected = []
    per_region = b0_size // n_b0_regions if n_b0_regions > 0 else b0_size
    remainder = b0_size - per_region * n_b0_regions

    for ri, rid in enumerate(selected_regions):
        n_take = per_region + (1 if ri < remainder else 0)
        n_take = min(n_take, len(region_to_episodes[rid]))
        chosen = rng.choice(region_to_episodes[rid], size=n_take, replace=False).tolist()
        selected.extend(chosen)

    # If still not enough, fill from remaining
    if len(selected) < b0_size:
        remaining = [ep for ep in episode_to_region.keys() if ep not in selected]
        n_needed = b0_size - len(selected)
        if remaining:
            additional = rng.choice(remaining, size=min(n_needed, len(remaining)), replace=False).tolist()
            selected.extend(additional)

    selected.sort()
    print(f"\n[Initial B0] Selected {len(selected)} episodes from {n_b0_regions} regions: {selected}")
    return selected


def compute_region_priority(
    region_id: int,
    episode_to_region: Dict[int, int],
    selected_ids: List[int],
    episode_data: Dict[int, dict],
    coverage_weight: float = 0.5,
    region_visual_weight: float = 0.3,
    region_action_weight: float = 0.2,
) -> Dict[str, float]:
    """
    计算区域优先级：coverage_gap + visual_uncertainty + action_uncertainty
    三个分量和默认权重保持原版。
    """
    region_episodes = [ep for ep, rid in episode_to_region.items() if rid == region_id]
    total_in_region = len(region_episodes)
    if total_in_region == 0:
        return {"priority": 0.0, "coverage_gap": 0.0, "visual_uncertainty": 0.0, "action_uncertainty": 0.0}

    selected_in_region = [ep for ep in region_episodes if ep in selected_ids]
    coverage_gap = 1.0 - (len(selected_in_region) / total_in_region)

    # Visual uncertainty: average pairwise distance of visual embeddings in region
    visual_uncertainty = 0.0
    vis_features = []
    for ep in region_episodes:
        if ep in episode_data and "visual_embedding" in episode_data[ep]:
            vis_features.append(episode_data[ep]["visual_embedding"])
    if len(vis_features) >= 2:
        arr = np.array(vis_features)
        dists = []
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                dists.append(np.linalg.norm(arr[i] - arr[j]))
        visual_uncertainty = float(np.mean(dists)) if dists else 0.0
    elif len(vis_features) == 1:
        visual_uncertainty = float(np.linalg.norm(vis_features[0]))

    # Action uncertainty: average pairwise distance of action descriptors in region
    action_uncertainty = 0.0
    act_features = []
    for ep in region_episodes:
        if ep in episode_data and "action_descriptor" in episode_data[ep]:
            act_features.append(episode_data[ep]["action_descriptor"])
    if len(act_features) >= 2:
        arr = np.array(act_features)
        dists = []
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                dists.append(np.linalg.norm(arr[i] - arr[j]))
        action_uncertainty = float(np.mean(dists)) if dists else 0.0
    elif len(act_features) == 1:
        action_uncertainty = float(np.linalg.norm(act_features[0]))

    # Normalize
    vis_norm = min(visual_uncertainty / 10.0, 1.0) if visual_uncertainty > 0 else 0.0
    act_norm = min(action_uncertainty / 10.0, 1.0) if action_uncertainty > 0 else 0.0

    priority = coverage_weight * coverage_gap + region_visual_weight * vis_norm + region_action_weight * act_norm

    return {
        "priority": priority,
        "coverage_gap": coverage_gap,
        "visual_uncertainty": vis_norm,
        "action_uncertainty": act_norm,
    }


def score_candidate_in_region(
    candidate_idx: int,
    selected_ids: List[int],
    episode_data: Dict[int, dict],
    visual_weight: float = 0.5,
    action_weight: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """
    Region 内候选 episode 评分：visual diversity + action diversity
    默认 visual_weight=0.5, action_weight=0.5，保持原版。
    """
    if candidate_idx not in episode_data:
        return 0.0, {}

    selected_vis = []
    selected_act = []
    for ep in selected_ids:
        if ep in episode_data:
            if "visual_embedding" in episode_data[ep]:
                selected_vis.append(episode_data[ep]["visual_embedding"])
            if "action_descriptor" in episode_data[ep]:
                selected_act.append(episode_data[ep]["action_descriptor"])

    cand = episode_data[candidate_idx]
    vis_score = 0.0
    if "visual_embedding" in cand and selected_vis:
        dists = [np.linalg.norm(cand["visual_embedding"] - s) for s in selected_vis]
        vis_score = float(np.min(dists))
    elif "visual_embedding" in cand:
        vis_score = float(np.linalg.norm(cand["visual_embedding"]))

    act_score = 0.0
    if "action_descriptor" in cand and selected_act:
        dists = [np.linalg.norm(cand["action_descriptor"] - s) for s in selected_act]
        act_score = float(np.min(dists))
    elif "action_descriptor" in cand:
        act_score = float(np.linalg.norm(cand["action_descriptor"]))

    final_score = visual_weight * vis_score + action_weight * act_score
    return final_score, {"visual_score": vis_score, "action_score": act_score, "final_score": final_score}


def select_episodes_our_v5_real(
    episode_data: Dict[int, dict],
    pixel_vecs: Dict[int, np.ndarray],
    episode_indices: List[int],
    num_select: int = 20,
    b0_size: int = 4,
    region_ratio: float = 0.1,
    min_regions: int = 4,
    max_regions: int = 16,
    b0_region_ratio: float = 0.5,
    coverage_weight: float = 0.5,
    region_visual_weight: float = 0.3,
    region_action_weight: float = 0.2,
    candidate_visual_weight: float = 0.5,
    candidate_action_weight: float = 0.5,
    seed: int = 42,
    use_pca: bool = False,
    pca_dim: int = 64,
) -> Tuple[List[int], List[Dict], Dict[int, int]]:
    """
    Our V5 核心选择算法，只把 rand_vec 替换为 pixel_vec。
    """
    selection_log = []

    # Step 1: Build pixel regions
    episode_to_region, num_regions = build_pixel_regions(
        pixel_vecs=pixel_vecs,
        episode_indices=episode_indices,
        region_ratio=region_ratio,
        min_regions=min_regions,
        max_regions=max_regions,
        seed=seed,
        use_pca=use_pca,
        pca_dim=pca_dim,
    )

    # Step 2: Initial B0
    selected = select_initial_b0_by_pixel_regions(
        episode_to_region=episode_to_region,
        num_regions=num_regions,
        b0_region_ratio=b0_region_ratio,
        b0_size=b0_size,
        seed=seed,
    )

    for i, ep_idx in enumerate(selected):
        selection_log.append({
            "step": i + 1,
            "episode_index": ep_idx,
            "phase": "initial_batch",
            "region_id": episode_to_region.get(ep_idx, -1),
        })

    # Step 3: Iterative selection
    print(f"\n[Iterative selection] Target: {num_select} episodes...")

    while len(selected) < num_select:
        step = len(selected) + 1

        # Compute region priorities
        region_priorities = {}
        for rid in range(num_regions):
            region_priorities[rid] = compute_region_priority(
                region_id=rid,
                episode_to_region=episode_to_region,
                selected_ids=selected,
                episode_data=episode_data,
                coverage_weight=coverage_weight,
                region_visual_weight=region_visual_weight,
                region_action_weight=region_action_weight,
            )

        # Pick highest priority region
        best_region = max(region_priorities.keys(), key=lambda r: region_priorities[r]["priority"])

        # Get candidates from that region
        candidates = [ep for ep, rid in episode_to_region.items()
                      if rid == best_region and ep not in selected]

        if not candidates:
            # Fallback: pick from any region with remaining episodes
            remaining = [ep for ep in episode_to_region.keys() if ep not in selected]
            if not remaining:
                print(f"  No more candidates, stopping at {len(selected)} episodes")
                break
            rng = np.random.RandomState(seed + step)
            chosen = int(rng.choice(remaining, size=1)[0])
            selected.append(chosen)
            selection_log.append({
                "step": step,
                "episode_index": chosen,
                "phase": "random_fallback",
                "note": "Region exhausted, random fallback",
            })
            continue

        # Score candidates
        best_candidate = None
        best_score = -1.0
        best_scores = None
        for cand in candidates:
            score, scores_dict = score_candidate_in_region(
                candidate_idx=cand,
                selected_ids=selected,
                episode_data=episode_data,
                visual_weight=candidate_visual_weight,
                action_weight=candidate_action_weight,
            )
            if score > best_score:
                best_score = score
                best_candidate = cand
                best_scores = scores_dict

        if best_candidate is None:
            remaining = [ep for ep in episode_to_region.keys() if ep not in selected]
            if not remaining:
                break
            rng = np.random.RandomState(seed + step)
            chosen = int(rng.choice(remaining, size=1)[0])
            selected.append(chosen)
            selection_log.append({
                "step": step,
                "episode_index": chosen,
                "phase": "random_fallback",
            })
        else:
            selected.append(best_candidate)
            selection_log.append({
                "step": step,
                "episode_index": best_candidate,
                "phase": "iterative_selection",
                "region_id": best_region,
                "region_priority": region_priorities[best_region]["priority"],
                "coverage_gap": region_priorities[best_region]["coverage_gap"],
                "visual_uncertainty": region_priorities[best_region]["visual_uncertainty"],
                "action_uncertainty": region_priorities[best_region]["action_uncertainty"],
                "candidate_scores": best_scores,
            })
            if step % 4 == 0 or step <= b0_size + 2:
                print(f"  Step {step}: episode {best_candidate} (region {best_region}, "
                      f"priority={region_priorities[best_region]['priority']:.3f}, "
                      f"score={best_score:.4f})")

    print(f"\n  Final: {len(selected)} episodes selected")
    return selected, selection_log, episode_to_region


def run_selection_for_dataset(
    dataset_path: str,
    dataset_name: str,
    output_dir: Path,
    num_selected: int = 20,
    b0_size: int = 4,
    seed: int = 42,
    pixel_camera_key: str = "observation.images.top",
    pixel_size: int = 64,
    region_ratio: float = 0.1,
    min_regions: int = 4,
    max_regions: int = 16,
    b0_region_ratio: float = 0.5,
    coverage_weight: float = 0.5,
    region_visual_weight: float = 0.3,
    region_action_weight: float = 0.2,
    candidate_visual_weight: float = 0.5,
    candidate_action_weight: float = 0.5,
    use_pca: bool = False,
    pca_dim: int = 64,
) -> Dict:
    """
    对单个 dataset 执行完整的 our_v5_real 选择流程。
    """
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    print(f"{'='*80}")

    # Load dataset
    dataset = LeRobotDataset(repo_id=dataset_name, root=dataset_path)
    total_episodes = dataset.meta.total_episodes
    episode_indices = list(range(total_episodes))
    print(f"  Total episodes: {total_episodes}")
    print(f"  Camera keys: {dataset.meta.camera_keys}")

    # Ensure visual embeddings and action descriptors via V5 flow
    print(f"\n[1/4] Ensuring visual embeddings and action descriptors...")
    visual_emb_dir, action_desc_dir = ensure_v5_embeddings_exist(
        dataset_root=dataset_path,
        dataset_name=dataset_name,
        gpu_id=0,  # Will be set by launcher via CUDA_VISIBLE_DEVICES
        pca_dim=32,
    )

    # Load visual embeddings from cache
    visual_embeddings = {}
    for f in sorted(visual_emb_dir.glob("*.npy")):
        if f.name.startswith("(") and f.name.endswith(".npy"):
            try:
                data = np.load(str(f), allow_pickle=True).item()
                ep_idx = int(data.get("episode_index", -1))
                if ep_idx >= 0:
                    phi = data.get("phi_global")
                    phi_w = data.get("phi_wrist")
                    if phi is not None and phi_w is not None:
                        visual_embeddings[ep_idx] = np.concatenate([phi, phi_w])
            except Exception:
                continue
    print(f"  Loaded visual embeddings for {len(visual_embeddings)} episodes")

    # Load action descriptors
    action_descriptors = load_action_descriptors(action_desc_dir)
    print(f"  Loaded action descriptors for {len(action_descriptors)} episodes")

    # Extract pixel vectors
    print(f"\n[2/4] Extracting pixel vectors...")
    pixel_vecs = load_pixel_vecs(dataset, episode_indices, camera_key=pixel_camera_key, pixel_size=pixel_size)
    print(f"  Extracted pixel vectors for {len(pixel_vecs)} episodes")

    # Log pixel diagnostics
    log_pixel_diagnostics(pixel_vecs, episode_indices)

    # Validate episode coverage
    valid_episodes = set(episode_indices)
    missing_pixel = [ep for ep in episode_indices if ep not in pixel_vecs]
    missing_vis = [ep for ep in episode_indices if ep not in visual_embeddings]
    missing_act = [ep for ep in episode_indices if ep not in action_descriptors]

    if missing_pixel:
        print(f"  WARNING: Missing pixel_vec for episodes: {missing_pixel[:10]}...")
    if missing_vis:
        print(f"  WARNING: Missing visual_embedding for episodes: {missing_vis[:10]}...")
    if missing_act:
        print(f"  WARNING: Missing action_descriptor for episodes: {missing_act[:10]}...")

    if missing_pixel or missing_vis or missing_act:
        print(f"  ERROR: Missing data for some episodes. Exiting.")
        sys.exit(1)

    # Build episode_data
    episode_data = {}
    for ep in episode_indices:
        episode_data[ep] = {
            "pixel_vec": pixel_vecs[ep],
            "visual_embedding": visual_embeddings[ep],
            "action_descriptor": action_descriptors[ep],
        }

    # Run selection
    print(f"\n[3/4] Running our_v5 selection...")
    selected, selection_log, episode_to_region = select_episodes_our_v5_real(
        episode_data=episode_data,
        pixel_vecs=pixel_vecs,
        episode_indices=episode_indices,
        num_select=num_selected,
        b0_size=b0_size,
        region_ratio=region_ratio,
        min_regions=min_regions,
        max_regions=max_regions,
        b0_region_ratio=b0_region_ratio,
        coverage_weight=coverage_weight,
        region_visual_weight=region_visual_weight,
        region_action_weight=region_action_weight,
        candidate_visual_weight=candidate_visual_weight,
        candidate_action_weight=candidate_action_weight,
        seed=seed,
        use_pca=use_pca,
        pca_dim=pca_dim,
    )

    # Save results
    subsets_dir = output_dir / "subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)

    subset_file = subsets_dir / f"{normalize_dataset_name(dataset_name)}_selected.json"
    subset_data = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "method": "our_v5_real",
        "num_selected": len(selected),
        "selected_episode_indices": selected,
        "selection_log": selection_log,
        "episode_to_region": {str(k): v for k, v in episode_to_region.items()},
        "parameters": {
            "num_selected": num_selected,
            "b0_size": b0_size,
            "seed": seed,
            "pixel_camera_key": pixel_camera_key,
            "pixel_size": pixel_size,
            "region_ratio": region_ratio,
            "min_regions": min_regions,
            "max_regions": max_regions,
            "b0_region_ratio": b0_region_ratio,
            "coverage_weight": coverage_weight,
            "region_visual_weight": region_visual_weight,
            "region_action_weight": region_action_weight,
            "candidate_visual_weight": candidate_visual_weight,
            "candidate_action_weight": candidate_action_weight,
            "use_pca": use_pca,
            "pca_dim": pca_dim,
        },
    }
    with open(subset_file, "w") as f:
        json.dump(subset_data, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved subset: {subset_file}")

    # Save episodes.txt for training
    episodes_str = ",".join(map(str, sorted(selected)))
    episodes_file = output_dir / f"{normalize_dataset_name(dataset_name)}_episodes.txt"
    with open(episodes_file, "w") as f:
        f.write(episodes_str)
    print(f"  Saved episodes list: {episodes_file}")

    return {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "selected_episode_indices": selected,
        "num_selected": len(selected),
        "subset_file": str(subset_file),
        "episode_to_region": episode_to_region,
    }


def main():
    parser = argparse.ArgumentParser(description="Our V5 Real Episode Selection")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-selected", type=int, default=20)
    parser.add_argument("--b0-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pixel-camera-key", type=str, default="observation.images.top")
    parser.add_argument("--pixel-size", type=int, default=64)
    parser.add_argument("--region-ratio", type=float, default=0.1)
    parser.add_argument("--min-regions", type=int, default=4)
    parser.add_argument("--max-regions", type=int, default=16)
    parser.add_argument("--b0-region-ratio", type=float, default=0.5)
    parser.add_argument("--coverage-weight", type=float, default=0.5)
    parser.add_argument("--region-visual-weight", type=float, default=0.3)
    parser.add_argument("--region-action-weight", type=float, default=0.2)
    parser.add_argument("--candidate-visual-weight", type=float, default=0.5)
    parser.add_argument("--candidate-action-weight", type=float, default=0.5)
    parser.add_argument("--use-pca", action="store_true", help="Use PCA on pixel vectors (default: off)")
    parser.add_argument("--pca-dim", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_selection_for_dataset(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        output_dir=output_dir,
        num_selected=args.num_selected,
        b0_size=args.b0_size,
        seed=args.seed,
        pixel_camera_key=args.pixel_camera_key,
        pixel_size=args.pixel_size,
        region_ratio=args.region_ratio,
        min_regions=args.min_regions,
        max_regions=args.max_regions,
        b0_region_ratio=args.b0_region_ratio,
        coverage_weight=args.coverage_weight,
        region_visual_weight=args.region_visual_weight,
        region_action_weight=args.region_action_weight,
        candidate_visual_weight=args.candidate_visual_weight,
        candidate_action_weight=args.candidate_action_weight,
        use_pca=args.use_pca,
        pca_dim=args.pca_dim,
    )

    # Save selection manifest entry
    manifest_file = output_dir / "selection_manifest.json"
    if manifest_file.exists():
        with open(manifest_file) as f:
            manifest = json.load(f)
    else:
        manifest = {"datasets": []}

    manifest["datasets"].append({
        "name": result["dataset_name"],
        "root": result["dataset_path"],
        "selected_episode_indices": result["selected_episode_indices"],
        "num_selected": result["num_selected"],
        "subset_file": result["subset_file"],
    })
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Updated selection manifest: {manifest_file}")

    print(f"\n{'='*80}")
    print(f"Selection complete for {result['dataset_name']}: {result['num_selected']} episodes")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()