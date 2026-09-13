#!/usr/bin/env python
"""
dOur V5 Real Episode Selection - 基于VLM embedding的自适应覆盖选择

保留our_v5核心思路，但适配真实机器人数据：
- 不用rand_vec（真实环境没有精确配置），改用第一帧像素特征进行区域划分
- 视觉特征使用VLM embedding（phi_global + phi_wrist），跟our_v5一样
- 区域优先级和episode打分都用VLM embedding

核心算法：
1. 提取每个episode的VLM embedding（phi_global + phi_wrist）
2. 用像素特征进行KMeans聚类分区（区域数动态计算）
3. 初始B0：从不同区域均匀选择
4. 迭代选择：
   a. 区域优先级 = coverage_weight * coverage_gap + visual_weight * visual_uncertainty
   b. 选最高优先级区域
   c. 区域内episode打分 = min_dist(VLM embedding, 已选集)

Usage:
    python select_our_v5_real.py \
        --dataset-root /data/zhonglinye/jun/ep10_30episodes_standing40/ep10_30episodes_standing40 \
        --output-dir /data/zhonglinye/jun/lerobot/personal/work1/our_v5_real_results \
        --num-selected 20 \
        --batch-size 4 \
        --seed 42 \
        --gpu-id 0
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
import joblib

# 添加lerobot到path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# VLM模型配置
VLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
PCA_DIM = 32
TASK_TEXT = "Pick and place a puck to a goal"


class NumpyEncoder(json.JSONEncoder):
    """JSON编码器，支持numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_vlm_model(gpu_id: int = 0):
    """
    加载VLM模型（SmolVLM2）
    
    Returns:
        (model, processor)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    import torch
    from transformers import AutoModel, AutoProcessor
    from PIL import Image
    
    print(f"\n{'='*60}")
    print(f"加载VLM模型: {VLM_MODEL_ID}")
    print(f"GPU: {gpu_id}")
    print(f"{'='*60}")
    
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    model = AutoModel.from_pretrained(
        VLM_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda().eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"VLM模型加载完成")
    print(f"  vision_model 类型: {type(model.vision_model).__name__}")
    print(f"  connector 类型: {type(model.connector).__name__}")
    
    return model, processor, Image


def extract_vlm_embeddings_for_task(
    dataset: LeRobotDataset,
    episode_indices: List[int],
    model,
    processor,
    Image_class,
    gpu_id: int = 0,
    pca_dim: int = PCA_DIM,
    cache_dir: Optional[Path] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    为任务的episodes提取VLM embedding
    
    跟our_v5一样：
    - phi_global: top相机前5帧embedding均值
    - phi_wrist: wrist相机20%-70%进度区间帧均值
    - PCA降维到32维
    
    Args:
        dataset: LeRobotDataset
        episode_indices: episode索引列表
        model: VLM模型
        processor: VLM处理器
        Image_class: PIL Image类
        gpu_id: GPU ID
        pca_dim: PCA降维维度
        cache_dir: 缓存目录
        
    Returns:
        Tuple of:
            - Dict[episode_index, phi_global]
            - Dict[episode_index, phi_wrist]
    """
    import torch
    
    # 检查缓存
    if cache_dir and cache_dir.exists():
        cached = _load_cached_embeddings(cache_dir, episode_indices)
        if cached:
            print(f"  从缓存加载 {len(cached[0])} 个episodes的VLM embedding")
            return cached
    
    print(f"\n[Step 1] 提取VLM embedding...")
    print(f"  模型: {VLM_MODEL_ID}")
    print(f"  PCA维度: {pca_dim}")
    print(f"  Episodes: {len(episode_indices)}")
    
    phi_global_dict = {}
    phi_wrist_dict = {}
    
    image_token = processor.tokenizer.image_token if hasattr(processor.tokenizer, 'image_token') else "<image>"
    text = f"{image_token}\n{TASK_TEXT}"
    
    # 批量处理
    batch_size = 8
    
    for ep_idx in episode_indices:
        ep_meta = dataset.meta.episodes[ep_idx]
        from_idx = ep_meta["dataset_from_index"]
        to_idx = ep_meta["dataset_to_index"]
        n_frames = to_idx - from_idx + 1
        
        # 获取top相机帧
        top_frames = []
        wrist_frames = []
        
        for i in range(n_frames):
            frame = dataset[from_idx + i]
            
            if "observation.images.top" in frame:
                img = frame["observation.images.top"]
                if hasattr(img, 'numpy'):
                    img = img.numpy()
                if img.ndim == 3 and img.shape[0] < img.shape[-1]:
                    img = np.transpose(img, (1, 2, 0))
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                top_frames.append(Image_class.fromarray(img))
            
            if "observation.images.wrist" in frame:
                img = frame["observation.images.wrist"]
                if hasattr(img, 'numpy'):
                    img = img.numpy()
                if img.ndim == 3 and img.shape[0] < img.shape[-1]:
                    img = np.transpose(img, (1, 2, 0))
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                wrist_frames.append(Image_class.fromarray(img))
        
        # 提取phi_global（top相机前5帧）
        if len(top_frames) > 0:
            n_global = min(5, len(top_frames))
            global_embs = []
            
            for i in range(0, n_global, batch_size):
                batch = top_frames[i:i+batch_size]
                with torch.no_grad():
                    inputs = processor(
                        images=batch,
                        return_tensors="pt",
                    ).to("cuda")
                    
                    pixel_values = inputs["pixel_values"]
                    if pixel_values.ndim == 5:
                        pixel_values = pixel_values[:, 0]
                    pixel_values = pixel_values.to(dtype=model.vision_model.dtype)
                    
                    vision_output = model.vision_model(pixel_values=pixel_values)
                    vision_hidden = vision_output.last_hidden_state
                    
                    connector_output = model.connector(vision_hidden)
                    global_embs.append(connector_output.squeeze(0).cpu().numpy())
            
            if global_embs:
                phi_global = np.mean(global_embs, axis=0)
                phi_global_dict[ep_idx] = phi_global
        
        # 提取phi_wrist（wrist相机20%-70%进度）
        if len(wrist_frames) > 0:
            start_idx = int(len(wrist_frames) * 0.2)
            end_idx = int(len(wrist_frames) * 0.7)
            wrist_selected = wrist_frames[start_idx:end_idx]
            
            if len(wrist_selected) > 0:
                wrist_embs = []
                
                for i in range(0, len(wrist_selected), batch_size):
                    batch = wrist_selected[i:i+batch_size]
                    with torch.no_grad():
                        inputs = processor(
                            images=batch,
                            return_tensors="pt",
                        ).to("cuda")
                        
                        pixel_values = inputs["pixel_values"]
                        if pixel_values.ndim == 5:
                            pixel_values = pixel_values[:, 0]
                        pixel_values = pixel_values.to(dtype=model.vision_model.dtype)
                        
                        vision_output = model.vision_model(pixel_values=pixel_values)
                        vision_hidden = vision_output.last_hidden_state
                        
                        connector_output = model.connector(vision_hidden)
                        wrist_embs.append(connector_output.squeeze(0).cpu().numpy())
                
                if wrist_embs:
                    phi_wrist = np.mean(wrist_embs, axis=0)
                    phi_wrist_dict[ep_idx] = phi_wrist
        
        if (ep_idx + 1) % 5 == 0:
            print(f"  已处理 {ep_idx + 1}/{len(episode_indices)} episodes")
    
    print(f"  成功提取 {len(phi_global_dict)} 个episodes的VLM embedding")
    
    # PCA降维
    if len(phi_global_dict) > 0:
        global_embs_array = np.array([phi_global_dict[ep] for ep in sorted(phi_global_dict.keys())])
        wrist_embs_array = np.array([phi_wrist_dict[ep] for ep in sorted(phi_wrist_dict.keys())])
        
        # 拟合PCA
        pca_global = joblib.PCA(n_components=pca_dim, random_state=42)
        pca_wrist = joblib.PCA(n_components=pca_dim, random_state=42)
        
        pca_global.fit(global_embs_array)
        pca_wrist.fit(wrist_embs_array)
        
        # 转换
        for i, ep_idx in enumerate(sorted(phi_global_dict.keys())):
            phi_global_dict[ep_idx] = pca_global.transform(global_embs_array[i:i+1])[0]
            phi_wrist_dict[ep_idx] = pca_wrist.transform(wrist_embs_array[i:i+1])[0]
        
        # 保存缓存
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for ep_idx in phi_global_dict:
                np.save(
                    cache_dir / f"episode_{ep_idx}.npy",
                    {
                        "episode_index": ep_idx,
                        "phi_global": phi_global_dict[ep_idx],
                        "phi_wrist": phi_wrist_dict[ep_idx],
                    },
                    allow_pickle=True,
                )
    
    return phi_global_dict, phi_wrist_dict


def _load_cached_embeddings(
    cache_dir: Path,
    episode_indices: List[int],
) -> Optional[Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]]:
    """加载缓存的VLM embedding"""
    phi_global_dict = {}
    phi_wrist_dict = {}
    
    for ep_idx in episode_indices:
        cache_file = cache_dir / f"episode_{ep_idx}.npy"
        if not cache_file.exists():
            return None
        
        try:
            data = np.load(str(cache_file), allow_pickle=True).item()
            phi_global_dict[ep_idx] = data["phi_global"]
            phi_wrist_dict[ep_idx] = data["phi_wrist"]
        except Exception:
            return None
    
    return phi_global_dict, phi_wrist_dict


def extract_first_frame_features(
    dataset: LeRobotDataset,
    episode_indices: List[int],
    camera_key: str = "observation.images.top",
) -> Dict[int, np.ndarray]:
    """
    提取每个episode第一帧的top相机图像特征（用于区域划分）
    
    Args:
        dataset: LeRobotDataset
        episode_indices: 要提取的episode索引列表
        camera_key: 相机key
        
    Returns:
        Dict[episode_index, flattened_normalized_pixel_vector]
    """
    features = {}
    
    for ep_idx in episode_indices:
        ep_meta = dataset.meta.episodes[ep_idx]
        from_idx = ep_meta["dataset_from_index"]
        
        frame = dataset[from_idx]
        
        if camera_key not in frame:
            print(f"  ⚠️ Episode {ep_idx}: 没有找到 {camera_key}")
            continue
            
        img_tensor = frame[camera_key]
        
        if hasattr(img_tensor, 'numpy'):
            img_np = img_tensor.numpy()
        else:
            img_np = np.array(img_tensor)
        
        # 确保是 (H, W, C) 格式
        if img_np.ndim == 3 and img_np.shape[0] < img_np.shape[-1]:
            img_np = np.transpose(img_np, (1, 2, 0))
        
        # 展平并归一化到 [0, 1]
        img_flat = img_np.flatten().astype(np.float32)
        img_normalized = img_flat / 255.0
        
        features[int(ep_idx)] = img_normalized
        
        if (ep_idx + 1) % 10 == 0:
            print(f"  已处理 {ep_idx + 1}/{len(episode_indices)} episodes")
    
    print(f"  成功提取 {len(features)} 个episodes的像素特征")
    return features


def build_pixel_regions(
    episode_features: Dict[int, np.ndarray],
    region_ratio: float = 0.1,
    min_regions: int = 4,
    max_regions: int = 16,
    seed: int = 42,
) -> Tuple[Dict[int, int], int]:
    """
    基于像素特征构建区域（替代our_v5的rand_vec区域）
    
    使用KMeans对像素特征进行聚类分区。
    由于像素维度很高（921600），先用随机投影降维到合理维度。
    
    Args:
        episode_features: Dict[episode_index, pixel_feature_vector]
        region_ratio: 区域数计算比例
        min_regions: 最小区域数
        max_regions: 最大区域数
        seed: 随机种子
        
    Returns:
        Tuple of:
            - Dict[episode_index, region_id]: episode到区域的映射
            - int: 区域数量
    """
    print(f"\n{'='*60}")
    print(f"基于像素特征构建区域（KMeans聚类）...")
    print(f"{'='*60}")
    
    valid_episodes = sorted(episode_features.keys())
    n_episodes = len(valid_episodes)
    
    if n_episodes == 0:
        raise ValueError("没有有效的episode特征")
    
    # 动态计算区域数
    num_regions = max(min_regions, min(max_regions, int(n_episodes * region_ratio)))
    num_regions = min(num_regions, n_episodes)
    
    print(f"  Episodes: {n_episodes}")
    print(f"  区域比例: {region_ratio}")
    print(f"  计算区域数: {num_regions} (min={min_regions}, max={max_regions})")
    
    # 提取特征矩阵
    feature_list = [episode_features[ep_idx] for ep_idx in valid_episodes]
    feature_matrix = np.array(feature_list)
    
    print(f"  原始特征维度: {feature_matrix.shape}")
    
    # 随机投影降维（避免KMeans在高维空间失效）
    rng = np.random.RandomState(seed)
    target_dim = min(128, feature_matrix.shape[1])
    n_samples = min(5000, feature_matrix.shape[1])
    
    sample_indices = rng.choice(feature_matrix.shape[1], n_samples, replace=False)
    feature_sampled = feature_matrix[:, sample_indices]
    
    # 随机投影
    projection_matrix = rng.randn(n_samples, target_dim).astype(np.float32)
    projection_matrix /= np.linalg.norm(projection_matrix, axis=0)
    
    features_projected = feature_sampled @ projection_matrix
    
    # L2归一化
    norms = np.linalg.norm(features_projected, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    features_normalized = features_projected / norms
    
    # KMeans聚类
    kmeans = KMeans(
        n_clusters=num_regions,
        random_state=seed,
        n_init=10,
    )
    region_ids = kmeans.fit_predict(features_normalized)
    
    # 构建episode到区域的映射
    episode_to_region = {}
    for i, ep_idx in enumerate(valid_episodes):
        episode_to_region[ep_idx] = int(region_ids[i])
    
    # 打印区域统计
    region_counts = {}
    for region_id in region_ids:
        region_id = int(region_id)
        region_counts[region_id] = region_counts.get(region_id, 0) + 1
    
    print(f"  创建区域数: {num_regions}")
    print(f"  区域大小 - 最小: {min(region_counts.values())}, 最大: {max(region_counts.values())}, "
          f"平均: {np.mean(list(region_counts.values())):.1f}")
    
    return episode_to_region, num_regions


def select_initial_b0(
    episode_to_region: Dict[int, int],
    episode_features: Dict[int, np.ndarray],
    num_regions: int,
    b0_size: int = 4,
    seed: int = 42,
) -> List[int]:
    """
    初始B0选择：从不同区域均匀采样
    
    Args:
        episode_to_region: episode到区域的映射
        episode_features: episode特征
        num_regions: 区域总数
        b0_size: B0大小
        seed: 随机种子
        
    Returns:
        选中的episode索引列表
    """
    rng = np.random.RandomState(seed)
    
    # 按区域分组
    region_to_episodes = {}
    for ep_idx, region_id in episode_to_region.items():
        if region_id not in region_to_episodes:
            region_to_episodes[region_id] = []
        region_to_episodes[region_id].append(ep_idx)
    
    # 从不同区域选择
    selected = []
    region_ids = sorted(region_to_episodes.keys())
    
    # 第一轮：每个区域选一个，直到B0填满
    for region_id in region_ids:
        if len(selected) >= b0_size:
            break
        region_episodes = region_to_episodes[region_id]
        chosen = rng.choice(region_episodes, size=1, replace=False).tolist()[0]
        selected.append(chosen)
    
    # 如果B0还没填满，从大区域继续选
    if len(selected) < b0_size:
        remaining_episodes = [ep for ep in episode_features.keys() if ep not in selected]
        n_needed = b0_size - len(selected)
        additional = rng.choice(remaining_episodes, size=min(n_needed, len(remaining_episodes)), replace=False).tolist()
        selected.extend(additional)
    
    selected.sort()
    print(f"\n[Step 2] 初始B0选择: {len(selected)} episodes (来自{num_regions}个区域)")
    print(f"  B0 episodes: {selected}")
    
    return selected


def compute_region_priority(
    region_id: int,
    episode_to_region: Dict[int, int],
    selected_ids: List[int],
    vlm_embeddings: Dict[int, np.ndarray],
    coverage_weight: float = 0.5,
    visual_weight: float = 0.3,
) -> Dict[str, float]:
    """
    计算区域优先级（跟our_v5一样，用VLM embedding）
    
    优先级 = coverage_weight * coverage_gap + visual_weight * visual_uncertainty
    
    Args:
        region_id: 区域ID
        episode_to_region: episode到区域的映射
        selected_ids: 已选中的episode列表
        vlm_embeddings: VLM embedding字典
        coverage_weight: 覆盖缺口权重
        visual_weight: 视觉不确定性权重
        
    Returns:
        Dict with keys: "priority", "coverage_gap", "visual_uncertainty"
    """
    # 获取该区域的所有episodes
    region_episodes = [ep for ep, rid in episode_to_region.items() if rid == region_id]
    total_in_region = len(region_episodes)
    
    if total_in_region == 0:
        return {
            "priority": 0.0,
            "coverage_gap": 0.0,
            "visual_uncertainty": 0.0,
        }
    
    # 1. 覆盖缺口：该区域选中的越少，优先级越高
    selected_in_region = [ep for ep in region_episodes if ep in selected_ids]
    n_selected_in_region = len(selected_in_region)
    
    coverage_gap = 1.0 - (n_selected_in_region / total_in_region)
    
    # 2. 视觉不确定性：区域内VLM embedding的平均两两距离
    visual_uncertainty = 0.0
    region_features = []
    for ep in region_episodes:
        if ep in vlm_embeddings:
            region_features.append(vlm_embeddings[ep])
    
    if len(region_features) >= 2:
        # 计算平均两两距离
        features_arr = np.array(region_features)
        n = len(features_arr)
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += np.linalg.norm(features_arr[i] - features_arr[j])
                count += 1
        visual_uncertainty = total_dist / count if count > 0 else 0.0
    elif len(region_features) == 1:
        visual_uncertainty = np.linalg.norm(region_features[0])
    
    # 归一化到[0, 1]
    visual_uncertainty_norm = min(visual_uncertainty / 10.0, 1.0)
    
    # 计算最终优先级
    priority = coverage_weight * coverage_gap + visual_weight * visual_uncertainty_norm
    
    return {
        "priority": priority,
        "coverage_gap": coverage_gap,
        "visual_uncertainty": visual_uncertainty_norm,
    }


def compute_vlm_coverage_score(
    candidate_feature: np.ndarray,
    selected_features: np.ndarray,
) -> float:
    """
    计算VLM embedding覆盖分数（跟our_v5一样）
    
    分数 = 候选episode与已选episodes的最小VLM embedding距离
    分数越高 = 与已选episodes差异越大 = 覆盖越好
    
    Args:
        candidate_feature: 候选episode的VLM embedding
        selected_features: 已选episodes的VLM embedding, shape (n_selected, dim)
        
    Returns:
        覆盖分数（越高越好）
    """
    if len(selected_features) == 0:
        return 1.0
    
    distances = np.linalg.norm(selected_features - candidate_feature, axis=1)
    min_distance = float(np.min(distances))
    
    return min_distance


def select_best_episode_from_region(
    region_id: int,
    episode_to_region: Dict[int, int],
    selected_ids: List[int],
    vlm_embeddings: Dict[int, np.ndarray],
) -> Optional[Tuple[int, Dict[str, float]]]:
    """
    从指定区域选择最佳episode（基于VLM embedding覆盖分数，跟our_v5一样）
    
    Args:
        region_id: 区域ID
        episode_to_region: episode到区域的映射
        selected_ids: 已选中的episode列表
        vlm_embeddings: VLM embedding字典
        
    Returns:
        Tuple of (best_episode_index, scores_dict) or None
    """
    # 获取候选episodes（该区域中未选中的）
    candidates = [ep for ep, rid in episode_to_region.items() 
                  if rid == region_id and ep not in selected_ids]
    
    if not candidates:
        return None
    
    # 准备已选episodes的VLM embedding
    selected_features = np.array([
        vlm_embeddings[ep]
        for ep in selected_ids
        if ep in vlm_embeddings
    ])
    
    # 计算所有候选的分数
    candidate_scores = []
    for candidate_idx in candidates:
        if candidate_idx not in vlm_embeddings:
            continue
        
        vlm_score = compute_vlm_coverage_score(
            vlm_embeddings[candidate_idx],
            selected_features
        )
        
        candidate_scores.append((candidate_idx, vlm_score))
    
    if not candidate_scores:
        return None
    
    # 归一化分数到[0, 1]
    vlm_scores = [s[1] for s in candidate_scores]
    vlm_min, vlm_max = min(vlm_scores), max(vlm_scores)
    vlm_range = vlm_max - vlm_min if vlm_max > vlm_min else 1.0
    
    # 选择分数最高的
    best_candidate = None
    best_score = -1.0
    best_scores = None
    
    for candidate_idx, vs in candidate_scores:
        norm_vs = (vs - vlm_min) / vlm_range
        
        if norm_vs > best_score:
            best_score = norm_vs
            best_candidate = candidate_idx
            best_scores = {
                "vlm_coverage": vs,
                "vlm_coverage_normalized": norm_vs,
                "final_score": norm_vs,
            }
    
    return (best_candidate, best_scores)


def select_episodes_v5_real(
    episode_features: Dict[int, np.ndarray],
    vlm_embeddings: Dict[int, np.ndarray],
    num_select: int = 20,
    b0_size: int = 4,
    region_ratio: float = 0.1,
    min_regions: int = 4,
    max_regions: int = 16,
    coverage_weight: float = 0.5,
    visual_weight: float = 0.3,
    seed: int = 42,
) -> Tuple[List[int], List[Dict]]:
    """
    Our V5 Real核心选择算法（跟our_v5一样的结构）
    
    Args:
        episode_features: 像素特征字典（用于区域划分）
        vlm_embeddings: VLM embedding字典（用于打分）
        num_select: 目标选择数量
        b0_size: 初始B0大小
        region_ratio: 区域数计算比例
        min_regions: 最小区域数
        max_regions: 最大区域数
        coverage_weight: 覆盖缺口权重
        visual_weight: 视觉不确定性权重
        seed: 随机种子
        
    Returns:
        Tuple of:
            - 选中的episode索引列表（按选择顺序）
            - 选择日志列表
    """
    selection_log = []
    
    # Step 1: 构建区域（用像素特征）
    episode_to_region, num_regions = build_pixel_regions(
        episode_features=episode_features,
        region_ratio=region_ratio,
        min_regions=min_regions,
        max_regions=max_regions,
        seed=seed,
    )
    
    # Step 2: 初始B0选择
    selected = select_initial_b0(
        episode_to_region=episode_to_region,
        episode_features=episode_features,
        num_regions=num_regions,
        b0_size=b0_size,
        seed=seed,
    )
    
    for i, ep_idx in enumerate(selected):
        selection_log.append({
            "step": i + 1,
            "episode_index": ep_idx,
            "phase": "initial_batch",
            "note": f"初始B0选择",
        })
    
    # Step 3: 迭代选择
    print(f"\n[Step 3] 迭代选择（目标: {num_select} episodes）...")
    
    while len(selected) < num_select:
        step = len(selected) + 1
        
        # 计算所有区域的优先级（用VLM embedding）
        region_priorities = {}
        for region_id in range(num_regions):
            priority_info = compute_region_priority(
                region_id=region_id,
                episode_to_region=episode_to_region,
                selected_ids=selected,
                vlm_embeddings=vlm_embeddings,
                coverage_weight=coverage_weight,
                visual_weight=visual_weight,
            )
            region_priorities[region_id] = priority_info
        
        # 选择优先级最高的区域
        best_region = max(region_priorities.keys(), 
                         key=lambda r: region_priorities[r]["priority"])
        
        # 从该区域选择最佳episode（用VLM embedding打分）
        result = select_best_episode_from_region(
            region_id=best_region,
            episode_to_region=episode_to_region,
            selected_ids=selected,
            vlm_embeddings=vlm_embeddings,
        )
        
        if result is None:
            # 该区域没有候选了，标记为已满
            print(f"  区域{best_region}没有候选episodes，跳过")
            # 从剩余区域中随机选择
            remaining_episodes = [ep for ep in episode_features.keys() if ep not in selected]
            if remaining_episodes:
                rng = np.random.RandomState(seed + step)
                chosen = rng.choice(remaining_episodes, size=1, replace=False).tolist()[0]
                selected.append(chosen)
                selection_log.append({
                    "step": step,
                    "episode_index": chosen,
                    "phase": "random_fallback",
                    "note": "区域已满，随机补充",
                })
            else:
                print(f"  没有更多候选episodes，提前结束")
                break
        else:
            chosen_ep, scores = result
            selected.append(chosen_ep)
            
            selection_log.append({
                "step": step,
                "episode_index": chosen_ep,
                "phase": "iterative_selection",
                "region_id": best_region,
                "region_priority": region_priorities[best_region]["priority"],
                "coverage_gap": region_priorities[best_region]["coverage_gap"],
                "visual_uncertainty": region_priorities[best_region]["visual_uncertainty"],
                "vlm_coverage_score": scores["vlm_coverage"],
                "note": f"从区域{best_region}选择（优先级={region_priorities[best_region]['priority']:.3f}）",
            })
            
            if step % 4 == 0 or step <= b0_size + 2:
                print(f"  [Step {step}] 选择 episode {chosen_ep} "
                      f"(区域{best_region}, VLM覆盖分数={scores['vlm_coverage']:.4f})")
    
    print(f"\n  最终选择: {len(selected)} episodes（按选择顺序）")
    
    return selected, selection_log


def select_episodes_for_task(
    dataset: LeRobotDataset,
    task_episode_indices: List[int],
    task_name: str,
    model,
    processor,
    Image_class,
    num_select: int = 20,
    b0_size: int = 4,
    seed: int = 42,
    camera_key: str = "observation.images.top",
    region_ratio: float = 0.3,
    min_regions: int = 8,
    max_regions: int = 32,
    coverage_weight: float = 0.5,
    visual_weight: float = 0.3,
    gpu_id: int = 0,
    pca_dim: int = PCA_DIM,
    cache_dir: Optional[Path] = None,
) -> Dict:
    """
    为单个任务选择episodes
    
    Args:
        dataset: LeRobotDataset
        task_episode_indices: 该任务包含的episode索引
        task_name: 任务名称
        model: VLM模型
        processor: VLM处理器
        Image_class: PIL Image类
        num_select: 目标选择数量
        b0_size: 初始B0大小
        seed: 随机种子
        camera_key: 相机key
        region_ratio: 区域数计算比例
        min_regions: 最小区域数
        max_regions: 最大区域数
        coverage_weight: 覆盖缺口权重
        visual_weight: 视觉不确定性权重
        gpu_id: GPU ID
        pca_dim: PCA降维维度
        cache_dir: 缓存目录
        
    Returns:
        选择结果字典
    """
    print(f"\n{'='*60}")
    print(f"任务: {task_name}")
    print(f"Episode范围: {task_episode_indices[0]} ~ {task_episode_indices[-1]} (共{len(task_episode_indices)}集)")
    print(f"{'='*60}")
    
    # Step 1: 提取VLM embedding（phi_global + phi_wrist）
    print(f"\n[Step 1] 提取VLM embedding...")
    phi_global_dict, phi_wrist_dict = extract_vlm_embeddings_for_task(
        dataset=dataset,
        episode_indices=task_episode_indices,
        model=model,
        processor=processor,
        Image_class=Image_class,
        gpu_id=gpu_id,
        pca_dim=pca_dim,
        cache_dir=cache_dir,
    )
    
    # 合并phi_global和phi_wrist
    vlm_embeddings = {}
    for ep_idx in phi_global_dict:
        if ep_idx in phi_wrist_dict:
            vlm_embeddings[ep_idx] = np.concatenate([phi_global_dict[ep_idx], phi_wrist_dict[ep_idx]])
    
    if len(vlm_embeddings) == 0:
        print(f"  ⚠️ 没有有效的VLM embedding，跳过该任务")
        return {
            "task_name": task_name,
            "selected_episodes": [],
            "selection_order": [],
            "selection_log": [],
            "total_episodes": len(task_episode_indices),
        }
    
    # Step 2: 提取像素特征（用于区域划分）
    print(f"\n[Step 2] 提取第一帧像素特征（用于区域划分）...")
    pixel_features = extract_first_frame_features(dataset, task_episode_indices, camera_key)
    
    if len(pixel_features) == 0:
        print(f"  ⚠️ 没有有效的像素特征，跳过该任务")
        return {
            "task_name": task_name,
            "selected_episodes": [],
            "selection_order": [],
            "selection_log": [],
            "total_episodes": len(task_episode_indices),
        }
    
    # Step 3: V5 Real选择
    print(f"\n[Step 3] Our V5 Real选择算法...")
    selected_episodes, selection_log = select_episodes_v5_real(
        episode_features=pixel_features,
        vlm_embeddings=vlm_embeddings,
        num_select=num_select,
        b0_size=b0_size,
        seed=seed,
        region_ratio=region_ratio,
        min_regions=min_regions,
        max_regions=max_regions,
        coverage_weight=coverage_weight,
        visual_weight=visual_weight,
    )
    
    # Step 4: 输出结果
    print(f"\n[Step 4] 选择结果:")
    print(f"  选中episodes（按选择顺序）: {selected_episodes}")
    print(f"  Episode范围: {selected_episodes[0]} ~ {selected_episodes[-1]} (共{len(selected_episodes)}集)")
    
    return {
        "task_name": task_name,
        "selected_episodes": selected_episodes,
        "selection_order": selected_episodes.copy(),
        "selection_log": selection_log,
        "total_episodes": len(task_episode_indices),
        "num_selected": len(selected_episodes),
    }


def get_task_episode_ranges() -> Dict[str, List[int]]:
    """
    获取每个任务的episode索引范围（去掉bushing_old）
    """
    task_ranges = {
        "bushing": list(range(0, 30)),
        # "bushing_old": list(range(30, 70)),  # 跳过
        "charger": list(range(70, 100)),
        "cylinder_lay": list(range(100, 130)),
        "cylinder_standing": list(range(130, 170)),
        "plug": list(range(170, 200)),
        "rectangular": list(range(200, 230)),
        "ring": list(range(230, 260)),
        "ushape": list(range(260, 290)),
        "valve": list(range(290, 320)),
    }
    return task_ranges


def main():
    parser = argparse.ArgumentParser(description="Our V5 Real - 基于VLM embedding的Episode选择")
    parser.add_argument("--dataset-root", type=str, required=True,
                       help="数据集根目录")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--num-selected", type=int, default=20,
                       help="每个任务选择的episode数量 (default: 20)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="初始B0大小 (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子 (default: 42)")
    parser.add_argument("--camera-key", type=str, default="observation.images.top",
                       help="相机key (default: observation.images.top)")
    parser.add_argument("--region-ratio", type=float, default=0.3,
                       help="区域数计算比例 (default: 0.3)")
    parser.add_argument("--min-regions", type=int, default=8,
                       help="最小区域数 (default: 8)")
    parser.add_argument("--max-regions", type=int, default=32,
                       help="最大区域数 (default: 32)")
    parser.add_argument("--coverage-weight", type=float, default=0.5,
                       help="覆盖缺口权重 (default: 0.5)")
    parser.add_argument("--visual-weight", type=float, default=0.3,
                       help="视觉不确定性权重 (default: 0.3)")
    parser.add_argument("--gpu-id", type=int, default=0,
                       help="GPU ID (default: 0)")
    parser.add_argument("--pca-dim", type=int, default=PCA_DIM,
                       help=f"PCA降维维度 (default: {PCA_DIM})")
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"Our V5 Real Episode Selection")
    print(f"{'='*60}")
    print(f"数据集: {dataset_root}")
    print(f"输出目录: {output_dir}")
    print(f"每任务选择: {args.num_selected} episodes")
    print(f"初始B0: {args.batch_size}")
    print(f"随机种子: {args.seed}")
    print(f"相机: {args.camera_key}")
    print(f"区域参数: ratio={args.region_ratio}, min={args.min_regions}, max={args.max_regions}")
    print(f"权重: coverage={args.coverage_weight}, visual={args.visual_weight}")
    print(f"GPU: {args.gpu_id}")
    print(f"PCA维度: {args.pca_dim}")
    
    # 加载VLM模型
    model, processor, Image_class = load_vlm_model(args.gpu_id)
    
    # 加载数据集
    print(f"\n加载数据集...")
    dataset = LeRobotDataset(
        repo_id="ep10/standing40",
        root=str(dataset_root),
    )
    print(f"  总episodes: {dataset.meta.total_episodes}")
    print(f"  总帧数: {dataset.meta.total_frames}")
    print(f"  相机keys: {dataset.meta.camera_keys}")
    
    # 获取任务分布
    task_ranges = get_task_episode_ranges()
    
    # 为每个任务选择episodes
    all_results = []
    all_selected_episodes = []
    task_mapping = {}
    
    # 创建缓存目录
    cache_dir = output_dir / "vlm_embeddings_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for task_name, episode_indices in task_ranges.items():
        result = select_episodes_for_task(
            dataset=dataset,
            task_episode_indices=episode_indices,
            task_name=task_name,
            model=model,
            processor=processor,
            Image_class=Image_class,
            num_select=args.num_selected,
            b0_size=args.batch_size,
            seed=args.seed,
            camera_key=args.camera_key,
            region_ratio=args.region_ratio,
            min_regions=args.min_regions,
            max_regions=args.max_regions,
            coverage_weight=args.coverage_weight,
            visual_weight=args.visual_weight,
            gpu_id=args.gpu_id,
            pca_dim=args.pca_dim,
            cache_dir=cache_dir,
        )
        all_results.append(result)
        all_selected_episodes.extend(result["selected_episodes"])
        
        for ep_idx in result["selected_episodes"]:
            task_mapping[ep_idx] = task_name
    
    # 输出汇总
    print(f"\n{'='*60}")
    print(f"选择完成 - 汇总")
    print(f"{'='*60}")
    
    global_idx = 0
    for result in all_results:
        task_name = result["task_name"]
        n_selected = result["num_selected"]
        if n_selected > 0:
            print(f"{task_name}: episode {global_idx} ~ {global_idx + n_selected - 1} (共 {n_selected} 集)")
            global_idx += n_selected
    
    total_selected = len(all_selected_episodes)
    print(f"\n总计: {total_selected} episodes")
    
    # 保存结果
    subsets_dir = output_dir / "subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)
    
    for result in all_results:
        task_name = result["task_name"]
        subset_file = subsets_dir / f"{task_name}_selected.json"
        subset_data = {
            "task_name": task_name,
            "method": "our_v5_real",
            "num_episodes": result["num_selected"],
            "selected_episode_indices": result["selected_episodes"],
            "selection_order": result.get("selection_order", result["selected_episodes"]),
            "selection_log": result.get("selection_log", []),
            "parameters": {
                "num_selected": args.num_selected,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "camera_key": args.camera_key,
                "region_ratio": args.region_ratio,
                "min_regions": args.min_regions,
                "max_regions": args.max_regions,
                "coverage_weight": args.coverage_weight,
                "visual_weight": args.visual_weight,
                "gpu_id": args.gpu_id,
                "pca_dim": args.pca_dim,
                "vlm_model": VLM_MODEL_ID,
            }
        }
        with open(subset_file, "w") as f:
            json.dump(subset_data, f, indent=2, cls=NumpyEncoder)
        print(f"  保存: {subset_file}")
    
    # 保存合并后的episode列表
    all_selected_episodes_sorted = sorted(all_selected_episodes)
    merged_subset_file = output_dir / "subsets" / "all_tasks_selected.json"
    merged_data = {
        "method": "our_v5_real",
        "total_episodes": total_selected,
        "selected_episode_indices": all_selected_episodes_sorted,
        "selection_order": all_selected_episodes,
        "task_mapping": {str(k): v for k, v in task_mapping.items()},
        "tasks": [r["task_name"] for r in all_results],
        "parameters": {
            "num_selected_per_task": args.num_selected,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "camera_key": args.camera_key,
            "gpu_id": args.gpu_id,
            "pca_dim": args.pca_dim,
            "vlm_model": VLM_MODEL_ID,
        }
    }
    with open(merged_subset_file, "w") as f:
        json.dump(merged_data, f, indent=2, cls=NumpyEncoder)
    print(f"  保存: {merged_subset_file}")
    
    # 保存episode字符串（排序后，用于训练）
    episodes_str_sorted = ",".join(map(str, all_selected_episodes_sorted))
    episodes_file = output_dir / "episodes.txt"
    with open(episodes_file, "w") as f:
        f.write(episodes_str_sorted)
    print(f"  保存: {episodes_file}")
    
    # 保存选择顺序（不排序）
    episodes_str_order = ",".join(map(str, all_selected_episodes))
    episodes_order_file = output_dir / "episodes_selection_order.txt"
    with open(episodes_order_file, "w") as f:
        f.write(episodes_str_order)
    print(f"  保存: {episodes_order_file}")
    
    # 保存详细选择日志
    selection_log_file = output_dir / "selection_log.json"
    full_selection_log = []
    for result in all_results:
        if result.get("selection_log"):
            full_selection_log.extend(result["selection_log"])
    with open(selection_log_file, "w") as f:
        json.dump(full_selection_log, f, indent=2, cls=NumpyEncoder)
    print(f"  保存: {selection_log_file}")
    
    # 保存可视化文本日志
    log_text_file = output_dir / "selection_log.txt"
    with open(log_text_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("Our V5 Real - Episode Selection Log\n")
        f.write("="*80 + "\n\n")
        
        for result in all_results:
            task_name = result["task_name"]
            selection_log = result.get("selection_log", [])
            if not selection_log:
                continue
            
            f.write(f"\n{'='*60}\n")
            f.write(f"任务: {task_name}\n")
            f.write(f"{'='*60}\n\n")
            
            initial_batch = [log for log in selection_log if log["phase"] == "initial_batch"]
            iterative = [log for log in selection_log if log["phase"] == "iterative_selection"]
            random_fallback = [log for log in selection_log if log["phase"] == "random_fallback"]
            
            f.write(f"初始B0 (前{len(initial_batch)}个):\n")
            for log in initial_batch:
                f.write(f"  Step {log['step']:2d}: Episode {log['episode_index']:3d}\n")
            
            f.write(f"\n迭代选择 ({len(iterative)}个):\n")
            for log in iterative:
                f.write(f"  Step {log['step']:2d}: Episode {log['episode_index']:3d} "
                       f"(区域{log.get('region_id', '?')}, "
                       f"优先级={log.get('region_priority', 0):.3f}, "
                       f"VLM覆盖={log.get('vlm_coverage_score', 0):.4f})\n")
            
            if random_fallback:
                f.write(f"\n随机补充 ({len(random_fallback)}个):\n")
                for log in random_fallback:
                    f.write(f"  Step {log['step']:2d}: Episode {log['episode_index']:3d}\n")
            
            f.write(f"\n选择顺序: {[log['episode_index'] for log in selection_log]}\n")
            f.write(f"排序后: {sorted([log['episode_index'] for log in selection_log])}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"总计: {total_selected} episodes\n")
        f.write(f"{'='*80}\n")
    print(f"  保存: {log_text_file}")
    
    print(f"\n{'='*60}")
    print(f"完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()