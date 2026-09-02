"""
嵌入提取模块

加载 HuggingFaceTB/SmolVLM2-500M-Video-Instruct 模型，
对每个 episode 提取 global 嵌入（前5帧平均）和 wrist 嵌入（20%-70%区间帧平均），
进行 PCA 降维（16或32维），缓存到本地。
"""

import sys
import os
import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# 强制禁用 Python 输出缓冲，确保日志实时输出
sys.stdout.reconfigure(line_buffering=True)

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from sklearn.decomposition import PCA

# Import constants from shared config
from embedding_utils.config import MODEL_NAME, PROMPT_TEXT, GLOBAL_FRAME_RULE, WRIST_START_RATIO, WRIST_END_RATIO


def load_vlm_model(device: str = "cuda"):
    """
    加载冻结的VLM模型
    
    参数:
        device: 计算设备
    
    返回:
        冻结的VLM模型
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor
    
    print(f"加载模型: {MODEL_NAME}")
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print("处理器加载完成")
    sys.stdout.flush()
    
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    print("模型权重加载完成，正在初始化...")
    sys.stdout.flush()
    
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    print(f"模型加载完成，设备: {device}")
    sys.stdout.flush()
    return model, processor


def extract_frame_embeddings(
    model,
    processor,
    frames: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 256
) -> np.ndarray:
    """
    批量提取帧的嵌入（优化版，使用 tensor 直接输入）
    
    参数:
        model: VLM模型
        processor: 处理器
        frames: torch.Tensor, shape=(n_frames, C, H, W) 或 (n_frames, H, W, C)
        device: 计算设备
        batch_size: 批量处理大小
    
    返回:
        np.ndarray, shape=(n_frames, embedding_dim)
    """
    n_frames = len(frames)
    n_batches = (n_frames + batch_size - 1) // batch_size
    if n_batches > 1:
        print(f"    → 批量处理 {n_frames} 帧 (共{n_batches}批)...")
    else:
        print(f"    → 处理 {n_frames} 帧...")
    sys.stdout.flush()
    
    # 确保帧格式正确：如果是 numpy 数组，转为 tensor
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    
    # 确保是 (N, C, H, W) 格式
    if frames.ndim == 4 and frames.shape[-1] in [1, 3, 4]:
        # (N, H, W, C) -> (N, C, H, W)
        frames = frames.permute(0, 3, 1, 2)
    
    # 确保在 CPU 上（processor 需要 CPU tensor）
    frames = frames.cpu()
    
    embeddings = []
    
    with torch.no_grad():
        batch_times = []
        for batch_idx in range(n_batches):
            batch_start = time.time()
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_frames)
            batch_frames = frames[start_idx:end_idx]
            
            # 逐帧处理，每帧对应一个文本
            batch_embs_list = []
            for frame in batch_frames:
                # 单张图像，添加一个 <image> token
                inputs = processor(
                    text=PROMPT_TEXT,
                    images=frame,
                    return_tensors="pt"
                ).to(device)
                
                outputs = model(
                    **inputs,
                    output_hidden_states=True
                )
                
                # 提取嵌入
                hidden_states = outputs.hidden_states[-1]
                batch_embs_list.append(hidden_states.mean(dim=1).cpu().numpy())
            
            embeddings.append(np.concatenate(batch_embs_list, axis=0))
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # 显示进度
            if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
                processed = end_idx
                avg_batch_time = sum(batch_times) / len(batch_times)
                print(f"    → 批次进度: {batch_idx + 1}/{n_batches} (已处理 {processed}/{n_frames} 帧, 当前批耗时: {batch_time:.2f}s, 平均: {avg_batch_time:.2f}s)")
                sys.stdout.flush()
        
        total_batch_time = sum(batch_times)
        print(f"    ✓ 所有批次完成: 总耗时 {total_batch_time:.2f}s, 平均 {total_batch_time/len(batch_times):.2f}s/批")
        sys.stdout.flush()
    
    return np.concatenate(embeddings, axis=0).squeeze()


def extract_episode_embeddings(
    model,
    processor,
    episode_data: Dict,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取单个episode的global和wrist嵌入
    
    global嵌入: 前5帧平均
    wrist嵌入: 20%-70%区间帧平均
    
    参数:
        model: VLM模型
        processor: 处理器
        episode_data: Dict，包含 "observation.images.top" 和 "observation.images.wrist"
        device: 计算设备
    
    返回:
        (phi_global, phi_wrist) - 两个视角的嵌入
    """
    top_frames = episode_data["observation.images.top"]
    wrist_frames = episode_data["observation.images.wrist"]
    
    n_frames = len(top_frames)
    
    global_start = 0
    global_end = min(5, n_frames)
    global_frames = top_frames[global_start:global_end]
    
    wrist_start = int(n_frames * 0.2)
    wrist_end = int(n_frames * 0.7)
    wrist_frames_selected = wrist_frames[wrist_start:wrist_end]
    
    print(f"  提取global嵌入: 帧 {global_start}-{global_end}")
    global_embs = extract_frame_embeddings(model, processor, global_frames, device)
    phi_global = global_embs.mean(axis=0)
    
    print(f"  提取wrist嵌入: 帧 {wrist_start}-{wrist_end}")
    wrist_embs = extract_frame_embeddings(model, processor, wrist_frames_selected, device)
    phi_wrist = wrist_embs.mean(axis=0)
    
    return phi_global, phi_wrist


def fit_pca(
    embeddings_list: np.ndarray,
    n_components: int = 32
) -> PCA:
    """
    拟合PCA降维
    
    参数:
        embeddings_list: shape=(n_samples, embedding_dim)
        n_components: 目标维度
    
    返回:
        拟合好的PCA对象
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(embeddings_list)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  PCA降维到 {n_components} 维，解释方差: {explained_var:.4f}")
    
    return pca


def process_dataset(
    dataset_dir: Path,
    output_dir: Path,
    n_components: int = 32,
    device: str = "cuda",
    allow_partial_cache: bool = False,
) -> Dict:
    """
    处理整个数据集，提取嵌入并缓存
    
    参数:
        dataset_dir: 数据集目录
        output_dir: 输出目录
        n_components: PCA降维维度
        device: 计算设备
        allow_partial_cache: 如果为False（默认），检测到部分缓存时报错；
                           如果为True，允许跳过已有缓存（仅用于旧CLI兼容）
    
    返回:
        包含处理信息的字典，包括episode_indices、pca_global、pca_wrist等
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    process_start_time = time.time()
    print(f"\n处理数据集: {dataset_dir}")
    
    print("加载 LeRobotDataset...")
    dataset = LeRobotDataset(
        repo_id="work2/metaworld_pick_place",
        root=str(dataset_dir)
    )
    print(f"数据集加载完成，共 {dataset.num_episodes} 个 episodes，{dataset.num_frames} 帧")
    
    # 获取真实episode数量
    num_episodes = dataset.num_episodes
    
    # 检查已存在的缓存文件
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_files = {f.stem for f in output_dir.glob("*.npy")}
    existing_count = len(existing_files)
    
    if existing_count > 0 and existing_count < num_episodes:
        # 部分缓存：不允许混合新旧PCA
        if not allow_partial_cache:
            print(f"\n错误: 检测到部分缓存 ({existing_count}/{num_episodes} episodes)")
            print(f"  不允许在缺失episode上拟合新PCA然后与旧embedding混合")
            print(f"  请使用全新output-dir或使用shared ensure流程")
            print(f"  当前output-dir: {output_dir}")
            sys.exit(1)
        else:
            print(f"\n发现 {existing_count}/{num_episodes} 个已存在的缓存文件 (跳过模式)")
    elif existing_count >= num_episodes:
        # 检查PCA模型是否也完整
        pca_dir = output_dir / "pca_models"
        pca_global_file = pca_dir / f"pca_global_{n_components}.joblib"
        pca_wrist_file = pca_dir / f"pca_wrist_{n_components}.joblib"
        if pca_global_file.exists() and pca_wrist_file.exists():
            print(f"\n缓存已完整 ({existing_count} episodes + PCA models)，跳过提取")
            return {
                "episode_indices": list(range(num_episodes)),
                "num_episodes": num_episodes,
                "output_dir": str(output_dir),
                "skipped": True,
            }
        else:
            print(f"\n错误: 检测到完整episode缓存但PCA模型缺失")
            print(f"  请使用全新output-dir或使用shared ensure流程")
            sys.exit(1)
    else:
        print(f"\n未发现缓存文件，将提取全部 {num_episodes} 个 episodes")
    
    print("\n加载 VLM 模型...")
    model, processor = load_vlm_model(device)
    print("模型加载完成，开始提取嵌入...\n")
    
    episode_embeddings = {}
    global_embs_list = []
    wrist_embs_list = []
    episode_coords = []
    
    skipped_count = existing_count if allow_partial_cache else 0
    
    # 预构建 episode 索引映射，使用 meta.episodes 的 dataset_from_index/to_index
    print("\n构建 episode 帧索引映射...")
    sys.stdout.flush()
    ep_indices_map = {}
    for ep_idx in range(dataset.num_episodes):
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]
        ep_indices_map[ep_idx] = list(range(from_idx, to_idx))
    print(f"✓ 索引映射构建完成，共 {len(ep_indices_map)} 个 episodes")
    sys.stdout.flush()
    
    for ep_idx in range(dataset.num_episodes):
        ep_start_time = time.time()
        
        # 检查是否已存在缓存
        coord_key = f"({ep_idx})"
        if coord_key in existing_files and allow_partial_cache:
            print(f"\n跳过 episode {ep_idx}：已存在缓存")
            sys.stdout.flush()
            skipped_count += 1
            continue
        
        print(f"\n处理 episode {ep_idx + 1}/{dataset.num_episodes} (索引 {ep_idx})")
        sys.stdout.flush()
        
        # 直接从映射获取帧索引
        ep_load_start = time.time()
        print(f"  [1/3] 正在收集 episode 帧数据...")
        sys.stdout.flush()
        
        episode_indices = ep_indices_map.get(ep_idx, [])
        
        if not episode_indices:
            print(f"  跳过 episode {ep_idx}：没有有效的帧")
            continue
        
        print(f"  找到 {len(episode_indices)} 帧 (索引 {episode_indices[0]}-{episode_indices[-1]})，正在从数据集加载...")
        sys.stdout.flush()
        
        # 批量加载帧（保持 tensor 格式）
        episode_frames_top = []
        episode_frames_wrist = []
        for idx in episode_indices:
            frame = dataset[idx]
            episode_frames_top.append(frame["observation.images.top"])
            episode_frames_wrist.append(frame["observation.images.wrist"])
        
        episode_frames_top = torch.stack(episode_frames_top)
        episode_frames_wrist = torch.stack(episode_frames_wrist)
        
        load_time = time.time() - ep_load_start
        print(f"  ✓ 帧加载完成: top={episode_frames_top.shape}, wrist={episode_frames_wrist.shape}, 耗时: {load_time:.2f}s")
        sys.stdout.flush()
        
        # 计算帧范围
        n_frames = len(episode_frames_top)
        global_start = 0
        global_end = min(int(GLOBAL_FRAME_RULE.replace("first", "")), n_frames) if GLOBAL_FRAME_RULE.startswith("first") else min(5, n_frames)
        wrist_start = int(n_frames * WRIST_START_RATIO)
        wrist_end = int(n_frames * WRIST_END_RATIO)
        wrist_frames_selected = episode_frames_wrist[wrist_start:wrist_end]
        
        episode_data = {
            "observation.images.top": episode_frames_top,
            "observation.images.wrist": episode_frames_wrist
        }
        
        print(f"  [2/3] 开始提取 VLM 嵌入...")
        sys.stdout.flush()
        
        extract_start = time.time()
        
        # 提取 global 嵌入
        global_start_time = time.time()
        print(f"    → 提取 global 嵌入 (前{global_end}帧)...")
        sys.stdout.flush()
        global_embs = extract_frame_embeddings(model, processor, episode_frames_top[:global_end], device)
        phi_global = global_embs.mean(axis=0)
        global_time = time.time() - global_start_time
        print(f"    ✓ Global 嵌入完成: {global_time:.2f}s, 维度={phi_global.shape}")
        sys.stdout.flush()
        
        # 提取 wrist 嵌入
        wrist_start_time = time.time()
        print(f"    → 提取 wrist 嵌入 (帧 {wrist_start}-{wrist_end}, 共{len(wrist_frames_selected)}帧)...")
        sys.stdout.flush()
        wrist_embs = extract_frame_embeddings(model, processor, wrist_frames_selected, device)
        phi_wrist = wrist_embs.mean(axis=0)
        wrist_time = time.time() - wrist_start_time
        print(f"    ✓ Wrist 嵌入完成: {wrist_time:.2f}s, 维度={phi_wrist.shape}")
        sys.stdout.flush()
        
        extract_time = time.time() - extract_start
        ep_total_time = time.time() - ep_start_time
        
        episode_embeddings[ep_idx] = {
            "phi_global": phi_global,
            "phi_wrist": phi_wrist
        }
        
        global_embs_list.append(phi_global)
        wrist_embs_list.append(phi_wrist)
        episode_coords.append(ep_idx)
        
        completed = len(episode_coords)
        progress = completed / dataset.num_episodes * 100
        avg_time = ep_total_time / completed if completed > 0 else 0
        eta = avg_time * (dataset.num_episodes - completed)
        
        print(f"  ✓ Episode {ep_idx} 完成！")
        print(f"    嵌入维度: global={phi_global.shape}, wrist={phi_wrist.shape}")
        print(f"    耗时: 加载={load_time:.1f}s, 提取={extract_time:.1f}s, 总计={ep_total_time:.1f}s")
        print(f"    进度: {completed}/{dataset.num_episodes} ({progress:.1f}%), ETA: {eta/60:.1f}分钟")
        sys.stdout.flush()
    
    if not global_embs_list:
        print("没有找到有效的episode数据")
        return {
            "episode_indices": [],
            "num_episodes": 0,
            "output_dir": str(output_dir),
            "skipped": False,
        }
    
    print(f"\n{'='*60}")
    print(f"所有 episode 嵌入提取完成！")
    print(f"共处理 {len(global_embs_list)} 个 episodes")
    print(f"{'='*60}")
    
    total_extract_time = time.time() - process_start_time
    print(f"总提取时间: {total_extract_time:.2f}s ({total_extract_time/60:.1f} 分钟)")
    if len(global_embs_list) > 0:
        print(f"平均每个 episode: {total_extract_time/len(global_embs_list):.2f}s")
    print(f"跳过已缓存: {skipped_count} 个 episodes")
    sys.stdout.flush()
    
    global_embs_array = np.array(global_embs_list)
    wrist_embs_array = np.array(wrist_embs_list)
    
    print(f"\nGlobal 嵌入数组形状: {global_embs_array.shape}")
    print(f"Wrist 嵌入数组形状: {wrist_embs_array.shape}")
    
    print(f"\n拟合PCA (global, {n_components}维)...")
    pca_global = fit_pca(global_embs_array, n_components)
    print("PCA (global) 拟合完成")
    
    print(f"\n拟合PCA (wrist, {n_components}维)...")
    pca_wrist = fit_pca(wrist_embs_array, n_components)
    print("PCA (wrist) 拟合完成")
    
    print(f"\n保存嵌入到: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for ep_idx, coord in enumerate(episode_coords):
        phi_global_pca = pca_global.transform(global_embs_array[ep_idx:ep_idx+1])[0]
        phi_wrist_pca = pca_wrist.transform(wrist_embs_array[ep_idx:ep_idx+1])[0]
        
        coord_key = f"({coord})"
        output_file = output_dir / f"{coord_key}.npy"
        
        np.save(output_file, {
            "phi_global": phi_global_pca,
            "phi_wrist": phi_wrist_pca,
            "episode_index": ep_idx
        }, allow_pickle=True)
        
        if (ep_idx + 1) % 10 == 0 or ep_idx == len(episode_coords) - 1:
            print(f"  已保存 {ep_idx + 1}/{len(episode_coords)} 个嵌入文件")
    
    print(f"\n保存 PCA 模型...")
    pca_output_dir = output_dir / "pca_models"
    pca_output_dir.mkdir(parents=True, exist_ok=True)
    
    import joblib
    joblib.dump(pca_global, pca_output_dir / f"pca_global_{n_components}.joblib")
    joblib.dump(pca_wrist, pca_output_dir / f"pca_wrist_{n_components}.joblib")
    
    print(f"\n{'='*60}")
    print(f"嵌入提取全部完成！")
    print(f"嵌入文件保存到: {output_dir}")
    print(f"PCA 模型保存到: {pca_output_dir}")
    print(f"{'='*60}")
    
    return {
        "episode_indices": episode_coords,
        "num_episodes": len(episode_coords),
        "output_dir": str(output_dir),
        "pca_global": pca_global,
        "pca_wrist": pca_wrist,
        "skipped": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取VLM嵌入")
    parser.add_argument("--dataset-dir", type=str, required=True,
                       help="数据集目录路径")
    parser.add_argument("--output-dir", type=str, 
                       default="personal/work2/our/embeddings/cache",
                       help="嵌入缓存输出目录")
    parser.add_argument("--n-components", type=int, default=32,
                       help="PCA降维目标维度 (16或32)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="批量处理大小 (默认128，显存充足可增大到256)")
    parser.add_argument("--metadata-output", type=str, default=None,
                       help="可选：输出metadata.json路径供ensure_embeddings.py使用")
    parser.add_argument("--allow-partial-cache", action="store_true",
                       help="允许跳过已有缓存（仅用于旧CLI兼容）")
    args = parser.parse_args()
    
    result = process_dataset(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        n_components=args.n_components,
        device=args.device,
        allow_partial_cache=args.allow_partial_cache,
    )
    
    if args.metadata_output and not result.get("skipped", False):
        from embedding_utils.config import build_expected_metadata
        meta = build_expected_metadata(
            dataset_root=str(args.dataset_dir),
            dataset_name=Path(args.dataset_dir).name.replace("pick_place_", ""),
            pca_dim=args.n_components,
            episode_indices=result["episode_indices"],
            source="cli_extract",
        )
        meta_path = Path(args.metadata_output)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata written to: {meta_path}")