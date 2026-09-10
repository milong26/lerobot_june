#!/usr/bin/env python3
"""
验证 MiniVLA checkpoint 加载后，模型推理的 action 是否与训练数据接近。

逻辑：
1. 从训练数据集中加载一个 episode 的 sample
2. 使用训练时的预处理流程处理 observation
3. 用加载的 checkpoint 推理 action
4. 比较推理 action 和数据集中的真实 action
5. 如果 loss=0.05，推理 action 应该和真实 action 很接近

用法：
python verify_minivla_action_accuracy.py \
    --checkpoint_path personal/work2/differentvlm/minivla/experiments/random_200_seed42_disassemblev3corner_minivla_random/checkpoints/checkpoints/020000/pretrained_model \
    --dataset_root personal/work2/differentvlm/datasets/disassemble-v3 \
    --episode_index 0 \
    --device cuda
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedConfig
from lerobot.utils.constants import ACTION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
    force=True,  # 强制重新配置，覆盖已有的配置
)
logger = logging.getLogger(__name__)

# 确保 logger 级别是 INFO
logger.setLevel(logging.INFO)


def load_dataset_sample(dataset_root: str, episode_index: int, frame_index: int = None):
    """从数据集中加载一个 sample（使用 LeRobot 标准方式）"""
    repo_id = "work2/disassemble-v3_corner"
    ds = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=[episode_index],
    )
    
    logger.info(f"数据集加载: {ds.num_frames} frames, {ds.num_episodes} episodes")
    logger.info(f"Features: {list(ds.features.keys())}")
    logger.info(f"Action feature: {ds.features.get(ACTION, 'NOT FOUND')}")
    
    # 如果没有指定 frame_index，使用 episode 中间的帧
    if frame_index is None:
        from_idx = ds.meta.episodes["dataset_from_index"][0]
        to_idx = ds.meta.episodes["dataset_to_index"][0]
        frame_index = (from_idx + to_idx) // 2
    
    sample = ds[frame_index]
    logger.info(f"加载 frame index {frame_index} from episode {episode_index}")
    logger.info(f"Sample keys: {list(sample.keys())}")
    
    if ACTION in sample:
        logger.info(f"Action shape: {sample[ACTION].shape}")
        logger.info(f"Action values: {sample[ACTION]}")
    
    return sample, ds


def load_policy(checkpoint_path: str, device: str):
    """加载 MiniVLA policy"""
    logger.info(f"\n加载 MiniVLA checkpoint: {checkpoint_path}")
    
    # 加载配置
    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    config.device = device
    
    # 加载 policy
    policy_cls = get_policy_class("minivla")
    policy = policy_cls.from_pretrained(checkpoint_path)
    
    # 统一使用 float32
    policy = policy.float()
    policy.to(device)
    policy.eval()
    
    logger.info(f"Policy 加载完成: {type(policy).__name__}")
    logger.info(f"Model dtype: float32")
    
    # 加载 pre/post processors
    config.pretrained_path = Path(checkpoint_path)
    if not hasattr(config, "type"):
        config.type = "minivla"
    
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint_path,
        dataset_stats=None,  # 不使用额外的 dataset stats
    )
    
    logger.info(f"Preprocessor: {type(preprocessor).__name__}, steps={len(preprocessor.steps)}")
    logger.info(f"Postprocessor: {type(postprocessor).__name__}, steps={len(postprocessor.steps)}")
    
    # 详细检查 postprocessor 中的每个 step
    for i, step in enumerate(postprocessor.steps):
        logger.info(f"  Postprocessor step {i}: {type(step).__name__}")
        if hasattr(step, '_tensor_stats'):
            if step._tensor_stats:
                logger.info(f"    _tensor_stats keys: {list(step._tensor_stats.keys())}")
                if 'action' in step._tensor_stats:
                    action_stats = step._tensor_stats['action']
                    logger.info(f"    action stats keys: {list(action_stats.keys())}")
                    if 'q01' in action_stats:
                        q01 = action_stats['q01']
                        if hasattr(q01, 'cpu'):
                            q01 = q01.cpu().numpy()
                        logger.info(f"    action q01: {q01}")
                    if 'q99' in action_stats:
                        q99 = action_stats['q99']
                        if hasattr(q99, 'cpu'):
                            q99 = q99.cpu().numpy()
                        logger.info(f"    action q99: {q99}")
            else:
                logger.info(f"    _tensor_stats: EMPTY (not loaded)")
        else:
            logger.info(f"    (no _tensor_stats attribute)")
    
    return policy, preprocessor, postprocessor, config


def patch_minivla_bfloat16(policy):
    """修复 MiniVLA VQ action tokenizer 的 bfloat16 兼容性问题"""
    try:
        core = policy.model
        if hasattr(core, "action_tokenizer"):
            action_tokenizer = core.action_tokenizer
            if hasattr(action_tokenizer, "vq_vae"):
                original_get_action = action_tokenizer.vq_vae.get_action_from_latent

                def patched_get_action(latent):
                    result = original_get_action(latent)
                    if result.dtype == torch.bfloat16:
                        return result.float()
                    return result

                action_tokenizer.vq_vae.get_action_from_latent = patched_get_action
                logger.info("已修复 MiniVLA vq_vae.get_action_from_latent 的 bfloat16 兼容性")
    except Exception as e:
        logger.warning(f"修复 MiniVLA action tokenizer 失败: {e}")


def tensor_stats(tensor, name="tensor"):
    """打印 tensor 的统计信息"""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    
    tensor = tensor.float()
    stats = {
        "name": name,
        "shape": tuple(tensor.shape),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
    }
    
    logger.info(f"[{name}] shape={stats['shape']}, min={stats['min']:.6f}, "
                f"max={stats['max']:.6f}, mean={stats['mean']:.6f}, std={stats['std']:.6f}")
    
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--frame_index", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info("=" * 80)
    logger.info("MiniVLA Action Accuracy Verification")
    logger.info("=" * 80)
    
    # 1. 加载数据集 sample
    logger.info("\n[Step 1] 加载数据集 sample...")
    sample, dataset = load_dataset_sample(
        args.dataset_root,
        args.episode_index,
        args.frame_index,
    )
    
    # 获取真实 action
    if ACTION not in sample:
        logger.error(f"Sample 中没有 action 字段!")
        sys.exit(1)
    
    real_action = sample[ACTION]
    if isinstance(real_action, torch.Tensor):
        real_action = real_action.cpu().numpy()
    
    logger.info(f"真实 action shape: {real_action.shape}")
    logger.info(f"真实 action values: {real_action}")
    
    # 2. 加载 policy 和 processors
    logger.info("\n[Step 2] 加载 Policy 和 Processors...")
    policy, preprocessor, postprocessor, config = load_policy(
        args.checkpoint_path,
        args.device,
    )
    
    # 修复 bfloat16 问题
    patch_minivla_bfloat16(policy)
    
    # 检查是否是 VQ 模式
    is_vq = config.action_tokenizer_type and "vq" in config.action_tokenizer_type.lower()
    logger.info(f"VQ mode: {is_vq}")
    logger.info(f"Action tokenizer: {config.action_tokenizer_type}")
    
    # 获取 action shape（PolicyFeature 对象）
    action_feature = config.output_features.get(ACTION)
    if action_feature is not None:
        if hasattr(action_feature, 'shape'):
            action_shape = action_feature.shape
        elif isinstance(action_feature, dict):
            action_shape = action_feature.get('shape')
        else:
            action_shape = None
        logger.info(f"Action shape: {action_shape}")
    else:
        logger.warning("Action feature not found in output_features")
    
    # 3. 预处理 observation
    logger.info("\n[Step 3] 预处理 observation...")
    
    # 构建 batch（添加 batch 维度）
    batch = {}
    for key, value in sample.items():
        if key in ["observation.images.top", "observation.images.wrist", "observation.state"]:
            if isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0).float().to(args.device)
            else:
                batch[key] = torch.from_numpy(value).unsqueeze(0).float().to(args.device)
    
    # 添加 task
    if "task" in sample:
        batch["task"] = [sample["task"]] if isinstance(sample["task"], str) else [""]
    else:
        batch["task"] = [""]
    
    # 添加真实 action（用于对比）
    batch[ACTION] = torch.from_numpy(real_action).unsqueeze(0).unsqueeze(0).float().to(args.device)
    
    logger.info(f"Batch keys: {list(batch.keys())}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # 应用 preprocessor
    preprocessed = preprocessor(batch)
    logger.info(f"Preprocessed batch keys: {list(preprocessed.keys())}")
    for key, value in preprocessed.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # 4. 模型推理（原始输出，未经 postprocessor）
    logger.info("\n[Step 4] 模型推理...")
    policy.reset()
    
    with torch.no_grad():
        # 使用 predict_action_chunk 获取模型输出
        raw_pred_action = policy.predict_action_chunk(preprocessed)
    
    logger.info(f"模型原始输出 shape: {raw_pred_action.shape}")
    logger.info(f"模型原始输出 dtype: {raw_pred_action.dtype}")
    raw_pred_stats = tensor_stats(raw_pred_action, "model_raw_prediction")
    
    # 5. 应用 postprocessor（反归一化）
    logger.info("\n[Step 5] 应用 postprocessor（反归一化）...")
    final_action_obj = postprocessor(raw_pred_action)
    
    if isinstance(final_action_obj, torch.Tensor):
        final_pred_action = final_action_obj
    elif isinstance(final_action_obj, dict) and "action" in final_action_obj:
        final_pred_action = final_action_obj["action"]
    else:
        final_pred_action = raw_pred_action
    
    if isinstance(final_pred_action, torch.Tensor):
        final_pred_stats = tensor_stats(final_pred_action, "final_prediction_after_postprocess")
    else:
        final_pred_stats = tensor_stats(torch.from_numpy(final_pred_action), "final_prediction_after_postprocess")
    
    # 6. 对比真实 action 和预测 action
    logger.info("\n[Step 6] 对比真实 action 和预测 action...")
    
    # 确保形状一致
    pred_np = final_pred_action.cpu().numpy() if isinstance(final_pred_action, torch.Tensor) else final_pred_action
    real_np = real_action
    
    # 处理形状差异 - 提取第一步 (index 0)
    if pred_np.ndim == 3:
        pred_np_first_step = pred_np[0, 0, :]  # [B, T, A] -> [A] (first step)
        pred_np_full = pred_np  # 保留完整 chunk 用于日志
    elif pred_np.ndim == 2:
        pred_np_first_step = pred_np[0, :]  # [B, A] -> [A]
        pred_np_full = pred_np
    else:
        pred_np_first_step = pred_np
        pred_np_full = pred_np
    
    if real_np.ndim == 2:
        real_np = real_np[0, :]  # [1, A] -> [A]
    
    logger.info(f"预测 action chunk shape: {pred_np.shape}")
    logger.info(f"预测 action 第一步 (反归一化后): {pred_np_first_step}")
    logger.info(f"真实 action (原始值): {real_np}")
    
    # 计算误差（反归一化后的对比，使用第一步）
    abs_error = np.abs(pred_np_first_step - real_np)
    max_abs_error = float(abs_error.max())
    mean_abs_error = float(abs_error.mean())
    
    logger.info(f"\n误差分析 (反归一化后第一步 vs 原始真实 action):")
    logger.info(f"  每维绝对误差: {abs_error}")
    logger.info(f"  最大绝对误差: {max_abs_error:.6f}")
    logger.info(f"  平均绝对误差: {mean_abs_error:.6f}")
    
    # 额外检查：将真实 action 归一化后与模型原始输出对比
    logger.info("\n[Step 6b] 将真实 action 归一化后与模型原始输出对比...")
    
    # 获取归一化统计信息
    unnormalizer_step = None
    for step in postprocessor.steps:
        if hasattr(step, '_tensor_stats') and 'action' in step._tensor_stats:
            unnormalizer_step = step
            break
    
    if unnormalizer_step is not None:
        action_stats = unnormalizer_step._tensor_stats['action']
        q01 = action_stats.get('q01')
        q99 = action_stats.get('q99')
        
        if q01 is not None and q99 is not None:
            # 将 numpy 或 tensor 转换为 numpy
            if hasattr(q01, 'cpu'):
                q01 = q01.cpu().numpy()
            if hasattr(q99, 'cpu'):
                q99 = q99.cpu().numpy()
            if isinstance(q01, torch.Tensor):
                q01 = q01.numpy()
            if isinstance(q99, torch.Tensor):
                q99 = q99.numpy()
            
            q01 = q01.flatten()
            q99 = q99.flatten()
            
            logger.info(f"归一化统计: q01={q01}, q99={q99}")
            
            # 将真实 action 归一化到 [-1, 1] 范围
            real_normalized = 2 * (real_np - q01) / (q99 - q01) - 1
            
            logger.info(f"真实 action (归一化后): {real_normalized}")
            
            # 对比归一化后的值 - 使用模型原始输出的第一步
            raw_pred_np = raw_pred_action.cpu().numpy()
            if raw_pred_np.ndim == 3:
                raw_pred_first_step = raw_pred_np[0, 0, :]  # [B, T, A] -> [A]
            elif raw_pred_np.ndim == 2:
                raw_pred_first_step = raw_pred_np[0, :]
            else:
                raw_pred_first_step = raw_pred_np
            
            logger.info(f"模型原始输出第一步: {raw_pred_first_step}")
            logger.info(f"模型原始输出整个 chunk mean: {raw_pred_stats['mean']:.6f}")
            
            abs_error_normalized = np.abs(raw_pred_first_step - real_normalized)
            max_abs_error_normalized = float(abs_error_normalized.max())
            mean_abs_error_normalized = float(abs_error_normalized.mean())
            
            logger.info(f"\n误差分析 (归一化后对比，第一步):")
            logger.info(f"  每维绝对误差: {abs_error_normalized}")
            logger.info(f"  最大绝对误差: {max_abs_error_normalized:.6f}")
            logger.info(f"  平均绝对误差: {mean_abs_error_normalized:.6f}")
            
            # 使用归一化后的误差作为主要判断标准
            mean_abs_error = mean_abs_error_normalized
            max_abs_error = max_abs_error_normalized
        else:
            logger.warning("未找到 q01/q99 统计信息，使用原始对比")
    else:
        logger.warning("未找到 unnormalizer step，使用原始对比")
    
    # 7. 判断是否合理
    logger.info("\n[Step 7] 结果判断...")
    
    # 如果 loss=0.05，平均绝对误差应该比较小（< 0.2）
    # 但考虑到这是 VQ 模式，会有一定的量化误差
    if mean_abs_error < 0.2:
        logger.info("✅ 模型加载正确！推理 action 与训练数据接近")
        logger.info(f"   平均绝对误差 {mean_abs_error:.4f} < 0.2，符合预期")
    elif mean_abs_error < 0.5:
        logger.warning("⚠️  模型可能加载有问题，或者后处理流程有问题")
        logger.warning(f"   平均绝对误差 {mean_abs_error:.4f} 在 0.2-0.5 之间，需要检查")
    else:
        logger.error("❌ 模型加载或推理流程有严重问题！")
        logger.error(f"   平均绝对误差 {mean_abs_error:.4f} > 0.5，远超预期")
    
    # 8. 输出详细诊断信息
    logger.info("\n" + "=" * 80)
    logger.info("诊断摘要:")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Episode: {args.episode_index}, Frame: {args.frame_index}")
    logger.info(f"VQ Mode: {is_vq}")
    logger.info(f"Action Tokenizer: {config.action_tokenizer_type}")
    logger.info(f"Real Action: {real_np}")
    logger.info(f"Predicted Action (raw): {raw_pred_stats['mean']:.4f} (mean)")
    logger.info(f"Predicted Action (final): {final_pred_stats['mean']:.4f} (mean)")
    logger.info(f"Max Absolute Error: {max_abs_error:.6f}")
    logger.info(f"Mean Absolute Error: {mean_abs_error:.6f}")
    
    # 保存结果
    output_dir = Path(args.checkpoint_path).parent / "verification_results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"action_accuracy_ep{args.episode_index}_frame{args.frame_index}.json"
    
    results = {
        "checkpoint_path": args.checkpoint_path,
        "episode_index": args.episode_index,
        "frame_index": args.frame_index,
        "is_vq": is_vq,
        "action_tokenizer": config.action_tokenizer_type,
        "real_action": real_np.tolist(),
        "predicted_action_raw": {
            "shape": raw_pred_stats["shape"],
            "min": raw_pred_stats["min"],
            "max": raw_pred_stats["max"],
            "mean": raw_pred_stats["mean"],
            "std": raw_pred_stats["std"],
        },
        "predicted_action_final": {
            "shape": final_pred_stats["shape"],
            "min": final_pred_stats["min"],
            "max": final_pred_stats["max"],
            "mean": final_pred_stats["mean"],
            "std": final_pred_stats["std"],
        },
        "error": {
            "abs_error_per_dim": abs_error.tolist(),
            "max_absolute_error": max_abs_error,
            "mean_absolute_error": mean_abs_error,
        },
        "verdict": "PASS" if mean_abs_error < 0.2 else ("WARN" if mean_abs_error < 0.5 else "FAIL"),
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n结果已保存到: {output_file}")
    
    return 0 if mean_abs_error < 0.2 else 1


if __name__ == "__main__":
    sys.exit(main())