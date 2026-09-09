#!/usr/bin/env python3
"""
TinyVLA Diffusion Action Head 训练-推理对称性诊断工具

核心问题:
  训练数据 normalized action: mean~0.90, std~2.20
  模型输出 normalized action:  mean~0.08, std~0.71

  模型预测严重偏向零，无法输出有效控制动作。

诊断目标:
  1. 训练时 action 进入 diffusion head 前的处理流程
  2. 推理时 diffusion sampling 后的处理流程
  3. 两条链是否完全对称
  4. diffusion loss 计算是否正确
  5. scheduler 配置训练/推理是否一致
  6. action 是否经过 clip/tanh/scale
  7. prediction target 是 epsilon 还是 velocity

设计原则:
  - 尽量使用已有 API (policy.forward, policy.predict_action_chunk, preprocessor, postprocessor)
  - 不自己实现 diffusion 逻辑，只通过已有 API 调用并观察输入输出
  - 通过 hook/monkey-patch 在关键位置插入日志

Usage:
  python verify_diffusion_action_head_symmetry.py \
    --policy_type tinyvla_s \
    --checkpoint_path /path/to/checkpoint \
    --dataset_root /path/to/dataset \
    --gpu_id 0
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from functools import wraps

import numpy as np
import torch
import torch.nn.functional as F

LEROBOT_ROOT = "/data/zhonglinye/jun/lerobot"
if LEROBOT_ROOT not in sys.path:
    sys.path.insert(0, LEROBOT_ROOT + "/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedConfig, PreTrainedPolicy
from lerobot.policies.factory import get_policy_class
from lerobot.policies import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_IMAGE

OUTPUT_FILE = str(Path(__file__).parent / "diffusion_head_symmetry_results.json")


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("diffusion_diag")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


def print_sep(title: str, logger: logging.Logger) -> None:
    logger.info(f"\n{'='*70}")
    logger.info(f"  {title}")
    logger.info(f"{'='*70}")


def tensor_info(x, name="tensor"):
    if isinstance(x, torch.Tensor):
        return {
            "name": name,
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "std": float(x.std()),
        }
    return {"name": name, "value": str(x)}


def load_policy(policy_type: str, checkpoint_path: str, device: str,
                dataset_stats: dict | None = None, logger: logging.Logger | None = None):
    log = logger or logging.getLogger("diffusion_diag")
    log.info(f"Loading {policy_type} from {checkpoint_path}")

    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    config.device = device
    config.pretrained_path = Path(checkpoint_path)
    if not hasattr(config, "type"):
        config.type = policy_type

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint_path,
        dataset_stats=dataset_stats,
    )

    policy_cls = get_policy_class(policy_type)
    policy = PreTrainedPolicy.from_pretrained.__func__(policy_cls, checkpoint_path)

    model_dtype = getattr(config, "dtype", "float32")
    if model_dtype == "bfloat16":
        policy = policy.to(dtype=torch.bfloat16)
    elif model_dtype == "float16":
        policy = policy.to(dtype=torch.float16)

    policy.to(device)
    policy.eval()

    log.info(f"Policy loaded: {type(policy).__name__}, dtype={model_dtype}")
    return policy, preprocessor, postprocessor, config


# ============================================================================
# Hook utilities to trace internal calls
# ============================================================================

class CallTracer:
    """Trace calls to specific methods and log input/output shapes and stats."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.calls = []

    def hook_method(self, obj, method_name, label=None):
        """Wrap a method or register forward hook on an nn.Module."""
        import torch.nn as nn

        target = getattr(obj, method_name, None)
        label = label or f"{type(obj).__name__}.{method_name}"

        # If target is an nn.Module, use forward hook instead of monkey-patching
        if isinstance(target, nn.Module):
            def forward_hook(module, args, kwargs, output):
                self.logger.info(f"\n[HOOK] {label} called")
                for i, arg in enumerate(args):
                    if isinstance(arg, torch.Tensor):
                        self.logger.info(f"  arg[{i}] shape={arg.shape}, dtype={arg.dtype}, "
                                         f"mean={arg.float().mean():.4f}, std={arg.float().std():.4f}")
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        self.logger.info(f"  kwarg[{k}] shape={v.shape}, dtype={v.dtype}, "
                                         f"mean={v.float().mean():.4f}, std={v.float().std():.4f}")
                if isinstance(output, torch.Tensor):
                    self.logger.info(f"  => output shape={output.shape}, dtype={output.dtype}, "
                                     f"mean={output.float().mean():.4f}, std={output.float().std():.4f}")
                elif isinstance(output, dict):
                    for k, v in output.items():
                        if isinstance(v, torch.Tensor):
                            self.logger.info(f"  => output[{k}] shape={v.shape}, dtype={v.dtype}, "
                                             f"mean={v.float().mean():.4f}, std={v.float().std():.4f}")
                elif isinstance(output, tuple):
                    for i, item in enumerate(output):
                        if isinstance(item, torch.Tensor):
                            self.logger.info(f"  => output[{i}] shape={item.shape}, dtype={item.dtype}, "
                                             f"mean={item.float().mean():.4f}, std={item.float().std():.4f}")
                self.calls.append({"label": label, "result": output})
            target.register_forward_hook(forward_hook, with_kwargs=True)
            return target

        # For regular methods, use monkey-patching
        original = target
        @wraps(original)
        def wrapped(*args, **kwargs):
            self.logger.info(f"\n[HOOK] {label} called")

            # Log inputs
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    self.logger.info(f"  arg[{i}] shape={arg.shape}, dtype={arg.dtype}, "
                                     f"mean={arg.float().mean():.4f}, std={arg.float().std():.4f}")
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    self.logger.info(f"  kwarg[{k}] shape={v.shape}, dtype={v.dtype}, "
                                     f"mean={v.float().mean():.4f}, std={v.float().std():.4f}")

            result = original(*args, **kwargs)

            # Log outputs
            if isinstance(result, torch.Tensor):
                self.logger.info(f"  => output shape={result.shape}, dtype={result.dtype}, "
                                 f"mean={result.float().mean():.4f}, std={result.float().std():.4f}")
            elif isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, torch.Tensor):
                        self.logger.info(f"  => output[{k}] shape={v.shape}, dtype={v.dtype}, "
                                         f"mean={v.float().mean():.4f}, std={v.float().std():.4f}")
            elif isinstance(result, tuple):
                for i, item in enumerate(result):
                    if isinstance(item, torch.Tensor):
                        self.logger.info(f"  => output[{i}] shape={item.shape}, dtype={item.dtype}, "
                                         f"mean={item.float().mean():.4f}, std={item.float().std():.4f}")

            self.calls.append({"label": label, "result": result})
            return result

        setattr(obj, method_name, wrapped)
        return original


# ============================================================================
# 1. SCHEDULER CONFIGURATION CONSISTENCY
# ============================================================================

def check_scheduler_consistency(policy, config, dataset_stats: dict | None = None,
                                 logger: logging.Logger | None = None) -> dict:
    """检查训练和推理的 scheduler 配置是否一致

    关键检查:
    - noise_scheduler 的 num_train_timesteps
    - beta_schedule
    - prediction_type (epsilon vs velocity)
    - clip_sample
    - inference 时 set_timesteps() 是否正确
    """
    log = logger
    print_sep("1. SCHEDULER CONFIGURATION CONSISTENCY", log)

    results = {}
    issues = []

    base_model = policy._get_base_model()

    # 触发 lazy init
    if hasattr(base_model, '_diffusion_initialized') and not base_model._diffusion_initialized:
        base_model._init_diffusion_head()

    if not hasattr(base_model, 'noise_scheduler'):
        log.info("  ERROR: noise_scheduler not found")
        results["error"] = "noise_scheduler not found"
        return results

    scheduler = base_model.noise_scheduler

    log.info(f"[Scheduler Type]")
    log.info(f"  Class: {type(scheduler).__name__}")
    results["scheduler_type"] = type(scheduler).__name__

    log.info(f"\n[Scheduler Config]")
    cfg = scheduler.config
    log.info(f"  num_train_timesteps: {cfg.num_train_timesteps}")
    log.info(f"  beta_schedule:       {cfg.beta_schedule}")
    log.info(f"  prediction_type:     {cfg.prediction_type}")
    log.info(f"  clip_sample:         {cfg.clip_sample}")
    log.info(f"  clip_sample_value:   {getattr(cfg, 'clip_sample_value', 'N/A')}")
    log.info(f"  set_alpha_to_one:    {cfg.set_alpha_to_one}")
    log.info(f"  steps_offset:        {cfg.steps_offset}")
    log.info(f"  trained_betas:       {cfg.trained_betas}")

    results["scheduler_config"] = {
        "num_train_timesteps": cfg.num_train_timesteps,
        "beta_schedule": cfg.beta_schedule,
        "prediction_type": cfg.prediction_type,
        "clip_sample": cfg.clip_sample,
        "clip_sample_value": getattr(cfg, 'clip_sample_value', None),
        "set_alpha_to_one": cfg.set_alpha_to_one,
        "steps_offset": cfg.steps_offset,
        "trained_betas": cfg.trained_betas,
    }

    log.info(f"\n[Inference Config]")
    num_inference = getattr(base_model, 'num_inference_timesteps',
                            getattr(config, 'num_inference_timesteps', None))
    log.info(f"  num_inference_timesteps: {num_inference}")
    results["num_inference_timesteps"] = num_inference

    # 检查 prediction_type 一致性
    log.info(f"\n[Critical: Prediction Type]")
    if cfg.prediction_type == "epsilon":
        log.info(f"  Training target: epsilon (noise)")
        log.info(f"  Training loss: MSE(noise_pred, noise)")
        log.info(f"  Inference: scheduler.step() uses epsilon prediction")
        log.info(f"  => CONSISTENT")
    elif cfg.prediction_type == "v_prediction":
        log.info(f"  Training target: velocity (v)")
        log.info(f"  Training loss: MSE(noise_pred, v)")
        log.info(f"  Inference: scheduler.step() uses v_prediction")
        log.info(f"  => CONSISTENT")
    elif cfg.prediction_type == "sample":
        log.info(f"  Training target: x_0 (clean sample)")
        log.info(f"  => This is unusual for diffusion policy")
        issues.append(f"Unusual prediction_type: {cfg.prediction_type}")
    else:
        log.info(f"  WARNING: Unknown prediction_type: {cfg.prediction_type}")
        issues.append(f"Unknown prediction_type: {cfg.prediction_type}")

    results["prediction_type_consistent"] = cfg.prediction_type in ["epsilon", "v_prediction"]

    # 检查 clip_sample
    log.info(f"\n[Critical: Clip Sample]")
    log.info(f"  clip_sample: {cfg.clip_sample}")
    if cfg.clip_sample:
        log.info(f"  => During inference, samples are clipped to [-1, 1] at each step")
        log.info(f"  => If normalized actions exceed [-1, 1], they will be clipped")
        log.info(f"  => This could cause action distribution shrinkage!")

        # 检查 dataset action stats 是否超出 [-1, 1]
        if dataset_stats and ACTION in dataset_stats:
            action_stats = dataset_stats[ACTION]
            if "min" in action_stats and "max" in action_stats:
                a_min = action_stats["min"]
                a_max = action_stats["max"]
                if hasattr(a_min, 'tolist'):
                    a_min = a_min.tolist()
                    a_max = a_max.tolist()
                a_min_np = np.array(a_min)
                a_max_np = np.array(a_max)

                log.info(f"  Dataset action min: {a_min_np}")
                log.info(f"  Dataset action max: {a_max_np}")

                if (a_min_np < -1.0).any() or (a_max_np > 1.0).any():
                    log.info(f"  WARNING: Dataset action range exceeds [-1, 1]!")
                    log.info(f"  => clip_sample=True will clip actions during inference")
                    log.info(f"  => This will shrink the action distribution!")
                    issues.append(f"clip_sample=True but action range [{a_min_np.min()}, {a_max_np.max()}] exceeds [-1, 1]")
                else:
                    log.info(f"  OK: Dataset action range within [-1, 1]")
    else:
        log.info(f"  => No clipping during inference")

    results["clip_sample_issue"] = cfg.clip_sample

    # 检查 beta_schedule
    log.info(f"\n[Critical: Beta Schedule]")
    log.info(f"  beta_schedule: {cfg.beta_schedule}")
    if cfg.beta_schedule == "squaredcos_cap_v2":
        log.info(f"  => squaredcos_cap_v2: common for diffusion policy")
    elif cfg.beta_schedule == "linear":
        log.info(f"  => linear: standard DDPM schedule")
    elif cfg.beta_schedule == "scaled_linear":
        log.info(f"  => scaled_linear: common for stable diffusion")

    # 验证 betas
    betas = scheduler.betas.cpu().numpy()
    log.info(f"  betas range: [{betas.min():.6f}, {betas.max():.6f}]")
    log.info(f"  betas mean: {betas.mean():.6f}")

    results["betas"] = {
        "min": float(betas.min()),
        "max": float(betas.max()),
        "mean": float(betas.mean()),
    }

    # 检查 alphas_cumprod
    alphas_cumprod = scheduler.alphas_cumprod.cpu().numpy()
    log.info(f"  alphas_cumprod[0]: {alphas_cumprod[0]:.6f}")
    log.info(f"  alphas_cumprod[-1]: {alphas_cumprod[-1]:.6f}")
    log.info(f"  alphas_cumprod[50]: {alphas_cumprod[50]:.6f}")

    results["alphas_cumprod"] = {
        "t0": float(alphas_cumprod[0]),
        "t50": float(alphas_cumprod[50]),
        "t99": float(alphas_cumprod[-1]),
    }

    # Summary
    results["issues"] = issues
    if issues:
        log.info(f"\n  ISSUES FOUND: {len(issues)}")
        for issue in issues:
            log.info(f"    - {issue}")
    else:
        log.info(f"\n  OK: No scheduler consistency issues found")

    return results


# ============================================================================
# 2. TRAINING ACTION FLOW (using policy.forward API)
# ============================================================================

def make_delta_timestamps(delta_indices: list[int], fps: float) -> list[float]:
    """Convert delta indices to delta timestamps in seconds."""
    return [idx / fps for idx in delta_indices]


def build_training_batch(policy, config, dataset: LeRobotDataset,
                          preprocessor, batch_size: int = 2,
                          logger: logging.Logger | None = None) -> dict | None:
    """构建训练 batch，使用 delta_timestamps 加载 chunked action

    训练时 dataloader 通过 delta_timestamps 获取 chunked action:
      action shape: [B, chunk_size, action_dim]
    这与训练脚本中 LeRobotDataset(delta_timestamps=...) 的行为一致
    """
    log = logger or logging.getLogger("diffusion_diag")

    chunk_size = getattr(config, "chunk_size", 16)
    fps = dataset.fps

    # 构建 delta_timestamps (与训练脚本一致)
    action_delta_indices = list(range(chunk_size))
    obs_delta_indices = [0]  # 单帧观测

    delta_timestamps = {
        ACTION: make_delta_timestamps(action_delta_indices, fps),
    }

    # 添加 image features
    image_keys = [k for k in config.image_features] if hasattr(config, "image_features") and config.image_features else []
    if not image_keys:
        # 从 dataset meta 推断
        for key in dataset.meta.features:
            if key.startswith("observation.images."):
                image_keys.append(key)

    for img_key in image_keys:
        delta_timestamps[img_key] = make_delta_timestamps(obs_delta_indices, fps)

    # 添加 state
    delta_timestamps["observation.state"] = make_delta_timestamps(obs_delta_indices, fps)

    log.info(f"  delta_timestamps: { {k: len(v) for k, v in delta_timestamps.items()} }")

    # 用 delta_timestamps 重新加载 dataset
    try:
        chunked_dataset = LeRobotDataset(
            dataset.repo_id if hasattr(dataset, "repo_id") else str(dataset.root),
            root=str(dataset.root) if hasattr(dataset, "root") else None,
            delta_timestamps=delta_timestamps,
        )
        log.info(f"  Chunked dataset loaded: {chunked_dataset.num_frames} frames")
    except Exception as e:
        log.info(f"  WARNING: Failed to load chunked dataset: {e}")
        return None

    # 取 sample
    samples = []
    for i in range(batch_size):
        try:
            samples.append(chunked_dataset[i])
        except IndexError:
            break

    if len(samples) == 0:
        log.info(f"  WARNING: No samples from chunked dataset")
        return None

    # Preprocess
    preprocessed_list = [preprocessor(s) for s in samples]

    # Stack into batch
    batch = {}
    for key in preprocessed_list[0].keys():
        vals = [p[key] for p in preprocessed_list if key in p]
        if vals and isinstance(vals[0], torch.Tensor):
            batch[key] = torch.stack(vals, dim=0)

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device=policy.config.device)

    # 确保有 task 字段
    if "task" not in batch:
        batch["task"] = [""] * len(samples)

    # 确保有 action_is_pad
    if "action_is_pad" not in batch and ACTION in batch:
        actions = batch[ACTION]
        batch["action_is_pad"] = torch.zeros(actions.shape[:2], dtype=torch.bool, device=policy.config.device)

    log.info(f"  Batch built: { {k: v.shape if isinstance(v, torch.Tensor) else type(v).__name__ for k, v in batch.items()} }")
    return batch


def check_training_action_flow(policy, config, dataset: LeRobotDataset,
                                preprocessor, dataset_stats: dict | None,
                                logger: logging.Logger) -> dict:
    """使用 policy.forward() API 诊断训练时 action 的完整流程

    流程:
      dataset (with delta_timestamps) -> chunked action [B, chunk_size, action_dim]
      -> preprocessor.normalize -> batch[ACTION] -> policy.forward()
      -> 内部: actions.to(dtype) -> forward_diffusion_head() -> add_noise -> UNet -> loss
    """
    log = logger
    print_sep("2. TRAINING ACTION FLOW (via policy.forward API)", log)

    results = {}
    issues = []

    # 2.1 从 dataset 取一个 sample (单步)
    sample = dataset[0]
    raw_action = sample[ACTION]
    if isinstance(raw_action, np.ndarray):
        raw_action_t = torch.from_numpy(raw_action).float()
    else:
        raw_action_t = raw_action.float()

    if raw_action_t.ndim == 1:
        raw_action_t = raw_action_t.unsqueeze(0)

    log.info(f"[Raw Action from Dataset (Single Step)]")
    log.info(f"  Shape: {raw_action_t.shape}")
    log.info(f"  Mean: {raw_action_t.mean():.6f}, Std: {raw_action_t.std():.6f}")
    log.info(f"  Min: {raw_action_t.min():.6f}, Max: {raw_action_t.max():.6f}")

    results["raw_action"] = tensor_info(raw_action_t, "raw_action")

    # 2.2 通过 preprocessor (使用已有 API)
    preprocessed = preprocessor(sample)

    if ACTION in preprocessed:
        pre_action = preprocessed[ACTION]
        if isinstance(pre_action, torch.Tensor):
            pre_action = pre_action.float()
            if pre_action.ndim == 3:
                pre_action_flat = pre_action[0]
            elif pre_action.ndim == 2:
                pre_action_flat = pre_action
            else:
                pre_action_flat = pre_action.view(-1, pre_action.shape[-1])

            log.info(f"\n[Action After Preprocessor (Normalized)]")
            log.info(f"  Shape: {pre_action.shape}")
            log.info(f"  Mean: {pre_action_flat.mean():.6f}, Std: {pre_action_flat.std():.6f}")
            log.info(f"  Min: {pre_action_flat.min():.6f}, Max: {pre_action_flat.max():.6f}")

            results["normalized_action"] = tensor_info(pre_action, "normalized_action")

            # 验证 normalization 公式
            if dataset_stats and ACTION in dataset_stats:
                action_stats = dataset_stats[ACTION]
                if "mean" in action_stats and "std" in action_stats:
                    ds_mean = torch.tensor(action_stats["mean"], dtype=torch.float32)
                    ds_std = torch.tensor(action_stats["std"], dtype=torch.float32)

                    expected = (raw_action_t - ds_mean) / ds_std
                    actual = pre_action_flat.to(expected.device)

                    diff = (actual - expected).abs()
                    max_diff = float(diff.max())

                    log.info(f"\n[Normalization Formula Verification]")
                    log.info(f"  Expected (raw - mean) / std: {expected.tolist()}")
                    log.info(f"  Actual preprocessed:         {actual.tolist()}")
                    log.info(f"  Max diff: {max_diff:.8f}")

                    if max_diff > 1e-5:
                        log.info(f"  WARNING: Normalization formula mismatch!")
                        issues.append(f"Normalization formula mismatch (max diff: {max_diff:.8f})")
                    else:
                        log.info(f"  OK: Normalization formula correct")

                    results["normalization_check"] = {
                        "expected": expected.tolist(),
                        "actual": actual.tolist(),
                        "max_diff": max_diff,
                        "is_correct": max_diff < 1e-5,
                    }
    else:
        log.info(f"  WARNING: '{ACTION}' not in preprocessed output")
        issues.append("Action not found after preprocessing")

    # 2.3 构建训练 batch 并调用 policy.forward() (使用已有 API)
    # 使用 delta_timestamps 加载 chunked action，与训练脚本一致
    log.info(f"\n[Building Training Batch with delta_timestamps]")
    log.info(f"  chunk_size: {getattr(config, 'chunk_size', 16)}")
    log.info(f"  fps: {dataset.fps}")

    batch = build_training_batch(policy, config, dataset, preprocessor, batch_size=2, logger=log)

    if batch is not None and ACTION in batch:
        # delta_timestamps 加载的 image 形状是 [B, T, C, H, W]，而 policy.forward()
        # 的 _process_images 期望 [B, C, H, W]，需要 squeeze 掉 T 维度
        for key in list(batch.keys()):
            if isinstance(batch[key], torch.Tensor) and "observation.images." in key and batch[key].ndim == 5:
                batch[key] = batch[key].squeeze(1)  # [B, T, C, H, W] -> [B, C, H, W]

        chunked_action = batch[ACTION]
        log.info(f"\n[Chunked Action from Training Batch]")
        log.info(f"  Shape: {chunked_action.shape}")
        log.info(f"  Mean: {chunked_action.float().mean():.6f}, Std: {chunked_action.float().std():.6f}")
        log.info(f"  Min: {chunked_action.float().min():.6f}, Max: {chunked_action.float().max():.6f}")

        results["chunked_action"] = tensor_info(chunked_action, "chunked_action")

        # 设置 hook 来追踪内部调用
        base_model = policy._get_base_model()
        if hasattr(base_model, '_diffusion_initialized') and not base_model._diffusion_initialized:
            base_model._init_diffusion_head()

        tracer = CallTracer(log)
        scheduler = base_model.noise_scheduler
        tracer.hook_method(scheduler, 'add_noise', 'noise_scheduler.add_noise')
        if hasattr(base_model, 'embed_out'):
            tracer.hook_method(base_model, 'embed_out', 'base_model.embed_out (UNet)')

        # 调用 policy.forward() (已有 API)
        log.info(f"\n[Calling policy.forward() API]")
        policy.train()

        with torch.no_grad():
            loss, info = policy.forward(batch)

        policy.eval()

        log.info(f"\n[policy.forward() Output]")
        log.info(f"  Loss: {loss.item():.6f}")
        log.info(f"  Info keys: {list(info.keys())}")

        results["forward_loss"] = float(loss.item())
        results["forward_info_keys"] = list(info.keys())
        results["forward_skipped"] = False

        # 分析 loss
        loss_val = float(loss.item())
        log.info(f"\n[Loss Analysis]")
        log.info(f"  Expected loss for random model: ~1.0 (MSE of N(0,1) noise)")
        log.info(f"  Actual loss: {loss_val:.6f}")

        if loss_val > 5.0:
            log.info(f"  WARNING: Loss is very high!")
            log.info(f"  => Model predictions are very poor")
            issues.append(f"Training loss very high: {loss_val:.4f}")
        elif loss_val > 2.0:
            log.info(f"  NOTE: Loss is moderately high")
            log.info(f"  => Model is still learning, may need more steps")
        elif loss_val > 0.5:
            log.info(f"  OK: Loss is in reasonable range")
            log.info(f"  => Model is learning but not yet converged")
        else:
            log.info(f"  OK: Loss is low, model may be well-trained")

        results["loss_analysis"] = {
            "value": loss_val,
            "expected_random": 1.0,
            "status": "high" if loss_val > 2.0 else ("moderate" if loss_val > 0.5 else "low"),
        }

        # 检查 info 中是否有 action 相关信息
        for key in info:
            if "action" in key.lower() or "noise" in key.lower() or "pred" in key.lower():
                val = info[key]
                if isinstance(val, torch.Tensor):
                    log.info(f"  info['{key}']: shape={val.shape}, mean={val.float().mean():.4f}, std={val.float().std():.4f}")

        # 分析 hook 收集的信息
        log.info(f"\n[Hook Analysis]")
        log.info(f"  Total calls traced: {len(tracer.calls)}")

        add_noise_calls = [c for c in tracer.calls if 'add_noise' in c['label']]
        if add_noise_calls:
            log.info(f"  add_noise calls: {len(add_noise_calls)}")
            for call in add_noise_calls:
                result = call['result']
                if isinstance(result, torch.Tensor):
                    log.info(f"    Noisy action shape: {result.shape}")
                    log.info(f"    Noisy action mean: {result.float().mean():.4f}")
                    log.info(f"    Noisy action std: {result.float().std():.4f}")

        embed_out_calls = [c for c in tracer.calls if 'embed_out' in c['label']]
        if embed_out_calls:
            log.info(f"  embed_out (UNet) calls: {len(embed_out_calls)}")
            for call in embed_out_calls:
                result = call['result']
                if isinstance(result, torch.Tensor):
                    log.info(f"    UNet output shape: {result.shape}")
                    log.info(f"    UNet output mean: {result.float().mean():.4f}")
                    log.info(f"    UNet output std: {result.float().std():.4f}")
    else:
        log.info(f"\n[Skipping policy.forward() - failed to build chunked batch]")
        results["forward_loss"] = None
        results["forward_info_keys"] = []
        results["forward_skipped"] = True

    # 2.4 多帧分析: 使用 preprocessor API 检查 normalized action 分布
    log.info(f"\n[Multi-Frame Normalized Action Distribution (via preprocessor API)]")
    n_frames = min(100, dataset.num_frames)
    all_normalized_actions = []

    for i in range(n_frames):
        sample_i = dataset[i]
        pre_i = preprocessor(sample_i)
        if ACTION in pre_i:
            act = pre_i[ACTION]
            if isinstance(act, torch.Tensor):
                if act.ndim == 3:
                    act = act[0]
                all_normalized_actions.append(act.float().cpu().numpy())

    if all_normalized_actions:
        all_actions_np = np.concatenate(all_normalized_actions, axis=0)
        log.info(f"  Analyzed {len(all_actions_np)} action vectors")
        log.info(f"  Overall mean: {all_actions_np.mean(axis=0)}")
        log.info(f"  Overall std:  {all_actions_np.std(axis=0)}")

        results["multi_frame_distribution"] = {
            "n_frames": len(all_actions_np),
            "mean_per_dim": all_actions_np.mean(axis=0).tolist(),
            "std_per_dim": all_actions_np.std(axis=0).tolist(),
            "overall_mean": float(all_actions_np.mean()),
            "overall_std": float(all_actions_np.std()),
        }

        overall_mean = float(all_actions_np.mean())
        overall_std = float(all_actions_np.std())

        log.info(f"\n[Distribution Quality Check]")
        log.info(f"  Expected for MEAN_STD normalization: mean~0, std~1")
        log.info(f"  Actual: mean={overall_mean:.4f}, std={overall_std:.4f}")

        if abs(overall_mean) > 0.5:
            log.info(f"  WARNING: Overall mean is far from 0!")
            issues.append(f"Overall normalized action mean far from 0: {overall_mean:.4f}")
        else:
            log.info(f"  OK: Overall mean is close to 0")

        if abs(overall_std - 1.0) > 0.5:
            log.info(f"  WARNING: Overall std is far from 1!")
            issues.append(f"Overall normalized action std far from 1: {overall_std:.4f}")
        else:
            log.info(f"  OK: Overall std is close to 1")

    # Summary
    results["issues"] = issues
    if issues:
        log.info(f"\n  ISSUES FOUND: {len(issues)}")
        for issue in issues:
            log.info(f"    - {issue}")
    else:
        log.info(f"\n  OK: No training action flow issues found")

    return results


# ============================================================================
# 3. INFERENCE ACTION FLOW (using policy.predict_action_chunk API)
# ============================================================================

def check_inference_action_flow(policy, config, dataset: LeRobotDataset,
                                 preprocessor, postprocessor, dataset_stats: dict | None,
                                 logger: logging.Logger) -> dict:
    """使用 policy.predict_action_chunk() API 诊断推理时 action 的完整流程

    流程:
      preprocessed batch -> policy.predict_action_chunk() -> normalized action
      -> postprocessor.unnormalize() -> robot action
    """
    log = logger
    print_sep("3. INFERENCE ACTION FLOW (via policy.predict_action_chunk API)", log)

    results = {}
    issues = []

    # 3.1 准备输入 (使用 preprocessor API)
    sample = dataset[0]
    preprocessed = preprocessor(sample)

    # 3.2 调用 policy.predict_action_chunk() (已有 API)
    log.info(f"[Calling policy.predict_action_chunk() API]")
    policy.reset()

    with torch.no_grad():
        pred_actions = policy.predict_action_chunk(preprocessed)

    log.info(f"  Policy raw output shape: {pred_actions.shape}")
    log.info(f"  Policy raw output dtype: {pred_actions.dtype}")

    pred_actions_fp32 = pred_actions.float()
    log.info(f"  Policy raw output (fp32):")
    log.info(f"    Shape: {pred_actions_fp32.shape}")
    if pred_actions_fp32.ndim >= 2:
        log.info(f"    First step: {pred_actions_fp32[0, 0].tolist()}")
    log.info(f"    Mean: {pred_actions_fp32.mean():.6f}")
    log.info(f"    Std:  {pred_actions_fp32.std():.6f}")
    log.info(f"    Min:  {pred_actions_fp32.min():.6f}")
    log.info(f"    Max:  {pred_actions_fp32.max():.6f}")

    results["policy_raw_output"] = tensor_info(pred_actions_fp32, "policy_raw_output")

    # 3.3 分析 raw output 分布
    if pred_actions_fp32.ndim >= 2:
        raw_first_step = pred_actions_fp32[0, 0]
    else:
        raw_first_step = pred_actions_fp32

    raw_mean = float(raw_first_step.mean())
    raw_std = float(raw_first_step.std())

    log.info(f"\n[Raw Output Distribution Analysis]")
    log.info(f"  Mean: {raw_mean:.6f}")
    log.info(f"  Std:  {raw_std:.6f}")
    log.info(f"  Range: [{raw_first_step.min():.6f}, {raw_first_step.max():.6f}]")

    # 关键诊断: 模型输出是否偏向零
    if abs(raw_mean) < 0.2 and raw_std < 1.0:
        log.info(f"  WARNING: Model output is concentrated near zero!")
        log.info(f"  => This indicates the diffusion head has not learned the action distribution")
        log.info(f"  => Possible causes:")
        log.info(f"     1. Training steps insufficient")
        log.info(f"     2. Learning rate too low")
        log.info(f"     3. Action normalization stats mismatch")
        log.info(f"     4. UNet architecture issue")
        issues.append("Model output concentrated near zero (mean={:.4f}, std={:.4f})".format(raw_mean, raw_std))
    else:
        log.info(f"  OK: Model output has reasonable distribution")

    results["output_distribution_check"] = {
        "mean": raw_mean,
        "std": raw_std,
        "is_near_zero": abs(raw_mean) < 0.2 and raw_std < 1.0,
    }

    # 3.4 通过 postprocessor (使用已有 API)
    log.info(f"\n[Calling postprocessor() API]")
    final_actions = postprocessor(pred_actions)

    if isinstance(final_actions, torch.Tensor):
        final_actions_fp32 = final_actions.float()
        log.info(f"  Postprocessed action shape: {final_actions_fp32.shape}")
        if final_actions_fp32.ndim >= 2:
            log.info(f"  Postprocessed first step: {final_actions_fp32[0, 0].tolist()}")
        log.info(f"  Postprocessed mean: {final_actions_fp32.mean():.6f}")
        log.info(f"  Postprocessed std:  {final_actions_fp32.std():.6f}")

        results["postprocessed_action"] = tensor_info(final_actions_fp32, "postprocessed_action")

        # 验证 unnormalization (使用 dataset stats)
        if dataset_stats and ACTION in dataset_stats:
            action_stats = dataset_stats[ACTION]
            if "mean" in action_stats and "std" in action_stats:
                ds_mean = torch.tensor(action_stats["mean"], dtype=torch.float32, device=raw_first_step.device)
                ds_std = torch.tensor(action_stats["std"], dtype=torch.float32, device=raw_first_step.device)

                if final_actions_fp32.ndim >= 2:
                    final_first = final_actions_fp32[0, 0]
                else:
                    final_first = final_actions_fp32

                expected = raw_first_step * ds_std + ds_mean
                actual = final_first.to(expected.device)

                diff = (actual - expected).abs()
                max_diff = float(diff.max())

                log.info(f"\n[Unnormalization Verification]")
                log.info(f"  Expected (raw * std + mean): {expected.tolist()}")
                log.info(f"  Actual postprocessed:        {actual.tolist()}")
                log.info(f"  Max diff: {max_diff:.8f}")

                if max_diff > 1e-3:
                    log.info(f"  WARNING: Unnormalization mismatch!")
                    issues.append(f"Unnormalization mismatch (max diff: {max_diff:.8f})")
                else:
                    log.info(f"  OK: Unnormalization correct")

                results["unnormalization_check"] = {
                    "expected": expected.tolist(),
                    "actual": actual.tolist(),
                    "max_diff": max_diff,
                    "is_correct": max_diff < 1e-3,
                }

    # 3.5 对比训练和推理的 action 分布
    log.info(f"\n[Training vs Inference Action Distribution Comparison]")

    # 训练时的 normalized action (从 preprocessor)
    if ACTION in preprocessed:
        train_action = preprocessed[ACTION].float()
        if train_action.ndim == 3:
            train_action = train_action[0]

        log.info(f"  Training normalized action:")
        log.info(f"    Mean: {train_action.mean():.6f}, Std: {train_action.std():.6f}")
        log.info(f"    Range: [{train_action.min():.6f}, {train_action.max():.6f}]")

        # 对比推理输出
        if pred_actions_fp32.ndim >= 2:
            inf_action = pred_actions_fp32[0, 0]
        else:
            inf_action = pred_actions_fp32

        log.info(f"  Inference normalized action (first step):")
        log.info(f"    Mean: {inf_action.mean():.6f}, Std: {inf_action.std():.6f}")
        log.info(f"    Range: [{inf_action.min():.6f}, {inf_action.max():.6f}]")

        # 计算分布差异
        if train_action.shape == inf_action.shape:
            mean_diff = abs(train_action.mean() - inf_action.mean())
            std_diff = abs(train_action.std() - inf_action.std())

            log.info(f"\n  Distribution Difference:")
            log.info(f"    Mean diff: {mean_diff:.6f}")
            log.info(f"    Std diff:  {std_diff:.6f}")

            results["distribution_comparison"] = {
                "train_mean": float(train_action.mean()),
                "train_std": float(train_action.std()),
                "inf_mean": float(inf_action.mean()),
                "inf_std": float(inf_action.std()),
                "mean_diff": float(mean_diff),
                "std_diff": float(std_diff),
            }

            if mean_diff > 1.0 or std_diff > 1.0:
                log.info(f"  WARNING: Large distribution gap between training and inference!")
                log.info(f"  => The model is not outputting actions similar to training targets")
                issues.append(f"Large train/inference distribution gap (mean_diff={mean_diff:.2f}, std_diff={std_diff:.2f})")
            else:
                log.info(f"  OK: Distribution gap is small")

    # Summary
    results["issues"] = issues
    if issues:
        log.info(f"\n  ISSUES FOUND: {len(issues)}")
        for issue in issues:
            log.info(f"    - {issue}")
    else:
        log.info(f"\n  OK: No inference action flow issues found")

    return results


# ============================================================================
# MAIN
# ============================================================================

def run_full_diagnostic(policy_type: str, checkpoint_path: str, dataset_root: str,
                        device: str, logger: logging.Logger) -> dict:
    log = logger
    log.info(f"\n{'#'*70}")
    log.info(f"# TinyVLA Diffusion Action Head Symmetry Diagnostic: {policy_type}")
    log.info(f"{'#'*70}")

    dataset = LeRobotDataset(dataset_root)
    log.info(f"Dataset loaded: {dataset.num_frames} frames, {dataset.num_episodes} episodes")

    ds_stats = dataset.meta.stats if hasattr(dataset.meta, "stats") else None

    policy, preprocessor, postprocessor, config = load_policy(
        policy_type, checkpoint_path, device, dataset_stats=ds_stats, logger=log
    )

    results = {
        "policy_type": policy_type,
        "checkpoint_path": checkpoint_path,
        "dataset_root": dataset_root,
        "config_summary": {
            "action_head_type": getattr(config, 'action_head_type', None),
            "chunk_size": getattr(config, 'chunk_size', None),
            "n_action_steps": getattr(config, 'n_action_steps', None),
            "action_dim": getattr(config, 'action_dim', None),
            "state_dim": getattr(config, 'state_dim', None),
            "num_inference_timesteps": getattr(config, 'num_inference_timesteps', None),
        },
    }

    # Run all diagnostics
    results["scheduler_consistency"] = check_scheduler_consistency(
        policy, config, ds_stats, log
    )

    results["training_action_flow"] = check_training_action_flow(
        policy, config, dataset, preprocessor, ds_stats, log
    )

    results["inference_action_flow"] = check_inference_action_flow(
        policy, config, dataset, preprocessor, postprocessor, ds_stats, log
    )

    # Overall diagnosis
    all_issues = []
    for check_name in ["scheduler_consistency", "training_action_flow",
                       "inference_action_flow"]:
        if check_name in results:
            check_issues = results[check_name].get("issues", [])
            all_issues.extend(check_issues)

    results["all_issues"] = all_issues
    results["total_issues"] = len(all_issues)

    # 关键诊断结论
    log.info(f"\n{'='*70}")
    log.info(f"  OVERALL DIAGNOSIS")
    log.info(f"{'='*70}")

    if all_issues:
        log.info(f"  FOUND {len(all_issues)} ISSUES:")
        for i, issue in enumerate(all_issues, 1):
            log.info(f"    {i}. {issue}")
    else:
        log.info(f"  No structural issues found.")
        log.info(f"  If model output is still near zero, the likely cause is:")
        log.info(f"    1. Insufficient training steps")
        log.info(f"    2. Learning rate too low")
        log.info(f"    3. Dataset quality issue")

    # 最终建议
    log.info(f"\n{'='*70}")
    log.info(f"  RECOMMENDATIONS")
    log.info(f"{'='*70}")

    # 检查 model output 是否 near zero
    if "inference_action_flow" in results:
        output_check = results["inference_action_flow"].get("output_distribution_check", {})
        if output_check.get("is_near_zero", False):
            log.info(f"  Model output IS concentrated near zero.")
            log.info(f"  Since training/inference flow is consistent, this means:")
            log.info(f"    -> The diffusion head has NOT learned the action distribution")
            log.info(f"    -> RECOMMEND: Increase training steps significantly")
            log.info(f"    -> Current: 12k steps, Try: 50k-100k steps")
            log.info(f"    -> Also check: learning rate, batch size, dataset quality")

    return results


def main():
    parser = argparse.ArgumentParser(description="TinyVLA Diffusion Action Head Symmetry Diagnostic")
    parser.add_argument("--policy_type", type=str, required=True,
                        choices=["tinyvla_s", "tinyvla_b"])
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset_root", type=str,
                        default="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/disassemble-v3_corner")
    parser.add_argument("--gpu_id", type=int, default=2)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(__file__).parent
    log_file = str(output_dir / "diffusion_head_symmetry.log")
    logger = setup_logging(log_file)

    logger.info(f"TinyVLA Diffusion Action Head Symmetry Diagnostic")
    logger.info(f"Policy type: {args.policy_type}")
    logger.info(f"Dataset: {args.dataset_root}")
    logger.info(f"Device: {device}")

    default_checkpoints = {
        "tinyvla_s": "/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/experiments/tinyvla_s_disassemble-v3_corner/checkpoints/tinyvla_tinyvla_s_disassemble-v3_corner/checkpoints/012000/pretrained_model",
        "tinyvla_b": "/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/experiments/tinyvla_b_disassemble-v3_corner/checkpoints/tinyvla_tinyvla_b_disassemble-v3_corner/checkpoints/012000/pretrained_model",
    }

    checkpoint_path = args.checkpoint_path or default_checkpoints.get(args.policy_type)
    if not checkpoint_path:
        logger.error(f"No default checkpoint for {args.policy_type}")
        return

    output_file = args.output_file or OUTPUT_FILE

    try:
        result = run_full_diagnostic(
            args.policy_type, checkpoint_path, args.dataset_root,
            device, logger
        )

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"\nResults saved to: {output_file}")
        logger.info(f"Log saved to: {log_file}")

    except Exception as e:
        logger.error(f"Error during diagnostic: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()