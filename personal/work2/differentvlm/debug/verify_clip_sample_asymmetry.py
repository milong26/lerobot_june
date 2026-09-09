#!/usr/bin/env python3
"""
TinyVLA: clip_sample=True 与 MEAN_STD 归一化不兼容问题验证工具

核心问题:
  训练: normalized action 可达 [-1.51, 3.39] (z-score 标准化)
  推理: DDIMScheduler(clip_sample=True) 每步将 sample 限制在 [-1, 1]
  结果: 推理输出被系统性压缩，无法恢复训练时学习到的大幅值动作

验证目标:
  1. 确认 normalized action 的真实范围 (应显著超出 [-1, 1])
  2. 确认 scheduler 的 clip_sample=True 配置
  3. 对比 clip_sample=True/False 时推理输出的差异
  4. 追踪 DDIM 采样过程中 sample 被 clip 的具体步骤和幅度
  5. 量化 clip 对最终动作分布的影响

Usage:
  python verify_clip_sample_asymmetry.py \
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
from copy import deepcopy

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

OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = str(OUTPUT_DIR / "clip_sample_asymmetry_results.json")


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("clip_sample_diag")
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


def tensor_stats(x, name="tensor"):
    if isinstance(x, torch.Tensor):
        x = x.float()
        return {
            "name": name,
            "shape": list(x.shape),
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "std": float(x.std()),
        }
    return {"name": name, "value": str(x)}


def load_policy(policy_type: str, checkpoint_path: str, device: str,
                dataset_stats: dict | None = None, logger: logging.Logger | None = None):
    log = logger or logging.getLogger("clip_sample_diag")
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
# 1. 验证 normalized action 的真实范围
# ============================================================================

def verify_normalized_action_range(dataset: LeRobotDataset, preprocessor,
                                    dataset_stats: dict, logger: logging.Logger) -> dict:
    """验证经过 MEAN_STD 归一化后 action 的真实范围

    关键: z-score 标准化 (x - mean) / std 不保证输出在 [-1, 1]
    如果原始 action 的 (max - mean) / std > 1，则 normalized action 会超出 1
    """
    log = logger
    print_sep("1. NORMALIZED ACTION RANGE VERIFICATION", log)

    results = {}

    action_stats = dataset_stats[ACTION]
    mean = torch.tensor(action_stats["mean"], dtype=torch.float32)
    std = torch.tensor(action_stats["std"], dtype=torch.float32)

    log.info(f"[Dataset Action Statistics]")
    log.info(f"  Mean: {mean.tolist()}")
    log.info(f"  Std:  {std.tolist()}")
    log.info(f"  Min:  {action_stats.get('min', 'N/A')}")
    log.info(f"  Max:  {action_stats.get('max', 'N/A')}")

    results["dataset_action_stats"] = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": action_stats.get("min", None),
        "max": action_stats.get("max", None),
    }

    # 理论计算: 原始 action 经过 z-score 后的范围
    if "min" in action_stats and "max" in action_stats:
        raw_min = torch.tensor(action_stats["min"], dtype=torch.float32)
        raw_max = torch.tensor(action_stats["max"], dtype=torch.float32)

        expected_norm_min = (raw_min - mean) / std
        expected_norm_max = (raw_max - mean) / std

        log.info(f"\n[Theoretical Normalized Range (z-score)]")
        log.info(f"  Expected norm min: {expected_norm_min.tolist()}")
        log.info(f"  Expected norm max: {expected_norm_max.tolist()}")
        log.info(f"  Expected norm range: [{expected_norm_min.min():.4f}, {expected_norm_max.max():.4f}]")

        results["theoretical_normalized_range"] = {
            "min": expected_norm_min.tolist(),
            "max": expected_norm_max.tolist(),
            "overall_min": float(expected_norm_min.min()),
            "overall_max": float(expected_norm_max.max()),
        }

        exceeds_1 = (expected_norm_max > 1.0).any() or (expected_norm_min < -1.0).any()
        log.info(f"\n[Critical: Does normalized range exceed [-1, 1]?]")
        log.info(f"  Exceeds 1: {exceeds_1}")
        if exceeds_1:
            log.info(f"  => clip_sample=True WILL clip actions during DDIM sampling!")
            log.info(f"  => This will systematically shrink the action amplitude!")
        results["exceeds_clip_range"] = bool(exceeds_1)

    # 实际测量: 通过 preprocessor 获取 normalized action
    log.info(f"\n[Empirical Normalized Action Range]")
    n_samples = min(200, dataset.num_frames)
    all_norm_actions = []

    for i in range(n_samples):
        sample = dataset[i]
        preprocessed = preprocessor(sample)
        if ACTION in preprocessed:
            act = preprocessed[ACTION]
            if isinstance(act, torch.Tensor):
                if act.ndim == 3:
                    act = act[0]
                all_norm_actions.append(act.float().cpu().numpy())

    if all_norm_actions:
        all_norm_np = np.concatenate(all_norm_actions, axis=0)
        emp_min = float(all_norm_np.min())
        emp_max = float(all_norm_np.max())
        emp_mean = float(all_norm_np.mean())
        emp_std = float(all_norm_np.std())

        log.info(f"  Samples analyzed: {len(all_norm_np)}")
        log.info(f"  Overall min:  {emp_min:.4f}")
        log.info(f"  Overall max:  {emp_max:.4f}")
        log.info(f"  Overall mean: {emp_mean:.4f}")
        log.info(f"  Overall std:  {emp_std:.4f}")

        results["empirical_normalized_range"] = {
            "n_samples": len(all_norm_np),
            "min": emp_min,
            "max": emp_max,
            "mean": emp_mean,
            "std": emp_std,
            "per_dim_min": all_norm_np.min(axis=0).tolist(),
            "per_dim_max": all_norm_np.max(axis=0).tolist(),
            "per_dim_mean": all_norm_np.mean(axis=0).tolist(),
            "per_dim_std": all_norm_np.std(axis=0).tolist(),
        }

        exceeds_1_emp = emp_max > 1.0 or emp_min < -1.0
        log.info(f"\n[Critical: Empirical range exceeds [-1, 1]?]")
        log.info(f"  Exceeds: {exceeds_1_emp}")
        if exceeds_1_emp:
            log.info(f"  => CONFIRMED: Normalized actions exceed [-1, 1]")
            log.info(f"  => clip_sample=True will cause information loss during inference")
        results["empirical_exceeds_clip_range"] = bool(exceeds_1_emp)

    return results


# ============================================================================
# 2. 验证 scheduler 配置
# ============================================================================

def verify_scheduler_config(policy, config, logger: logging.Logger) -> dict:
    """验证 scheduler 的 clip_sample 配置及其影响"""
    log = logger
    print_sep("2. SCHEDULER CONFIGURATION VERIFICATION", log)

    results = {}

    base_model = policy._get_base_model()

    # 触发 lazy init
    if hasattr(base_model, '_diffusion_initialized') and not base_model._diffusion_initialized:
        base_model._init_diffusion_head()

    if not hasattr(base_model, 'noise_scheduler'):
        log.info("  ERROR: noise_scheduler not found")
        results["error"] = "noise_scheduler not found"
        return results

    scheduler = base_model.noise_scheduler
    cfg = scheduler.config

    log.info(f"[Scheduler Configuration]")
    log.info(f"  Type:              {type(scheduler).__name__}")
    log.info(f"  num_train_timesteps: {cfg.num_train_timesteps}")
    log.info(f"  beta_schedule:     {cfg.beta_schedule}")
    log.info(f"  prediction_type:   {cfg.prediction_type}")
    log.info(f"  clip_sample:       {cfg.clip_sample}")
    log.info(f"  clip_sample_value: {getattr(cfg, 'clip_sample_value', 'N/A')}")
    log.info(f"  set_alpha_to_one:  {cfg.set_alpha_to_one}")
    log.info(f"  steps_offset:      {cfg.steps_offset}")

    results["scheduler_config"] = {
        "type": type(scheduler).__name__,
        "num_train_timesteps": cfg.num_train_timesteps,
        "beta_schedule": cfg.beta_schedule,
        "prediction_type": cfg.prediction_type,
        "clip_sample": cfg.clip_sample,
        "clip_sample_value": getattr(cfg, 'clip_sample_value', None),
        "set_alpha_to_one": cfg.set_alpha_to_one,
        "steps_offset": cfg.steps_offset,
    }

    num_inference = getattr(base_model, 'num_inference_timesteps',
                            getattr(config, 'num_inference_timesteps', None))
    log.info(f"\n  num_inference_timesteps: {num_inference}")
    results["num_inference_timesteps"] = num_inference

    # 关键分析
    log.info(f"\n[Critical Analysis]")
    if cfg.clip_sample:
        log.info(f"  clip_sample=True means:")
        log.info(f"    During each DDIM step, scheduler.step() clips the predicted x_0 to [-clip_sample_value, clip_sample_value]")
        log.info(f"    Default clip_sample_value = 1.0")
        log.info(f"    This assumes the clean sample x_0 is in [-1, 1]")
        log.info(f"    But with MEAN_STD normalization, x_0 can be outside [-1, 1]")
        log.info(f"    => CLIPPING WILL OCCUR AND SHRINK THE ACTION DISTRIBUTION")
    else:
        log.info(f"  clip_sample=False: no clipping, safe for MEAN_STD normalization")

    results["clip_sample_enabled"] = cfg.clip_sample

    return results


# ============================================================================
# 3. 对比 clip_sample=True vs False 的推理输出
# ============================================================================

def run_diffusion_sampling_with_clipping(base_model, hidden_states, states,
                                          clip_sample: bool, num_inference_steps: int = 10) -> torch.Tensor:
    """运行 diffusion 采样，可控制 clip_sample 设置

    这是从 forward_diffusion_head 提取的推理逻辑，但允许修改 clip_sample
    """
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    B = hidden_states.size(0)
    Tp = base_model.num_queries
    action_dim = base_model.action_dim

    # 创建 scheduler 副本以控制 clip_sample
    scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=clip_sample,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type='epsilon'
    )

    noisy_action = torch.randn((B, Tp, action_dim), device=hidden_states.device, dtype=hidden_states.dtype)
    naction = noisy_action
    scheduler.set_timesteps(num_inference_steps)

    for k in scheduler.timesteps:
        noise_pred = base_model.embed_out(naction, k, global_cond=hidden_states, states=states)
        naction = scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=naction
        ).prev_sample

    return naction


def run_diffusion_sampling_with_tracing(base_model, hidden_states, states,
                                         num_inference_steps: int = 10, logger: logging.Logger | None = None) -> dict:
    """运行 diffusion 采样并追踪每步的 clip 情况"""
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    log = logger

    B = hidden_states.size(0)
    Tp = base_model.num_queries
    action_dim = base_model.action_dim

    scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type='epsilon'
    )

    noisy_action = torch.randn((B, Tp, action_dim), device=hidden_states.device, dtype=hidden_states.dtype)
    naction = noisy_action
    scheduler.set_timesteps(num_inference_steps)

    trace = []

    for idx, k in enumerate(scheduler.timesteps):
        sample_before = naction.clone()
        sample_before_range = float(sample_before.max()) - float(sample_before.min())

        noise_pred = base_model.embed_out(naction, k, global_cond=hidden_states, states=states)

        output = scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=naction
        )
        naction = output.prev_sample

        sample_after = naction
        sample_after_range = float(sample_after.max()) - float(sample_after.min())

        was_clipped = (sample_after.max() > 1.0) or (sample_after.min() < -1.0)
        range_shrink = sample_before_range - sample_after_range

        step_info = {
            "step": idx,
            "timestep": int(k),
            "sample_before_min": float(sample_before.min()),
            "sample_before_max": float(sample_before.max()),
            "sample_before_range": sample_before_range,
            "sample_after_min": float(sample_after.min()),
            "sample_after_max": float(sample_after.max()),
            "sample_after_range": sample_after_range,
            "range_shrink": range_shrink,
            "noise_pred_mean": float(noise_pred.float().mean()),
            "noise_pred_std": float(noise_pred.float().std()),
        }

        trace.append(step_info)

        if log:
            clip_marker = " [CLIPPED]" if range_shrink > 0 else ""
            log.info(f"  Step {idx:2d} (t={int(k):3d}): "
                     f"range [{step_info['sample_before_min']:.4f}, {step_info['sample_before_max']:.4f}] "
                     f"-> [{step_info['sample_after_min']:.4f}, {step_info['sample_after_max']:.4f}] "
                     f"(shrink={range_shrink:.4f}){clip_marker}")

    return {
        "final_action": naction,
        "trace": trace,
    }


def _extract_hidden_states(policy, base_model, preprocessed, config, image_chunks, num_cams):
    """从模型中提取 multimodal hidden_states，复用 predict_action_chunk 的输入处理逻辑"""
    states = preprocessed[OBS_STATE].to(policy.config.device) if OBS_STATE in preprocessed else None
    raw_lang = preprocessed.get("task", [""])
    if isinstance(raw_lang, torch.Tensor):
        raw_lang = [""] * 1
    elif isinstance(raw_lang, str):
        raw_lang = [raw_lang]

    input_ids, labels = policy._tokenize_language(raw_lang)

    model_dtype = base_model.get_model().mm_projector[0].weight.dtype
    if states is not None:
        states = states.to(dtype=model_dtype)

    # 通过 multimodal 处理获取 hidden_states
    with torch.no_grad():
        input_ids_mm, attention_mask_mm, _, inputs_embeds_mm, _ = base_model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, labels, image_chunks[0],
            images_r=image_chunks[1] if num_cams > 1 else None,
            images_top=image_chunks[2] if num_cams > 2 else None,
            visual_concat=base_model.visual_concat,
            states=states,
        )
        outputs_backbone = base_model.get_model()(
            input_ids=input_ids_mm,
            attention_mask=attention_mask_mm,
            inputs_embeds=inputs_embeds_mm,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = outputs_backbone.last_hidden_state

    return hidden_states, states


def compare_clipping_impact(policy, config, dataset: LeRobotDataset,
                             preprocessor, logger: logging.Logger) -> dict:
    """对比 clip_sample=True vs False 对推理输出的影响"""
    log = logger
    print_sep("3. CLIPPING IMPACT COMPARISON (True vs False)", log)

    results = {}

    base_model = policy._get_base_model()
    if hasattr(base_model, '_diffusion_initialized') and not base_model._diffusion_initialized:
        base_model._init_diffusion_head()

    num_inference = getattr(base_model, 'num_inference_timesteps',
                            getattr(config, 'num_inference_timesteps', 10))

    # 准备输入
    sample = dataset[0]
    preprocessed = preprocessor(sample)

    image_keys = sorted([k for k in config.image_features if k in preprocessed])
    if not image_keys:
        image_keys = sorted([k for k in preprocessed if k.startswith("observation.images.")])
    images = [preprocessed[k].to(policy.config.device) for k in image_keys]

    processed_images = policy._process_images(images)
    num_cams = len(images)
    image_chunks = torch.chunk(processed_images, num_cams, dim=0)

    # 提取 hidden_states 和 states
    hidden_states, states = _extract_hidden_states(
        policy, base_model, preprocessed, config, image_chunks, num_cams
    )

    log.info(f"[Running Diffusion Sampling Comparison]")
    log.info(f"  hidden_states shape: {hidden_states.shape}")
    log.info(f"  states shape: {states.shape if states is not None else 'None'}")
    log.info(f"  num_inference_steps: {num_inference}")

    # 方式1: clip_sample=True (当前配置)
    log.info(f"\n[Sampling with clip_sample=True]")
    with torch.no_grad():
        actions_clipped = run_diffusion_sampling_with_clipping(
            base_model, hidden_states, states,
            clip_sample=True, num_inference_steps=num_inference
        )

    actions_clipped_fp = actions_clipped.float()
    log.info(f"  Output shape: {actions_clipped_fp.shape}")
    log.info(f"  Min: {actions_clipped_fp.min():.6f}")
    log.info(f"  Max: {actions_clipped_fp.max():.6f}")
    log.info(f"  Mean: {actions_clipped_fp.mean():.6f}")
    log.info(f"  Std:  {actions_clipped_fp.std():.6f}")

    results["with_clip"] = tensor_stats(actions_clipped_fp, "actions_with_clip")

    # 方式2: clip_sample=False (建议配置)
    log.info(f"\n[Sampling with clip_sample=False]")
    with torch.no_grad():
        actions_unclipped = run_diffusion_sampling_with_clipping(
            base_model, hidden_states, states,
            clip_sample=False, num_inference_steps=num_inference
        )

    actions_unclipped_fp = actions_unclipped.float()
    log.info(f"  Output shape: {actions_unclipped_fp.shape}")
    log.info(f"  Min: {actions_unclipped_fp.min():.6f}")
    log.info(f"  Max: {actions_unclipped_fp.max():.6f}")
    log.info(f"  Mean: {actions_unclipped_fp.mean():.6f}")
    log.info(f"  Std:  {actions_unclipped_fp.std():.6f}")

    results["without_clip"] = tensor_stats(actions_unclipped_fp, "actions_without_clip")

    # 对比
    log.info(f"\n[Comparison: clip_sample=True vs False]")
    range_clipped = float(actions_clipped_fp.max()) - float(actions_clipped_fp.min())
    range_unclipped = float(actions_unclipped_fp.max()) - float(actions_unclipped_fp.min())
    std_clipped = float(actions_clipped_fp.std())
    std_unclipped = float(actions_unclipped_fp.std())

    log.info(f"  Range (clipped):    {range_clipped:.4f}")
    log.info(f"  Range (unclipped):  {range_unclipped:.4f}")
    log.info(f"  Range difference:   {range_unclipped - range_clipped:.4f}")
    log.info(f"  Std (clipped):      {std_clipped:.4f}")
    log.info(f"  Std (unclipped):    {std_unclipped:.4f}")
    log.info(f"  Std difference:     {std_unclipped - std_clipped:.4f}")

    results["comparison"] = {
        "range_clipped": range_clipped,
        "range_unclipped": range_unclipped,
        "range_difference": range_unclipped - range_clipped,
        "std_clipped": std_clipped,
        "std_unclipped": std_unclipped,
        "std_difference": std_unclipped - std_clipped,
        "clipped_is_smaller_range": range_clipped < range_unclipped,
        "clipped_is_smaller_std": std_clipped < std_unclipped,
    }

    if range_clipped < range_unclipped:
        log.info(f"\n  => CONFIRMED: clip_sample=True shrinks the action range!")
        log.info(f"  => Range reduced by: {((range_unclipped - range_clipped) / range_unclipped * 100):.1f}%")
        log.info(f"  => Std reduced by:   {((std_unclipped - std_clipped) / std_unclipped * 100):.1f}%")

    # 与训练目标对比
    if ACTION in preprocessed:
        train_action = preprocessed[ACTION].float()
        if train_action.ndim == 3:
            train_action = train_action[0]

        log.info(f"\n[Comparison with Training Target]")
        log.info(f"  Training target range: [{train_action.min():.4f}, {train_action.max():.4f}]")
        log.info(f"  Training target std:   {train_action.std():.4f}")
        log.info(f"  Inference (clipped) range:   [{actions_clipped_fp.min():.4f}, {actions_clipped_fp.max():.4f}]")
        log.info(f"  Inference (clipped) std:     {actions_clipped_fp.std():.4f}")
        log.info(f"  Inference (unclipped) range: [{actions_unclipped_fp.min():.4f}, {actions_unclipped_fp.max():.4f}]")
        log.info(f"  Inference (unclipped) std:   {actions_unclipped_fp.std():.4f}")

        results["training_target"] = tensor_stats(train_action, "training_target")

    return results


# ============================================================================
# 4. 追踪 DDIM 采样过程中的 clip 行为
# ============================================================================

def trace_ddim_clipping(policy, config, dataset: LeRobotDataset,
                         preprocessor, logger: logging.Logger) -> dict:
    """追踪 DDIM 采样过程中每一步的 clip 行为"""
    log = logger
    print_sep("4. DDIM SAMPLING CLIP TRACING", log)

    results = {}

    base_model = policy._get_base_model()
    if hasattr(base_model, '_diffusion_initialized') and not base_model._diffusion_initialized:
        base_model._init_diffusion_head()

    num_inference = getattr(base_model, 'num_inference_timesteps',
                            getattr(config, 'num_inference_timesteps', 10))

    # 准备输入
    sample = dataset[0]
    preprocessed = preprocessor(sample)

    states = preprocessed[OBS_STATE].to(policy.config.device) if OBS_STATE in preprocessed else None
    image_keys = sorted([k for k in config.image_features if k in preprocessed])
    if not image_keys:
        image_keys = sorted([k for k in preprocessed if k.startswith("observation.images.")])
    images = [preprocessed[k].to(policy.config.device) for k in image_keys]

    processed_images = policy._process_images(images)
    num_cams = len(images)
    image_chunks = torch.chunk(processed_images, num_cams, dim=0)

    model_dtype = base_model.get_model().mm_projector[0].weight.dtype
    if states is not None:
        states = states.to(dtype=model_dtype)

    # 提取 hidden_states 和 states
    hidden_states, states = _extract_hidden_states(
        policy, base_model, preprocessed, config, image_chunks, num_cams
    )

    log.info(f"[Tracing DDIM Sampling Steps (clip_sample=True)]")
    log.info(f"  Total steps: {num_inference}")
    log.info(f"  hidden_states shape: {hidden_states.shape}")

    with torch.no_grad():
        trace_result = run_diffusion_sampling_with_tracing(
            base_model, hidden_states, states,
            num_inference_steps=num_inference, logger=log
        )

    results["trace"] = trace_result["trace"]
    results["final_action"] = tensor_stats(trace_result["final_action"].float(), "final_action")

    # 分析哪些步骤发生了 clip
    clipped_steps = [t for t in trace_result["trace"] if t["range_shrink"] > 0]
    log.info(f"\n[Clip Analysis]")
    log.info(f"  Total steps: {num_inference}")
    log.info(f"  Steps with clipping: {len(clipped_steps)}")
    log.info(f"  Clipped step indices: {[t['step'] for t in clipped_steps]}")

    if clipped_steps:
        total_shrink = sum(t["range_shrink"] for t in clipped_steps)
        log.info(f"  Total range shrink: {total_shrink:.4f}")
        log.info(f"  => Clipping is actively reducing action range during sampling")

    results["clipped_steps_count"] = len(clipped_steps)
    results["total_shrink"] = sum(t["range_shrink"] for t in clipped_steps) if clipped_steps else 0

    return results


# ============================================================================
# 5. 多帧统计: 量化 clip 对动作分布的系统性影响
# ============================================================================

def multi_frame_clipping_analysis(policy, config, dataset: LeRobotDataset,
                                   preprocessor, postprocessor, logger: logging.Logger) -> dict:
    """在多个样本上统计 clip 对动作分布的影响"""
    log = logger
    print_sep("5. MULTI-FRAME CLIPPING IMPACT ANALYSIS", log)

    results = {}

    base_model = policy._get_base_model()
    if hasattr(base_model, '_diffusion_initialized') and not base_model._diffusion_initialized:
        base_model._init_diffusion_head()

    num_inference = getattr(base_model, 'num_inference_timesteps',
                            getattr(config, 'num_inference_timesteps', 10))

    n_frames = min(50, dataset.num_frames)

    all_train_actions = []
    all_clipped_actions = []
    all_unclipped_actions = []

    log.info(f"[Analyzing {n_frames} samples]")

    for i in range(n_frames):
        if i % 10 == 0:
            log.info(f"  Processing sample {i}/{n_frames}...")

        sample = dataset[i]
        preprocessed = preprocessor(sample)

        if ACTION in preprocessed:
            train_act = preprocessed[ACTION].float()
            if train_act.ndim == 3:
                train_act = train_act[0]
            all_train_actions.append(train_act.cpu().numpy())

        states = preprocessed[OBS_STATE].to(policy.config.device) if OBS_STATE in preprocessed else None
        image_keys = sorted([k for k in config.image_features if k in preprocessed])
        if not image_keys:
            image_keys = sorted([k for k in preprocessed if k.startswith("observation.images.")])
        images = [preprocessed[k].to(policy.config.device) for k in image_keys]

        if len(images) == 0:
            continue

        processed_images = policy._process_images(images)
        num_cams = len(images)
        image_chunks = torch.chunk(processed_images, num_cams, dim=0)

        model_dtype = base_model.get_model().mm_projector[0].weight.dtype
        if states is not None:
            states = states.to(dtype=model_dtype)

        # 提取 hidden_states 和 states
        hidden_states, states = _extract_hidden_states(
            policy, base_model, preprocessed, config, image_chunks, num_cams
        )

        with torch.no_grad():
            actions_clipped = run_diffusion_sampling_with_clipping(
                base_model, hidden_states, states,
                clip_sample=True, num_inference_steps=num_inference
            )
            actions_unclipped = run_diffusion_sampling_with_clipping(
                base_model, hidden_states, states,
                clip_sample=False, num_inference_steps=num_inference
            )

        all_clipped_actions.append(actions_clipped.float().cpu().numpy())
        all_unclipped_actions.append(actions_unclipped.float().cpu().numpy())

    if all_train_actions and all_clipped_actions and all_unclipped_actions:
        train_np = np.concatenate(all_train_actions, axis=0)
        clipped_np = np.concatenate(all_clipped_actions, axis=0)
        unclipped_np = np.concatenate(all_unclipped_actions, axis=0)

        log.info(f"\n[Aggregate Statistics over {len(train_np)} action vectors]")
        log.info(f"  Training target:  mean={train_np.mean():.4f}, std={train_np.std():.4f}, "
                 f"range=[{train_np.min():.4f}, {train_np.max():.4f}]")
        log.info(f"  Inference (clip=True):  mean={clipped_np.mean():.4f}, std={clipped_np.std():.4f}, "
                 f"range=[{clipped_np.min():.4f}, {clipped_np.max():.4f}]")
        log.info(f"  Inference (clip=False): mean={unclipped_np.mean():.4f}, std={unclipped_np.std():.4f}, "
                 f"range=[{unclipped_np.min():.4f}, {unclipped_np.max():.4f}]")

        results["aggregate"] = {
            "n_actions": len(train_np),
            "training": {
                "mean": float(train_np.mean()),
                "std": float(train_np.std()),
                "min": float(train_np.min()),
                "max": float(train_np.max()),
                "range": float(train_np.max() - train_np.min()),
            },
            "inference_clipped": {
                "mean": float(clipped_np.mean()),
                "std": float(clipped_np.std()),
                "min": float(clipped_np.min()),
                "max": float(clipped_np.max()),
                "range": float(clipped_np.max() - clipped_np.min()),
            },
            "inference_unclipped": {
                "mean": float(unclipped_np.mean()),
                "std": float(unclipped_np.std()),
                "min": float(unclipped_np.min()),
                "max": float(unclipped_np.max()),
                "range": float(unclipped_np.max() - unclipped_np.min()),
            },
        }

        # 计算 clip 导致的分布压缩
        std_ratio_clipped = clipped_np.std() / train_np.std() if train_np.std() > 0 else 0
        std_ratio_unclipped = unclipped_np.std() / train_np.std() if train_np.std() > 0 else 0
        range_ratio_clipped = (clipped_np.max() - clipped_np.min()) / (train_np.max() - train_np.min()) if (train_np.max() - train_np.min()) > 0 else 0
        range_ratio_unclipped = (unclipped_np.max() - unclipped_np.min()) / (train_np.max() - train_np.min()) if (train_np.max() - train_np.min()) > 0 else 0

        log.info(f"\n[Distribution Compression Ratios (relative to training target)]")
        log.info(f"  Std ratio (clipped):    {std_ratio_clipped:.4f}")
        log.info(f"  Std ratio (unclipped):  {std_ratio_unclipped:.4f}")
        log.info(f"  Range ratio (clipped):    {range_ratio_clipped:.4f}")
        log.info(f"  Range ratio (unclipped):  {range_ratio_unclipped:.4f}")

        results["compression_ratios"] = {
            "std_ratio_clipped": std_ratio_clipped,
            "std_ratio_unclipped": std_ratio_unclipped,
            "range_ratio_clipped": range_ratio_clipped,
            "range_ratio_unclipped": range_ratio_unclipped,
        }

        if std_ratio_clipped < std_ratio_unclipped:
            log.info(f"\n  => CONFIRMED: clip_sample=True compresses the action distribution")
            log.info(f"  => Std compressed by {(1 - std_ratio_clipped) * 100:.1f}% relative to training target")
            log.info(f"  => Without clip, Std is {(std_ratio_unclipped) * 100:.1f}% of training target")

    return results


# ============================================================================
# MAIN
# ============================================================================

def run_full_diagnostic(policy_type: str, checkpoint_path: str, dataset_root: str,
                        device: str, logger: logging.Logger) -> dict:
    log = logger
    log.info(f"\n{'#'*70}")
    log.info(f"# TinyVLA clip_sample=True vs MEAN_STD Asymmetry Diagnostic")
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
            "action_dim": getattr(config, 'action_dim', None),
            "normalization_mapping": str(getattr(config, 'normalization_mapping', None)),
        },
    }

    # 1. 验证 normalized action 范围
    results["normalized_action_range"] = verify_normalized_action_range(
        dataset, preprocessor, ds_stats, log
    )

    # 2. 验证 scheduler 配置
    results["scheduler_config"] = verify_scheduler_config(policy, config, log)

    # 3. 对比 clip_sample=True vs False
    results["clipping_impact_comparison"] = compare_clipping_impact(
        policy, config, dataset, preprocessor, log
    )

    # 4. 追踪 DDIM 采样过程中的 clip 行为
    results["ddim_clip_tracing"] = trace_ddim_clipping(
        policy, config, dataset, preprocessor, log
    )

    # 5. 多帧统计分析
    results["multi_frame_analysis"] = multi_frame_clipping_analysis(
        policy, config, dataset, preprocessor, postprocessor, log
    )

    # 最终结论
    log.info(f"\n{'='*70}")
    log.info(f"  OVERALL CONCLUSION")
    log.info(f"{'='*70}")

    norm_range = results.get("normalized_action_range", {})
    scheduler = results.get("scheduler_config", {})
    comparison = results.get("clipping_impact_comparison", {})
    multi_frame = results.get("multi_frame_analysis", {})

    issues = []

    # 检查 normalized action 是否超出 [-1, 1]
    if norm_range.get("empirical_exceeds_clip_range", False):
        emp = norm_range.get("empirical_normalized_range", {})
        issues.append(
            f"CONFIRMED: Normalized action range [{emp.get('min', 'N/A')}, {emp.get('max', 'N/A')}] "
            f"exceeds [-1, 1]"
        )

    # 检查 clip_sample 是否启用
    if scheduler.get("clip_sample_enabled", False):
        issues.append("CONFIRMED: scheduler clip_sample=True is enabled")

    # 检查 clip 是否压缩了分布
    comp = comparison.get("comparison", {})
    if comp.get("clipped_is_smaller_std", False):
        issues.append(
            f"CONFIRMED: clip_sample=True reduces std from {comp['std_unclipped']:.4f} "
            f"to {comp['std_clipped']:.4f} (reduction: {(comp['std_difference']/comp['std_unclipped']*100):.1f}%)"
        )

    # 多帧统计确认
    compression = multi_frame.get("compression_ratios", {})
    if compression:
        issues.append(
            f"CONFIRMED: Over {multi_frame.get('aggregate', {}).get('n_actions', 0)} samples, "
            f"clip_sample=True compresses std to {compression.get('std_ratio_clipped', 0)*100:.1f}% "
            f"of training target"
        )

    if issues:
        log.info(f"\n  FOUND {len(issues)} CONFIRMED ISSUES:")
        for i, issue in enumerate(issues, 1):
            log.info(f"    {i}. {issue}")

        log.info(f"\n  RECOMMENDATION:")
        log.info(f"    Change clip_sample=True to clip_sample=False in _init_diffusion_head()")
        log.info(f"    Location: src/lerobot/policies/tinyvla/llava_pythia/model/language_model/pythia/llava_pythia.py")
        log.info(f"    Line: self.noise_scheduler = DDIMScheduler(..., clip_sample=True, ...)")
        log.info(f"    Change to: clip_sample=False")
    else:
        log.info(f"\n  No clip_sample asymmetry issues detected.")

    return results


def main():
    parser = argparse.ArgumentParser(description="TinyVLA clip_sample=True vs MEAN_STD Asymmetry Diagnostic")
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

    log_file = str(OUTPUT_DIR / "clip_sample_asymmetry.log")
    logger = setup_logging(log_file)

    logger.info(f"TinyVLA clip_sample=True vs MEAN_STD Asymmetry Diagnostic")
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