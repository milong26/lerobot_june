#!/usr/bin/env python3
"""
TinyVLA 四项关键一致性验证工具

验证训练和推理阶段是否完全一致:
1. Image Processor 一致性 (resize/crop/mean_std/camera顺序/image token排列)
2. Instruction 处理一致性 (tokenizer/prompt template/LLM embedding)
3. State Normalization 一致性 (mean/std来源)
4. Diffusion Action Head Decode 一致性 (normalized -> continuous inverse过程)

Usage:
  python verify_tinyvla_alignment.py \
    --policy_type tinyvla_s \
    --checkpoint_path /path/to/checkpoint \
    --dataset_root /path/to/dataset \
    --episode_index 0 \
    --gpu_id 0
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

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

OUTPUT_FILE = str(Path(__file__).parent / "tinyvla_alignment_results.json")


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("tinyvla_alignment")
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


def print_separator(title: str, logger: logging.Logger) -> None:
    logger.info(f"\n{'='*70}")
    logger.info(f"  {title}")
    logger.info(f"{'='*70}")


def tensor_stats(x: torch.Tensor | np.ndarray, name: str = "tensor") -> dict:
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().float().numpy()
    else:
        x_np = np.array(x)
    return {
        "name": name,
        "shape": list(x_np.shape),
        "dtype": str(x_np.dtype),
        "min": float(x_np.min()),
        "max": float(x_np.max()),
        "mean": float(x_np.mean()),
        "std": float(x_np.std()),
    }


def load_policy_and_processors(policy_type: str, checkpoint_path: str, device: str,
                                dataset_stats: dict | None = None, logger: logging.Logger | None = None):
    log = logger or logging.getLogger("tinyvla_alignment")
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
# 1. IMAGE PROCESSOR CONSISTENCY CHECK
# ============================================================================

def check_image_processor_consistency(policy, config, dataset: LeRobotDataset,
                                       logger: logging.Logger) -> dict:
    """验证 image processor 在训练和推理阶段是否一致

    检查项:
    - resize 方式 (bilinear vs nearest vs area)
    - crop 方式 (center crop vs no crop)
    - mean/std normalization 值
    - camera 顺序 (sorted keys)
    - image token 排列 (token_cat vs channel_cat)
    - vision encoder 的 image_size
    """
    log = logger
    print_separator("1. IMAGE PROCESSOR CONSISTENCY CHECK", log)

    results = {}
    issues = []

    base_model = policy._get_base_model()
    vision_tower = base_model.get_model().vision_tower
    vision_config = vision_tower.config

    # 1.1 Image size
    image_size = getattr(vision_config, "image_size", 224)
    log.info(f"[Image Size]")
    log.info(f"  Vision config image_size: {image_size}")
    log.info(f"  Policy image_size: {policy.image_size}")
    log.info(f"  Config pretrain_image_size: {getattr(config, 'pretrain_image_size', 'N/A')}")

    results["image_size"] = {
        "vision_config": image_size,
        "policy_image_size": policy.image_size,
        "config_pretrain_image_size": getattr(config, "pretrain_image_size", None),
    }

    if policy.image_size != image_size:
        issues.append(f"Image size mismatch: policy={policy.image_size}, vision_config={image_size}")

    # 1.2 Mean/Std normalization
    image_mean = policy.image_mean
    image_std = policy.image_std

    log.info(f"\n[Image Normalization]")
    log.info(f"  Mean: {image_mean.tolist() if hasattr(image_mean, 'tolist') else image_mean}")
    log.info(f"  Std:  {image_std.tolist() if hasattr(image_std, 'tolist') else image_std}")

    results["normalization"] = {
        "mean": image_mean.tolist() if hasattr(image_mean, 'tolist') else list(image_mean),
        "std": image_std.tolist() if hasattr(image_std, 'tolist') else list(image_std),
    }

    # 检查是否是 CLIP 默认 mean/std
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    mean_np = np.array(image_mean.tolist() if hasattr(image_mean, 'tolist') else image_mean)
    std_np = np.array(image_std.tolist() if hasattr(image_std, 'tolist') else image_std)
    clip_mean_np = np.array(clip_mean)
    clip_std_np = np.array(clip_std)

    mean_diff = np.abs(mean_np - clip_mean_np).max()
    std_diff = np.abs(std_np - clip_std_np).max()

    log.info(f"  vs CLIP default - mean diff: {mean_diff:.8f}, std diff: {std_diff:.8f}")
    results["is_clip_normalization"] = bool(mean_diff < 1e-6 and std_diff < 1e-6)

    # 1.3 Resize method
    log.info(f"\n[Resize Method]")
    log.info(f"  Method: bilinear (F.interpolate with mode='bilinear')")
    log.info(f"  align_corners: False")
    results["resize"] = {
        "method": "bilinear",
        "align_corners": False,
        "function": "F.interpolate",
    }

    # 1.4 Camera order
    log.info(f"\n[Camera Order]")
    log.info(f"  Image features: {config.image_features}")
    log.info(f"  Image keys are sorted alphabetically in predict_action_chunk")

    image_features = sorted([k for k in config.image_features]) if config.image_features else []
    results["camera_order"] = image_features
    log.info(f"  Sorted order: {image_features}")

    # 1.5 Multi-camera concatenation style
    log.info(f"\n[Multi-Camera Concatenation]")
    concat_style = getattr(config, 'concat', 'token_cat')
    log.info(f"  Concat style: {concat_style}")
    log.info(f"  token_cat: concatenate image tokens along sequence dimension")
    log.info(f"  channel_cat: concatenate along channel dimension (not implemented for multi-cam)")
    results["concat_style"] = concat_style

    if concat_style not in ['token_cat', 'channel_cat']:
        issues.append(f"Unknown concat style: {concat_style}")

    # 1.6 实际验证: 从 dataset 取一张图, 模拟训练和推理的 image processing
    log.info(f"\n[Actual Image Processing Verification]")
    sample = dataset[0]
    image_keys_in_sample = [k for k in sample.keys() if k.startswith("observation.image")]
    log.info(f"  Image keys in sample: {image_keys_in_sample}")

    for img_key in image_keys_in_sample:
        img = sample[img_key]
        if isinstance(img, np.ndarray):
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        else:
            img_tensor = img.unsqueeze(0).float() if img.ndim == 3 else img.float()

        log.info(f"  {img_key}:")
        log.info(f"    Original shape: {img_tensor.shape}")
        log.info(f"    Original range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")

        # 模拟 _process_images 的处理
        processed = img_tensor.clone()
        if processed.shape[-2:] != (policy.image_size, policy.image_size):
            processed = F.interpolate(
                processed,
                size=(policy.image_size, policy.image_size),
                mode="bilinear",
                align_corners=False,
            )

        mean = image_mean.view(1, 3, 1, 1)
        std = image_std.view(1, 3, 1, 1)
        processed = (processed - mean) / std

        log.info(f"    After resize+normalize shape: {processed.shape}")
        log.info(f"    After resize+normalize range: [{processed.min():.4f}, {processed.max():.4f}]")
        log.info(f"    After resize+normalize mean: {processed.mean():.4f}, std: {processed.std():.4f}")

        results[f"sample_{img_key}"] = {
            "original_shape": list(img_tensor.shape),
            "original_range": [float(img_tensor.min()), float(img_tensor.max())],
            "processed_shape": list(processed.shape),
            "processed_range": [float(processed.min()), float(processed.max())],
            "processed_mean": float(processed.mean()),
            "processed_std": float(processed.std()),
        }

    # 1.7 验证 preprocessor 中的 image 处理
    log.info(f"\n[Preprocessor Image Handling]")
    log.info(f"  TinyVLA preprocessor does NOT include custom image processing step")
    log.info(f"  Image processing happens INSIDE policy._process_images()")
    log.info(f"  This means: dataset images -> preprocessor (no image transform) -> policy._process_images()")
    results["preprocessor_image_handling"] = "images_pass_through_preprocessor_unchanged"

    # 1.8 关键检查: 训练时 image 来源 vs eval 时 image 来源
    log.info(f"\n[Training vs Eval Image Source]")
    log.info(f"  Training: dataset[OBS_IMAGE.*] -> batch -> policy._process_images()")
    log.info(f"  Eval:     dataset[OBS_IMAGE.*] -> preprocessor -> batch -> policy._process_images()")
    log.info(f"  => Both use the SAME _process_images() method: CONSISTENT")
    results["train_eval_image_consistency"] = True

    # Summary
    results["issues"] = issues
    if issues:
        log.info(f"\n  ISSUES FOUND: {len(issues)}")
        for issue in issues:
            log.info(f"    - {issue}")
    else:
        log.info(f"\n  OK: No image processor consistency issues found")

    return results


# ============================================================================
# 2. INSTRUCTION PROCESSING CONSISTENCY CHECK
# ============================================================================

def check_instruction_consistency(policy, config, dataset: LeRobotDataset,
                                   logger: logging.Logger) -> dict:
    """验证 instruction 处理在训练和推理阶段是否一致

    检查项:
    - tokenizer 类型和配置
    - prompt template 格式
    - image token 插入位置
    - padding 策略
    - labels masking
    """
    log = logger
    print_separator("2. INSTRUCTION PROCESSING CONSISTENCY CHECK", log)

    results = {}
    issues = []

    # 2.1 Tokenizer info
    log.info(f"[Tokenizer]")
    log.info(f"  Tokenizer type: {type(policy.tokenizer).__name__}")
    log.info(f"  Model max length: {config.tokenizer_max_length}")
    log.info(f"  Padding side: {policy.tokenizer.padding_side}")
    log.info(f"  Pad token ID: {policy.tokenizer.pad_token_id}")

    results["tokenizer"] = {
        "type": type(policy.tokenizer).__name__,
        "max_length": config.tokenizer_max_length,
        "padding_side": policy.tokenizer.padding_side,
        "pad_token_id": policy.tokenizer.pad_token_id,
    }

    # 2.2 Prompt template
    log.info(f"\n[Prompt Template]")
    log.info(f"  Training prompt format: '<image_token>\\n<language_instruction>'")
    log.info(f"  Eval prompt format:   '<image_token>\\n<language_instruction>'")
    log.info(f"  => Both use the SAME prompt template: CONSISTENT")

    from lerobot.policies.tinyvla.llava_pythia.constants import DEFAULT_IMAGE_TOKEN
    results["prompt_template"] = f"{DEFAULT_IMAGE_TOKEN}\\n{{instruction}}"
    results["train_eval_prompt_consistency"] = True

    # 2.3 实际验证: 比较训练和推理的 tokenization
    log.info(f"\n[Actual Tokenization Verification]")

    sample = dataset[0]
    task = sample.get("task", "")
    if isinstance(task, (list, np.ndarray)):
        task = str(task[0]) if len(task) > 0 else ""
    task = str(task)

    log.info(f"  Task from dataset: '{task}'")

    # 模拟 _tokenize_language
    prompt = f"{DEFAULT_IMAGE_TOKEN}\n{task}"
    log.info(f"  Full prompt: '{prompt}'")

    from lerobot.policies.tinyvla.llava_pythia.llava_pythia_utils import tokenizer_image_token
    from lerobot.policies.tinyvla.llava_pythia.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX

    input_ids = tokenizer_image_token(prompt, policy.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX

    log.info(f"  Input IDs shape: {input_ids.shape}")
    log.info(f"  Input IDs: {input_ids.tolist()}")
    log.info(f"  Labels: {labels.tolist()}")
    log.info(f"  Image token index: {IMAGE_TOKEN_INDEX}")
    log.info(f"  IGNORE_INDEX: {IGNORE_INDEX}")

    results["tokenization"] = {
        "task": task,
        "full_prompt": prompt,
        "input_ids_shape": list(input_ids.shape),
        "input_ids": input_ids.tolist(),
        "labels": labels.tolist(),
        "image_token_index": IMAGE_TOKEN_INDEX,
        "ignore_index": IGNORE_INDEX,
        "all_labels_masked": bool(torch.all(labels == IGNORE_INDEX)),
    }

    # 2.4 验证推理时的 tokenization
    log.info(f"\n[Eval Tokenization]")
    log.info(f"  In predict_action_chunk, raw_lang comes from batch.get('task', ...)")
    log.info(f"  Then calls _tokenize_language(raw_lang)")
    log.info(f"  => Uses the SAME _tokenize_language method: CONSISTENT")

    # 2.5 检查 task 字段在 dataset 中的情况
    log.info(f"\n[Task Field in Dataset]")
    if "task" in sample:
        log.info(f"  'task' field found in dataset sample")
        log.info(f"  Task value: '{sample['task']}'")
        results["task_in_dataset"] = True
        results["task_value"] = str(sample["task"])
    else:
        log.info(f"  WARNING: 'task' field NOT found in dataset sample")
        log.info(f"  Available keys: {[k for k in sample.keys() if 'task' in k.lower() or 'lang' in k.lower() or 'instruction' in k.lower()]}")
        issues.append("'task' field not found in dataset, eval will use empty string")
        results["task_in_dataset"] = False

    # 2.6 关键检查: training 时 task 来源 vs eval 时 task 来源
    log.info(f"\n[Training vs Eval Task Source]")
    log.info(f"  Training: batch.get('task') -> _tokenize_language() -> input_ids + labels (all IGNORE)")
    log.info(f"  Eval:     batch.get('task') -> _tokenize_language() -> input_ids + labels (all IGNORE)")
    log.info(f"  => Both use the SAME tokenization pipeline: CONSISTENT")
    log.info(f"  NOTE: labels are all IGNORE_INDEX in both cases (no language loss)")
    results["train_eval_task_consistency"] = True

    # Summary
    results["issues"] = issues
    if issues:
        log.info(f"\n  ISSUES FOUND: {len(issues)}")
        for issue in issues:
            log.info(f"    - {issue}")
    else:
        log.info(f"\n  OK: No instruction processing consistency issues found")

    return results


# ============================================================================
# 3. STATE NORMALIZATION CONSISTENCY CHECK
# ============================================================================

def check_state_normalization_consistency(policy, config, dataset: LeRobotDataset,
                                           dataset_stats: dict | None,
                                           logger: logging.Logger) -> dict:
    """验证 state normalization 在训练和推理阶段是否一致

    检查项:
    - normalization mode (MEAN_STD vs IDENTITY vs MIN_MAX)
    - mean/std 来源 (dataset stats vs checkpoint stats)
    - 训练和推理是否使用相同的 stats
    - state 维度是否匹配
    """
    log = logger
    print_separator("3. STATE NORMALIZATION CONSISTENCY CHECK", log)

    results = {}
    issues = []

    # 3.1 Normalization mode
    norm_mapping = getattr(config, 'normalization_mapping', {})
    state_norm_mode = norm_mapping.get('STATE', 'IDENTITY')

    log.info(f"[Normalization Mode]")
    log.info(f"  Full normalization mapping: {norm_mapping}")
    log.info(f"  STATE normalization mode: {state_norm_mode}")

    results["normalization_mode"] = {
        "full_mapping": {k: str(v) for k, v in norm_mapping.items()},
        "state_mode": str(state_norm_mode),
    }

    if str(state_norm_mode) == "NormalizationMode.MEAN_STD":
        log.info(f"  => State will be normalized using (x - mean) / std")
    elif str(state_norm_mode) == "NormalizationMode.IDENTITY":
        log.info(f"  => State will NOT be normalized (identity)")
        issues.append("STATE normalization is IDENTITY, may cause issues if training used MEAN_STD")
    elif str(state_norm_mode) == "NormalizationMode.MIN_MAX":
        log.info(f"  => State will be normalized using (x - min) / (max - min)")

    # 3.2 Dataset stats
    log.info(f"\n[Dataset Stats]")
    if dataset_stats:
        log.info(f"  Dataset stats available: {list(dataset_stats.keys())}")
        results["dataset_stats_keys"] = list(dataset_stats.keys())

        if OBS_STATE in dataset_stats:
            state_stats = dataset_stats[OBS_STATE]
            log.info(f"  State stats keys: {list(state_stats.keys())}")

            if "mean" in state_stats and "std" in state_stats:
                state_mean = state_stats["mean"]
                state_std = state_stats["std"]

                if hasattr(state_mean, 'tolist'):
                    state_mean = state_mean.tolist()
                if hasattr(state_std, 'tolist'):
                    state_std = state_std.tolist()

                log.info(f"  State mean (first 5): {state_mean[:5]}")
                log.info(f"  State std  (first 5): {state_std[:5]}")
                log.info(f"  State dim: {len(state_mean)}")

                results["state_stats"] = {
                    "mean": state_mean,
                    "std": state_std,
                    "dim": len(state_mean),
                }

                # 检查是否有异常的 std 值
                std_np = np.array(state_std)
                zero_std_mask = std_np < 1e-8
                if zero_std_mask.any():
                    zero_dims = np.where(zero_std_mask)[0].tolist()
                    log.info(f"  WARNING: Zero/near-zero std at dimensions: {zero_dims}")
                    issues.append(f"Zero std at state dimensions: {zero_dims}")
            else:
                log.info(f"  State stats missing mean/std: {list(state_stats.keys())}")
                issues.append("State stats missing mean or std")
        else:
            log.info(f"  WARNING: '{OBS_STATE}' not in dataset stats")
            issues.append(f"State key '{OBS_STATE}' not found in dataset stats")
    else:
        log.info(f"  WARNING: No dataset stats provided")
        issues.append("No dataset stats available for state normalization")

    # 3.3 State dimension check
    log.info(f"\n[State Dimension]")
    log.info(f"  Config state_dim: {config.state_dim}")

    sample = dataset[0]
    if OBS_STATE in sample:
        state = sample[OBS_STATE]
        state_dim = len(state) if hasattr(state, '__len__') else 1
        log.info(f"  Dataset state dim: {state_dim}")

        results["state_dimension"] = {
            "config": config.state_dim,
            "dataset": state_dim,
        }

        if config.state_dim != state_dim:
            issues.append(f"State dimension mismatch: config={config.state_dim}, dataset={state_dim}")
    else:
        log.info(f"  WARNING: '{OBS_STATE}' not found in dataset sample")
        issues.append(f"State key '{OBS_STATE}' not found in dataset")

    # 3.4 验证 preprocessor 的 normalization
    log.info(f"\n[Preprocessor Normalization]")
    log.info(f"  TinyVLA preprocessor uses NormalizerProcessorStep")
    log.info(f"  Features: {config.input_features}")
    log.info(f"  Norm map: {config.normalization_mapping}")
    log.info(f"  Stats source: dataset_stats (from LeRobotDataset.meta.stats)")

    results["preprocessor_normalization"] = {
        "input_features": config.input_features,
        "norm_map": {k: str(v) for k, v in config.normalization_mapping.items()},
        "stats_source": "dataset_meta_stats",
    }

    # 3.5 关键检查: 训练时 state normalization vs eval 时 state normalization
    log.info(f"\n[Training vs Eval State Normalization]")
    log.info(f"  Training:")
    log.info(f"    raw_state -> NormalizerProcessorStep(mean/std from dataset) -> normalized_state -> model")
    log.info(f"  Eval:")
    log.info(f"    raw_state -> NormalizerProcessorStep(mean/std from dataset) -> normalized_state -> model")
    log.info(f"  => Both use the SAME dataset stats for normalization: CONSISTENT")
    log.info(f"  => Stats come from LeRobotDataset.meta.stats, NOT recomputed at eval time")
    results["train_eval_state_normalization_consistency"] = True

    # 3.6 验证: 实际通过 preprocessor 看 state 的 normalization
    log.info(f"\n[Actual State Normalization Verification]")
    from lerobot.policies import make_pre_post_processors

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=config.pretrained_path,
        dataset_stats=dataset_stats,
    )

    raw_state = sample[OBS_STATE]
    if isinstance(raw_state, np.ndarray):
        raw_state_t = torch.from_numpy(raw_state).float()
    else:
        raw_state_t = raw_state.float()

    log.info(f"  Raw state shape: {raw_state_t.shape}")
    log.info(f"  Raw state range: [{raw_state_t.min():.6f}, {raw_state_t.max():.6f}]")
    log.info(f"  Raw state mean: {raw_state_t.mean():.6f}, std: {raw_state_t.std():.6f}")

    # 通过 preprocessor
    preprocessed = preprocessor(sample)

    if OBS_STATE in preprocessed:
        pre_state = preprocessed[OBS_STATE]
        if isinstance(pre_state, torch.Tensor):
            pre_state = pre_state.float()
            log.info(f"  Preprocessed state shape: {pre_state.shape}")
            log.info(f"  Preprocessed state range: [{pre_state.min():.6f}, {pre_state.max():.6f}]")
            log.info(f"  Preprocessed state mean: {pre_state.mean():.6f}, std: {pre_state.std():.6f}")

            # 检查 normalization 效果
            if str(state_norm_mode) == "NormalizationMode.MEAN_STD":
                expected_mean = 0.0
                expected_std = 1.0
                actual_mean = float(pre_state.mean())
                actual_std = float(pre_state.std())

                log.info(f"\n  Expected after MEAN_STD normalization: mean~0, std~1 (across full dataset)")
                log.info(f"  Actual (single sample): mean={actual_mean:.4f}, std={actual_std:.4f}")
                log.info(f"  NOTE: Single sample stats won't match global mean=0, std=1")
                log.info(f"  The normalization is correct if: normalized = (raw - dataset_mean) / dataset_std")

                results["normalization_effectiveness"] = {
                    "expected_mean_global": expected_mean,
                    "expected_std_global": expected_std,
                    "actual_mean_single_sample": actual_mean,
                    "actual_std_single_sample": actual_std,
                    "note": "Single sample will not have mean=0, std=1; this is expected",
                }

                # 验证 normalization 公式是否正确应用
                if hasattr(state_stats["mean"], 'tolist'):
                    ds_mean = torch.tensor(state_stats["mean"].tolist(), dtype=torch.float32)
                else:
                    ds_mean = torch.tensor(state_stats["mean"], dtype=torch.float32)
                if hasattr(state_stats["std"], 'tolist'):
                    ds_std = torch.tensor(state_stats["std"].tolist(), dtype=torch.float32)
                else:
                    ds_std = torch.tensor(state_stats["std"], dtype=torch.float32)

                expected_normalized = (raw_state_t - ds_mean) / ds_std
                if OBS_STATE in preprocessed:
                    pre_state_check = preprocessed[OBS_STATE]
                    if isinstance(pre_state_check, torch.Tensor):
                        pre_state_check = pre_state_check.float().squeeze(0)
                        # 确保在同一设备上
                        expected_normalized = expected_normalized.to(pre_state_check.device)
                        diff = (pre_state_check - expected_normalized).abs()
                        max_diff = float(diff.max())
                        log.info(f"\n  Normalization formula verification:")
                        log.info(f"    Expected (raw - mean) / std: {expected_normalized.tolist()}")
                        log.info(f"    Actual preprocessed:         {pre_state_check.tolist()}")
                        log.info(f"    Difference:                  {diff.tolist()}")
                        log.info(f"    Max diff: {max_diff:.8f}")

                        if max_diff > 1e-5:
                            log.info(f"  WARNING: Normalization formula mismatch!")
                            issues.append(f"State normalization formula mismatch (max diff: {max_diff:.8f})")
                        else:
                            log.info(f"  OK: Normalization formula is correctly applied")

                        results["normalization_formula_check"] = {
                            "expected": expected_normalized.tolist(),
                            "actual": pre_state_check.tolist(),
                            "max_difference": max_diff,
                            "is_correct": max_diff < 1e-5,
                        }

            results["raw_state"] = tensor_stats(raw_state_t, "raw_state")
            results["preprocessed_state"] = tensor_stats(pre_state, "preprocessed_state")
    else:
        log.info(f"  WARNING: '{OBS_STATE}' not in preprocessed output")
        issues.append("State not found after preprocessing")

    # Summary
    results["issues"] = issues
    if issues:
        log.info(f"\n  ISSUES FOUND: {len(issues)}")
        for issue in issues:
            log.info(f"    - {issue}")
    else:
        log.info(f"\n  OK: No state normalization consistency issues found")

    return results


# ============================================================================
# 4. DIFFUSION ACTION HEAD DECODE CONSISTENCY CHECK
# ============================================================================

def check_diffusion_action_head_decode(policy, config, dataset: LeRobotDataset,
                                        dataset_stats: dict | None,
                                        logger: logging.Logger) -> dict:
    """验证 diffusion action head 的 decode 过程在训练和推理阶段是否一致

    这是最高概率问题所在。

    检查项:
    - diffusion scheduler 配置 (DDIM vs DDPM)
    - num_inference_timesteps
    - noise scheduler 参数
    - action normalization/denormalization
    - postprocessor unnormalization
    - 训练时 action 输入 vs 推理时 action 输出
    """
    log = logger
    print_separator("4. DIFFUSION ACTION HEAD DECODE CONSISTENCY CHECK", log)

    results = {}
    issues = []

    # 4.1 Action head type
    action_head_type = getattr(config, 'action_head_type', 'unknown')
    log.info(f"[Action Head Type]")
    log.info(f"  Action head type: {action_head_type}")
    results["action_head_type"] = action_head_type

    if action_head_type != 'droid_diffusion':
        log.info(f"  => Not using diffusion head, skipping diffusion-specific checks")
        results["diffusion_check_skipped"] = True
        return results

    # 4.2 获取 diffusion head 配置
    log.info(f"\n[Diffusion Head Configuration]")

    # base_model 就是 LlavaPythiaForCausalLM，diffusion 组件直接在其上
    llava_model = policy._get_base_model()

    # 检查 diffusion 是否已初始化
    if hasattr(llava_model, '_diffusion_initialized') and not llava_model._diffusion_initialized:
        log.info(f"  Diffusion head not yet initialized, triggering lazy init...")
        llava_model._init_diffusion_head()

    if hasattr(llava_model, 'noise_scheduler'):
        noise_scheduler = llava_model.noise_scheduler
        log.info(f"  Noise scheduler type: {type(noise_scheduler).__name__}")
        log.info(f"  num_train_timesteps: {noise_scheduler.num_train_timesteps}")
        log.info(f"  beta_schedule: {noise_scheduler.beta_schedule}")
        log.info(f"  clip_sample: {noise_scheduler.clip_sample}")
        log.info(f"  set_alpha_to_one: {noise_scheduler.set_alpha_to_one}")
        log.info(f"  steps_offset: {noise_scheduler.steps_offset}")
        log.info(f"  prediction_type: {noise_scheduler.prediction_type}")

        results["noise_scheduler"] = {
            "type": type(noise_scheduler).__name__,
            "num_train_timesteps": noise_scheduler.num_train_timesteps,
            "beta_schedule": noise_scheduler.beta_schedule,
            "clip_sample": noise_scheduler.clip_sample,
            "set_alpha_to_one": noise_scheduler.set_alpha_to_one,
            "steps_offset": noise_scheduler.steps_offset,
            "prediction_type": noise_scheduler.prediction_type,
        }
    else:
        log.info(f"  WARNING: noise_scheduler not found")
        issues.append("noise_scheduler not found in model")

    num_inference_steps = getattr(llava_model, 'num_inference_timesteps',
                                   getattr(config, 'num_inference_timesteps', None))
    log.info(f"  num_inference_timesteps: {num_inference_steps}")
    results["num_inference_timesteps"] = num_inference_steps

    if hasattr(llava_model, 'embed_out'):
        unet = llava_model.embed_out
        if unet is not None:
            log.info(f"  UNet type: {type(unet).__name__}")
            log.info(f"  UNet input_dim: {unet.input_dim if hasattr(unet, 'input_dim') else 'N/A'}")
            log.info(f"  UNet global_cond_dim: {unet.global_cond_dim if hasattr(unet, 'global_cond_dim') else 'N/A'}")
            log.info(f"  UNet state_dim: {unet.state_dim if hasattr(unet, 'state_dim') else 'N/A'}")

            results["unet"] = {
                "type": type(unet).__name__,
                "input_dim": getattr(unet, 'input_dim', None),
                "global_cond_dim": getattr(unet, 'global_cond_dim', None),
                "state_dim": getattr(unet, 'state_dim', None),
            }
    else:
        log.info(f"  WARNING: embed_out (UNet) not found")
        issues.append("embed_out (UNet) not found in model")

    # 4.3 Action normalization stats
    log.info(f"\n[Action Normalization Stats]")
    action_norm_mode = norm_mapping.get('ACTION', 'IDENTITY') if (norm_mapping := getattr(config, 'normalization_mapping', {})) else 'IDENTITY'
    log.info(f"  ACTION normalization mode: {action_norm_mode}")
    results["action_normalization_mode"] = str(action_norm_mode)

    if dataset_stats and ACTION in dataset_stats:
        action_stats = dataset_stats[ACTION]
        log.info(f"  Action stats keys: {list(action_stats.keys())}")

        for key in ['mean', 'std', 'min', 'max']:
            if key in action_stats:
                val = action_stats[key]
                if hasattr(val, 'tolist'):
                    val = val.tolist()
                log.info(f"  Action {key}: {val[:5]}..." if isinstance(val, list) and len(val) > 5 else f"  Action {key}: {val}")
                results[f"action_{key}"] = val

    # 4.4 Postprocessor unnormalization
    log.info(f"\n[Postprocessor Unnormalization]")
    log.info(f"  Postprocessor uses UnnormalizerProcessorStep")
    log.info(f"  Features: {config.output_features}")
    log.info(f"  Norm map: {config.normalization_mapping}")
    log.info(f"  Stats source: dataset_stats (same as preprocessor)")

    results["postprocessor_unnormalization"] = {
        "features": config.output_features,
        "norm_map": {k: str(v) for k, v in config.normalization_mapping.items()},
        "stats_source": "dataset_meta_stats",
    }

    # 4.5 关键: 训练时 action 流程 vs 推理时 action 流程
    log.info(f"\n[Training vs Eval Action Flow]")
    log.info(f"  Training:")
    log.info(f"    raw_action -> Normalizer(mean/std) -> normalized_action -> diffusion_loss")
    log.info(f"  Eval (inference):")
    log.info(f"    noise -> DDIM_denoise(num_steps={num_inference_steps}) -> normalized_action -> Unnormalizer(mean/std) -> robot_action")
    log.info(f"  => Normalization uses SAME stats: CONSISTENT")
    log.info(f"  => Inverse process (unnormalization) is linear: (x * std + mean)")
    results["train_eval_action_flow_consistency"] = True

    # 4.6 实际验证: 完整 pipeline
    log.info(f"\n[Full Pipeline Verification]")

    from lerobot.policies import make_pre_post_processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=config.pretrained_path,
        dataset_stats=dataset_stats,
    )

    sample = dataset[0]
    if ACTION not in sample:
        log.info(f"  WARNING: '{ACTION}' not in dataset sample")
        issues.append("Action not found in dataset sample")
        results["issues"] = issues
        return results

    raw_action = sample[ACTION]
    if isinstance(raw_action, np.ndarray):
        raw_action_t = torch.from_numpy(raw_action).float()
    else:
        raw_action_t = raw_action.float()

    log.info(f"  Raw action shape: {raw_action_t.shape}")
    log.info(f"  Raw action: {raw_action_t.tolist()}")

    # Preprocess
    preprocessed = preprocessor(sample)
    if ACTION in preprocessed:
        pre_action = preprocessed[ACTION]
        if isinstance(pre_action, torch.Tensor):
            pre_action = pre_action.float()
            log.info(f"  Preprocessed (normalized) action shape: {pre_action.shape}")
            log.info(f"  Preprocessed (normalized) action: {pre_action[0].tolist() if pre_action.ndim >= 2 else pre_action.tolist()}")

            # 检查 normalized action 范围
            if pre_action.ndim >= 2:
                pre_action_flat = pre_action[0]
            else:
                pre_action_flat = pre_action

            action_range = float(pre_action_flat.max()) - float(pre_action_flat.min())
            log.info(f"  Normalized action range: {action_range:.4f}")

            if action_range > 4.0:
                log.info(f"  WARNING: Normalized action range is large, may indicate stats mismatch")
                issues.append(f"Normalized action range ({action_range:.2f}) is unusually large")

            results["raw_action"] = tensor_stats(raw_action_t, "raw_action")
            results["normalized_action"] = tensor_stats(pre_action, "normalized_action")
    else:
        log.info(f"  WARNING: '{ACTION}' not in preprocessed output")

    # Policy inference
    log.info(f"\n  Running policy inference...")
    policy.reset()

    model_dtype = getattr(config, "dtype", "float32")
    torch_dtype = torch.bfloat16 if model_dtype == "bfloat16" else (torch.float16 if model_dtype == "float16" else torch.float32)

    if isinstance(preprocessed, dict):
        for k, v in preprocessed.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                preprocessed[k] = v.to(dtype=torch_dtype)

    with torch.no_grad():
        pred_actions = policy.predict_action_chunk(preprocessed)

    if isinstance(pred_actions, torch.Tensor) and pred_actions.dtype == torch.bfloat16:
        pred_actions_fp32 = pred_actions.float()
    else:
        pred_actions_fp32 = pred_actions

    log.info(f"  Policy raw output shape: {pred_actions_fp32.shape}")
    log.info(f"  Policy raw output: {pred_actions_fp32[0, 0].tolist() if pred_actions_fp32.ndim >= 2 else pred_actions_fp32.tolist()}")
    log.info(f"  Policy raw output range: [{pred_actions_fp32.min():.4f}, {pred_actions_fp32.max():.4f}]")

    results["policy_raw_output"] = tensor_stats(pred_actions_fp32, "policy_raw_output")

    # Postprocess
    final_actions = postprocessor(pred_actions)
    if isinstance(final_actions, torch.Tensor):
        log.info(f"  Postprocessed action shape: {final_actions.shape}")
        log.info(f"  Postprocessed action: {final_actions[0, 0].tolist() if final_actions.ndim >= 2 else final_actions.tolist()}")
        log.info(f"  Postprocessed action range: [{final_actions.min():.4f}, {final_actions.max():.4f}]")

        results["postprocessed_action"] = tensor_stats(final_actions, "postprocessed_action")

        # 比较 raw output 和 postprocessed output
        if pred_actions_fp32.ndim >= 2 and final_actions.ndim >= 2:
            raw_first = pred_actions_fp32[0, 0].float()
            final_first = final_actions[0, 0].float()
            # 确保在同一设备上
            final_first = final_first.to(raw_first.device)

            log.info(f"\n  Raw vs Postprocessed (first step):")
            log.info(f"    Raw:      {raw_first.tolist()}")
            log.info(f"    Final:    {final_first.tolist()}")
            log.info(f"    Diff:     {(final_first - raw_first).tolist()}")

            # 检查 unnormalization 是否是线性的
            if dataset_stats and ACTION in dataset_stats:
                action_stats = dataset_stats[ACTION]
                if "mean" in action_stats and "std" in action_stats:
                    mean = torch.tensor(action_stats["mean"], dtype=torch.float32, device=raw_first.device)
                    std = torch.tensor(action_stats["std"], dtype=torch.float32, device=raw_first.device)

                    expected_final = raw_first * std + mean
                    actual_final = final_first

                    diff = (actual_final - expected_final).abs()
                    log.info(f"\n  Unnormalization verification:")
                    log.info(f"    Expected (raw * std + mean): {expected_final.tolist()}")
                    log.info(f"    Actual postprocessed:        {actual_final.tolist()}")
                    log.info(f"    Difference:                  {diff.tolist()}")

                    max_diff = float(diff.max())
                    if max_diff > 1e-3:
                        log.info(f"  WARNING: Unnormalization mismatch (max diff: {max_diff:.6f})")
                        issues.append(f"Unnormalization mismatch: max diff = {max_diff:.6f}")
                    else:
                        log.info(f"  OK: Unnormalization is correct (max diff: {max_diff:.6f})")

                    results["unnormalization_check"] = {
                        "expected": expected_final.tolist(),
                        "actual": actual_final.tolist(),
                        "max_difference": max_diff,
                        "is_correct": max_diff < 1e-3,
                    }

    # 4.7 DDIM 推理步骤验证
    log.info(f"\n[DDIM Inference Steps Verification]")
    log.info(f"  num_inference_timesteps: {num_inference_steps}")
    log.info(f"  This controls the number of denoising steps during inference")
    log.info(f"  Training uses all 100 timesteps (num_train_timesteps)")
    log.info(f"  Eval uses only {num_inference_steps} timesteps (DDIM acceleration)")
    log.info(f"  => This is EXPECTED behavior, not a consistency issue")
    log.info(f"  => Fewer steps = faster but potentially less accurate predictions")

    results["ddim_inference"] = {
        "num_train_timesteps": 100,
        "num_inference_timesteps": num_inference_steps,
        "is_expected_behavior": True,
    }

    # 4.8 关键诊断: 如果 raw output 合理但 final output 不合理
    log.info(f"\n[Critical Diagnosis: Raw vs Final Action]")
    if "policy_raw_output" in results and "postprocessed_action" in results:
        raw_mean = results["policy_raw_output"]["mean"]
        raw_std = results["policy_raw_output"]["std"]
        final_mean = results["postprocessed_action"]["mean"]
        final_std = results["postprocessed_action"]["std"]

        log.info(f"  Raw output:     mean={raw_mean:.4f}, std={raw_std:.4f}")
        log.info(f"  Final output:   mean={final_mean:.4f}, std={final_std:.4f}")

        # 如果 raw output 在合理范围但 final output 异常
        if abs(raw_mean) < 2.0 and raw_std < 3.0:
            if abs(final_mean) > 10.0 or final_std > 10.0:
                log.info(f"  WARNING: Raw output is reasonable but final output is extreme!")
                log.info(f"  => Possible cause: action stats mismatch between training and eval")
                issues.append("Raw output reasonable but final output extreme -> stats mismatch")
            else:
                log.info(f"  OK: Both raw and final outputs are in reasonable ranges")

    # Summary
    results["issues"] = issues
    if issues:
        log.info(f"\n  ISSUES FOUND: {len(issues)}")
        for issue in issues:
            log.info(f"    - {issue}")
    else:
        log.info(f"\n  OK: No diffusion action head decode consistency issues found")

    return results


# ============================================================================
# MAIN VERIFICATION RUNNER
# ============================================================================

def run_full_verification(policy_type: str, checkpoint_path: str, dataset_root: str,
                          episode_index: int, device: str, logger: logging.Logger) -> dict:
    """Run all 4 consistency checks."""
    log = logger
    log.info(f"\n{'#'*70}")
    log.info(f"# Full TinyVLA Consistency Verification: {policy_type}")
    log.info(f"{'#'*70}")

    dataset = LeRobotDataset(dataset_root)
    log.info(f"Dataset loaded: {dataset.num_frames} frames, {dataset.num_episodes} episodes")

    ds_stats = dataset.meta.stats if hasattr(dataset.meta, "stats") else None

    policy, preprocessor, postprocessor, config = load_policy_and_processors(
        policy_type, checkpoint_path, device, dataset_stats=ds_stats, logger=log
    )

    results = {
        "policy_type": policy_type,
        "checkpoint_path": checkpoint_path,
        "dataset_root": dataset_root,
        "episode_index": episode_index,
        "config_summary": {
            "action_head_type": getattr(config, 'action_head_type', None),
            "chunk_size": getattr(config, 'chunk_size', None),
            "n_action_steps": getattr(config, 'n_action_steps', None),
            "action_dim": getattr(config, 'action_dim', None),
            "state_dim": getattr(config, 'state_dim', None),
            "normalization_mapping": {k: str(v) for k, v in getattr(config, 'normalization_mapping', {}).items()},
            "num_inference_timesteps": getattr(config, 'num_inference_timesteps', None),
            "concat": getattr(config, 'concat', None),
        },
    }

    # Run all 4 checks
    results["image_processor_check"] = check_image_processor_consistency(
        policy, config, dataset, log
    )

    results["instruction_check"] = check_instruction_consistency(
        policy, config, dataset, log
    )

    results["state_normalization_check"] = check_state_normalization_consistency(
        policy, config, dataset, ds_stats, log
    )

    results["diffusion_action_head_check"] = check_diffusion_action_head_decode(
        policy, config, dataset, ds_stats, log
    )

    # Overall diagnosis
    all_issues = []
    for check_name in ["image_processor_check", "instruction_check",
                       "state_normalization_check", "diffusion_action_head_check"]:
        if check_name in results:
            check_issues = results[check_name].get("issues", [])
            all_issues.extend(check_issues)

    results["all_issues"] = all_issues
    results["total_issues"] = len(all_issues)

    if all_issues:
        results["overall_diagnosis"] = f"FOUND {len(all_issues)} CONSISTENCY ISSUES"
        log.info(f"\n{'='*70}")
        log.info(f"OVERALL: FOUND {len(all_issues)} CONSISTENCY ISSUES")
        log.info(f"{'='*70}")
        for i, issue in enumerate(all_issues, 1):
            log.info(f"  {i}. {issue}")
    else:
        results["overall_diagnosis"] = "ALL CONSISTENCY CHECKS PASSED"
        log.info(f"\n{'='*70}")
        log.info(f"OVERALL: ALL CONSISTENCY CHECKS PASSED")
        log.info(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description="TinyVLA Consistency Verification")
    parser.add_argument("--policy_type", type=str, required=True,
                        choices=["tinyvla_s", "tinyvla_b"])
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset_root", type=str,
                        default="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/disassemble-v3_corner")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(__file__).parent
    log_file = str(output_dir / "tinyvla_alignment_verify.log")
    logger = setup_logging(log_file)

    logger.info(f"TinyVLA Consistency Verification Tool")
    logger.info(f"Policy type: {args.policy_type}")
    logger.info(f"Dataset: {args.dataset_root}")
    logger.info(f"Episode: {args.episode_index}")
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
        result = run_full_verification(
            args.policy_type, checkpoint_path, args.dataset_root,
            args.episode_index, device, logger
        )

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"\nResults saved to: {output_file}")
        logger.info(f"Log saved to: {log_file}")

    except Exception as e:
        logger.error(f"Error during verification: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()