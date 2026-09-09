#!/usr/bin/env python3
"""
Complete Action Pipeline Verification for TinyVLA and MiniVLA.

Traces the full pipeline:
  raw dataset action -> action preprocessor -> policy inference -> raw predicted action -> action postprocessor -> final robot action

Purpose: Distinguish between
  1) 12k steps insufficient -> action distribution reasonable but control能力不足
  2) Pipeline error -> action distribution already wrong, further training meaningless

Usage:
  python verify_action_space_alignment.py \
    --policy_type tinyvla_s \
    --dataset_root /path/to/dataset \
    --episode_index 0 \
    --gpu_id 0

  Or check all models:
  python verify_action_space_alignment.py \
    --policy_type all \
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

# Add lerobot to path
LEROBOT_ROOT = "/data/zhonglinye/jun/lerobot"
if LEROBOT_ROOT not in sys.path:
    sys.path.insert(0, LEROBOT_ROOT + "/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedConfig, PreTrainedPolicy
from lerobot.policies.factory import get_policy_class
from lerobot.policies import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

ACTION = "action"

DEFAULT_CHECKPOINTS = {
    "tinyvla_s": "/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/experiments/tinyvla_s_disassemble-v3_corner/checkpoints/tinyvla_tinyvla_s_disassemble-v3_corner/checkpoints/012000/pretrained_model",
    "tinyvla_b": "/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/experiments/tinyvla_b_disassemble-v3_corner/checkpoints/tinyvla_tinyvla_b_disassemble-v3_corner/checkpoints/012000/pretrained_model",
    "minivla": "/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/minivla/experiments/random_112_seed42_disassemblev3corner_minivla_random/checkpoints/checkpoints/012000/pretrained_model",
}

DEFAULT_DATASET = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/disassemble-v3_corner"

OUTPUT_FILE = str(Path(__file__).parent / "action_space_alignment_results.json")


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("action_space_verify")
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


def print_stats(stats: dict, prefix: str = "", logger: logging.Logger | None = None) -> None:
    log = logger or logging.getLogger("action_space_verify")
    tag = f"[{prefix}] " if prefix else ""
    log.info(f"  {tag}{stats['name']}")
    log.info(f"    shape : {stats['shape']}")
    log.info(f"    dtype : {stats['dtype']}")
    log.info(f"    min   : {stats['min']:.6f}")
    log.info(f"    max   : {stats['max']:.6f}")
    log.info(f"    mean  : {stats['mean']:.6f}")
    log.info(f"    std   : {stats['std']:.6f}")


def print_separator(title: str, logger: logging.Logger | None = None) -> None:
    log = logger or logging.getLogger("action_space_verify")
    log.info(f"\n{'='*60}")
    log.info(f"  {title}")
    log.info(f"{'='*60}")


def compute_comparison_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pred = np.array(pred).flatten()
    target = np.array(target).flatten()
    min_len = min(len(pred), len(target))
    pred = pred[:min_len]
    target = target[:min_len]

    mae = float(np.mean(np.abs(pred - target)))
    max_err = float(np.max(np.abs(pred - target)))

    if np.std(pred) > 1e-8 and np.std(target) > 1e-8:
        correlation = float(np.corrcoef(pred, target)[0, 1])
    else:
        correlation = 0.0

    dist_diff = float(np.abs(np.mean(pred) - np.mean(target)) + np.abs(np.std(pred) - np.std(target)))

    return {
        "mean_absolute_error": mae,
        "max_absolute_error": max_err,
        "correlation_coefficient": correlation,
        "distribution_difference": dist_diff,
    }


def load_policy_and_processors(policy_type: str, checkpoint_path: str, device: str, dataset_stats: dict | None = None, logger: logging.Logger | None = None):
    log = logger or logging.getLogger("action_space_verify")
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


def check_dataset_schema(dataset: LeRobotDataset, logger: logging.Logger | None = None) -> dict:
    log = logger or logging.getLogger("action_space_verify")
    print_separator("DATASET SCHEMA CHECK", log)

    features = dataset.features
    log.info(f"Dataset features: {list(features.keys())}")

    action_feature = features.get(ACTION, None)
    state_feature = features.get(OBS_STATE, None)

    result = {
        "observation_keys": list(features.keys()),
        "action_feature": str(action_feature) if action_feature else None,
        "state_feature": str(state_feature) if state_feature else None,
        "num_episodes": dataset.num_episodes,
        "num_frames": dataset.num_frames,
        "fps": dataset.fps,
    }

    if action_feature:
        action_shape = action_feature.get("shape", ())
        action_names = action_feature.get("names", {})
        log.info(f"Action shape: {action_shape}")
        log.info(f"Action axes: {action_names}")
        result["action_shape"] = list(action_shape) if hasattr(action_shape, "__iter__") else [action_shape]
        result["action_axes"] = action_names

    if state_feature:
        state_shape = state_feature.get("shape", ())
        log.info(f"State shape: {state_shape}")
        result["state_shape"] = list(state_shape) if hasattr(state_shape, "__iter__") else [state_shape]

    if hasattr(dataset.meta, "stats"):
        stats = dataset.meta.stats
        log.info(f"Dataset stats keys: {list(stats.keys())}")
        result["stats_keys"] = list(stats.keys())

        if ACTION in stats:
            action_stats = stats[ACTION]
            log.info(f"Action stats: {list(action_stats.keys())}")
            for k, v in action_stats.items():
                if hasattr(v, "tolist"):
                    v = v.tolist()
                log.info(f"  {k}: {v}")
            result["action_stats"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in action_stats.items()}

    return result


def check_action_chunk_alignment(config, dataset: LeRobotDataset, logger: logging.Logger | None = None) -> dict:
    log = logger or logging.getLogger("action_space_verify")
    print_separator("ACTION CHUNK ALIGNMENT CHECK", log)

    chunk_size = getattr(config, "chunk_size", None)
    n_action_steps = getattr(config, "n_action_steps", None)

    log.info(f"Config chunk_size: {chunk_size}")
    log.info(f"Config n_action_steps: {n_action_steps}")

    sample_first = dataset[0]
    if ACTION in sample_first:
        raw_act = sample_first[ACTION]
        if isinstance(raw_act, np.ndarray):
            act_shape = raw_act.shape
        else:
            act_shape = list(raw_act.shape)
        log.info(f"Dataset action shape per frame: {act_shape}")
    else:
        act_shape = None
        log.info("No action found in dataset sample")

    result = {
        "dataset_action_chunk_length": act_shape[0] if act_shape and len(act_shape) > 1 else 1,
        "dataset_action_dim": act_shape[-1] if act_shape else None,
        "config_chunk_size": chunk_size,
        "config_n_action_steps": n_action_steps,
        "policy_action_chunk_length": chunk_size,
        "executed_action_horizon": n_action_steps,
    }

    if chunk_size and n_action_steps:
        if n_action_steps > chunk_size:
            log.info(f"  MISMATCH: n_action_steps ({n_action_steps}) > chunk_size ({chunk_size})")
            result["mismatch"] = True
            result["mismatch_detail"] = "n_action_steps exceeds chunk_size"
        else:
            log.info(f"  OK: n_action_steps ({n_action_steps}) <= chunk_size ({chunk_size})")
            result["mismatch"] = False
    else:
        log.info("  WARNING: chunk_size or n_action_steps not available")
        result["mismatch"] = None

    return result


def verify_tinyvla_pipeline(sample: dict, policy, preprocessor, postprocessor, config, device: str, logger: logging.Logger | None = None) -> dict:
    log = logger or logging.getLogger("action_space_verify")
    print_separator(f"TinyVLA ({config.type}) ACTION PIPELINE VERIFICATION", log)

    results = {}

    raw_act = sample[ACTION]
    if isinstance(raw_act, np.ndarray):
        raw_act_t = torch.from_numpy(raw_act).float().to(device)
    else:
        raw_act_t = raw_act.float().to(device)

    raw_stats = tensor_stats(raw_act_t, "raw_dataset_action")
    print_stats(raw_stats, "RAW", log)
    results["raw_dataset_action"] = raw_stats

    preprocessed = preprocessor(sample)
    if isinstance(preprocessed, dict):
        for k, v in preprocessed.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                model_dtype = getattr(config, "dtype", "float32")
                if model_dtype == "bfloat16":
                    preprocessed[k] = v.to(dtype=torch.bfloat16)
                elif model_dtype == "float16":
                    preprocessed[k] = v.to(dtype=torch.float16)

    if ACTION in preprocessed:
        pre_act = preprocessed[ACTION]
        if isinstance(pre_act, torch.Tensor):
            pre_stats = tensor_stats(pre_act, "preprocessed_action_normalized")
            print_stats(pre_stats, "PRE", log)
            results["preprocessed_action"] = pre_stats

            normalization_check = {}
            if raw_act_t.ndim == 1 and pre_act.ndim == 2:
                raw_flat = raw_act_t.float().cpu().numpy()
                pre_flat = pre_act[0].float().cpu().numpy()
                normalization_check["raw_to_normalized_mae"] = float(np.mean(np.abs(raw_flat - pre_flat)))
                normalization_check["raw_range"] = [float(raw_act_t.min()), float(raw_act_t.max())]
                normalization_check["normalized_range"] = [float(pre_act.min()), float(pre_act.max())]

                if abs(normalization_check["normalized_range"][1] - normalization_check["normalized_range"][0]) > 2.5:
                    log.info("  WARNING: Normalized action range exceeds [-1, 1] significantly")
                    normalization_check["normalization_warning"] = "Range exceeds expected [-1, 1]"
                else:
                    log.info("  OK: Normalized action within expected range")
                    normalization_check["normalization_ok"] = True
            results["normalization_check"] = normalization_check
    else:
        log.info("  [PRE] No action in preprocessed batch")
        results["preprocessed_action"] = None

    policy.reset()
    with torch.no_grad():
        pred_actions = policy.predict_action_chunk(preprocessed)

    if isinstance(pred_actions, torch.Tensor) and pred_actions.dtype == torch.bfloat16:
        pred_actions_fp32 = pred_actions.float()
    else:
        pred_actions_fp32 = pred_actions

    pred_stats = tensor_stats(pred_actions_fp32, "policy_raw_output_continuous_action")
    print_stats(pred_stats, "PRED_RAW", log)
    results["policy_raw_output"] = pred_stats

    if pred_actions_fp32.ndim >= 2:
        log.info(f"    chunk_length: {pred_actions_fp32.shape[-2]}")
        results["policy_raw_output"]["chunk_length"] = int(pred_actions_fp32.shape[-2])

    final_actions = postprocessor(pred_actions)
    if isinstance(final_actions, torch.Tensor):
        final_stats = tensor_stats(final_actions, "postprocessed_final_robot_action")
        print_stats(final_stats, "FINAL", log)
        results["postprocessed_action"] = final_stats

        if final_actions.ndim == 3:
            pred_step0 = final_actions[0, 0, :]
            pred_chunk = final_actions[0, :, :]
        elif final_actions.ndim == 2:
            pred_step0 = final_actions[0, :]
            pred_chunk = final_actions
        else:
            pred_step0 = final_actions
            pred_chunk = final_actions

        raw_on_device = raw_act_t.to(final_actions.device).float()
        abs_error = (raw_on_device - pred_step0.float()).abs()

        comparison = {
            "real_action": raw_on_device.tolist(),
            "predicted_step0": pred_step0.float().tolist(),
            "abs_error_per_dim": abs_error.tolist(),
            "max_absolute_error": float(abs_error.max()),
            "mean_absolute_error": float(abs_error.mean()),
        }

        if pred_chunk.ndim == 2:
            pred_np = pred_chunk.float().cpu().numpy()
            if raw_act_t.ndim == 1:
                target_np = raw_act_t.float().cpu().numpy()
            else:
                target_np = raw_act_t.float().cpu().numpy()
            comparison["distribution_comparison"] = compute_comparison_metrics(pred_np, target_np)

        results["prediction_vs_real"] = comparison

        log.info(f"\n  Real action     : {raw_on_device.tolist()}")
        log.info(f"  Predicted (step0): {pred_step0.float().tolist()}")
        log.info(f"  Max error       : {comparison['max_absolute_error']:.6f}")
        log.info(f"  Mean error      : {comparison['mean_absolute_error']:.6f}")

        if "distribution_comparison" in comparison:
            dc = comparison["distribution_comparison"]
            log.info(f"  MAE             : {dc['mean_absolute_error']:.6f}")
            log.info(f"  Correlation     : {dc['correlation_coefficient']:.6f}")
            log.info(f"  Distribution diff: {dc['distribution_difference']:.6f}")
    else:
        results["prediction_vs_real"] = None

    results["config_info"] = {
        "action_head_type": getattr(config, "action_head_type", None),
        "chunk_size": getattr(config, "chunk_size", None),
        "n_action_steps": getattr(config, "n_action_steps", None),
        "action_dim": getattr(config, "action_dim", None),
        "normalization_mapping": {k: str(v) for k, v in getattr(config, "normalization_mapping", {}).items()},
    }

    return results


def verify_minivla_pipeline(sample: dict, policy, preprocessor, postprocessor, config, device: str, logger: logging.Logger | None = None) -> dict:
    log = logger or logging.getLogger("action_space_verify")
    print_separator(f"MiniVLA ACTION PIPELINE VERIFICATION", log)

    results = {}

    raw_act = sample[ACTION]
    if isinstance(raw_act, np.ndarray):
        raw_act_t = torch.from_numpy(raw_act).float().to(device)
    else:
        raw_act_t = raw_act.float().to(device)

    raw_stats = tensor_stats(raw_act_t, "raw_dataset_action")
    print_stats(raw_stats, "RAW", log)
    results["raw_dataset_action"] = raw_stats

    preprocessed = preprocessor(sample)
    if isinstance(preprocessed, dict):
        for k, v in preprocessed.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                model_dtype = getattr(config, "dtype", "float32")
                if model_dtype == "bfloat16":
                    preprocessed[k] = v.to(dtype=torch.bfloat16)
                elif model_dtype == "float16":
                    preprocessed[k] = v.to(dtype=torch.float16)

    if ACTION in preprocessed:
        pre_act = preprocessed[ACTION]
        if isinstance(pre_act, torch.Tensor):
            pre_stats = tensor_stats(pre_act, "preprocessed_action_normalized")
            print_stats(pre_stats, "PRE", log)
            results["preprocessed_action"] = pre_stats
    else:
        log.info("  [PRE] No action in preprocessed batch")
        results["preprocessed_action"] = None

    core = policy.model
    is_vq = getattr(core.config, "is_vq_mode", False)
    log.info(f"\n  [VQ] VQ mode: {is_vq}")
    results["vq_mode"] = is_vq

    if is_vq and hasattr(core, "action_tokenizer"):
        action_tokenizer = core.action_tokenizer
        vq_vae = action_tokenizer.vq_vae

        log.info(f"  VQ config: input_dim_h={vq_vae.input_dim_h}, input_dim_w={vq_vae.input_dim_w}")
        log.info(f"  VQ config: vqvae_groups={vq_vae.vqvae_groups}, n_latent_dims={vq_vae.n_latent_dims}, n_embed={vq_vae.vqvae_n_embed}")

        results["vq_config"] = {
            "input_dim_h": vq_vae.input_dim_h,
            "input_dim_w": vq_vae.input_dim_w,
            "vqvae_groups": vq_vae.vqvae_groups,
            "n_latent_dims": vq_vae.n_latent_dims,
            "vqvae_n_embed": vq_vae.vqvae_n_embed,
        }

        log.info(f"  [VQ] Note: Independent VQ encode/decode test skipped.")
        log.info(f"    Reason: VQ encoder expects chunk shape [{vq_vae.input_dim_h}, {vq_vae.input_dim_w}],")
        log.info(f"    but dataset single-frame action has shape {raw_act_t.shape}.")
        log.info(f"    During inference, VQ decoding happens inside predict_action_chunk via token IDs.")
        log.info(f"    VQ reconstruction error should be checked during training, not inference verification.")
        results["vq_independent_test"] = "skipped (shape mismatch between single-frame action and VQ chunk input)"

    # === Hook MiniVLA VQ decode inside predict_action_chunk to capture intermediate values ===
    vq_internal = {}
    if is_vq and hasattr(core, "action_tokenizer"):
        action_tokenizer = core.action_tokenizer
        original_decode = action_tokenizer.decode_token_ids_to_actions

        def hooked_decode(action_token_ids):
            if isinstance(action_token_ids, torch.Tensor):
                action_token_ids_np = action_token_ids.detach().cpu().numpy()
            else:
                action_token_ids_np = np.array(action_token_ids)

            vq_internal["token_ids_raw"] = action_token_ids_np.copy()
            vq_internal["token_ids_shape"] = list(action_token_ids_np.shape)

            tokenizer_len = action_tokenizer.tokenizer_len
            vq_codes = tokenizer_len - 1 - action_token_ids_np
            vq_codes = np.clip(vq_codes, 0, action_tokenizer.n_bins - 1)
            vq_internal["vq_codes"] = vq_codes.copy()
            vq_internal["vq_codes_shape"] = list(vq_codes.shape)

            # Patch VQ decoder to handle bfloat16 -> float32 conversion
            vq_vae = action_tokenizer.vq_vae
            original_get_action = vq_vae.get_action_from_latent

            def patched_get_action(latent):
                result = original_get_action(latent)
                if result.dtype == torch.bfloat16:
                    return result.float()
                return result

            vq_vae.get_action_from_latent = patched_get_action

            try:
                result = original_decode(action_token_ids)
            finally:
                vq_vae.get_action_from_latent = original_get_action

            vq_internal["decoded_action"] = result.copy() if hasattr(result, "copy") else np.array(result)
            return result

        action_tokenizer.decode_token_ids_to_actions = hooked_decode

    policy.reset()
    with torch.no_grad():
        pred_actions = policy.predict_action_chunk(preprocessed)

    # Restore original method
    if is_vq and hasattr(core, "action_tokenizer"):
        action_tokenizer.decode_token_ids_to_actions = original_decode

    if isinstance(pred_actions, torch.Tensor) and pred_actions.dtype == torch.bfloat16:
        pred_actions_fp32 = pred_actions.float()
    else:
        pred_actions_fp32 = pred_actions

    pred_stats = tensor_stats(pred_actions_fp32, "policy_raw_output")
    print_stats(pred_stats, "PRED_RAW", log)
    results["policy_raw_output"] = pred_stats

    if is_vq and vq_internal:
        log.info(f"\n  [VQ INTERNAL] predict_action_chunk VQ decode details:")
        log.info(f"    token_ids shape : {vq_internal['token_ids_shape']}")
        log.info(f"    token_ids values: {vq_internal['token_ids_raw'].flatten().tolist()}")
        log.info(f"    vq_codes shape  : {vq_internal['vq_codes_shape']}")
        log.info(f"    vq_codes values : {vq_internal['vq_codes'].flatten().tolist()}")

        decoded_np = vq_internal["decoded_action"]
        decoded_stats = tensor_stats(decoded_np, "vq_decoded_action_before_postprocess")
        print_stats(decoded_stats, "VQ_DECODED", log)
        results["vq_decoded_action_internal"] = decoded_stats

        if decoded_np.ndim == 3:
            decoded_first_step = decoded_np[0, 0, :]
        elif decoded_np.ndim == 2:
            decoded_first_step = decoded_np[0, :]
        else:
            decoded_first_step = decoded_np.flatten()

        log.info(f"    decoded action (step0): {decoded_first_step.tolist()}")
        results["vq_decoded_action_step0"] = decoded_first_step.tolist()

        log.info(f"  MiniVLA: action token/latent -> VQ decoder -> continuous action")
        log.info(f"  Raw output shape: {pred_actions_fp32.shape}")

    final_actions = postprocessor(pred_actions)
    if isinstance(final_actions, torch.Tensor):
        final_stats = tensor_stats(final_actions, "postprocessed_final_robot_action")
        print_stats(final_stats, "FINAL", log)
        results["postprocessed_action"] = final_stats

        if final_actions.ndim == 3:
            pred_step0 = final_actions[0, 0, :]
            pred_chunk = final_actions[0, :, :]
        elif final_actions.ndim == 2:
            pred_step0 = final_actions[0, :]
            pred_chunk = final_actions
        else:
            pred_step0 = final_actions
            pred_chunk = final_actions

        raw_on_device = raw_act_t.to(final_actions.device).float()
        abs_error = (raw_on_device - pred_step0.float()).abs()

        comparison = {
            "real_action": raw_on_device.tolist(),
            "predicted_step0": pred_step0.float().tolist(),
            "abs_error_per_dim": abs_error.tolist(),
            "max_absolute_error": float(abs_error.max()),
            "mean_absolute_error": float(abs_error.mean()),
        }

        if pred_chunk.ndim == 2:
            pred_np = pred_chunk.float().cpu().numpy()
            target_np = raw_act_t.float().cpu().numpy() if raw_act_t.ndim == 1 else raw_act_t.float().cpu().numpy()
            comparison["distribution_comparison"] = compute_comparison_metrics(pred_np, target_np)

        results["prediction_vs_real"] = comparison

        log.info(f"\n  Real action     : {raw_on_device.tolist()}")
        log.info(f"  Predicted (step0): {pred_step0.float().tolist()}")
        log.info(f"  Max error       : {comparison['max_absolute_error']:.6f}")
        log.info(f"  Mean error      : {comparison['mean_absolute_error']:.6f}")

        if "distribution_comparison" in comparison:
            dc = comparison["distribution_comparison"]
            log.info(f"  MAE             : {dc['mean_absolute_error']:.6f}")
            log.info(f"  Correlation     : {dc['correlation_coefficient']:.6f}")
            log.info(f"  Distribution diff: {dc['distribution_difference']:.6f}")
    else:
        results["prediction_vs_real"] = None

    results["config_info"] = {
        "is_vq_mode": is_vq,
        "chunk_size": getattr(config, "chunk_size", None),
        "n_action_steps": getattr(config, "n_action_steps", None),
        "action_dim": getattr(config, "action_dim", None),
    }

    return results


def run_multi_timestep_analysis(policy_type: str, policy, preprocessor, postprocessor, config, dataset, episode_index: int, device: str, num_frames: int = 10, logger: logging.Logger | None = None) -> dict:
    log = logger or logging.getLogger("action_space_verify")

    # Patch MiniVLA VQ decoder for bfloat16 compatibility
    if policy_type == "minivla":
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
                    log.info("Patched MiniVLA vq_vae.get_action_from_latent for bfloat16 in multi-timestep analysis")
        except Exception as e:
            log.warning(f"Failed to patch MiniVLA for multi-timestep: {e}")
    print_separator(f"MULTI-TIMESTEP DISTRIBUTION ANALYSIS ({policy_type})", log)

    episode_indices = np.array(dataset.hf_dataset["episode_index"])
    mask = episode_indices == episode_index
    frame_indices = np.where(mask)[0]

    if len(frame_indices) == 0:
        log.error(f"Episode {episode_index} not found")
        return {"error": "Episode not found"}

    frames_to_check = min(num_frames, len(frame_indices))
    log.info(f"Checking {frames_to_check} frames from episode {episode_index}")

    all_raw_actions = []
    all_pred_actions = []
    all_postprocessed_actions = []

    model_dtype = getattr(config, "dtype", "float32")
    if model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif model_dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    for i in range(frames_to_check):
        frame_idx = frame_indices[i]
        sample = dataset[frame_idx]

        if ACTION not in sample:
            continue

        raw_act = sample[ACTION]
        if isinstance(raw_act, np.ndarray):
            raw_act_t = torch.from_numpy(raw_act).float()
        else:
            raw_act_t = raw_act.float()

        if raw_act_t.ndim == 2:
            raw_act_first_step = raw_act_t[0, :]
        elif raw_act_t.ndim == 1:
            raw_act_first_step = raw_act_t
        else:
            raw_act_first_step = raw_act_t.flatten()

        all_raw_actions.append(raw_act_first_step.numpy())

        preprocessed = preprocessor(sample)
        if isinstance(preprocessed, dict):
            for k, v in preprocessed.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    preprocessed[k] = v.to(dtype=torch_dtype)

        policy.reset()
        with torch.no_grad():
            pred_actions = policy.predict_action_chunk(preprocessed)

        if isinstance(pred_actions, torch.Tensor) and pred_actions.dtype == torch.bfloat16:
            pred_actions = pred_actions.float()

        if isinstance(pred_actions, torch.Tensor):
            if pred_actions.ndim == 3:
                all_pred_actions.append(pred_actions[0, 0, :].cpu().numpy())
            elif pred_actions.ndim == 2:
                all_pred_actions.append(pred_actions[0, :].cpu().numpy())
            else:
                all_pred_actions.append(pred_actions.cpu().numpy())

        final_actions = postprocessor(pred_actions)
        if isinstance(final_actions, torch.Tensor):
            if final_actions.ndim == 3:
                all_postprocessed_actions.append(final_actions[0, 0, :].cpu().numpy())
            elif final_actions.ndim == 2:
                all_postprocessed_actions.append(final_actions[0, :].cpu().numpy())
            else:
                all_postprocessed_actions.append(final_actions.cpu().numpy())

    results = {}

    if len(all_raw_actions) > 0:
        raw_np = np.array(all_raw_actions)
        results["raw_action_distribution"] = {
            "mean_per_dim": raw_np.mean(axis=0).tolist(),
            "std_per_dim": raw_np.std(axis=0).tolist(),
            "min_per_dim": raw_np.min(axis=0).tolist(),
            "max_per_dim": raw_np.max(axis=0).tolist(),
            "overall_mean": float(raw_np.mean()),
            "overall_std": float(raw_np.std()),
        }
        log.info(f"\n  Raw action distribution:")
        log.info(f"    Mean per dim: {results['raw_action_distribution']['mean_per_dim']}")
        log.info(f"    Std per dim : {results['raw_action_distribution']['std_per_dim']}")

    if len(all_pred_actions) > 0:
        pred_np = np.array(all_pred_actions)
        results["raw_prediction_distribution"] = {
            "mean_per_dim": pred_np.mean(axis=0).tolist(),
            "std_per_dim": pred_np.std(axis=0).tolist(),
            "min_per_dim": pred_np.min(axis=0).tolist(),
            "max_per_dim": pred_np.max(axis=0).tolist(),
            "overall_mean": float(pred_np.mean()),
            "overall_std": float(pred_np.std()),
        }
        log.info(f"\n  Raw prediction distribution (before postprocess):")
        log.info(f"    Mean per dim: {results['raw_prediction_distribution']['mean_per_dim']}")
        log.info(f"    Std per dim : {results['raw_prediction_distribution']['std_per_dim']}")

    if len(all_postprocessed_actions) > 0:
        post_np = np.array(all_postprocessed_actions)
        results["postprocessed_prediction_distribution"] = {
            "mean_per_dim": post_np.mean(axis=0).tolist(),
            "std_per_dim": post_np.std(axis=0).tolist(),
            "min_per_dim": post_np.min(axis=0).tolist(),
            "max_per_dim": post_np.max(axis=0).tolist(),
            "overall_mean": float(post_np.mean()),
            "overall_std": float(post_np.std()),
        }
        log.info(f"\n  Postprocessed prediction distribution:")
        log.info(f"    Mean per dim: {results['postprocessed_prediction_distribution']['mean_per_dim']}")
        log.info(f"    Std per dim : {results['postprocessed_prediction_distribution']['std_per_dim']}")

    if len(all_raw_actions) > 0 and len(all_postprocessed_actions) > 0:
        raw_np = np.array(all_raw_actions)
        post_np = np.array(all_postprocessed_actions)

        dist_check = {}
        raw_mean = float(raw_np.mean())
        post_mean = float(post_np.mean())
        raw_std = float(raw_np.std())
        post_std = float(post_np.std())

        dist_check["mean_difference"] = abs(raw_mean - post_mean)
        dist_check["std_difference"] = abs(raw_std - post_std)
        dist_check["raw_mean"] = raw_mean
        dist_check["post_mean"] = post_mean
        dist_check["raw_std"] = raw_std
        dist_check["post_std"] = post_std

        if dist_check["mean_difference"] > 0.5 or dist_check["std_difference"] > 0.5:
            dist_check["distribution_mismatch"] = True
            log.info(f"\n  WARNING: Distribution mismatch detected!")
            log.info(f"    Mean diff: {dist_check['mean_difference']:.4f}")
            log.info(f"    Std diff : {dist_check['std_difference']:.4f}")
            log.info(f"  => Action pipeline error likely, further training may be meaningless")
        else:
            dist_check["distribution_mismatch"] = False
            log.info(f"\n  OK: Distributions are reasonably aligned")
            log.info(f"  => If success=0, may be due to insufficient training steps")

        results["distribution_alignment_check"] = dist_check

    return results


def run_verification(policy_type: str, checkpoint_path: str, dataset_root: str, episode_index: int, device: str, logger: logging.Logger | None = None) -> dict:
    log = logger or logging.getLogger("action_space_verify")
    log.info(f"\n{'#'*60}")
    log.info(f"# Verifying: {policy_type}")
    log.info(f"{'#'*60}")

    dataset = LeRobotDataset(dataset_root)
    log.info(f"Dataset loaded: {dataset.num_frames} frames, {dataset.num_episodes} episodes")

    ds_stats = dataset.meta.stats if hasattr(dataset.meta, "stats") else None

    schema_check = check_dataset_schema(dataset, log)

    policy, preprocessor, postprocessor, config = load_policy_and_processors(
        policy_type, checkpoint_path, device, dataset_stats=ds_stats, logger=log
    )

    chunk_check = check_action_chunk_alignment(config, dataset, log)

    episode_indices = np.array(dataset.hf_dataset["episode_index"])
    mask = episode_indices == episode_index
    frame_indices = np.where(mask)[0]
    if len(frame_indices) == 0:
        log.error(f"Episode {episode_index} not found")
        return {"error": "Episode not found"}

    frame_idx = int(frame_indices[len(frame_indices) // 2])
    sample = dataset[frame_idx]
    log.info(f"Loaded frame {frame_idx} from episode {episode_index}")

    if policy_type.startswith("tinyvla"):
        pipeline_results = verify_tinyvla_pipeline(sample, policy, preprocessor, postprocessor, config, device, log)
    elif policy_type == "minivla":
        pipeline_results = verify_minivla_pipeline(sample, policy, preprocessor, postprocessor, config, device, log)
    else:
        log.error(f"Unknown policy type: {policy_type}")
        return {"error": f"Unknown policy type: {policy_type}"}

    multi_step_results = run_multi_timestep_analysis(
        policy_type, policy, preprocessor, postprocessor, config, dataset, episode_index, device, num_frames=10, logger=log
    )

    final_result = {
        "policy_type": policy_type,
        "checkpoint_path": checkpoint_path,
        "dataset_root": dataset_root,
        "episode_index": episode_index,
        "dataset_schema": schema_check,
        "chunk_alignment": chunk_check,
        "single_frame_pipeline": pipeline_results,
        "multi_timestep_distribution": multi_step_results,
    }

    if "distribution_alignment_check" in multi_step_results:
        dist_check = multi_step_results["distribution_alignment_check"]
        if dist_check.get("distribution_mismatch", False):
            final_result["diagnosis"] = "PIPELINE_ERROR: Action distribution mismatch detected. Further training unlikely to help. Check normalization, VQ decoder, or postprocessing."
        else:
            final_result["diagnosis"] = "TRAINING_INSUFFICIENT: Action distribution is reasonable but control能力不足. More training steps may help."
    else:
        final_result["diagnosis"] = "UNKNOWN: Could not determine distribution alignment"

    return final_result


def main():
    parser = argparse.ArgumentParser(description="Action Space Alignment Verification")
    parser.add_argument("--policy_type", type=str, required=True, choices=["tinyvla_s", "tinyvla_b", "minivla", "all"])
    parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(__file__).parent
    log_file = str(output_dir / "action_space_alignment_verify.log")
    logger = setup_logging(log_file)

    logger.info(f"Action Space Alignment Verification Tool")
    logger.info(f"Dataset: {args.dataset_root}")
    logger.info(f"Episode: {args.episode_index}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {args.output_file or OUTPUT_FILE}")

    policy_types = ["tinyvla_s", "tinyvla_b", "minivla"] if args.policy_type == "all" else [args.policy_type]

    # Load existing results to skip already-completed models
    output_file = args.output_file or OUTPUT_FILE
    existing_results = {}
    if Path(output_file).exists():
        try:
            with open(output_file, "r") as f:
                existing_results = json.load(f)
            logger.info(f"Found existing results at {output_file}")
        except Exception:
            logger.warning(f"Failed to read existing results, will run all models")
            existing_results = {}

    # Filter out already-completed models
    completed = [pt for pt in policy_types if pt in existing_results and "error" not in existing_results[pt]]
    if completed:
        logger.info(f"Skipping already-completed models: {completed}")
        policy_types = [pt for pt in policy_types if pt not in completed]

    if not policy_types:
        logger.info(f"All requested models already completed. Nothing to run.")
        logger.info(f"Existing results: {list(existing_results.keys())}")
        return

    logger.info(f"Models to run: {policy_types}")

    all_results = dict(existing_results)

    for pt in policy_types:
        checkpoint_path = args.checkpoint_path or DEFAULT_CHECKPOINTS.get(pt)
        if not checkpoint_path:
            logger.error(f"No default checkpoint for {pt}")
            continue

        try:
            result = run_verification(pt, checkpoint_path, args.dataset_root, args.episode_index, device, logger)
            all_results[pt] = result
        except Exception as e:
            logger.error(f"Error verifying {pt}: {e}")
            logger.error(traceback.format_exc())
            all_results[pt] = {"error": str(e), "traceback": traceback.format_exc()}

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")

    for pt, result in all_results.items():
        if "error" in result:
            logger.info(f"{pt:12s}: ERROR - {result['error']}")
        else:
            diagnosis = result.get("diagnosis", "UNKNOWN")
            logger.info(f"{pt:12s}: {diagnosis}")

            if "single_frame_pipeline" in result:
                pipeline = result["single_frame_pipeline"]
                if "prediction_vs_real" in pipeline:
                    pvr = pipeline["prediction_vs_real"]
                    logger.info(f"              Max error: {pvr.get('max_absolute_error', 'N/A'):.4f}")
                    logger.info(f"              Mean error: {pvr.get('mean_absolute_error', 'N/A'):.4f}")

    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()