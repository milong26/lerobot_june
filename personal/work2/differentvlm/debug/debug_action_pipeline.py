#!/usr/bin/env python3
"""
Action Pipeline Debug Tool for TinyVLA / MiniVLA / SmolVLA

Loads an existing checkpoint and runs a single dataset sample through the
full action pipeline, printing every intermediate tensor for diagnostic
comparison across policy types.

Usage:
  python debug_action_pipeline.py \
    --policy_type tinyvla_s \
    --checkpoint_path /path/to/pretrained_model \
    --dataset_root /path/to/dataset_view/disassemble-v3_corner \
    --episode_index 0 \
    --device cuda

Supported policy types: tinyvla_s, tinyvla_b, minivla, smolvla

Output:
  - Console: detailed step-by-step tensor statistics
  - File: action_pipeline_debug_results.json (same directory as script)
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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[5]  # lerobot root
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

# ---------------------------------------------------------------------------
# Logging setup: dual output to console + file
# ---------------------------------------------------------------------------

def setup_logging(log_file: str | None = None) -> logging.Logger:
    """Create a logger that writes to both console and file."""
    logger = logging.getLogger("action_pipeline_debug")
    logger.setLevel(logging.DEBUG)
    # Clear existing handlers
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_stats(x: torch.Tensor | np.ndarray, name: str = "tensor") -> dict:
    """Return a JSON-serialisable dict of shape/dtype/min/max/mean/std."""
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
    """Pretty-print a stats dict."""
    log = logger or logging.getLogger("action_pipeline_debug")
    tag = f"[{prefix}] " if prefix else ""
    log.info(f"  {tag}{stats['name']}")
    log.info(f"    shape : {stats['shape']}")
    log.info(f"    dtype : {stats['dtype']}")
    log.info(f"    min   : {stats['min']:.6f}")
    log.info(f"    max   : {stats['max']:.6f}")
    log.info(f"    mean  : {stats['mean']:.6f}")
    log.info(f"    std   : {stats['std']:.6f}")


def print_separator(title: str, logger: logging.Logger | None = None) -> None:
    log = logger or logging.getLogger("action_pipeline_debug")
    width = 72
    log.info(f"\n{'=' * width}")
    log.info(f"  {title}")
    log.info(f"{'=' * width}")


# ---------------------------------------------------------------------------
# Dataset sample loader
# ---------------------------------------------------------------------------

def load_dataset_sample(
    dataset_root: str,
    episode_index: int,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, Any], dict, LeRobotDataset]:
    """Load a single frame from the specified episode.

    Returns (sample_dict, dataset_features, dataset_object).
    """
    log = logger or logging.getLogger("action_pipeline_debug")
    repo_id = "debug/local"
    ds = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=[episode_index],
    )
    log.info(f"Dataset loaded: {ds.num_frames} frames, {ds.num_episodes} episode(s)")
    log.info(f"Features: {list(ds.features.keys())}")
    log.info(f"Action feature: {ds.features.get(ACTION, 'NOT FOUND')}")

    # Pick the middle frame of the episode for a representative sample
    frame_idx = len(ds) // 2
    sample = ds[frame_idx]
    log.info(f"Loaded frame index {frame_idx} from episode {episode_index}")
    log.info(f"Sample keys: {list(sample.keys())}")

    return sample, ds.features, ds


# ---------------------------------------------------------------------------
# Policy loading (reuses eval-style checkpoint loading)
# ---------------------------------------------------------------------------

def load_policy_and_processors(
    policy_type: str,
    checkpoint_path: str,
    device: str,
    dataset_stats: dict | None = None,
    logger: logging.Logger | None = None,
):
    """Load policy + pre/post processors from checkpoint, matching eval scripts.

    Returns (policy, preprocessor, postprocessor, config).
    """
    log = logger or logging.getLogger("action_pipeline_debug")
    log.info(f"\nLoading policy type '{policy_type}' from {checkpoint_path}")

    # Load config from checkpoint (same as eval scripts)
    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    config.device = device

    # Override pretrained_path so make_pre_post_processors loads saved processors
    config.pretrained_path = Path(checkpoint_path)

    # Create a minimal config object compatible with make_pre_post_processors
    # We need the .type attribute to dispatch to the right processor factory
    if not hasattr(config, "type"):
        config.type = policy_type

    # Load pre/post processors from checkpoint
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint_path,
        dataset_stats=dataset_stats,
    )
    log.info(f"Preprocessor loaded: {type(preprocessor).__name__}, steps={len(preprocessor.steps)}")
    log.info(f"Postprocessor loaded: {type(postprocessor).__name__}, steps={len(postprocessor.steps)}")

    # Load the policy itself.
    # We use the base PreTrainedPolicy.from_pretrained directly to avoid bugs
    # in policy-specific from_pretrained overrides (e.g. TinyVLA missing import).
    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.policies.factory import get_policy_class

    policy_cls = get_policy_class(policy_type)
    # Call the base class method directly, bypassing broken overrides
    policy = PreTrainedPolicy.from_pretrained.__func__(policy_cls, checkpoint_path)
    policy.to(device)
    policy.eval()
    log.info(f"Policy loaded: {type(policy).__name__}")

    return policy, preprocessor, postprocessor, config


# ---------------------------------------------------------------------------
# Per-policy debug runners
# ---------------------------------------------------------------------------

def _build_observation_batch(sample: dict, policy_type: str, config) -> dict[str, Any]:
    """Build a raw observation dict from a dataset sample for the given policy."""
    batch = {}

    # Copy observation.state
    if OBS_STATE in sample:
        batch[OBS_STATE] = sample[OBS_STATE]

    # Copy action if present
    if ACTION in sample:
        batch[ACTION] = sample[ACTION]

    # Copy task / language instruction
    if "task" in sample:
        batch["task"] = sample["task"]
    elif "observation.language" in sample:
        batch["task"] = sample["observation.language"]

    # Copy image observations
    for key in sample:
        if key.startswith("observation.images."):
            batch[key] = sample[key]

    return batch


def _print_raw_sample_info(sample: dict, policy_type: str, logger: logging.Logger | None = None) -> dict:
    """Print raw dataset sample statistics before any processing."""
    log = logger or logging.getLogger("action_pipeline_debug")
    print_separator("RAW DATASET SAMPLE", log)

    result = {}

    # Action stats
    if ACTION in sample:
        act = sample[ACTION]
        if isinstance(act, np.ndarray):
            act_t = torch.from_numpy(act)
        else:
            act_t = act
        stats = tensor_stats(act_t, "raw_action")
        print_stats(stats, "RAW", log)
        result["raw_action"] = stats

        # If action is a chunk (2D or 3D), report chunk length
        if act_t.ndim >= 2:
            chunk_len = act_t.shape[-2] if act_t.ndim >= 2 else 1
            log.info(f"    chunk_length: {chunk_len}")
            result["raw_action"]["chunk_length"] = chunk_len
    else:
        log.info("  [RAW] No action found in sample")
        result["raw_action"] = None

    # State stats
    if OBS_STATE in sample:
        st = sample[OBS_STATE]
        if isinstance(st, np.ndarray):
            st_t = torch.from_numpy(st)
        else:
            st_t = st
        stats = tensor_stats(st_t, "raw_state")
        print_stats(stats, "RAW", log)
        result["raw_state"] = stats
    else:
        log.info("  [RAW] No state found in sample")
        result["raw_state"] = None

    # Image keys
    img_keys = [k for k in sample if k.startswith("observation.images.")]
    log.info(f"  [RAW] Image keys: {img_keys}")
    for k in img_keys:
        img = sample[k]
        if isinstance(img, np.ndarray):
            img_t = torch.from_numpy(img)
        else:
            img_t = img
        stats = tensor_stats(img_t, f"raw_image_{k}")
        print_stats(stats, "RAW", log)
        result[f"raw_image_{k}"] = stats

    # Task
    task = sample.get("task", sample.get("observation.language", "N/A"))
    log.info(f"  [RAW] Task: {task}")
    result["task"] = str(task)

    return result


def debug_tinyvla(
    sample: dict,
    policy,
    preprocessor,
    postprocessor,
    config,
    device: str,
    logger: logging.Logger | None = None,
) -> dict:
    """Debug TinyVLA-S / TinyVLA-B action pipeline."""
    log = logger or logging.getLogger("action_pipeline_debug")
    print_separator("TinyVLA ACTION PIPELINE DEBUG", log)

    results = {}

    # 1. Raw sample
    results["raw"] = _print_raw_sample_info(sample, "tinyvla", log)

    # 2. Build observation batch
    raw_batch = _build_observation_batch(sample, "tinyvla", config)

    # 3. Preprocessor
    print_separator("PREPROCESSOR (TinyVLA)", log)
    pre_batch = preprocessor(raw_batch)
    log.info(f"  Preprocessor output keys: {list(pre_batch.keys())}")

    # Print preprocessed action (normalized)
    if ACTION in pre_batch:
        pre_act = pre_batch[ACTION]
        if isinstance(pre_act, torch.Tensor):
            stats = tensor_stats(pre_act, "preprocessed_action_normalized")
        else:
            stats = tensor_stats(torch.from_numpy(pre_act), "preprocessed_action_normalized")
        print_stats(stats, "PRE", log)
        results["preprocessed_action"] = stats
    else:
        log.info("  [PRE] No action in preprocessed batch")
        results["preprocessed_action"] = None

    # Print preprocessed state
    if OBS_STATE in pre_batch:
        pre_st = pre_batch[OBS_STATE]
        if isinstance(pre_st, torch.Tensor):
            stats = tensor_stats(pre_st, "preprocessed_state")
        else:
            stats = tensor_stats(torch.from_numpy(pre_st), "preprocessed_state")
        print_stats(stats, "PRE", log)
        results["preprocessed_state"] = stats

    # 4. Policy forward (predict_action_chunk)
    print_separator("POLICY predict_action_chunk (TinyVLA)", log)
    policy.reset()

    # TinyVLA predict_action_chunk expects OBS_STATE and image keys in batch
    # The preprocessor already put data on device and added batch dim
    with torch.no_grad():
        pred_actions = policy.predict_action_chunk(pre_batch)

    stats = tensor_stats(pred_actions, "model_prediction_action")
    print_stats(stats, "PRED", log)
    if pred_actions.ndim >= 2:
        log.info(f"    chunk_length: {pred_actions.shape[-2]}")
        stats["chunk_length"] = int(pred_actions.shape[-2])
    results["prediction_action"] = stats

    # 5. Postprocessor (unnormalize)
    print_separator("POSTPROCESSOR (TinyVLA)", log)

    # PolicyAction is a type alias for torch.Tensor, not a class
    # Pass the tensor directly to postprocessor
    final_action_obj = postprocessor(pred_actions)

    if isinstance(final_action_obj, torch.Tensor):
        final_actions = final_action_obj
    elif isinstance(final_action_obj, dict) and "action" in final_action_obj:
        final_actions = final_action_obj["action"]
    else:
        final_actions = pred_actions

    if isinstance(final_actions, torch.Tensor):
        stats = tensor_stats(final_actions, "final_action_unnormalized")
    else:
        stats = tensor_stats(torch.from_numpy(final_actions), "final_action_unnormalized")
    print_stats(stats, "FINAL", log)
    results["final_action"] = stats

    # 6. Action head info
    print_separator("TinyVLA ACTION HEAD INFO", log)
    log.info(f"  action_head_type : {config.action_head_type}")
    log.info(f"  chunk_size       : {config.chunk_size}")
    log.info(f"  n_action_steps   : {config.n_action_steps}")
    log.info(f"  action_dim       : {config.action_dim}")
    log.info(f"  normalization    : {config.normalization_mapping}")
    results["action_head_info"] = {
        "action_head_type": config.action_head_type,
        "chunk_size": config.chunk_size,
        "n_action_steps": config.n_action_steps,
        "action_dim": config.action_dim,
        "normalization_mapping": {k: str(v) for k, v in config.normalization_mapping.items()},
    }

    # 7. Compare model prediction vs real action
    print_separator("PREDICTION vs REAL ACTION COMPARISON", log)
    if ACTION in sample:
        raw_act = sample[ACTION]
        if isinstance(raw_act, np.ndarray):
            raw_act_t = torch.from_numpy(raw_act)
        else:
            raw_act_t = raw_act

        # Get first step of prediction (after unnormalization)
        # final_actions shape: [1, chunk_size, action_dim]
        if final_actions.ndim == 3:
            pred_step0 = final_actions[0, 0, :]  # First step
            pred_chunk = final_actions[0, :, :]  # Full chunk
        elif final_actions.ndim == 2:
            pred_step0 = final_actions[0, :]
            pred_chunk = final_actions
        else:
            pred_step0 = final_actions
            pred_chunk = final_actions

        # Compare raw action vs predicted first step
        raw_on_device = raw_act_t.to(final_actions.device).float()
        abs_error = (raw_on_device - pred_step0.float()).abs()
        max_abs_err = float(abs_error.max())
        mean_abs_err = float(abs_error.mean())

        log.info(f"  Real action (current frame)     : {raw_on_device.tolist()}")
        log.info(f"  Predicted action (step 0)       : {pred_step0.float().tolist()}")
        log.info(f"  Absolute error per dim          : {abs_error.tolist()}")
        log.info(f"  Max absolute error              : {max_abs_err:.6f}")
        log.info(f"  Mean absolute error             : {mean_abs_err:.6f}")

        results["prediction_vs_real"] = {
            "real_action": raw_on_device.tolist(),
            "predicted_step0": pred_step0.float().tolist(),
            "abs_error_per_dim": abs_error.tolist(),
            "max_absolute_error": max_abs_err,
            "mean_absolute_error": mean_abs_err,
        }

        # Also compare full chunk stats
        chunk_stats = tensor_stats(pred_chunk, "predicted_full_chunk")
        log.info(f"\n  Full predicted chunk ({pred_chunk.shape[0]} steps):")
        log.info(f"    min   : {chunk_stats['min']:.6f}")
        log.info(f"    max   : {chunk_stats['max']:.6f}")
        log.info(f"    mean  : {chunk_stats['mean']:.6f}")
        log.info(f"    std   : {chunk_stats['std']:.6f}")
        results["predicted_chunk_stats"] = chunk_stats
    else:
        log.info("  No real action available for comparison")
        results["prediction_vs_real"] = None

    return results


def debug_minivla(
    sample: dict,
    policy,
    preprocessor,
    postprocessor,
    config,
    device: str,
    logger: logging.Logger | None = None,
) -> dict:
    """Debug MiniVLA action pipeline, including VQ encode/decode."""
    log = logger or logging.getLogger("action_pipeline_debug")
    print_separator("MiniVLA ACTION PIPELINE DEBUG", log)

    results = {}

    # 1. Raw sample
    results["raw"] = _print_raw_sample_info(sample, "minivla", log)

    # 2. Build observation batch
    raw_batch = _build_observation_batch(sample, "minivla", config)

    # 3. Preprocessor
    print_separator("PREPROCESSOR (MiniVLA)", log)
    pre_batch = preprocessor(raw_batch)
    log.info(f"  Preprocessor output keys: {list(pre_batch.keys())}")

    if ACTION in pre_batch:
        pre_act = pre_batch[ACTION]
        if isinstance(pre_act, torch.Tensor):
            stats = tensor_stats(pre_act, "preprocessed_action_normalized")
        else:
            stats = tensor_stats(torch.from_numpy(pre_act), "preprocessed_action_normalized")
        print_stats(stats, "PRE", log)
        results["preprocessed_action"] = stats
    else:
        log.info("  [PRE] No action in preprocessed batch")
        results["preprocessed_action"] = None

    # 4. VQ action tokenizer diagnostics (if VQ mode)
    core = policy.model
    is_vq = core.config.is_vq_mode
    log.info(f"\n  [VQ] VQ mode: {is_vq}")
    results["vq_mode"] = is_vq

    if ACTION in sample and is_vq:
        print_separator("MiniVLA VQ ACTION ENCODE/DECODE", log)
        raw_act = sample[ACTION]
        if isinstance(raw_act, np.ndarray):
            raw_act_t = torch.from_numpy(raw_act).float()
        else:
            raw_act_t = raw_act.float()

        # Ensure shape [B, T, A]
        if raw_act_t.ndim == 1:
            raw_act_t = raw_act_t.unsqueeze(0).unsqueeze(0)
        elif raw_act_t.ndim == 2:
            raw_act_t = raw_act_t.unsqueeze(0)

        action_tokenizer = core.action_tokenizer
        vq_vae = action_tokenizer.vq_vae

        # Step 1: Raw action -> VQ codes via VqVae.get_code()
        with torch.no_grad():
            state_vq, vq_code = vq_vae.get_code(raw_act_t.to(device), required_recon=False)
        code_stats = tensor_stats(vq_code, "vq_encoded_codes")
        print_stats(code_stats, "VQ_ENCODE", log)
        results["vq_encoded"] = code_stats

        # Step 2: VQ codes -> decoded action via draw_code_forward + get_action_from_latent
        with torch.no_grad():
            latent = vq_vae.draw_code_forward(vq_code)
            decoded = vq_vae.get_action_from_latent(latent)
        decoded_stats = tensor_stats(decoded, "vq_decoded_action")
        print_stats(decoded_stats, "VQ_DECODE", log)
        results["vq_decoded"] = decoded_stats

        # Step 3: Reconstruction error
        raw_on_device = raw_act_t.to(device)
        min_t = min(raw_on_device.shape[1], decoded.shape[1])
        raw_slice = raw_on_device[:, :min_t, :]
        dec_slice = decoded[:, :min_t, :]
        max_abs_err = float((raw_slice - dec_slice).abs().max())
        mean_abs_err = float((raw_slice - dec_slice).abs().mean())
        log.info(f"\n  [VQ] Max absolute error : {max_abs_err:.6f}")
        log.info(f"  [VQ] Mean absolute error: {mean_abs_err:.6f}")
        results["vq_reconstruction_error"] = {
            "max_absolute_error": max_abs_err,
            "mean_absolute_error": mean_abs_err,
        }

    # 5. Policy forward
    print_separator("POLICY predict_action_chunk (MiniVLA)", log)
    policy.reset()

    with torch.no_grad():
        pred_actions = policy.predict_action_chunk(pre_batch)

    stats = tensor_stats(pred_actions, "model_prediction_action")
    print_stats(stats, "PRED", log)
    if pred_actions.ndim >= 2:
        log.info(f"    chunk_length: {pred_actions.shape[-2]}")
        stats["chunk_length"] = int(pred_actions.shape[-2])
    results["prediction_action"] = stats

    # 6. Postprocessor
    print_separator("POSTPROCESSOR (MiniVLA)", log)
    final_action_obj = postprocessor(pred_actions)

    if isinstance(final_action_obj, torch.Tensor):
        final_actions = final_action_obj
    elif isinstance(final_action_obj, dict) and "action" in final_action_obj:
        final_actions = final_action_obj["action"]
    else:
        final_actions = pred_actions

    if isinstance(final_actions, torch.Tensor):
        stats = tensor_stats(final_actions, "final_action_unnormalized")
    else:
        stats = tensor_stats(torch.from_numpy(final_actions), "final_action_unnormalized")
    print_stats(stats, "FINAL", log)
    results["final_action"] = stats

    # 7. Compare model prediction vs real action
    print_separator("PREDICTION vs REAL ACTION COMPARISON", log)
    if ACTION in sample:
        raw_act = sample[ACTION]
        if isinstance(raw_act, np.ndarray):
            raw_act_t = torch.from_numpy(raw_act)
        else:
            raw_act_t = raw_act

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
        max_abs_err = float(abs_error.max())
        mean_abs_err = float(abs_error.mean())

        log.info(f"  Real action (current frame)     : {raw_on_device.tolist()}")
        log.info(f"  Predicted action (step 0)       : {pred_step0.float().tolist()}")
        log.info(f"  Absolute error per dim          : {abs_error.tolist()}")
        log.info(f"  Max absolute error              : {max_abs_err:.6f}")
        log.info(f"  Mean absolute error             : {mean_abs_err:.6f}")

        results["prediction_vs_real"] = {
            "real_action": raw_on_device.tolist(),
            "predicted_step0": pred_step0.float().tolist(),
            "abs_error_per_dim": abs_error.tolist(),
            "max_absolute_error": max_abs_err,
            "mean_absolute_error": mean_abs_err,
        }

        chunk_stats = tensor_stats(pred_chunk, "predicted_full_chunk")
        log.info(f"\n  Full predicted chunk ({pred_chunk.shape[0]} steps):")
        log.info(f"    min   : {chunk_stats['min']:.6f}")
        log.info(f"    max   : {chunk_stats['max']:.6f}")
        log.info(f"    mean  : {chunk_stats['mean']:.6f}")
        log.info(f"    std   : {chunk_stats['std']:.6f}")
        results["predicted_chunk_stats"] = chunk_stats
    else:
        log.info("  No real action available for comparison")
        results["prediction_vs_real"] = None

    return results


def debug_smolvla(
    sample: dict,
    policy,
    preprocessor,
    postprocessor,
    config,
    device: str,
    logger: logging.Logger | None = None,
) -> dict:
    """Debug SmolVLA action pipeline (baseline)."""
    log = logger or logging.getLogger("action_pipeline_debug")
    print_separator("SmolVLA ACTION PIPELINE DEBUG", log)

    results = {}

    # 1. Raw sample
    results["raw"] = _print_raw_sample_info(sample, "smolvla", log)

    # 2. Build observation batch
    raw_batch = _build_observation_batch(sample, "smolvla", config)

    # 3. Preprocessor
    print_separator("PREPROCESSOR (SmolVLA)", log)
    pre_batch = preprocessor(raw_batch)
    log.info(f"  Preprocessor output keys: {list(pre_batch.keys())}")

    if ACTION in pre_batch:
        pre_act = pre_batch[ACTION]
        if isinstance(pre_act, torch.Tensor):
            stats = tensor_stats(pre_act, "preprocessed_action_normalized")
        else:
            stats = tensor_stats(torch.from_numpy(pre_act), "preprocessed_action_normalized")
        print_stats(stats, "PRE", log)
        results["preprocessed_action"] = stats
    else:
        log.info("  [PRE] No action in preprocessed batch")
        results["preprocessed_action"] = None

    if OBS_STATE in pre_batch:
        pre_st = pre_batch[OBS_STATE]
        if isinstance(pre_st, torch.Tensor):
            stats = tensor_stats(pre_st, "preprocessed_state")
        else:
            stats = tensor_stats(torch.from_numpy(pre_st), "preprocessed_state")
        print_stats(stats, "PRE", log)
        results["preprocessed_state"] = stats

    # 4. Policy forward
    print_separator("POLICY predict_action_chunk (SmolVLA)", log)
    policy.reset()

    with torch.no_grad():
        pred_actions = policy.predict_action_chunk(pre_batch)

    stats = tensor_stats(pred_actions, "model_prediction_action")
    print_stats(stats, "PRED", log)
    if pred_actions.ndim >= 2:
        log.info(f"    chunk_length: {pred_actions.shape[-2]}")
        stats["chunk_length"] = int(pred_actions.shape[-2])
    results["prediction_action"] = stats

    # 5. Postprocessor
    print_separator("POSTPROCESSOR (SmolVLA)", log)
    final_action_obj = postprocessor(pred_actions)

    if isinstance(final_action_obj, torch.Tensor):
        final_actions = final_action_obj
    elif isinstance(final_action_obj, dict) and "action" in final_action_obj:
        final_actions = final_action_obj["action"]
    else:
        final_actions = pred_actions

    if isinstance(final_actions, torch.Tensor):
        stats = tensor_stats(final_actions, "final_action_unnormalized")
    else:
        stats = tensor_stats(torch.from_numpy(final_actions), "final_action_unnormalized")
    print_stats(stats, "FINAL", log)
    results["final_action"] = stats

    # 6. Config info
    print_separator("SmolVLA CONFIG INFO", log)
    log.info(f"  chunk_size        : {config.chunk_size}")
    log.info(f"  n_action_steps    : {config.n_action_steps}")
    log.info(f"  max_action_dim    : {config.max_action_dim}")
    log.info(f"  adapt_to_pi_aloha : {config.adapt_to_pi_aloha}")
    log.info(f"  normalization     : {config.normalization_mapping}")
    results["config_info"] = {
        "chunk_size": config.chunk_size,
        "n_action_steps": config.n_action_steps,
        "max_action_dim": config.max_action_dim,
        "adapt_to_pi_aloha": config.adapt_to_pi_aloha,
        "normalization_mapping": {k: str(v) for k, v in config.normalization_mapping.items()},
    }

    # 7. Compare model prediction vs real action
    print_separator("PREDICTION vs REAL ACTION COMPARISON", log)
    if ACTION in sample:
        raw_act = sample[ACTION]
        if isinstance(raw_act, np.ndarray):
            raw_act_t = torch.from_numpy(raw_act)
        else:
            raw_act_t = raw_act

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
        max_abs_err = float(abs_error.max())
        mean_abs_err = float(abs_error.mean())

        log.info(f"  Real action (current frame)     : {raw_on_device.tolist()}")
        log.info(f"  Predicted action (step 0)       : {pred_step0.float().tolist()}")
        log.info(f"  Absolute error per dim          : {abs_error.tolist()}")
        log.info(f"  Max absolute error              : {max_abs_err:.6f}")
        log.info(f"  Mean absolute error             : {mean_abs_err:.6f}")

        results["prediction_vs_real"] = {
            "real_action": raw_on_device.tolist(),
            "predicted_step0": pred_step0.float().tolist(),
            "abs_error_per_dim": abs_error.tolist(),
            "max_absolute_error": max_abs_err,
            "mean_absolute_error": mean_abs_err,
        }

        chunk_stats = tensor_stats(pred_chunk, "predicted_full_chunk")
        log.info(f"\n  Full predicted chunk ({pred_chunk.shape[0]} steps):")
        log.info(f"    min   : {chunk_stats['min']:.6f}")
        log.info(f"    max   : {chunk_stats['max']:.6f}")
        log.info(f"    mean  : {chunk_stats['mean']:.6f}")
        log.info(f"    std   : {chunk_stats['std']:.6f}")
        results["predicted_chunk_stats"] = chunk_stats
    else:
        log.info("  No real action available for comparison")
        results["prediction_vs_real"] = None

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

POLICY_DEBUG_FN = {
    "tinyvla_s": debug_tinyvla,
    "tinyvla_b": debug_tinyvla,
    "minivla": debug_minivla,
    "smolvla": debug_smolvla,
}


def main():
    parser = argparse.ArgumentParser(description="Action Pipeline Debug Tool")
    parser.add_argument(
        "--policy_type",
        type=str,
        required=True,
        choices=list(POLICY_DEBUG_FN.keys()),
        help="Policy type: tinyvla_s, tinyvla_b, minivla, smolvla",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to pretrained_model directory (checkpoint)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory of the LeRobot dataset (e.g. personal/work2/dataset_view/disassemble-v3_corner)",
    )
    parser.add_argument(
        "--episode_index",
        type=int,
        default=0,
        help="Episode index to sample from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda / cpu)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON file path (default: action_pipeline_debug_results.json in script dir)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Log file path for console output (default: action_pipeline_debug.log in script dir)",
    )
    args = parser.parse_args()

    # Setup logging (console + file)
    log_file = args.log_file
    if log_file is None:
        log_file = str(Path(__file__).parent / "action_pipeline_debug.log")
    logger = setup_logging(log_file)

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Policy type: {args.policy_type}")
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Dataset root: {args.dataset_root}")
    logger.info(f"Episode index: {args.episode_index}")
    logger.info(f"Log file: {log_file}")

    # 1. Load dataset sample
    print_separator("LOADING DATASET SAMPLE", logger)
    sample, features, dataset = load_dataset_sample(args.dataset_root, args.episode_index, logger)

    # Get dataset stats for processor loading
    ds_stats = None
    if hasattr(dataset.meta, "stats"):
        ds_stats = dataset.meta.stats
        logger.info(f"Dataset stats keys: {list(ds_stats.keys())}")

    # 2. Load policy + processors
    policy, preprocessor, postprocessor, config = load_policy_and_processors(
        args.policy_type,
        args.checkpoint_path,
        device,
        dataset_stats=ds_stats,
        logger=logger,
    )

    # 3. Run debug
    debug_fn = POLICY_DEBUG_FN[args.policy_type]
    try:
        results = debug_fn(sample, policy, preprocessor, postprocessor, config, device, logger)
    except Exception as e:
        logger.error(f"\nERROR during debug run:")
        logger.error(traceback.format_exc())
        results = {"error": str(e), "traceback": traceback.format_exc()}

    # 4. Add metadata
    results["metadata"] = {
        "policy_type": args.policy_type,
        "checkpoint_path": args.checkpoint_path,
        "dataset_root": args.dataset_root,
        "episode_index": args.episode_index,
        "device": device,
        "dataset_features": {k: str(v) for k, v in features.items()},
        "log_file": log_file,
    }

    # 5. Save JSON
    output_file = args.output_file
    if output_file is None:
        output_file = str(Path(__file__).parent / "action_pipeline_debug_results.json")

    # Append to existing results if file exists
    existing = {}
    if Path(output_file).exists():
        try:
            with open(output_file, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    existing[args.policy_type] = results

    with open(output_file, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Log saved to: {log_file}")
    print_separator("DONE", logger)


if __name__ == "__main__":
    main()