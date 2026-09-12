#!/usr/bin/env python
"""
Action verification script for MiniVLA.

Diagnostic tool to determine:
1. Whether MiniVLA action prediction is close to expert actions
2. Whether action errors are acceptable for successful task execution

Key features:
- Auto-detects MiniVLA variant (minivla, minivla_t2, minivla_wrist, etc.)
- Properly normalizes expert actions using QUANTILE normalization (same as training)
- Reports both normalized-space and raw-space metrics separately
- Validates round-trip normalize->unnormalize correctness
- Supports one_step and chunk evaluation modes
- Handles VQ and non-VQ action tokenizers
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from torch.nn.functional import cosine_similarity
from collections import deque

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.minivla.modeling_minivla import MiniVLAPolicy
from lerobot.policies.minivla.configuration_minivla import MiniVLAConfig
from lerobot.envs import preprocess_observation
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.configs.types import FeatureType, NormalizationMode

# ============================================================
# Configuration
# ============================================================
CHECKPOINT_PATH = str(
    PROJECT_ROOT / "personal" / "work2" / "differentvlm" / "minivla" /
    "experiments" / "random_200_seed42_disassemblev3corner_minivla_random" /
    "checkpoints" / "checkpoints" / "050000" / "pretrained_model"
)
DATASET_PATH = str(PROJECT_ROOT / "personal" / "work2" / "dataset_view" / "disassemble-v3_corner")
TASK_DESCRIPTION = "Pick a nut out of a peg"

# Evaluation settings
NUM_SEEDS_PER_FRAME = 5
BASE_SEED = 42
SEED_STRIDE = 1


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def load_model_and_preprocessor(checkpoint_path, device="cuda"):
    """Load MiniVLA policy and preprocessor from checkpoint."""
    print(f"Loading MiniVLA from: {checkpoint_path}")
    
    # Load config from checkpoint
    print("\nLoading config from checkpoint...")
    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    
    # Detect policy type
    policy_name = getattr(config, 'name', 'unknown')
    print(f"Config class: {type(config).__name__}")
    print(f"Policy name: {policy_name}")
    
    # Load MiniVLA policy
    print("\nLoading MiniVLA policy via from_pretrained...")
    policy = MiniVLAPolicy.from_pretrained(
        pretrained_name_or_path=checkpoint_path,
    )
    
    policy = policy.to(device)
    policy.eval()
    
    # Create preprocessor and postprocessor
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_path,
    )
    
    # Print key configuration
    print(f"\n{'='*60}")
    print("Policy Configuration:")
    print(f"  Policy name: {policy_name}")
    print(f"  chunk_size: {config.chunk_size}")
    print(f"  n_action_steps: {config.n_action_steps}")
    print(f"  action_tokenizer_type: {config.action_tokenizer_type}")
    print(f"  is_vq_mode: {config.is_vq_mode}")
    print(f"  action normalization: {config.normalization_mapping.get(FeatureType.ACTION, 'unknown')}")
    print(f"  state normalization: {config.normalization_mapping.get(FeatureType.STATE, 'unknown')}")
    print(f"  primary_image_key: {config.primary_image_key}")
    print(f"  wrist_image_key: {config.wrist_image_key}")
    print(f"  use_wrist_image: {config.use_wrist_image}")
    print(f"{'='*60}")
    
    return policy, preprocessor, postprocessor


def validate_normalization_sanity(policy, preprocessor, postprocessor, dataset):
    """Sanity checks for action normalization."""
    print("\n" + "=" * 80)
    print("Sanity Check: Action Normalization")
    print("=" * 80)
    
    # Check ACTION normalization mode
    action_norm_mode = policy.config.normalization_mapping.get(FeatureType.ACTION)
    print(f"\nACTION normalization mode: {action_norm_mode}")
    
    # Get action stats from postprocessor
    for step in postprocessor.steps:
        if isinstance(step, UnnormalizerProcessorStep):
            action_stats = step.stats.get("action", {})
            if action_stats:
                print(f"\nAction stats from postprocessor:")
                if "min" in action_stats:
                    print(f"  min: {action_stats['min']}")
                if "max" in action_stats:
                    print(f"  max: {action_stats['max']}")
                if "mean" in action_stats:
                    print(f"  mean: {action_stats['mean']}")
                if "std" in action_stats:
                    print(f"  std: {action_stats['std']}")
                if "q01" in action_stats:
                    print(f"  q01: {action_stats['q01']}")
                if "q99" in action_stats:
                    print(f"  q99: {action_stats['q99']}")
            break
    
    # Find normalizer in preprocessor
    normalizer = None
    for step in preprocessor.steps:
        if isinstance(step, NormalizerProcessorStep):
            normalizer = step
            break
    
    if normalizer is None:
        print("ERROR: No NormalizerProcessorStep found in preprocessor!")
        return False
    
    # Round-trip test: expert_action -> normalize -> unnormalize -> should recover original
    print("\nRound-trip test: expert_action -> normalize -> unnormalize")
    frame = dataset[0]
    expert_action_raw = frame["action"]
    expert_action_tensor = torch.as_tensor(expert_action_raw, dtype=torch.float32)
    
    # Normalize
    action_normalized = normalizer._apply_transform(
        expert_action_tensor.unsqueeze(0).unsqueeze(0),  # [1, 1, action_dim]
        "action",
        FeatureType.ACTION,
        inverse=False
    ).squeeze()
    
    # Unnormalize using postprocessor
    for step in postprocessor.steps:
        if isinstance(step, UnnormalizerProcessorStep):
            action_unnormalized = step._apply_transform(
                action_normalized.unsqueeze(0).unsqueeze(0),
                "action",
                FeatureType.ACTION,
                inverse=True
            ).squeeze()
            break
    
    round_trip_error = (action_unnormalized - expert_action_tensor).abs().max().item()
    print(f"  Original:     {expert_action_tensor.numpy()}")
    print(f"  Normalized:   {action_normalized.numpy()}")
    print(f"  Unnormalized: {action_unnormalized.numpy()}")
    print(f"  Round-trip max absolute error: {round_trip_error:.2e}")
    
    if round_trip_error > 1e-2:
        print(f"ERROR: Round-trip error {round_trip_error:.2e} > 1e-2!")
        return False
    else:
        print(f"PASS: Round-trip error < 1e-2 (acceptable)")
    
    return True


def normalize_expert_action(normalizer, expert_action_raw):
    """Normalize a raw expert action using the SAME normalization as training."""
    expert_action_tensor = torch.as_tensor(expert_action_raw, dtype=torch.float32)
    action_normalized = normalizer._apply_transform(
        expert_action_tensor.unsqueeze(0).unsqueeze(0),
        "action",
        FeatureType.ACTION,
        inverse=False
    ).squeeze()
    return action_normalized


def postprocessor_unnormalize(postprocessor, action_normalized):
    """Unnormalize an action using postprocessor."""
    for step in postprocessor.steps:
        if isinstance(step, UnnormalizerProcessorStep):
            action_unnormalized = step._apply_transform(
                action_normalized.unsqueeze(0).unsqueeze(0),
                "action",
                FeatureType.ACTION,
                inverse=True
            ).squeeze()
            # Convert to numpy if it's a tensor
            if hasattr(action_unnormalized, 'numpy'):
                if action_unnormalized.dtype == torch.bfloat16:
                    action_unnormalized = action_unnormalized.float()
                return action_unnormalized.cpu().numpy()
            return action_unnormalized
    # If no unnormalizer found, just convert to numpy
    if hasattr(action_normalized, 'numpy'):
        if action_normalized.dtype == torch.bfloat16:
            action_normalized = action_normalized.float()
        return action_normalized.cpu().numpy()
    return action_normalized


def postprocessor_unnormalize_chunk(postprocessor, chunk_actions):
    """Unnormalize a chunk of actions [batch, chunk_size, action_dim]."""
    for step in postprocessor.steps:
        if isinstance(step, UnnormalizerProcessorStep):
            chunk_unnormalized = step._apply_transform(
                chunk_actions,
                "action",
                FeatureType.ACTION,
                inverse=True
            )
            # Convert to numpy if it's a tensor
            if hasattr(chunk_unnormalized, 'numpy'):
                if chunk_unnormalized.dtype == torch.bfloat16:
                    chunk_unnormalized = chunk_unnormalized.float()
                return chunk_unnormalized.cpu().numpy()
            return chunk_unnormalized
    # If no unnormalizer found, just convert to numpy
    if hasattr(chunk_actions, 'numpy'):
        if chunk_actions.dtype == torch.bfloat16:
            chunk_actions = chunk_actions.float()
        return chunk_actions.cpu().numpy()
    return chunk_actions


def build_observation(frame, config, task_description):
    """Build observation dict from dataset frame based on MiniVLA config."""
    observation = {
        "observation.state": frame["observation.state"],
        "task": task_description,
    }
    
    # Add primary image
    primary_key = config.primary_image_key
    if primary_key and primary_key in frame:
        observation["observation.images.top"] = frame[primary_key]
    elif "observation.images.corner" in frame:
        observation["observation.images.top"] = frame["observation.images.corner"]
    elif "observation.images.primary" in frame:
        observation["observation.images.top"] = frame["observation.images.primary"]
    
    # Add wrist image if configured
    if config.use_wrist_image:
        wrist_key = config.wrist_image_key
        if wrist_key and wrist_key in frame:
            observation["observation.images.wrist"] = frame[wrist_key]
        elif "observation.images.gripperPOV" in frame:
            observation["observation.images.wrist"] = frame["observation.images.gripperPOV"]
    
    return observation


def compute_seed_metrics(
    expert_action_raw,
    expert_action_normalized,
    model_normalized,
    model_raw,
    seed,
):
    """Compute metrics for a single seed prediction."""
    # 4D MAE
    normalized_mae_4d = np.abs(expert_action_normalized - model_normalized).mean()
    raw_mae_4d = np.abs(expert_action_raw - model_raw).mean()
    
    # XYZ MAE (first 3 dimensions)
    normalized_xyz_mae = np.abs(expert_action_normalized[:3] - model_normalized[:3]).mean()
    raw_xyz_mae = np.abs(expert_action_raw[:3] - model_raw[:3]).mean()
    
    # Gripper error (4th dimension)
    normalized_gripper_error = abs(expert_action_normalized[3] - model_normalized[3])
    raw_gripper_error = abs(expert_action_raw[3] - model_raw[3])
    
    # Cosine similarity for XYZ
    expert_xyz = torch.tensor(expert_action_normalized[:3], dtype=torch.float32)
    model_xyz = torch.tensor(model_normalized[:3], dtype=torch.float32)
    xyz_cosine = cosine_similarity(expert_xyz.unsqueeze(0), model_xyz.unsqueeze(0)).item()
    
    return {
        "seed": int(seed),
        "normalized_mae_4d": float(normalized_mae_4d),
        "raw_mae_4d": float(raw_mae_4d),
        "normalized_xyz_mae": float(normalized_xyz_mae),
        "raw_xyz_mae": float(raw_xyz_mae),
        "normalized_gripper_error": float(normalized_gripper_error),
        "raw_gripper_error": float(raw_gripper_error),
        "xyz_cosine_similarity": float(xyz_cosine),
        "predicted_action_normalized": model_normalized.tolist(),
        "predicted_action_raw": model_raw.tolist(),
    }


def compute_frame_distribution(
    expert_action_raw,
    expert_action_normalized,
    seed_results,
    seed_normalized_maes,
    seed_raw_maes,
    eval_mode,
):
    """Compute per-frame distribution statistics across seeds."""
    num_seeds = len(seed_results)
    all_normalized = np.array([s["predicted_action_normalized"] for s in seed_results])
    all_raw = np.array([s["predicted_action_raw"] for s in seed_results])
    
    # Sample statistics
    sample_mean_normalized = all_normalized.mean(axis=0)
    sample_std_normalized = all_normalized.std(axis=0)
    sample_mean_raw = all_raw.mean(axis=0)
    sample_std_raw = all_raw.std(axis=0)
    
    # Mean action to GT
    mean_action_normalized_mae = np.abs(sample_mean_normalized - expert_action_normalized).mean()
    mean_action_raw_mae = np.abs(sample_mean_raw - expert_action_raw).mean()
    
    # Best/median/worst based on 4D MAE
    normalized_maes_array = np.array(seed_normalized_maes)
    raw_maes_array = np.array(seed_raw_maes)
    
    best_idx = np.argmin(raw_maes_array)
    worst_idx = np.argmax(raw_maes_array)
    median_idx = np.argsort(raw_maes_array)[num_seeds // 2]
    
    # Seed-to-seed variance
    mean_sample_std_normalized_per_dim = sample_std_normalized.mean()
    mean_sample_std_raw_per_dim = sample_std_raw.mean()
    
    # Pairwise L1 distance
    pairwise_distances = []
    for i in range(num_seeds):
        for j in range(i + 1, num_seeds):
            dist = np.abs(all_normalized[i] - all_normalized[j]).mean()
            pairwise_distances.append(dist)
    mean_pairwise_distance = np.mean(pairwise_distances) if pairwise_distances else 0.0
    
    return {
        "expert_action_raw": expert_action_raw.tolist(),
        "expert_action_normalized": expert_action_normalized.tolist(),
        "seeds": seed_results,
        "sample_mean_normalized": sample_mean_normalized.tolist(),
        "sample_std_normalized": sample_std_normalized.tolist(),
        "sample_mean_raw": sample_mean_raw.tolist(),
        "sample_std_raw": sample_std_raw.tolist(),
        "mean_action_normalized_mae": float(mean_action_normalized_mae),
        "mean_action_raw_mae": float(mean_action_raw_mae),
        "best_seed": seed_results[best_idx]["seed"],
        "best_normalized_mae_4d": float(normalized_maes_array[best_idx]),
        "best_raw_mae_4d": float(raw_maes_array[best_idx]),
        "median_seed": seed_results[median_idx]["seed"],
        "median_normalized_mae_4d": float(normalized_maes_array[median_idx]),
        "median_raw_mae_4d": float(raw_maes_array[median_idx]),
        "worst_seed": seed_results[worst_idx]["seed"],
        "worst_normalized_mae_4d": float(normalized_maes_array[worst_idx]),
        "worst_raw_mae_4d": float(raw_maes_array[worst_idx]),
        "mean_sample_std_normalized_per_dim": float(mean_sample_std_normalized_per_dim),
        "mean_sample_std_raw_per_dim": float(mean_sample_std_raw_per_dim),
        "mean_pairwise_l1_distance": float(mean_pairwise_distance),
    }


def run_evaluation(
    policy,
    preprocessor,
    postprocessor,
    dataset,
    episode_index=0,
    num_seeds=NUM_SEEDS_PER_FRAME,
    base_seed=BASE_SEED,
    seed_stride=SEED_STRIDE,
    eval_mode="one_step",
    task_description=TASK_DESCRIPTION,
    device="cuda",
):
    """Run evaluation on a single episode."""
    
    # Find normalizer in preprocessor
    normalizer = None
    for step in preprocessor.steps:
        if isinstance(step, NormalizerProcessorStep):
            normalizer = step
            break
    
    if normalizer is None:
        raise ValueError("No NormalizerProcessorStep found in preprocessor!")
    
    # Get episode frame indices
    episode_indices = np.array(dataset.hf_dataset["episode_index"])
    mask = episode_indices == episode_index
    frame_indices = np.where(mask)[0]
    
    if len(frame_indices) == 0:
        print(f"ERROR: Episode {episode_index} not found in dataset")
        return [], [], []
    
    total_frames = len(frame_indices)
    chunk_size = policy.config.chunk_size
    
    print(f"\nEvaluating episode {episode_index} with {total_frames} frames")
    print(f"Number of seeds per frame: {num_seeds}")
    print(f"Seed stride: {seed_stride}")
    print(f"Evaluation mode: {eval_mode}")
    print(f"Task description: '{task_description}'")
    print(f"Base seed: {base_seed}")
    print(f"Chunk size: {chunk_size}")
    print(f"is_vq_mode: {policy.config.is_vq_mode}")
    
    all_results = []
    all_normalized_maes_4d = []
    all_raw_maes_4d = []
    
    for idx, frame_idx in enumerate(tqdm(frame_indices, desc=f"Episode {episode_index}")):
        frame = dataset[frame_idx]
        expert_action_raw = frame["action"].numpy()
        observation_state = frame["observation.state"]
        
        # Normalize expert action
        expert_action_normalized = normalize_expert_action(normalizer, expert_action_raw)
        
        # Build observation
        observation = build_observation(frame, policy.config, task_description)
        
        observation_tensor = preprocess_observation(observation)
        observation_tensor = preprocessor(observation_tensor)
        
        # Debug output for first frame
        debug_first_frame = (idx == 0)
        if debug_first_frame:
            print(f"\n  {'='*60}")
            print(f"  [DIAGNOSTIC] Frame {frame_idx} - First Frame Action Comparison")
            print(f"  {'='*60}")
        
        frame_seed_results = []
        frame_seed_normalized_maes_4d = []
        frame_seed_raw_maes_4d = []
        
        if eval_mode == "one_step":
            # One-step mode: each seed is independent single-step inference
            for seed_idx in range(num_seeds):
                current_seed = base_seed + seed_idx * seed_stride
                
                # Reset and set seed for each inference
                policy.reset()
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(current_seed)
                    torch.cuda.manual_seed_all(current_seed)
                
                with torch.inference_mode():
                    raw_action = policy.select_action(observation_tensor)
                
                # MiniVLA select_action returns RAW action [B, action_dim]
                # NOT normalized action (unlike TinyVLA diffusion)
                model_raw = raw_action.to("cpu").to(dtype=torch.float32).numpy()[0]
                
                # To get normalized action, we need to normalize the raw action
                model_normalized = normalize_expert_action(normalizer, model_raw).numpy()
                
                # Debug output for first frame
                if debug_first_frame and seed_idx == 0:
                    print(f"  GT raw action: {expert_action_raw}")
                    print(f"  GT normalized action: {expert_action_normalized.numpy()}")
                    print(f"  Model raw action: {model_raw}")
                    print(f"  Model normalized action: {model_normalized}")
                    normalized_error = np.abs(expert_action_normalized.numpy() - model_normalized).mean()
                    raw_error = np.abs(expert_action_raw - model_raw).mean()
                    print(f"\n  Normalized error (MAE): {normalized_error:.4f}")
                    print(f"  Raw error (MAE): {raw_error:.4f}")
                    print(f"  {'='*60}")
                    if normalized_error < 0.1 and raw_error < 0.1:
                        print(f"  => Both errors < 0.1: Model action prediction looks GOOD")
                    elif normalized_error < 0.2 and raw_error < 0.2:
                        print(f"  => Both errors < 0.2: Model action prediction is ACCEPTABLE")
                    else:
                        print(f"  => Both errors >= 0.2: Model action prediction is POOR")
                    print(f"  {'='*60}\n")
                
                seed_result = compute_seed_metrics(
                    expert_action_raw=expert_action_raw,
                    expert_action_normalized=expert_action_normalized.numpy(),
                    model_normalized=model_normalized,
                    model_raw=model_raw,
                    seed=current_seed,
                )
                
                frame_seed_results.append(seed_result)
                frame_seed_normalized_maes_4d.append(seed_result["normalized_mae_4d"])
                frame_seed_raw_maes_4d.append(seed_result["raw_mae_4d"])
        
        elif eval_mode == "chunk":
            # Chunk mode: each seed predicts a full chunk independently
            for seed_idx in range(num_seeds):
                current_seed = base_seed + seed_idx * seed_stride
                
                # Reset and set seed for each chunk prediction
                policy.reset()
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(current_seed)
                    torch.cuda.manual_seed_all(current_seed)
                
                with torch.inference_mode():
                    chunk_actions = policy.predict_action_chunk(observation_tensor)
                
                # chunk_actions: [1, chunk_size, action_dim] normalized
                chunk_normalized = chunk_actions.to("cpu").to(dtype=torch.float32).numpy()[0]
                chunk_raw = postprocessor_unnormalize_chunk(postprocessor, chunk_actions)[0]
                
                # Get GT chunk: future chunk_size frames from current frame
                gt_chunk_raw = []
                gt_chunk_normalized = []
                valid_mask = []
                
                for t in range(chunk_size):
                    future_frame_idx = frame_idx + t
                    if future_frame_idx < len(dataset) and dataset.hf_dataset[future_frame_idx]["episode_index"] == episode_index:
                        future_frame = dataset[future_frame_idx]
                        future_expert_raw = future_frame["action"].numpy()
                        future_expert_normalized = normalize_expert_action(normalizer, future_expert_raw).numpy()
                        gt_chunk_raw.append(future_expert_raw)
                        gt_chunk_normalized.append(future_expert_normalized)
                        valid_mask.append(True)
                    else:
                        gt_chunk_raw.append(np.zeros_like(expert_action_raw))
                        gt_chunk_normalized.append(np.zeros_like(expert_action_normalized.numpy()))
                        valid_mask.append(False)
                
                gt_chunk_raw = np.array(gt_chunk_raw)
                gt_chunk_normalized = np.array(gt_chunk_normalized)
                valid_mask = np.array(valid_mask)
                
                # Compute chunk-level metrics (only valid steps)
                chunk_result = compute_chunk_seed_metrics(
                    gt_chunk_raw=gt_chunk_raw,
                    gt_chunk_normalized=gt_chunk_normalized,
                    predicted_chunk_normalized=chunk_normalized,
                    predicted_chunk_raw=chunk_raw,
                    valid_mask=valid_mask,
                    seed=current_seed,
                )
                
                frame_seed_results.append(chunk_result)
                frame_seed_normalized_maes_4d.append(chunk_result["normalized_mae_4d"])
                frame_seed_raw_maes_4d.append(chunk_result["raw_mae_4d"])
        
        # Compute per-frame distribution statistics
        frame_result = compute_frame_distribution(
            expert_action_raw=expert_action_raw,
            expert_action_normalized=expert_action_normalized.numpy(),
            seed_results=frame_seed_results,
            seed_normalized_maes=frame_seed_normalized_maes_4d,
            seed_raw_maes=frame_seed_raw_maes_4d,
            eval_mode=eval_mode,
        )
        
        all_results.append(frame_result)
        all_normalized_maes_4d.extend(frame_seed_normalized_maes_4d)
        all_raw_maes_4d.extend(frame_seed_raw_maes_4d)
        
        # Print progress every 10 frames
        if idx % 10 == 0:
            print(f"\n  [{idx+1}/{total_frames}] Frame {frame_idx}")
            print(f"    Expert raw: {expert_action_raw}")
            print(f"    Expert normalized: {expert_action_normalized.numpy()}")
            print(f"    Mean normalized MAE: {frame_result['mean_action_normalized_mae']:.4f}")
            print(f"    Best normalized MAE: {frame_result['best_normalized_mae_4d']:.4f}")
            print(f"    Worst normalized MAE: {frame_result['worst_normalized_mae_4d']:.4f}")
            sample_std_normalized_mean = np.mean(frame_result['sample_std_normalized'])
            print(f"    Sample std (normalized): {sample_std_normalized_mean:.4f}")
    
    return all_results, all_normalized_maes_4d, all_raw_maes_4d


def compute_chunk_seed_metrics(
    gt_chunk_raw,
    gt_chunk_normalized,
    predicted_chunk_normalized,
    predicted_chunk_raw,
    valid_mask,
    seed,
):
    """Compute metrics for a chunk prediction."""
    valid_normalized_maes = []
    valid_raw_maes = []
    valid_xyz_maes = []
    valid_gripper_errors = []
    
    for t in range(len(valid_mask)):
        if not valid_mask[t]:
            continue
        
        gt_norm = gt_chunk_normalized[t]
        gt_raw = gt_chunk_raw[t]
        pred_norm = predicted_chunk_normalized[t]
        pred_raw = predicted_chunk_raw[t]
        
        valid_normalized_maes.append(np.abs(gt_norm - pred_norm).mean())
        valid_raw_maes.append(np.abs(gt_raw - pred_raw).mean())
        valid_xyz_maes.append(np.abs(gt_norm[:3] - pred_norm[:3]).mean())
        valid_gripper_errors.append(abs(gt_norm[3] - pred_norm[3]))
    
    return {
        "seed": int(seed),
        "normalized_mae_4d": float(np.mean(valid_normalized_maes)) if valid_normalized_maes else 0.0,
        "raw_mae_4d": float(np.mean(valid_raw_maes)) if valid_raw_maes else 0.0,
        "normalized_xyz_mae": float(np.mean(valid_xyz_maes)) if valid_xyz_maes else 0.0,
        "normalized_gripper_error": float(np.mean(valid_gripper_errors)) if valid_gripper_errors else 0.0,
        "valid_steps": int(sum(valid_mask)),
    }


def print_and_save_results(
    results,
    normalized_maes_4d,
    raw_maes_4d,
    episode_index,
    eval_mode,
    num_seeds,
    output_dir=None,
):
    """Print summary and save results to JSON."""
    
    normalized_maes_4d = np.array(normalized_maes_4d)
    raw_maes_4d = np.array(raw_maes_4d)
    total_samples = len(normalized_maes_4d)
    total_frames = len(results)
    
    print("\n" + "=" * 80)
    print("MiniVLA Action Comparison Summary")
    print("=" * 80)
    print(f"\nEpisode {episode_index} - Mode: {eval_mode}")
    print(f"Total frames: {total_frames}")
    print(f"Seeds per frame: {num_seeds}")
    print(f"Total samples: {total_samples}")
    
    # Sample-level metrics
    print(f"\n{'='*60}")
    print("Sample-Level Metrics (all seeds, all frames)")
    print(f"{'='*60}")
    
    print(f"\nNormalized Space:")
    normalized_xyz_maes = []
    normalized_gripper_errors = []
    normalized_cosines = []
    for frame in results:
        for seed in frame["seeds"]:
            normalized_xyz_maes.append(seed.get("normalized_xyz_mae", 0))
            normalized_gripper_errors.append(seed.get("normalized_gripper_error", 0))
            normalized_cosines.append(seed.get("xyz_cosine_similarity", 0))
    
    normalized_xyz_maes = np.array(normalized_xyz_maes)
    normalized_gripper_errors = np.array(normalized_gripper_errors)
    normalized_cosines = np.array(normalized_cosines)
    
    print(f"  4D MAE: mean={normalized_maes_4d.mean():.4f}, std={normalized_maes_4d.std():.4f}, min={normalized_maes_4d.min():.4f}, max={normalized_maes_4d.max():.4f}")
    print(f"  XYZ MAE: mean={normalized_xyz_maes.mean():.4f}, std={normalized_xyz_maes.std():.4f}")
    print(f"  Gripper error: mean={normalized_gripper_errors.mean():.4f}, std={normalized_gripper_errors.std():.4f}")
    print(f"  XYZ cosine: mean={normalized_cosines.mean():.4f}, std={normalized_cosines.std():.4f}")
    
    print(f"\nRaw Space:")
    raw_xyz_maes = []
    raw_gripper_errors = []
    for frame in results:
        for seed in frame["seeds"]:
            raw_xyz_maes.append(seed.get("raw_xyz_mae", 0))
            raw_gripper_errors.append(seed.get("raw_gripper_error", 0))
    
    raw_xyz_maes = np.array(raw_xyz_maes)
    raw_gripper_errors = np.array(raw_gripper_errors)
    
    print(f"  4D MAE: mean={raw_maes_4d.mean():.4f}, std={raw_maes_4d.std():.4f}, min={raw_maes_4d.min():.4f}, max={raw_maes_4d.max():.4f}")
    print(f"  XYZ MAE: mean={raw_xyz_maes.mean():.4f}, std={raw_xyz_maes.std():.4f}")
    print(f"  Gripper error: mean={raw_gripper_errors.mean():.4f}, std={raw_gripper_errors.std():.4f}")
    
    # Mean action to GT
    mean_normalized_maes = np.array([f["mean_action_normalized_mae"] for f in results])
    mean_raw_maes = np.array([f["mean_action_raw_mae"] for f in results])
    print(f"\nSample Mean to GT:")
    print(f"  Normalized MAE: mean={mean_normalized_maes.mean():.4f}, std={mean_normalized_maes.std():.4f}")
    print(f"  Raw MAE: mean={mean_raw_maes.mean():.4f}, std={mean_raw_maes.std():.4f}")
    
    # Best/Median/Worst
    best_normalized = np.array([f["best_normalized_mae_4d"] for f in results])
    best_raw = np.array([f["best_raw_mae_4d"] for f in results])
    median_normalized = np.array([f["median_normalized_mae_4d"] for f in results])
    median_raw = np.array([f["median_raw_mae_4d"] for f in results])
    worst_normalized = np.array([f["worst_normalized_mae_4d"] for f in results])
    worst_raw = np.array([f["worst_raw_mae_4d"] for f in results])
    
    print(f"\nBest-of-N / Median / Worst (per frame, based on raw 4D MAE):")
    print(f"  Best normalized MAE:  mean={best_normalized.mean():.4f}")
    print(f"  Median normalized MAE: mean={median_normalized.mean():.4f}")
    print(f"  Worst normalized MAE: mean={worst_normalized.mean():.4f}")
    print(f"  Best raw MAE:  mean={best_raw.mean():.4f}")
    print(f"  Median raw MAE: mean={median_raw.mean():.4f}")
    print(f"  Worst raw MAE: mean={worst_raw.mean():.4f}")
    
    # Per-seed metrics
    print(f"\nPer-Seed Average Metrics:")
    for seed_idx in range(num_seeds):
        seed_normalized_maes = []
        seed_raw_maes = []
        seed_cosines = []
        for frame in results:
            if seed_idx < len(frame["seeds"]):
                seed = frame["seeds"][seed_idx]
                seed_normalized_maes.append(seed["normalized_mae_4d"])
                seed_raw_maes.append(seed["raw_mae_4d"])
                seed_cosines.append(seed.get("xyz_cosine_similarity", 0))
        
        print(f"  Seed {seed_idx}: normalized={np.mean(seed_normalized_maes):.4f}, raw={np.mean(seed_raw_maes):.4f}, cosine={np.mean(seed_cosines):.4f}")
    
    # Seed variance
    sample_stds_normalized = np.array([f["mean_sample_std_normalized_per_dim"] for f in results])
    sample_stds_raw = np.array([f["mean_sample_std_raw_per_dim"] for f in results])
    pairwise_distances = np.array([f["mean_pairwise_l1_distance"] for f in results])
    
    print(f"\nSeed Variance Metrics:")
    print(f"  Mean sample std (normalized): {sample_stds_normalized.mean():.4f}")
    print(f"  Mean sample std (raw): {sample_stds_raw.mean():.4f}")
    print(f"  Mean pairwise L1 distance: {pairwise_distances.mean():.4f}")
    
    # Success ratios
    print(f"\nSample-Level Success Ratios:")
    for t in [0.1, 0.2, 0.3]:
        norm_ratio = (normalized_maes_4d < t).sum() / total_samples
        raw_ratio = (raw_maes_4d < t).sum() / total_samples
        print(f"  Normalized MAE < {t}: {norm_ratio*100:.1f}%")
        print(f"  Raw MAE < {t}: {raw_ratio*100:.1f}%")
    
    # Frame-level best-of-N success ratios
    print(f"\nFrame-Level Best-of-N Success Ratios:")
    for t in [0.1, 0.2, 0.3]:
        count_norm = sum(1 for f in results if f["best_normalized_mae_4d"] < t)
        count_raw = sum(1 for f in results if f["best_raw_mae_4d"] < t)
        print(f"  Best-of-N normalized MAE < {t}: {count_norm}/{total_frames} ({count_norm/total_frames*100:.1f}%)")
        print(f"  Best-of-N raw MAE < {t}: {count_raw}/{total_frames} ({count_raw/total_frames*100:.1f}%)")
    
    # Save JSON
    if output_dir is None:
        output_dir = Path(__file__).parent
    
    output_file = output_dir / f"minivla_action_ep{episode_index}_{eval_mode}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "episode_index": episode_index,
            "checkpoint_path": CHECKPOINT_PATH,
            "task_description": TASK_DESCRIPTION,
            "eval_mode": eval_mode,
            "num_seeds_per_frame": num_seeds,
            "base_seed": BASE_SEED,
            "seed_stride": SEED_STRIDE,
            "summary": {
                "total_frames": total_frames,
                "num_seeds": num_seeds,
                "total_samples": total_samples,
                "normalized_space": {
                    "mae_4d": {
                        "mean": float(normalized_maes_4d.mean()),
                        "std": float(normalized_maes_4d.std()),
                        "min": float(normalized_maes_4d.min()),
                        "max": float(normalized_maes_4d.max()),
                    },
                    "xyz_mae": {
                        "mean": float(normalized_xyz_maes.mean()),
                        "std": float(normalized_xyz_maes.std()),
                    },
                    "gripper_error": {
                        "mean": float(normalized_gripper_errors.mean()),
                        "std": float(normalized_gripper_errors.std()),
                    },
                    "normalized_cosine": {
                        "mean": float(normalized_cosines.mean()),
                        "std": float(normalized_cosines.std()),
                    },
                    "mean_action_to_gt_normalized": {
                        "mean": float(mean_normalized_maes.mean()),
                        "std": float(mean_normalized_maes.std()),
                    },
                    "mean_action_to_gt_raw": {
                        "mean": float(mean_raw_maes.mean()),
                        "std": float(mean_raw_maes.std()),
                    },
                    "best_of_n_normalized_mae": {
                        "mean": float(best_normalized.mean()),
                        "std": float(best_normalized.std()),
                    },
                    "best_of_n_raw_mae": {
                        "mean": float(best_raw.mean()),
                        "std": float(best_raw.std()),
                    },
                    "median_normalized_mae": {
                        "mean": float(median_normalized.mean()),
                        "std": float(median_normalized.std()),
                    },
                    "median_raw_mae": {
                        "mean": float(median_raw.mean()),
                        "std": float(median_raw.std()),
                    },
                    "worst_normalized_mae": {
                        "mean": float(worst_normalized.mean()),
                        "std": float(worst_normalized.std()),
                    },
                    "worst_raw_mae": {
                        "mean": float(worst_raw.mean()),
                        "std": float(worst_raw.std()),
                    },
                    "seed_variance": {
                        "mean_sample_std_normalized": float(sample_stds_normalized.mean()),
                        "mean_sample_std_raw": float(sample_stds_raw.mean()),
                        "mean_pairwise_l1_distance": float(pairwise_distances.mean()),
                    },
                    "success_ratios": {
                        f"normalized_mae<{t}": float((normalized_maes_4d < t).sum() / total_samples)
                        for t in [0.1, 0.2, 0.3]
                    },
                    "raw_success_ratios": {
                        f"raw_mae<{t}": float((raw_maes_4d < t).sum() / total_samples)
                        for t in [0.1, 0.2, 0.3]
                    },
                    "frame_level_best_of_n_success_ratios": {
                        f"best_normalized_mae<{t}": float((best_normalized < t).sum() / total_frames)
                        for t in [0.1, 0.2, 0.3]
                    },
                    "frame_level_best_of_n_raw_success_ratios": {
                        f"best_raw_mae<{t}": float((best_raw < t).sum() / total_frames)
                        for t in [0.1, 0.2, 0.3]
                    },
                },
            },
            "frames": results,
        }, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="MiniVLA action verification")
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
    parser.add_argument("--task", type=str, default=TASK_DESCRIPTION)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=NUM_SEEDS_PER_FRAME)
    parser.add_argument("--base_seed", type=int, default=BASE_SEED)
    parser.add_argument("--seed_stride", type=int, default=SEED_STRIDE)
    parser.add_argument("--eval_mode", type=str, default="one_step", choices=["one_step", "chunk"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Step 1: Load Model and Preprocessor")
    print("=" * 80)
    policy, preprocessor, postprocessor = load_model_and_preprocessor(
        args.checkpoint_path, args.device
    )
    
    print("\n" + "=" * 80)
    print("Step 2: Load Dataset")
    print("=" * 80)
    print(f"Loading dataset from: {args.dataset_path}")
    
    dataset = LeRobotDataset(
        repo_id="work2/disassemble-v3_corner",
        root=args.dataset_path,
        download_videos=False,
    )
    
    print(f"Dataset frames: {len(dataset)}")
    print(f"Dataset episodes: {dataset.num_episodes}")
    
    # Sanity check
    if not validate_normalization_sanity(policy, preprocessor, postprocessor, dataset):
        print("\nERROR: Normalization sanity check failed!")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Step 3: Action Evaluation")
    print("=" * 80)
    
    results, normalized_maes_4d, raw_maes_4d = run_evaluation(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        episode_index=args.episode_index,
        num_seeds=args.num_seeds,
        base_seed=args.base_seed,
        seed_stride=args.seed_stride,
        eval_mode=args.eval_mode,
        task_description=args.task,
        device=args.device,
    )
    
    print_and_save_results(
        results=results,
        normalized_maes_4d=normalized_maes_4d,
        raw_maes_4d=raw_maes_4d,
        episode_index=args.episode_index,
        eval_mode=args.eval_mode,
        num_seeds=args.num_seeds,
    )


if __name__ == "__main__":
    main()