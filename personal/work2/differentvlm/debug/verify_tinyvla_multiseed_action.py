#!/usr/bin/env python
"""
Multi-seed action verification script for TinyVLA.

Diagnostic tool to determine:
1. Whether the conditional action distribution center is close to expert
2. Whether diffusion sampling variance (seed sensitivity) is too large

Key features:
- Auto-detects policy type (tinyvla_s vs tinyvla_b) from checkpoint config
- Properly normalizes expert actions using the SAME normalization as training
- Reports both normalized-space and raw-space metrics separately
- Computes per-frame sample_mean, sample_std, best/median/worst across seeds
- Validates round-trip normalize->unnormalize correctness
- Supports one_step and chunk evaluation modes
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
from safetensors.torch import load_file

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.tinyvla.modeling_tinyvla import TinyVLAPolicy, TinyVLABPolicy
from lerobot.policies.tinyvla.configuration_tinyvla import TinyVLAConfig
from lerobot.envs import preprocess_observation
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep, PolicyAction, EnvTransition, TransitionKey
from lerobot.configs.types import FeatureType, NormalizationMode

# ============================================================
# Configuration
# ============================================================
CHECKPOINT_PATH = str(
    PROJECT_ROOT / "personal" / "work2" / "differentvlm" / "tinyvla" /
    "tinyvla_s_random_ep200_seed42_disassemble-v3_corner" / "checkpoints" /
    "050000" / "pretrained_model"
)
DATASET_PATH = str(PROJECT_ROOT / "personal" / "work2" / "dataset_view" / "disassemble-v3_corner")
TASK_DESCRIPTION = "Pick a nut out of a peg"

# Multi-seed settings
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
    """Load TinyVLA policy and preprocessor from checkpoint.
    
    Auto-detects policy type from config and uses the correct class.
    """
    print(f"Loading TinyVLA from: {checkpoint_path}")
    
    # Load config from checkpoint
    print("\nLoading config from checkpoint...")
    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    
    # Detect policy type
    policy_type = config._name_or_path if hasattr(config, '_name_or_path') else None
    if policy_type is None:
        # Try to infer from config class
        policy_type = config.__class__.__name__
    
    print(f"Config class: {type(config).__name__}")
    print(f"Policy type (registered name): {getattr(config, 'name', 'unknown')}")
    
    # Select correct policy class based on config
    if hasattr(config, 'name') and config.name == "tinyvla_b":
        print("\nLoading TinyVLA-B policy via from_pretrained...")
        policy = TinyVLABPolicy.from_pretrained(
            pretrained_name_or_path=checkpoint_path,
        )
    else:
        print("\nLoading TinyVLA-S policy via from_pretrained...")
        policy = TinyVLAPolicy.from_pretrained(
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
    print(f"  Policy type: {getattr(config, 'name', 'unknown')}")
    print(f"  chunk_size: {config.chunk_size}")
    print(f"  n_action_steps: {config.n_action_steps}")
    print(f"  action_dim: {config.action_dim}")
    print(f"  action normalization: {config.normalization_mapping.get(FeatureType.ACTION, 'unknown')}")
    print(f"  state normalization: {config.normalization_mapping.get(FeatureType.STATE, 'unknown')}")
    print(f"  num_inference_timesteps: {config.num_inference_timesteps}")
    print(f"  head_type: {config.action_head_type}")
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
    
    if action_norm_mode != NormalizationMode.MIN_MAX:
        print(f"WARNING: Expected MIN_MAX but got {action_norm_mode}")
    
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
        print(f"PASS: Round-trip error < 1e-2 (acceptable for MIN_MAX normalization)")
    
    return True


def normalize_expert_action(normalizer, expert_action_raw):
    """Normalize a raw expert action using the SAME normalization as training.
    
    Args:
        normalizer: NormalizerProcessorStep from preprocessor
        expert_action_raw: Raw action from dataset [action_dim]
    
    Returns:
        Normalized action tensor [action_dim]
    """
    expert_action_tensor = torch.as_tensor(expert_action_raw, dtype=torch.float32)
    action_normalized = normalizer._apply_transform(
        expert_action_tensor.unsqueeze(0).unsqueeze(0),  # [1, 1, action_dim]
        "action",
        FeatureType.ACTION,
        inverse=False
    ).squeeze()
    return action_normalized


def unnormalize_action(postprocessor, action_normalized):
    """Unnormalize an action using the postprocessor.
    
    Args:
        postprocessor: Postprocessor pipeline
        action_normalized: Normalized action tensor [action_dim]
    
    Returns:
        Unnormalized action numpy array [action_dim]
    """
    for step in postprocessor.steps:
        if isinstance(step, UnnormalizerProcessorStep):
            action_unnormalized = step._apply_transform(
                action_normalized.unsqueeze(0).unsqueeze(0),
                "action",
                FeatureType.ACTION,
                inverse=True
            ).squeeze()
            return action_unnormalized.numpy()
    return action_normalized.numpy()


def run_multiseed_evaluation(
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
    test_sampling_mode="random",
):
    """Run multi-seed evaluation on a single episode.
    
    For each frame, samples actions with different random seeds and compares
    with expert actions in both normalized and raw spaces.
    """
    
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
    print(f"Test sampling mode: {test_sampling_mode}")
    
    all_results = []
    all_normalized_maes_4d = []
    all_raw_maes_4d = []
    
    for idx, frame_idx in enumerate(tqdm(frame_indices, desc=f"Episode {episode_index}")):
        frame = dataset[frame_idx]
        expert_action_raw = frame["action"].numpy()  # Raw action from dataset
        observation_state = frame["observation.state"]
        
        # Normalize expert action using training normalization
        expert_action_normalized = normalize_expert_action(normalizer, expert_action_raw)
        
        # Build observation
        observation = {
            "observation.images.top": frame["observation.images.top"],
            "observation.images.wrist": frame["observation.images.wrist"],
            "observation.state": observation_state,
            "task": task_description,
        }
        
        observation_tensor = preprocess_observation(observation)
        observation_tensor = preprocessor(observation_tensor)
        
        frame_seed_results = []
        frame_seed_normalized_maes_4d = []
        frame_seed_raw_maes_4d = []
        
        if eval_mode == "one_step":
            # One-step mode: each seed is independent single-step inference
            for seed_idx in range(num_seeds):
                current_seed = base_seed + seed_idx * seed_stride
                
                # Reset and set seed for each inference
                policy.reset()
                
                # Handle different sampling modes
                if test_sampling_mode == "deterministic":
                    # Deterministic mode: use zero noise or fixed generator
                    torch.manual_seed(0)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(0)
                        torch.cuda.manual_seed_all(0)
                    np.random.seed(0)
                elif test_sampling_mode == "fixed_seed":
                    # Fixed seed mode: use the same seed for all frames
                    torch.manual_seed(base_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(base_seed)
                        torch.cuda.manual_seed_all(base_seed)
                    np.random.seed(base_seed)
                else:
                    # Random mode: use different seed for each frame
                    torch.manual_seed(current_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(current_seed)
                        torch.cuda.manual_seed_all(current_seed)
                    np.random.seed(current_seed)
                
                with torch.inference_mode():
                    raw_action = policy.select_action(observation_tensor)
                
                # raw_action is normalized action from model
                model_normalized = raw_action.to("cpu").to(dtype=torch.float32).numpy()[0]
                model_raw = postprocessor_unnormalize(postprocessor, raw_action)[0]
                
                # Debug output for first frame only
                debug_first_frame = (idx == 0)
                if debug_first_frame:
                    print(f"\n  {'='*60}")
                    print(f"  [DIAGNOSTIC] Frame {frame_idx} - First Frame Action Comparison")
                    print(f"  {'='*60}")
                    print(f"  GT raw action: {expert_action_raw}")
                    print(f"  GT normalized action: {expert_action_normalized.numpy()}")
                    print(f"  Model normalized action: {model_normalized}")
                    print(f"  Model raw action: {model_raw}")
                    normalized_error = np.abs(expert_action_normalized.numpy() - model_normalized).mean()
                    raw_error = np.abs(expert_action_raw - model_raw).mean()
                    print(f"\n  Normalized error (MAE): {normalized_error:.4f}")
                    print(f"  Raw error (MAE): {raw_error:.4f}")
                    print(f"  {'='*60}")
                    if normalized_error < 0.1 and raw_error < 0.1:
                        print(f"  => Both errors < 0.1: Model action prediction looks GOOD")
                        print(f"     If success rate is still 0, check chunk execution or environment eval")
                    elif normalized_error < 0.2 and raw_error < 0.2:
                        print(f"  => Both errors < 0.2: Model action prediction is ACCEPTABLE")
                        print(f"     May need more training or better data quality")
                    else:
                        print(f"  => Both errors >= 0.2: Model action prediction is POOR")
                        print(f"     Need to retrain or adjust training pipeline")
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
                chunk_normalized = chunk_actions.to("cpu").to(dtype=torch.float32).numpy()[0]  # [chunk_size, action_dim]
                chunk_raw = postprocessor_unnormalize_chunk(postprocessor, chunk_actions)[0]  # [chunk_size, action_dim]
                
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
                        # Out of episode, pad with zeros
                        gt_chunk_raw.append(np.zeros_like(expert_action_raw))
                        gt_chunk_normalized.append(np.zeros_like(expert_action_normalized.numpy()))
                        valid_mask.append(False)
                
                gt_chunk_raw = np.array(gt_chunk_raw)  # [chunk_size, action_dim]
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


def postprocessor_unnormalize(postprocessor, action_normalized):
    """Unnormalize a single action using postprocessor."""
    for step in postprocessor.steps:
        if isinstance(step, UnnormalizerProcessorStep):
            action_unnormalized = step._apply_transform(
                action_normalized,
                "action",
                FeatureType.ACTION,
                inverse=True
            )
            # Convert to numpy if it's a tensor
            if hasattr(action_unnormalized, 'numpy'):
                # Convert bfloat16 to float32 first, then to numpy
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
                # Convert bfloat16 to float32 first, then to numpy
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


def compute_seed_metrics(expert_action_raw, expert_action_normalized, model_normalized, model_raw, seed):
    """Compute metrics for a single seed prediction.
    
    Compares in both normalized space and raw space.
    """
    expert_normalized = np.asarray(expert_action_normalized)
    expert_raw = np.asarray(expert_action_raw)
    
    # Normalized space metrics
    normalized_xyz_mae = np.abs(expert_normalized[:3] - model_normalized[:3]).mean()
    normalized_gripper_error = abs(expert_normalized[3] - model_normalized[3])
    normalized_mae_4d = np.abs(expert_normalized - model_normalized).mean()
    
    # Raw space metrics
    raw_xyz_mae = np.abs(expert_raw[:3] - model_raw[:3]).mean()
    raw_gripper_error = abs(expert_raw[3] - model_raw[3])
    raw_mae_4d = np.abs(expert_raw - model_raw).mean()
    
    # XYZ cosine similarity (normalized space)
    norm_exp = np.linalg.norm(expert_normalized[:3])
    norm_model = np.linalg.norm(model_normalized[:3])
    if norm_exp > 0 and norm_model > 0:
        xyz_cosine = np.dot(expert_normalized[:3], model_normalized[:3]) / (norm_exp * norm_model)
    else:
        xyz_cosine = 0.0
    
    # Sign agreement (normalized space)
    sign_agreement = np.sum(np.sign(expert_normalized[:3]) == np.sign(model_normalized[:3])) / 3.0
    
    # Environment-clipped diagnostic (NOT used for main metrics)
    env_clipped_expert_xyz = np.clip(expert_raw[:3], -1, 1)
    env_clipped_model_xyz = np.clip(model_raw[:3], -1, 1)
    env_clipped_xyz_mae = np.abs(env_clipped_expert_xyz - env_clipped_model_xyz).mean()
    
    return {
        "seed": seed,
        "predicted_action_normalized": model_normalized.tolist(),
        "predicted_action_raw": model_raw.tolist(),
        "normalized_xyz_mae": float(normalized_xyz_mae),
        "normalized_gripper_error": float(normalized_gripper_error),
        "normalized_mae_4d": float(normalized_mae_4d),
        "raw_xyz_mae": float(raw_xyz_mae),
        "raw_gripper_error": float(raw_gripper_error),
        "raw_mae_4d": float(raw_mae_4d),
        "xyz_cosine_similarity": float(xyz_cosine),
        "sign_agreement": float(sign_agreement),
        "environment_clipped_xyz_mae": float(env_clipped_xyz_mae),
    }


def compute_chunk_seed_metrics(gt_chunk_raw, gt_chunk_normalized, predicted_chunk_normalized, predicted_chunk_raw, valid_mask, seed):
    """Compute metrics for a chunk prediction (single seed)."""
    valid_gt_normalized = gt_chunk_normalized[valid_mask]
    valid_pred_normalized = predicted_chunk_normalized[valid_mask]
    valid_gt_raw = gt_chunk_raw[valid_mask]
    valid_pred_raw = predicted_chunk_raw[valid_mask]
    
    if len(valid_gt_normalized) == 0:
        return {
            "seed": seed,
            "normalized_mae_4d": 0.0,
            "raw_mae_4d": 0.0,
            "valid_steps": 0,
        }
    
    # Normalized space
    normalized_mae_4d = np.abs(valid_gt_normalized - valid_pred_normalized).mean()
    normalized_xyz_mae = np.abs(valid_gt_normalized[:, :3] - valid_pred_normalized[:, :3]).mean()
    normalized_gripper_error = np.abs(valid_gt_normalized[:, 3] - valid_pred_normalized[:, 3]).mean()
    
    # Raw space
    raw_mae_4d = np.abs(valid_gt_raw - valid_pred_raw).mean()
    raw_xyz_mae = np.abs(valid_gt_raw[:, :3] - valid_pred_raw[:, :3]).mean()
    raw_gripper_error = np.abs(valid_gt_raw[:, 3] - valid_pred_raw[:, 3]).mean()
    
    # XYZ cosine (average over valid steps)
    cosines = []
    for t in range(len(valid_gt_normalized)):
        norm_exp = np.linalg.norm(valid_gt_normalized[t, :3])
        norm_pred = np.linalg.norm(valid_pred_normalized[t, :3])
        if norm_exp > 0 and norm_pred > 0:
            cos = np.dot(valid_gt_normalized[t, :3], valid_pred_normalized[t, :3]) / (norm_exp * norm_pred)
            cosines.append(cos)
    xyz_cosine = np.mean(cosines) if cosines else 0.0
    
    return {
        "seed": seed,
        "predicted_chunk_normalized": predicted_chunk_normalized.tolist(),
        "predicted_chunk_raw": predicted_chunk_raw.tolist(),
        "normalized_xyz_mae": float(normalized_xyz_mae),
        "normalized_gripper_error": float(normalized_gripper_error),
        "normalized_mae_4d": float(normalized_mae_4d),
        "raw_xyz_mae": float(raw_xyz_mae),
        "raw_gripper_error": float(raw_gripper_error),
        "raw_mae_4d": float(raw_mae_4d),
        "xyz_cosine_similarity": float(xyz_cosine),
        "valid_steps": int(valid_mask.sum()),
    }


def compute_frame_distribution(expert_action_raw, expert_action_normalized, seed_results, seed_normalized_maes, seed_raw_maes, eval_mode):
    """Compute per-frame distribution statistics across seeds."""
    num_seeds = len(seed_results)
    
    # Collect all normalized and raw predictions
    all_normalized = np.array([r["predicted_action_normalized"] for r in seed_results])  # [num_seeds, action_dim]
    all_raw = np.array([r["predicted_action_raw"] for r in seed_results])
    
    # Sample mean and std
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
    
    # Check if model output exceeds [-1, 1]
    exceeds_clip_ratio = (np.abs(all_normalized) > 1.0).mean()
    
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
        "exceeds_clip_ratio": float(exceeds_clip_ratio),
    }


def run_sampling_mode_comparison(
    policy,
    preprocessor,
    postprocessor,
    dataset,
    episode_index=0,
    num_seeds=20,
    base_seed=42,
    task_description=TASK_DESCRIPTION,
    device="cuda",
):
    """Run comparison across different sampling modes (random, fixed_seed, deterministic).
    
    For each frame, runs inference with different sampling modes and compares:
    - Random sampling mean error
    - Deterministic sampling error  
    - Best-of-N error
    - Whether sampling variance is the main issue
    """
    
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
        return {}
    
    total_frames = len(frame_indices)
    
    print(f"\n{'='*80}")
    print("Sampling Mode Comparison Experiment")
    print(f"{'='*80}")
    print(f"Evaluating episode {episode_index} with {total_frames} frames")
    print(f"Number of seeds per frame: {num_seeds}")
    print(f"Task description: '{task_description}'")
    print(f"Base seed: {base_seed}")
    
    # Collect results for each mode
    all_random_maes = []
    all_deterministic_maes = []
    all_best_of_n_maes = []
    
    for idx, frame_idx in enumerate(tqdm(frame_indices, desc=f"Episode {episode_index}")):
        frame = dataset[frame_idx]
        expert_action_raw = frame["action"].numpy()
        observation_state = frame["observation.state"]
        
        observation = {
            "observation.images.top": frame["observation.images.top"],
            "observation.images.wrist": frame["observation.images.wrist"],
            "observation.state": observation_state,
            "task": task_description,
        }
        
        observation_tensor = preprocess_observation(observation)
        observation_tensor = preprocessor(observation_tensor)
        
        # Normalize expert action
        expert_action_tensor = torch.as_tensor(expert_action_raw, dtype=torch.float32)
        expert_action_normalized = normalizer._apply_transform(
            expert_action_tensor.unsqueeze(0).unsqueeze(0),
            "action",
            FeatureType.ACTION,
            inverse=False
        ).squeeze()
        
        # Run random sampling (multiple seeds)
        random_seed_maes = []
        for seed_idx in range(num_seeds):
            current_seed = base_seed + seed_idx
            policy.reset()
            torch.manual_seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(current_seed)
                torch.cuda.manual_seed_all(current_seed)
            
            with torch.inference_mode():
                raw_action = policy.select_action(observation_tensor)
            
            model_normalized = raw_action.to("cpu").to(dtype=torch.float32).numpy()[0]
            model_raw = postprocessor_unnormalize(postprocessor, raw_action)[0]
            
            normalized_mae = np.abs(expert_action_normalized.numpy() - model_normalized).mean()
            random_seed_maes.append(normalized_mae)
        
        # Run deterministic sampling
        policy.reset()
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        
        with torch.inference_mode():
            raw_action = policy.select_action(observation_tensor)
        
        model_normalized = raw_action.to("cpu").to(dtype=torch.float32).numpy()[0]
        model_raw = postprocessor_unnormalize(postprocessor, raw_action)[0]
        
        deterministic_mae = np.abs(expert_action_normalized.numpy() - model_normalized).mean()
        
        # Compute statistics
        random_mean_mae = np.mean(random_seed_maes)
        best_of_n_mae = np.min(random_seed_maes)
        
        all_random_maes.append(random_mean_mae)
        all_deterministic_maes.append(deterministic_mae)
        all_best_of_n_maes.append(best_of_n_mae)
    
    # Aggregate results
    random_maes = np.array(all_random_maes)
    deterministic_maes = np.array(all_deterministic_maes)
    best_of_n_maes = np.array(all_best_of_n_maes)
    
    print(f"\n{'='*60}")
    print("Sampling Mode Comparison Results")
    print(f"{'='*60}")
    print(f"\nRandom sampling mean error: {random_maes.mean():.4f} (std={random_maes.std():.4f})")
    print(f"Deterministic sampling error: {deterministic_maes.mean():.4f} (std={deterministic_maes.std():.4f})")
    print(f"Best-of-N error: {best_of_n_maes.mean():.4f} (std={best_of_n_maes.std():.4f})")
    
    # Diagnosis
    deterministic_vs_best_gap = np.abs(deterministic_maes - best_of_n_maes).mean()
    random_vs_deterministic_gap = np.abs(random_maes - deterministic_maes).mean()
    
    print(f"\nDeterministic vs Best-of-N gap: {deterministic_vs_best_gap:.4f}")
    print(f"Random vs Deterministic gap: {random_vs_deterministic_gap:.4f}")
    
    sampling_variance_is_main_issue = False
    model_prediction_quality_issue = False
    
    if deterministic_vs_best_gap < 0.05:
        print(f"\n=> Deterministic MAE close to Best-of-N MAE")
        print(f"   Sampling variance is the MAIN ISSUE")
        print(f"   Model has learned action prediction, but diffusion sampling is unstable")
        sampling_variance_is_main_issue = True
    elif deterministic_maes.mean() > 0.2 and random_maes.mean() > 0.2:
        print(f"\n=> Both deterministic and random errors are large")
        print(f"   Model prediction quality is the MAIN ISSUE")
        print(f"   Action head needs retraining or better training data")
        model_prediction_quality_issue = True
    else:
        print(f"\n=> Mixed results, further analysis needed")
    
    results = {
        "random_sampling_mean_error": {
            "mean": float(random_maes.mean()),
            "std": float(random_maes.std()),
            "min": float(random_maes.min()),
            "max": float(random_maes.max()),
        },
        "deterministic_sampling_error": {
            "mean": float(deterministic_maes.mean()),
            "std": float(deterministic_maes.std()),
            "min": float(deterministic_maes.min()),
            "max": float(deterministic_maes.max()),
        },
        "best_of_n_error": {
            "mean": float(best_of_n_maes.mean()),
            "std": float(best_of_n_maes.std()),
            "min": float(best_of_n_maes.min()),
            "max": float(best_of_n_maes.max()),
        },
        "diagnosis": {
            "deterministic_vs_best_of_n_gap": float(deterministic_vs_best_gap),
            "random_vs_deterministic_gap": float(random_vs_deterministic_gap),
            "sampling_variance_is_main_issue": sampling_variance_is_main_issue,
            "model_prediction_quality_issue": model_prediction_quality_issue,
        }
    }
    
    return results


def run_chunk_step_comparison(
    policy,
    preprocessor,
    postprocessor,
    dataset,
    episode_index=0,
    task_description=TASK_DESCRIPTION,
    device="cuda",
    chunk_steps_list=None,
):
    """Run chunk evaluation with different n_action_steps.
    
    Compares execution with different action horizons to determine
    if rollout failures are due to chunk execution strategy.
    """
    if chunk_steps_list is None:
        chunk_steps_list = [1, 8, 16]
    
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
        return {}
    
    total_frames = len(frame_indices)
    
    print(f"\n{'='*80}")
    print("Chunk Step Comparison Experiment")
    print(f"{'='*80}")
    print(f"Evaluating episode {episode_index} with {total_frames} frames")
    print(f"Task description: '{task_description}'")
    print(f"Testing n_action_steps: {chunk_steps_list}")
    
    results_per_step = {}
    
    for n_action_steps in chunk_steps_list:
        print(f"\n{'='*60}")
        print(f"Testing n_action_steps={n_action_steps}")
        print(f"{'='*60}")
        
        # Store per-frame metrics
        frame_maes = []
        frame_errors = []
        
        # Reset policy
        policy.reset()
        
        # We'll simulate chunk execution by predicting and executing n_action_steps at a time
        current_frame_idx = 0
        step_idx = 0
        
        while step_idx < total_frames:
            frame_idx = frame_indices[step_idx]
            frame = dataset[frame_idx]
            expert_action_raw = frame["action"].numpy()
            observation_state = frame["observation.state"]
            
            observation = {
                "observation.images.top": frame["observation.images.top"],
                "observation.images.wrist": frame["observation.images.wrist"],
                "observation.state": observation_state,
                "task": task_description,
            }
            
            observation_tensor = preprocess_observation(observation)
            observation_tensor = preprocessor(observation_tensor)
            
            # Normalize expert action
            expert_action_tensor = torch.as_tensor(expert_action_raw, dtype=torch.float32)
            expert_action_normalized = normalizer._apply_transform(
                expert_action_tensor.unsqueeze(0).unsqueeze(0),
                "action",
                FeatureType.ACTION,
                inverse=False
            ).squeeze()
            
            # Predict action chunk
            with torch.inference_mode():
                chunk_actions = policy.predict_action_chunk(observation_tensor)
            
            # Execute first n_action_steps from the chunk
            chunk_normalized = chunk_actions.to("cpu").to(dtype=torch.float32).numpy()[0]
            
            # Compare predicted vs expert for executed steps
            step_maes = []
            for t in range(min(n_action_steps, total_frames - step_idx)):
                future_step_idx = step_idx + t
                future_frame_idx = frame_indices[future_step_idx]
                future_frame = dataset[future_frame_idx]
                future_expert_raw = future_frame["action"].numpy()
                
                # Normalize future expert action
                future_expert_tensor = torch.as_tensor(future_expert_raw, dtype=torch.float32)
                future_expert_normalized = normalizer._apply_transform(
                    future_expert_tensor.unsqueeze(0).unsqueeze(0),
                    "action",
                    FeatureType.ACTION,
                    inverse=False
                ).squeeze()
                
                predicted_normalized = chunk_normalized[t]
                mae = np.abs(future_expert_normalized.numpy() - predicted_normalized).mean()
                step_maes.append(mae)
            
            frame_maes.extend(step_maes)
            step_idx += n_action_steps
        
        # Compute statistics for this n_action_steps
        frame_maes = np.array(frame_maes)
        results_per_step[n_action_steps] = {
            "mean_mae": float(frame_maes.mean()),
            "std_mae": float(frame_maes.std()),
            "min_mae": float(frame_maes.min()),
            "max_mae": float(frame_maes.max()),
            "mae_below_0.1_ratio": float((frame_maes < 0.1).mean()),
            "mae_below_0.2_ratio": float((frame_maes < 0.2).mean()),
        }
        
        print(f"  Mean MAE: {frame_maes.mean():.4f}")
        print(f"  Std MAE: {frame_maes.std():.4f}")
        print(f"  MAE < 0.1: {(frame_maes < 0.1).mean()*100:.1f}%")
        print(f"  MAE < 0.2: {(frame_maes < 0.2).mean()*100:.1f}%")
    
    # Print comparison table
    print(f"\n{'='*60}")
    print("Chunk Step Comparison Summary")
    print(f"{'='*60}")
    print(f"{'n_action_steps':>15} | {'Mean MAE':>10} | {'Std MAE':>10} | {'MAE<0.1':>10} | {'MAE<0.2':>10}")
    print(f"{'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    for n_action_steps in chunk_steps_list:
        r = results_per_step[n_action_steps]
        print(f"{n_action_steps:>15} | {r['mean_mae']:>10.4f} | {r['std_mae']:>10.4f} | {r['mae_below_0.1_ratio']*100:>9.1f}% | {r['mae_below_0.2_ratio']*100:>9.1f}%")
    
    return results_per_step


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
    print("Multi-Seed Action Comparison Summary")
    print("=" * 80)
    print(f"\nEpisode {episode_index} - Mode: {eval_mode}")
    print(f"Total frames: {total_frames}")
    print(f"Seeds per frame: {num_seeds}")
    print(f"Total samples: {total_samples}")
    
    # Sample-level metrics
    print(f"\n{'='*60}")
    print("Sample-Level Metrics (all seeds, all frames)")
    print(f"{'='*60}")
    
    print(f"\nNormalized Space ([-1,1] action space):")
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
    
    print(f"\nRaw Space (dataset/action space):")
    raw_xyz_maes = []
    raw_gripper_errors = []
    raw_cosines = []
    for frame in results:
        for seed in frame["seeds"]:
            raw_xyz_maes.append(seed.get("raw_xyz_mae", 0))
            raw_gripper_errors.append(seed.get("raw_gripper_error", 0))
            raw_cosines.append(seed.get("xyz_cosine_similarity", 0))
    
    raw_xyz_maes = np.array(raw_xyz_maes)
    raw_gripper_errors = np.array(raw_gripper_errors)
    raw_cosines = np.array(raw_cosines)
    
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
                seed_cosines.append(seed["xyz_cosine_similarity"])
        
        print(f"  Seed {seed_idx+1}: norm_MAE={np.mean(seed_normalized_maes):.4f}, raw_MAE={np.mean(seed_raw_maes):.4f}, cosine={np.mean(seed_cosines):.4f}")
    
    # Seed variance
    sample_stds_normalized = np.array([f["mean_sample_std_normalized_per_dim"] for f in results])
    sample_stds_raw = np.array([f["mean_sample_std_raw_per_dim"] for f in results])
    pairwise_distances = np.array([f["mean_pairwise_l1_distance"] for f in results])
    
    print(f"\nSeed Variance (diffusion sensitivity):")
    print(f"  Mean sample std (normalized): {sample_stds_normalized.mean():.4f}")
    print(f"  Mean sample std (raw): {sample_stds_raw.mean():.4f}")
    print(f"  Mean pairwise L1 distance: {pairwise_distances.mean():.4f}")
    
    # Exceeds clip
    exceeds_ratios = np.array([f["exceeds_clip_ratio"] for f in results])
    if exceeds_ratios.mean() > 0:
        print(f"\nWARNING: Model outputs exceed [-1,1] in {exceeds_ratios.mean()*100:.1f}% of predictions")
    
    # Sample-level success ratios
    print(f"\n{'='*60}")
    print("Sample-Level Success Ratios")
    print(f"{'='*60}")
    for threshold in [0.1, 0.2, 0.3]:
        count_norm = (normalized_maes_4d < threshold).sum()
        count_raw = (raw_maes_4d < threshold).sum()
        print(f"  MAE < {threshold}: normalized={count_norm}/{total_samples} ({count_norm/total_samples*100:.1f}%), raw={count_raw}/{total_samples} ({count_raw/total_samples*100:.1f}%)")
    
    # Frame-level success ratios (best-of-N)
    print(f"\n{'='*60}")
    print("Frame-Level Success Ratios (best-of-N per frame)")
    print(f"{'='*60}")
    for threshold in [0.1, 0.2, 0.3]:
        count_norm = (best_normalized < threshold).sum()
        count_raw = (best_raw < threshold).sum()
        print(f"  Best-of-N MAE < {threshold}: normalized={count_norm}/{total_frames} ({count_norm/total_frames*100:.1f}%), raw={count_raw}/{total_frames} ({count_raw/total_frames*100:.1f}%)")
    
    # Save JSON
    if output_dir is None:
        output_dir = Path(__file__).parent
    
    output_file = output_dir / f"tinyvla_multiseed_ep{episode_index}_{eval_mode}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "episode_index": episode_index,
            "checkpoint_path": CHECKPOINT_PATH,
            "task_description": TASK_DESCRIPTION,
            "eval_mode": eval_mode,
            "num_seeds_per_frame": num_seeds,
            "base_seed": BASE_SEED,
            "seed_stride": SEED_STRIDE,
            "clip_sample": True,
            "total_frames": total_frames,
            "total_samples": total_samples,
            "summary": {
                "sample_level": {
                    "normalized_4d_mae": {
                        "mean": float(normalized_maes_4d.mean()),
                        "std": float(normalized_maes_4d.std()),
                        "min": float(normalized_maes_4d.min()),
                        "max": float(normalized_maes_4d.max()),
                    },
                    "raw_4d_mae": {
                        "mean": float(raw_maes_4d.mean()),
                        "std": float(raw_maes_4d.std()),
                        "min": float(raw_maes_4d.min()),
                        "max": float(raw_maes_4d.max()),
                    },
                    "normalized_xyz_mae": {
                        "mean": float(normalized_xyz_maes.mean()),
                        "std": float(normalized_xyz_maes.std()),
                    },
                    "raw_xyz_mae": {
                        "mean": float(raw_xyz_maes.mean()),
                        "std": float(raw_xyz_maes.std()),
                    },
                    "normalized_gripper_error": {
                        "mean": float(normalized_gripper_errors.mean()),
                        "std": float(normalized_gripper_errors.std()),
                    },
                    "raw_gripper_error": {
                        "mean": float(raw_gripper_errors.mean()),
                        "std": float(raw_gripper_errors.std()),
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
    parser = argparse.ArgumentParser(description="Multi-seed TinyVLA action verification")
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
    parser.add_argument("--task", type=str, default=TASK_DESCRIPTION)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=NUM_SEEDS_PER_FRAME)
    parser.add_argument("--base_seed", type=int, default=BASE_SEED)
    parser.add_argument("--seed_stride", type=int, default=SEED_STRIDE)
    parser.add_argument("--eval_mode", type=str, default="one_step", choices=["one_step", "chunk"])
    parser.add_argument("--device", type=str, default="cuda")
    # New parameters for sampling mode comparison
    parser.add_argument(
        "--test_sampling_mode",
        type=str,
        default="random",
        choices=["random", "fixed_seed", "deterministic"],
        help="Sampling mode: random (default), fixed_seed, or deterministic"
    )
    parser.add_argument(
        "--compare_chunk_steps",
        type=str,
        default=None,
        help="Comma-separated list of n_action_steps to compare in chunk mode, e.g., '1,8,16'"
    )
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
    print("Step 3: Multi-Seed Action Evaluation")
    print("=" * 80)
    
    # Check if we need to run special comparison experiments
    if args.test_sampling_mode != "random" and args.eval_mode == "one_step":
        # Run sampling mode comparison
        sampling_results = run_sampling_mode_comparison(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            episode_index=args.episode_index,
            num_seeds=args.num_seeds,
            base_seed=args.base_seed,
            task_description=args.task,
            device=args.device,
        )
        
        # Save sampling comparison results
        output_file = Path("outputs") / f"sampling_mode_comparison_ep{args.episode_index}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(sampling_results, f, indent=2, cls=NumpyEncoder)
        print(f"\nSampling comparison results saved to: {output_file}")
        
    elif args.compare_chunk_steps is not None and args.eval_mode == "chunk":
        # Run chunk step comparison
        chunk_steps_list = [int(x) for x in args.compare_chunk_steps.split(",")]
        chunk_results = run_chunk_step_comparison(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            episode_index=args.episode_index,
            task_description=args.task,
            device=args.device,
            chunk_steps_list=chunk_steps_list,
        )
        
        # Save chunk comparison results
        output_file = Path("outputs") / f"chunk_step_comparison_ep{args.episode_index}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(chunk_results, f, indent=2, cls=NumpyEncoder)
        print(f"\nChunk comparison results saved to: {output_file}")
        
    else:
        # Run standard multi-seed evaluation
        results, normalized_maes_4d, raw_maes_4d = run_multiseed_evaluation(
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
            test_sampling_mode=args.test_sampling_mode,
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