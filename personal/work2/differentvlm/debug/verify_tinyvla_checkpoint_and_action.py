#!/usr/bin/env python
"""
TinyVLA Action Accuracy Evaluation Script.

Evaluates TinyVLA model's action prediction accuracy against ground truth actions
from the dataset. Supports two evaluation modes to distinguish different types of errors.

Evaluation Modes:
1. one_step: Reset model for each frame, test strict single-step accuracy
   - Answers: "Given current observation, can the model predict the correct action?"
   - Uses: model.reset() -> select_action() for every frame
   
2. chunk: Use action queue mechanism, simulate actual evaluation behavior
   - Answers: "Can the model predict a 16-step action chunk that matches future expert trajectory?"
   - Uses: select_action() with internal queue (generates new chunk every 16 steps)

Key Features:
- Uses correct task description from training ("Pick a nut out of a peg")
- Clips both expert and model XYZ to [-1, 1] for fair comparison
- Keeps gripper values unclipped (original dataset values)
- Provides per-dimension error statistics
- Tracks frames with MAE < 0.2 as a quality metric

Usage:
  # One-step mode (default)
  python verify_tinyvla_checkpoint_and_action.py --eval_mode one_step --episode_index 0
  
  # Chunk mode
  python verify_tinyvla_checkpoint_and_action.py --eval_mode chunk --episode_index 0
  
  # Both modes with comparison
  python verify_tinyvla_checkpoint_and_action.py --eval_mode both --episode_index 0
  
  # Custom task description
  python verify_tinyvla_checkpoint_and_action.py --task "Pick a nut out of a peg"
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.tinyvla.configuration_tinyvla import TinyVLABConfig
from lerobot.policies.tinyvla.modeling_tinyvla import TinyVLABPolicy

CHECKPOINT_PATH = str(
    PROJECT_ROOT / "personal" / "work2" / "differentvlm" / "tinyvla" /
    "tinyvla_s_random_ep200_seed42_disassemble-v3_corner" / "checkpoints" /
    "050000" / "pretrained_model"
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


def step1_verify_checkpoint_loading():
    """Step 1: Verify checkpoint is correctly loaded using lerobot's from_pretrained."""
    print("=" * 80)
    print("Step 1: Verify Checkpoint Loading")
    print("=" * 80)

    checkpoint_path = Path(CHECKPOINT_PATH)
    model_file = checkpoint_path / "model.safetensors"

    if not model_file.exists():
        print(f"ERROR: model.safetensors not found at {model_file}")
        sys.exit(1)

    # Load checkpoint state dict directly
    print(f"\nLoading checkpoint state dict from: {model_file}")
    checkpoint_state = load_file(str(model_file))
    checkpoint_keys = set(checkpoint_state.keys())
    lora_keys_in_ckpt = [k for k in checkpoint_keys if 'lora' in k.lower()]
    action_head_keys = [k for k in checkpoint_keys if 'action' in k.lower() or 'diffusion' in k.lower()]

    print(f"Total keys in checkpoint: {len(checkpoint_keys)}")
    print(f"LoRA keys in checkpoint: {len(lora_keys_in_ckpt)}")
    print(f"Action/diffusion keys in checkpoint: {len(action_head_keys)}")

    # Load config from checkpoint
    print("\nLoading config from checkpoint...")
    config = PreTrainedConfig.from_pretrained(CHECKPOINT_PATH)
    print(f"Config type: {type(config).__name__}")

    # Load policy using lerobot's from_pretrained
    print("\nLoading policy via from_pretrained (lerobot standard way)...")
    model = TinyVLABPolicy.from_pretrained(
        pretrained_name_or_path=CHECKPOINT_PATH,
    )

    # Compare weights between loaded model and checkpoint
    print("\nComparing loaded model vs checkpoint weights...")
    mismatches = []
    shape_mismatches = []
    loaded_state = model.state_dict()

    # Check for missing and unexpected keys
    missing_keys = [k for k in checkpoint_keys if k not in loaded_state]
    unexpected_keys = [k for k in loaded_state if k not in checkpoint_keys]

    print(f"\nMissing keys: {len(missing_keys)}")
    if missing_keys:
        for k in missing_keys[:20]:
            print(f"  - {k}")

    print(f"\nUnexpected keys: {len(unexpected_keys)}")
    if unexpected_keys:
        for k in unexpected_keys[:20]:
            print(f"  - {k}")

    for key in checkpoint_keys:
        if key in loaded_state:
            ckpt_weight = checkpoint_state[key].cpu()
            loaded_weight = loaded_state[key].cpu()
            if ckpt_weight.shape != loaded_weight.shape:
                shape_mismatches.append((key, ckpt_weight.shape, loaded_weight.shape))
            else:
                diff = (ckpt_weight - loaded_weight).abs().max().item()
                if diff > 1e-5:
                    mismatches.append((key, diff))

    print(f"Shape mismatches: {len(shape_mismatches)}")
    for key, ckpt_shape, loaded_shape in shape_mismatches[:10]:
        print(f"  - {key}: checkpoint={ckpt_shape}, loaded={loaded_shape}")

    print(f"Weight mismatches (diff > 1e-5): {len(mismatches)}")
    for key, diff in sorted(mismatches, key=lambda x: -x[1])[:10]:
        print(f"  - {key}: max_diff={diff:.6e}")

    # Key weight summary
    print("\nKey weight summary:")
    for key in sorted(checkpoint_keys):
        if any(x in key.lower() for x in ['embed_out', 'action', 'diffusion', 'unet']):
            w = checkpoint_state[key]
            print(f"  {key}: shape={tuple(w.shape)}, mean={w.mean().item():.4f}, std={w.std().item():.4f}")

    # Check if LoRA adapters are actually applied
    print("\nChecking LoRA adapter status...")
    lora_applied = False
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
            lora_applied = True
            print(f"  LoRA module found: {name}")
            break

    if not lora_applied:
        print("  WARNING: No LoRA modules found in model!")

    result = {
        "total_keys": len(checkpoint_keys),
        "lora_keys": len(lora_keys_in_ckpt),
        "action_head_keys": len(action_head_keys),
        "missing_keys": len(missing_keys),
        "missing_keys_list": missing_keys,
        "unexpected_keys": len(unexpected_keys),
        "unexpected_keys_list": unexpected_keys,
        "shape_mismatches": len(shape_mismatches),
        "weight_mismatches": len(mismatches),
        "lora_applied": lora_applied,
    }

    print(f"\nStep 1 Result: {json.dumps(result, indent=2)}")
    return result, model


def step2_compare_actions_with_expert(model, episode_index=0, eval_mode="one_step", task_description=None):
    """Step 2: Compare TinyVLA actions with expert actions from dataset.
    
    Args:
        model: Loaded TinyVLA model
        episode_index: Which episode to evaluate (default: 0)
        eval_mode: Evaluation mode
            - "one_step": 每帧都reset后推理，测试单步精度
            - "chunk": 使用action queue机制，模拟实际eval行为（每16步生成一次chunk）
        task_description: Language instruction for the model. 
            If None, uses the correct training task description.
    """
    print("\n" + "=" * 80)
    print("Step 2: Compare TinyVLA vs Expert Actions (from dataset)")
    print("=" * 80)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.envs import preprocess_observation
    from tqdm import tqdm

    # Use the correct task description from training
    if task_description is None:
        task_description = "Pick a nut out of a peg"
    
    print(f"Task description: '{task_description}'")
    print(f"Evaluation mode: {eval_mode}")

    # Load dataset
    dataset_root = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/disassemble-v3_corner"
    print(f"Loading dataset from: {dataset_root}")
    dataset = LeRobotDataset(
        repo_id="work2/disassemble-v3_corner",
        root=dataset_root,
        download_videos=False,
    )
    print(f"Dataset frames: {len(dataset)}")
    print(f"Dataset features: {dataset.features}")
    print(f"Dataset episodes: {dataset.num_episodes}")

    # Get all frames for the specified episode
    episode_indices = np.array(dataset.hf_dataset["episode_index"])
    mask = episode_indices == episode_index
    frame_indices = np.where(mask)[0]
    
    if len(frame_indices) == 0:
        print(f"ERROR: Episode {episode_index} not found in dataset")
        return []
    
    print(f"\nEvaluating episode {episode_index} with {len(frame_indices)} frames")
    print(f"Frame indices range: {frame_indices[0]} to {frame_indices[-1]}")

    # Create preprocessor and postprocessor using lerobot's factory
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=CHECKPOINT_PATH,
    )

    comparison_data = []
    total_frames = len(frame_indices)
    
    # Clip bounds for both expert and model actions
    env_low = np.array([-1.0, -1.0, -1.0])
    env_high = np.array([1.0, 1.0, 1.0])

    if eval_mode == "one_step":
        # Mode 1: 每帧都reset，测试严格的单步精度
        print("\n[One-Step Mode] Resetting model for each frame...")
        
        for idx, frame_idx in enumerate(tqdm(frame_indices, desc=f"Processing episode {episode_index} (one-step)")):
            # Get frame from dataset
            frame = dataset[frame_idx]
            expert_action_raw = frame["action"].numpy()  # 原始expert action
            observation_state = frame["observation.state"]

            # Build observation dict for TinyVLA
            observation = {
                "observation.images.top": frame["observation.images.top"],
                "observation.images.wrist": frame["observation.images.wrist"],
                "observation.state": observation_state,
                "task": task_description,
            }

            # Preprocess observation
            observation_tensor = preprocess_observation(observation)
            observation_tensor = preprocessor(observation_tensor)

            # IMPORTANT: Reset model for each frame to get true one-step prediction
            model.reset()
            
            # Get TinyVLA action
            with torch.inference_mode():
                raw_action = model.select_action(observation_tensor)

            action = postprocessor(raw_action)
            action_numpy = action.to("cpu").to(dtype=torch.float32).numpy()

            # Clip both expert and model XYZ to [-1, 1] for fair comparison
            expert_xyz_clipped = np.clip(expert_action_raw[:3], -1, 1)
            model_xyz_clipped = np.clip(action_numpy[0, :3], -1, 1)
            
            # Gripper: keep original values (don't clip)
            expert_gripper = expert_action_raw[3]
            model_gripper = action_numpy[0, 3]

            # Calculate metrics with CLIPPED expert action
            norm_exp = np.linalg.norm(expert_xyz_clipped)
            norm_tv = np.linalg.norm(model_xyz_clipped)
            if norm_exp > 0 and norm_tv > 0:
                cosine_sim = np.dot(expert_xyz_clipped, model_xyz_clipped) / (norm_exp * norm_tv)
            else:
                cosine_sim = 0.0

            # MAE on clipped XYZ
            xyz_mae = np.abs(expert_xyz_clipped - model_xyz_clipped).mean()

            # Direction sign agreement
            sign_agree = np.sum(np.sign(expert_xyz_clipped) == np.sign(model_xyz_clipped)) / 3.0

            # Gripper difference
            gripper_diff = abs(model_gripper - expert_gripper)

            # Per-dimension absolute error (using CLIPPED expert XYZ + original gripper)
            expert_4d_clipped = np.concatenate([expert_xyz_clipped, [expert_gripper]])
            model_4d_clipped = np.clip(action_numpy[0], [-1, -1, -1, 0], [1, 1, 1, 1])
            per_dim_abs_error = np.abs(expert_4d_clipped - model_4d_clipped)
            overall_mae = per_dim_abs_error.mean()

            step_data = {
                "sample_index": idx,
                "frame_index": int(frame_idx),
                "expert_action_raw": expert_action_raw.tolist(),
                "expert_xyz_clipped": expert_xyz_clipped.tolist(),
                "expert_gripper": expert_gripper,
                "tinyvla_raw": raw_action.to("cpu").to(dtype=torch.float32).numpy().tolist(),
                "tinyvla_postproc": action_numpy.tolist(),
                "tinyvla_xyz_clipped": model_xyz_clipped.tolist(),
                "tinyvla_gripper": model_gripper,
                "xyz_cosine_sim": float(cosine_sim),
                "xyz_mae": float(xyz_mae),
                "sign_agreement": float(sign_agree),
                "gripper_diff": float(gripper_diff),
                "per_dim_abs_error": per_dim_abs_error.tolist(),
                "overall_mae": float(overall_mae),
            }

            comparison_data.append(step_data)

            # Print progress every 10 frames
            if (idx + 1) % 10 == 0 or idx == 0 or idx == total_frames - 1:
                print(f"\n  [{idx+1}/{total_frames}] Frame {frame_idx}")
                print(f"    Expert (clipped):     XYZ={expert_xyz_clipped}, Gripper={expert_gripper}")
                print(f"    TinyVLA (clipped):    XYZ={model_xyz_clipped}, Gripper={model_gripper}")
                print(f"    Cosine: {cosine_sim:.3f}, MAE: {xyz_mae:.3f}, Sign: {sign_agree:.2f}, Gripper diff: {gripper_diff:.3f}")

    elif eval_mode == "chunk":
        # Mode 2: 使用action queue，模拟实际eval行为
        print("\n[Chunk Mode] Using action queue (chunk_size=16, simulating real eval)...")
        
        model.reset()
        chunk_counter = 0
        
        for idx, frame_idx in enumerate(tqdm(frame_indices, desc=f"Processing episode {episode_index} (chunk)")):
            # Get frame from dataset
            frame = dataset[frame_idx]
            expert_action_raw = frame["action"].numpy()
            observation_state = frame["observation.state"]

            # Build observation dict for TinyVLA
            observation = {
                "observation.images.top": frame["observation.images.top"],
                "observation.images.wrist": frame["observation.images.wrist"],
                "observation.state": observation_state,
                "task": task_description,
            }

            # Preprocess observation
            observation_tensor = preprocess_observation(observation)
            observation_tensor = preprocessor(observation_tensor)

            # Get TinyVLA action (uses internal queue)
            with torch.inference_mode():
                raw_action = model.select_action(observation_tensor)
                
                # Track when new chunks are generated
                if len(model._action_queue) == model.config.n_action_steps - 1:
                    chunk_counter += 1
                    is_new_chunk = True
                else:
                    is_new_chunk = False

            action = postprocessor(raw_action)
            action_numpy = action.to("cpu").to(dtype=torch.float32).numpy()

            # Clip both expert and model XYZ to [-1, 1]
            expert_xyz_clipped = np.clip(expert_action_raw[:3], -1, 1)
            model_xyz_clipped = np.clip(action_numpy[0, :3], -1, 1)
            
            expert_gripper = expert_action_raw[3]
            model_gripper = action_numpy[0, 3]

            # Calculate metrics
            norm_exp = np.linalg.norm(expert_xyz_clipped)
            norm_tv = np.linalg.norm(model_xyz_clipped)
            if norm_exp > 0 and norm_tv > 0:
                cosine_sim = np.dot(expert_xyz_clipped, model_xyz_clipped) / (norm_exp * norm_tv)
            else:
                cosine_sim = 0.0

            xyz_mae = np.abs(expert_xyz_clipped - model_xyz_clipped).mean()
            sign_agree = np.sum(np.sign(expert_xyz_clipped) == np.sign(model_xyz_clipped)) / 3.0
            gripper_diff = abs(model_gripper - expert_gripper)

            expert_4d_clipped = np.concatenate([expert_xyz_clipped, [expert_gripper]])
            model_4d_clipped = np.clip(action_numpy[0], [-1, -1, -1, 0], [1, 1, 1, 1])
            per_dim_abs_error = np.abs(expert_4d_clipped - model_4d_clipped)
            overall_mae = per_dim_abs_error.mean()

            step_data = {
                "sample_index": idx,
                "frame_index": int(frame_idx),
                "chunk_index": chunk_counter,
                "is_new_chunk": is_new_chunk,
                "expert_action_raw": expert_action_raw.tolist(),
                "expert_xyz_clipped": expert_xyz_clipped.tolist(),
                "expert_gripper": expert_gripper,
                "tinyvla_raw": raw_action.to("cpu").to(dtype=torch.float32).numpy().tolist(),
                "tinyvla_postproc": action_numpy.tolist(),
                "tinyvla_xyz_clipped": model_xyz_clipped.tolist(),
                "tinyvla_gripper": model_gripper,
                "xyz_cosine_sim": float(cosine_sim),
                "xyz_mae": float(xyz_mae),
                "sign_agreement": float(sign_agree),
                "gripper_diff": float(gripper_diff),
                "per_dim_abs_error": per_dim_abs_error.tolist(),
                "overall_mae": float(overall_mae),
            }

            comparison_data.append(step_data)

            # Print progress every 10 frames or at chunk boundaries
            if (idx + 1) % 10 == 0 or idx == 0 or idx == total_frames - 1 or is_new_chunk:
                chunk_marker = " [NEW CHUNK]" if is_new_chunk else ""
                print(f"\n  [{idx+1}/{total_frames}] Frame {frame_idx}{chunk_marker}")
                print(f"    Expert (clipped):     XYZ={expert_xyz_clipped}, Gripper={expert_gripper}")
                print(f"    TinyVLA (clipped):    XYZ={model_xyz_clipped}, Gripper={model_gripper}")
                print(f"    Cosine: {cosine_sim:.3f}, MAE: {xyz_mae:.3f}, Sign: {sign_agree:.2f}, Gripper diff: {gripper_diff:.3f}")
        
        print(f"\nTotal chunks generated: {chunk_counter}")

    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}. Must be 'one_step' or 'chunk'")

    # Summary statistics
    print("\n" + "=" * 80)
    print("Action Comparison Summary")
    print("=" * 80)

    all_cosine = [d["xyz_cosine_sim"] for d in comparison_data]
    all_mae = [d["xyz_mae"] for d in comparison_data]
    all_sign = [d["sign_agreement"] for d in comparison_data]
    all_gripper_diff = [d["gripper_diff"] for d in comparison_data]
    all_overall_mae = [d["overall_mae"] for d in comparison_data]
    
    # Per-dimension statistics
    per_dim_errors = np.array([d["per_dim_abs_error"] for d in comparison_data])
    dim_names = ["x", "y", "z", "gripper"]

    print(f"\nEpisode {episode_index} - Total frames: {total_frames} - Mode: {eval_mode}")
    print(f"\nXYZ Cosine Similarity: mean={np.mean(all_cosine):.3f}, std={np.std(all_cosine):.3f}, "
          f"min={np.min(all_cosine):.3f}, max={np.max(all_cosine):.3f}")
    print(f"XYZ MAE: mean={np.mean(all_mae):.3f}, std={np.std(all_mae):.3f}, "
          f"min={np.min(all_mae):.3f}, max={np.max(all_mae):.3f}")
    print(f"Sign Agreement: mean={np.mean(all_sign):.3f}, std={np.std(all_sign):.3f}")
    print(f"Gripper Diff: mean={np.mean(all_gripper_diff):.3f}, std={np.std(all_gripper_diff):.3f}")
    print(f"Overall MAE (4D, clipped): mean={np.mean(all_overall_mae):.3f}, std={np.std(all_overall_mae):.3f}")
    
    print(f"\nPer-Dimension Absolute Error (clipped expert vs clipped model):")
    for i, name in enumerate(dim_names):
        print(f"  {name:8s}: mean={per_dim_errors[:, i].mean():.4f}, "
              f"std={per_dim_errors[:, i].std():.4f}, "
              f"min={per_dim_errors[:, i].min():.4f}, "
              f"max={per_dim_errors[:, i].max():.4f}")
    
    # Additional analysis: MAE < 0.2 ratio
    mae_below_02 = sum(1 for mae in all_overall_mae if mae < 0.2)
    mae_ratio = mae_below_02 / len(all_overall_mae) * 100
    print(f"\nFrames with overall MAE < 0.2: {mae_below_02}/{total_frames} ({mae_ratio:.1f}%)")

    # Save results
    output_file = Path(__file__).parent / f"tinyvla_action_comparison_ep{episode_index}_{eval_mode}.json"
    
    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
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
    
    with open(output_file, "w") as f:
        json.dump({
            "episode_index": episode_index,
            "checkpoint_path": CHECKPOINT_PATH,
            "task_description": task_description,
            "eval_mode": eval_mode,
            "total_frames": total_frames,
            "summary": {
                "xyz_cosine_sim": {
                    "mean": float(np.mean(all_cosine)), 
                    "std": float(np.std(all_cosine)),
                    "min": float(np.min(all_cosine)),
                    "max": float(np.max(all_cosine))
                },
                "xyz_mae": {
                    "mean": float(np.mean(all_mae)), 
                    "std": float(np.std(all_mae)),
                    "min": float(np.min(all_mae)),
                    "max": float(np.max(all_mae))
                },
                "sign_agreement": {
                    "mean": float(np.mean(all_sign)), 
                    "std": float(np.std(all_sign))
                },
                "gripper_diff": {
                    "mean": float(np.mean(all_gripper_diff)), 
                    "std": float(np.std(all_gripper_diff))
                },
                "overall_mae": {
                    "mean": float(np.mean(all_overall_mae)), 
                    "std": float(np.std(all_overall_mae))
                },
                "per_dimension_abs_error": {
                    name: {
                        "mean": float(per_dim_errors[:, i].mean()),
                        "std": float(per_dim_errors[:, i].std()),
                        "min": float(per_dim_errors[:, i].min()),
                        "max": float(per_dim_errors[:, i].max())
                    }
                    for i, name in enumerate(dim_names)
                },
                "frames_with_mae_below_02": {
                    "count": int(mae_below_02),
                    "ratio": float(mae_ratio)
                }
            },
            "samples": comparison_data,
        }, f, indent=2, cls=NumpyEncoder)
    print(f"\nDetailed results saved to: {output_file}")

    return comparison_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify TinyVLA checkpoint and compare actions with expert",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One-step mode (每帧reset，测试严格单步精度)
  python verify_tinyvla_checkpoint_and_action.py --eval_mode one_step --episode_index 0
  
  # Chunk mode (使用action queue，模拟实际eval)
  python verify_tinyvla_checkpoint_and_action.py --eval_mode chunk --episode_index 0
  
  # Run both modes
  python verify_tinyvla_checkpoint_and_action.py --eval_mode both --episode_index 0
  
  # Custom task description
  python verify_tinyvla_checkpoint_and_action.py --task "Pick a nut out of a peg"
        """
    )
    parser.add_argument("--episode_index", type=int, default=0, help="Episode index to evaluate (default: 0)")
    parser.add_argument("--eval_mode", type=str, default="one_step", choices=["one_step", "chunk", "both"],
                        help="Evaluation mode: one_step (per-frame reset), chunk (action queue), or both (default: one_step)")
    parser.add_argument("--task", type=str, default=None, 
                        help="Task description for the model (default: 'Pick a nut out of a peg')")
    parser.add_argument("--skip_step1", action="store_true", help="Skip step 1 (checkpoint verification)")
    args = parser.parse_args()
    
    # Step 1: Verify checkpoint loading (optional)
    if not args.skip_step1:
        step1_result, model = step1_verify_checkpoint_loading()
    else:
        print("Skipping Step 1, loading model directly...")
        model = TinyVLABPolicy.from_pretrained(
            pretrained_name_or_path=CHECKPOINT_PATH,
        )
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        step1_result = None

    # Step 2: Compare actions with expert
    task_desc = args.task if args.task else "Pick a nut out of a peg"
    
    if args.eval_mode == "both":
        # Run both modes
        print("\n" + "=" * 80)
        print("Running BOTH evaluation modes")
        print("=" * 80)
        
        step2_one_step = step2_compare_actions_with_expert(
            model, episode_index=args.episode_index, 
            eval_mode="one_step", task_description=task_desc
        )
        
        step2_chunk = step2_compare_actions_with_expert(
            model, episode_index=args.episode_index, 
            eval_mode="chunk", task_description=task_desc
        )
        
        print("\n" + "=" * 80)
        print("Comparison: One-Step vs Chunk Mode")
        print("=" * 80)
        
        # Quick comparison
        def calc_stats(data):
            return {
                "xyz_cosine": np.mean([d["xyz_cosine_sim"] for d in data]),
                "xyz_mae": np.mean([d["xyz_mae"] for d in data]),
                "overall_mae": np.mean([d["overall_mae"] for d in data]),
                "gripper_diff": np.mean([d["gripper_diff"] for d in data]),
            }
        
        stats_one = calc_stats(step2_one_step)
        stats_chunk = calc_stats(step2_chunk)
        
        print(f"\n{'Metric':<20} {'One-Step':<15} {'Chunk':<15}")
        print("-" * 50)
        print(f"{'XYZ Cosine':<20} {stats_one['xyz_cosine']:<15.3f} {stats_chunk['xyz_cosine']:<15.3f}")
        print(f"{'XYZ MAE':<20} {stats_one['xyz_mae']:<15.3f} {stats_chunk['xyz_mae']:<15.3f}")
        print(f"{'Overall MAE (4D)':<20} {stats_one['overall_mae']:<15.3f} {stats_chunk['overall_mae']:<15.3f}")
        print(f"{'Gripper Diff':<20} {stats_one['gripper_diff']:<15.3f} {stats_chunk['gripper_diff']:<15.3f}")
        
        step2_result = {"one_step": step2_one_step, "chunk": step2_chunk}
    else:
        step2_result = step2_compare_actions_with_expert(
            model, episode_index=args.episode_index, 
            eval_mode=args.eval_mode, task_description=task_desc
        )

    print("\n" + "=" * 80)
    print("All steps completed!")
    print("=" * 80)