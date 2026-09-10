#!/usr/bin/env python
"""
Step 1 & 2: Verify TinyVLA checkpoint loading and compare actions with expert.

Step 1: Verify checkpoint/action head is correctly loaded.
  - Check missing_keys, unexpected_keys, LoRA key count
  - Compare checkpoint weights with loaded model weights

Step 2: Compare TinyVLA actions with expert actions on same observations.
  - Run 2 episodes, record both TinyVLA and expert actions
  - Compare xyz cosine similarity, per-dim MAE, direction sign agreement, gripper action
  - Compare TinyVLA env action vs clip(expert_action[:3], -1, 1)
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
    PROJECT_ROOT / "personal" / "work2" / "differentvlm" / "experiments" /
    "tinyvla_b_disassemble-v3_corner" / "checkpoints" /
    "tinyvla_tinyvla_b_disassemble-v3_corner" / "checkpoints" / "012000" / "pretrained_model"
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


def step2_compare_actions_with_expert(model):
    """Step 2: Compare TinyVLA actions with expert actions from dataset."""
    print("\n" + "=" * 80)
    print("Step 2: Compare TinyVLA vs Expert Actions (from dataset)")
    print("=" * 80)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.envs import preprocess_observation

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

    # Sample N frames from dataset for comparison
    n_samples = 10
    sample_indices = np.random.choice(len(dataset), size=n_samples, replace=False)

    # Create preprocessor and postprocessor using lerobot's factory
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=CHECKPOINT_PATH,
    )

    comparison_data = []

    for idx, frame_idx in enumerate(sample_indices):
        print(f"\n--- Sample {idx}/{n_samples} (frame {frame_idx}) ---")

        # Get frame from dataset
        frame = dataset[frame_idx]
        expert_action = frame["action"]  # torch tensor, shape (4,)
        observation_state = frame["observation.state"]  # torch tensor, shape (4,)

        # Build observation dict for TinyVLA
        observation = {
            "observation.images.top": frame["observation.images.top"],  # (C, H, W)
            "observation.images.wrist": frame["observation.images.wrist"],  # (C, H, W)
            "observation.state": observation_state,
            "task": "disassemble",
        }

        # Preprocess observation
        observation_tensor = preprocess_observation(observation)
        observation_tensor = preprocessor(observation_tensor)

        # Get TinyVLA action
        with torch.inference_mode():
            raw_action = model.select_action(observation_tensor)

        action = postprocessor(raw_action)
        action_numpy = action.to("cpu").to(dtype=torch.float32).numpy()

        # Clip to env bounds
        env_low = np.array([-1.0, -1.0, -1.0, 0.0])
        env_high = np.array([1.0, 1.0, 1.0, 1.0])
        action_clipped = np.clip(action_numpy, env_low, env_high)

        # Compare with expert action
        expert_action_numpy = expert_action.numpy()
        expert_clipped = np.clip(expert_action_numpy[:3], -1, 1)
        tinyvla_xyz = action_clipped[0, :3]

        # Cosine similarity
        norm_exp = np.linalg.norm(expert_clipped)
        norm_tv = np.linalg.norm(tinyvla_xyz)
        if norm_exp > 0 and norm_tv > 0:
            cosine_sim = np.dot(expert_clipped, tinyvla_xyz) / (norm_exp * norm_tv)
        else:
            cosine_sim = 0.0

        # Per-dim MAE
        mae = np.abs(expert_clipped - tinyvla_xyz).mean()

        # Direction sign agreement
        sign_agree = np.sum(np.sign(expert_clipped) == np.sign(tinyvla_xyz)) / 3.0

        # Gripper comparison
        expert_gripper = expert_action_numpy[3]
        gripper_diff = abs(action_clipped[0, 3] - expert_gripper)

        step_data = {
            "sample_index": idx,
            "frame_index": int(frame_idx),
            "expert_action": expert_action_numpy.tolist(),
            "expert_clipped": expert_clipped.tolist(),
            "tinyvla_raw": raw_action.to("cpu").to(dtype=torch.float32).numpy().tolist(),
            "tinyvla_postproc": action_numpy.tolist(),
            "tinyvla_env_action": action_clipped.tolist(),
            "xyz_cosine_sim": float(cosine_sim),
            "xyz_mae": float(mae),
            "sign_agreement": float(sign_agree),
            "gripper_diff": float(gripper_diff),
        }

        comparison_data.append(step_data)

        print(f"  Expert action: {expert_action_numpy}")
        print(f"  TinyVLA action (clipped): {action_clipped[0]}")
        print(f"  Cosine sim: {cosine_sim:.3f}, MAE: {mae:.3f}, Sign: {sign_agree:.2f}, Gripper diff: {gripper_diff:.3f}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("Action Comparison Summary")
    print("=" * 80)

    all_cosine = [d["xyz_cosine_sim"] for d in comparison_data]
    all_mae = [d["xyz_mae"] for d in comparison_data]
    all_sign = [d["sign_agreement"] for d in comparison_data]
    all_gripper_diff = [d["gripper_diff"] for d in comparison_data]

    print(f"\nXYZ Cosine Similarity: mean={np.mean(all_cosine):.3f}, std={np.std(all_cosine):.3f}")
    print(f"XYZ MAE: mean={np.mean(all_mae):.3f}, std={np.std(all_mae):.3f}")
    print(f"Sign Agreement: mean={np.mean(all_sign):.3f}, std={np.std(all_sign):.3f}")
    print(f"Gripper Diff: mean={np.mean(all_gripper_diff):.3f}, std={np.std(all_gripper_diff):.3f}")

    # Save results
    output_file = Path(__file__).parent / "tinyvla_action_comparison.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": {
                "xyz_cosine_sim": {"mean": np.mean(all_cosine), "std": np.std(all_cosine)},
                "xyz_mae": {"mean": np.mean(all_mae), "std": np.std(all_mae)},
                "sign_agreement": {"mean": np.mean(all_sign), "std": np.std(all_sign)},
                "gripper_diff": {"mean": np.mean(all_gripper_diff), "std": np.std(all_gripper_diff)},
            },
            "samples": comparison_data,
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return comparison_data


if __name__ == "__main__":
    # Step 1
    step1_result, model = step1_verify_checkpoint_loading()

    # Step 2
    step2_result = step2_compare_actions_with_expert(model)

    print("\n" + "=" * 80)
    print("All steps completed!")
    print("=" * 80)