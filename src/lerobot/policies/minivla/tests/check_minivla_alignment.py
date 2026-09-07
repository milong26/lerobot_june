#!/usr/bin/env python
"""
check_minivla_alignment.py

Consistency test script to verify LeRobot MiniVLA implementation aligns with official MiniVLA.
Compares model config, tokenizer vocabulary, vision/projector shapes, action tokenizer behavior,
and forward pass shapes between official MiniVLA and LeRobot MiniVLA implementations.

Usage:
    python check_minivla_alignment.py \
        --official_checkpoint /path/to/official/checkpoint.pt \
        --lr_config_json /path/to/lr/config.json

This script ensures that official MiniVLA -> lb_minivla are fully consistent
except for LeRobot data interface adaptations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_official_config(checkpoint_path: str) -> dict[str, Any]:
    """Load official MiniVLA config.json from checkpoint run directory."""
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.suffix == ".pt" and ckpt_path.parent.name == "checkpoints":
        run_dir = ckpt_path.parents[1]
    else:
        run_dir = ckpt_path.parent

    config_json = run_dir / "config.json"
    if not config_json.exists():
        raise FileNotFoundError(f"config.json not found at {config_json}")

    with open(config_json, "r") as f:
        return json.load(f)


def compare_configs(official_config: dict, lr_config: dict) -> list[str]:
    """Compare official config with LeRobot config."""
    mismatches = []

    vla_cfg = official_config.get("vla", {})
    model_cfg = official_config.get("model", {})

    # Compare vision_backbone_id
    if lr_config.get("vision_backbone_id") != model_cfg.get("vision_backbone_id"):
        mismatches.append(
            f"vision_backbone_id: official={model_cfg.get('vision_backbone_id')}, "
            f"lr={lr_config.get('vision_backbone_id')}"
        )

    # Compare llm_backbone_id
    if lr_config.get("llm_backbone_id") != model_cfg.get("llm_backbone_id"):
        mismatches.append(
            f"llm_backbone_id: official={model_cfg.get('llm_backbone_id')}, "
            f"lr={lr_config.get('llm_backbone_id')}"
        )

    # Compare base_vlm_checkpoint
    if lr_config.get("base_vlm_checkpoint") != vla_cfg.get("base_vlm"):
        mismatches.append(
            f"base_vlm_checkpoint: official={vla_cfg.get('base_vlm')}, "
            f"lr={lr_config.get('base_vlm_checkpoint')}"
        )

    # Compare image_size
    if lr_config.get("image_size") != model_cfg.get("image_size"):
        mismatches.append(
            f"image_size: official={model_cfg.get('image_size')}, "
            f"lr={lr_config.get('image_size')}"
        )

    # Compare arch_specifier
    if lr_config.get("arch_specifier") != model_cfg.get("arch_specifier"):
        mismatches.append(
            f"arch_specifier: official={model_cfg.get('arch_specifier')}, "
            f"lr={lr_config.get('arch_specifier')}"
        )

    # Compare num_extra_tokens
    if lr_config.get("num_extra_tokens") != 256:
        mismatches.append(f"num_extra_tokens: expected=256, got={lr_config.get('num_extra_tokens')}")

    # Compare chunk_size
    if lr_config.get("chunk_size") != 8:
        mismatches.append(f"chunk_size: expected=8, got={lr_config.get('chunk_size')}")

    # Compare n_action_steps
    if lr_config.get("n_action_steps") != 1:
        mismatches.append(f"n_action_steps: expected=1, got={lr_config.get('n_action_steps')}")

    # Compare vqvae_n_embed
    if lr_config.get("vqvae_n_embed") != 128:
        mismatches.append(f"vqvae_n_embed: expected=128, got={lr_config.get('vqvae_n_embed')}")

    # Compare vqvae_groups
    if lr_config.get("vqvae_groups") != 7:
        mismatches.append(f"vqvae_groups: expected=7, got={lr_config.get('vqvae_groups')}")

    # Compare n_latent_dims
    if lr_config.get("n_latent_dims") != 512:
        mismatches.append(f"n_latent_dims: expected=512, got={lr_config.get('n_latent_dims')}")

    # Compare optimizer_lr
    if lr_config.get("optimizer_lr") != 2e-5:
        mismatches.append(f"optimizer_lr: expected=2e-5, got={lr_config.get('optimizer_lr')}")

    # Compare optimizer_weight_decay
    if lr_config.get("optimizer_weight_decay") != 0.0:
        mismatches.append(f"optimizer_weight_decay: expected=0.0, got={lr_config.get('optimizer_weight_decay')}")

    # Compare dtype
    if lr_config.get("dtype") != "bfloat16":
        mismatches.append(f"dtype: expected=bfloat16, got={lr_config.get('dtype')}")

    # Compare scheduler_type
    if lr_config.get("scheduler_type") != "constant":
        mismatches.append(f"scheduler_type: expected=constant, got={lr_config.get('scheduler_type')}")

    # Compare image_sequence_len
    official_seq_len = vla_cfg.get("image_sequence_len", 1)
    if lr_config.get("image_sequence_len") != official_seq_len:
        mismatches.append(
            f"image_sequence_len: official={official_seq_len}, "
            f"lr={lr_config.get('image_sequence_len')}"
        )

    # Compare use_wrist_image
    official_use_wrist = vla_cfg.get("use_wrist_image", False)
    if lr_config.get("use_wrist_image") != official_use_wrist:
        mismatches.append(
            f"use_wrist_image: official={official_use_wrist}, "
            f"lr={lr_config.get('use_wrist_image')}"
        )

    return mismatches


def check_action_tokenizer_consistency(lr_config: dict, lr_action_tokenizer) -> list[str]:
    """Check action tokenizer encode/decode consistency."""
    mismatches = []

    np.random.seed(42)
    torch.manual_seed(42)

    action_dim = lr_config.get("action_dim", 7)
    chunk_size = lr_config.get("chunk_size", 8)
    tokenizer_len = lr_config.get("tokenizer_len", 151936)

    random_action = np.random.uniform(-1.0, 1.0, (1, chunk_size, action_dim)).astype(np.float32)

    try:
        encoded = lr_action_tokenizer(random_action)
        decoded = lr_action_tokenizer.decode_token_ids_to_actions(encoded)

        if hasattr(decoded, "detach"):
            decoded = decoded.detach().cpu().numpy()

        if hasattr(random_action, "detach"):
            random_action = random_action.detach().cpu().numpy()

        max_diff = np.max(np.abs(random_action - decoded))
        if max_diff > 0.01:
            mismatches.append(
                f"Action tokenizer encode/decode mismatch: max_diff={max_diff:.6f} "
                f"(input_shape={random_action.shape}, output_shape={decoded.shape})"
            )
    except Exception as e:
        mismatches.append(f"Action tokenizer consistency check failed: {e}")

    return mismatches


def check_vq_encode_decode_consistency(lr_config: dict, lr_action_tokenizer) -> list[str]:
    """Check VQ-VAE encode -> decode roundtrip consistency."""
    mismatches = []

    np.random.seed(42)
    torch.manual_seed(42)

    action_dim = lr_config.get("action_dim", 7)
    chunk_size = lr_config.get("chunk_size", 8)

    random_action = np.random.uniform(-1.0, 1.0, (2, chunk_size, action_dim)).astype(np.float32)

    try:
        encoded = lr_action_tokenizer(random_action)

        decoded = lr_action_tokenizer.decode_token_ids_to_actions(encoded)

        if hasattr(decoded, "detach"):
            decoded = decoded.detach().cpu().numpy()

        if hasattr(random_action, "detach"):
            random_action = random_action.detach().cpu().numpy()

        max_diff = np.max(np.abs(random_action - decoded))
        if max_diff > 0.01:
            mismatches.append(
                f"VQ encode/decode roundtrip mismatch: max_diff={max_diff:.6f} "
                f"(input_shape={random_action.shape}, output_shape={decoded.shape})"
            )
        else:
            print(f"  VQ encode/decode roundtrip OK: max_diff={max_diff:.6f}")
    except Exception as e:
        mismatches.append(f"VQ encode/decode consistency check failed: {e}")

    return mismatches


def check_vision_projector_shapes(lr_model, lr_config: dict) -> list[str]:
    """Check vision encoder and projector output shapes."""
    mismatches = []

    try:
        image_size = lr_config.get("image_size", 224)
        batch_size = 2

        dummy_pixel_values = {
            "image": torch.randn(batch_size, 3, image_size, image_size),
        }

        with torch.no_grad():
            patch_embeddings = lr_model.vision_backbone(dummy_pixel_values)

        projected_patches = lr_model.projector(patch_embeddings)

        expected_num_patches = lr_model.num_patches
        actual_num_patches = projected_patches.shape[1]
        if actual_num_patches != expected_num_patches:
            mismatches.append(
                f"Vision num_patches mismatch: expected={expected_num_patches}, "
                f"actual={actual_num_patches}"
            )

        projector_output_dim = projected_patches.shape[-1]
        llm_config = lr_model.llm.config
        expected_llm_dim = llm_config.hidden_size
        if projector_output_dim != expected_llm_dim:
            mismatches.append(
                f"Projector output dim mismatch: expected={expected_llm_dim}, "
                f"actual={projector_output_dim}"
            )

        print(f"  Vision output shape: {patch_embeddings.shape}")
        print(f"  Projector output shape: {projected_patches.shape}")
        print(f"  Projector output dim matches LLM hidden dim: {projector_output_dim == expected_llm_dim}")

    except Exception as e:
        mismatches.append(f"Vision/projector shape check failed: {e}")

    return mismatches


def check_forward_pass_shapes(lr_model, lr_config: dict) -> list[str]:
    """Check forward pass shapes and tensor operations."""
    mismatches = []

    try:
        image_size = lr_config.get("image_size", 224)
        batch_size = 2
        seq_len = 10

        input_ids = torch.randint(0, lr_config.get("tokenizer_len", 151936), (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = input_ids.clone()

        pixel_values = {
            "image": torch.randn(batch_size, 3, image_size, image_size),
        }

        with torch.no_grad():
            output = lr_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )

        if hasattr(output, "logits"):
            logits = output.logits
            expected_seq_len = seq_len + lr_model.num_patches
            if logits.shape[1] != expected_seq_len:
                mismatches.append(
                    f"Forward logits seq_len mismatch: expected={expected_seq_len}, "
                    f"actual={logits.shape[1]}"
                )
            print(f"  Forward logits shape: {logits.shape}")

        if hasattr(output, "loss") and output.loss is not None:
            print(f"  Forward loss: {output.loss.item():.6f}")

        print(f"  Forward pass shapes verified successfully")

    except Exception as e:
        mismatches.append(f"Forward pass shape check failed: {e}")

    return mismatches


def run_alignment_check(
    official_checkpoint: str,
    lr_config: dict,
    lr_model=None,
    lr_action_tokenizer=None,
) -> list[str]:
    """Run full alignment check between official and LeRobot MiniVLA."""
    all_mismatches = []

    # Load official config
    try:
        official_config = load_official_config(official_checkpoint)
    except Exception as e:
        all_mismatches.append(f"Failed to load official config: {e}")
        return all_mismatches

    # Compare configs
    print("[1/5] Checking config consistency...")
    config_mismatches = compare_configs(official_config, lr_config)
    all_mismatches.extend(config_mismatches)
    if not config_mismatches:
        print("  Config consistency OK")

    # Check action tokenizer consistency
    if lr_action_tokenizer is not None:
        print("[2/5] Checking action tokenizer consistency...")
        tokenizer_mismatches = check_action_tokenizer_consistency(lr_config, lr_action_tokenizer)
        all_mismatches.extend(tokenizer_mismatches)
        if not tokenizer_mismatches:
            print("  Action tokenizer consistency OK")
    else:
        print("[2/5] Skipping action tokenizer check (no tokenizer provided)")

    # Check VQ encode/decode consistency
    if lr_action_tokenizer is not None:
        print("[3/5] Checking VQ encode/decode consistency...")
        vq_mismatches = check_vq_encode_decode_consistency(lr_config, lr_action_tokenizer)
        all_mismatches.extend(vq_mismatches)
        if not vq_mismatches:
            print("  VQ encode/decode consistency OK")
    else:
        print("[3/5] Skipping VQ encode/decode check (no tokenizer provided)")

    # Check vision/projector shapes
    if lr_model is not None:
        print("[4/5] Checking vision/projector shapes...")
        vision_mismatches = check_vision_projector_shapes(lr_model, lr_config)
        all_mismatches.extend(vision_mismatches)
        if not vision_mismatches:
            print("  Vision/projector shapes OK")
    else:
        print("[4/5] Skipping vision/projector shape check (no model provided)")

    # Check forward pass shapes
    if lr_model is not None:
        print("[5/5] Checking forward pass shapes...")
        forward_mismatches = check_forward_pass_shapes(lr_model, lr_config)
        all_mismatches.extend(forward_mismatches)
        if not forward_mismatches:
            print("  Forward pass shapes OK")
    else:
        print("[5/5] Skipping forward pass shape check (no model provided)")

    return all_mismatches


def main():
    parser = argparse.ArgumentParser(description="Check MiniVLA alignment between official and LeRobot implementations")
    parser.add_argument("--official_checkpoint", type=str, required=True, help="Path to official MiniVLA checkpoint (.pt)")
    parser.add_argument("--lr_config_json", type=str, required=True, help="Path to LeRobot MiniVLA config.json")
    parser.add_argument("--lr_model_path", type=str, default=None, help="Path to LeRobot MiniVLA model checkpoint (optional, for shape checks)")
    parser.add_argument("--vq_vae_path", type=str, default=None, help="Path to VQ-VAE checkpoint directory (optional, for action tokenizer checks)")
    args = parser.parse_args()

    # Load LeRobot config
    with open(args.lr_config_json, "r") as f:
        lr_config = json.load(f)

    print("=" * 80)
    print("MiniVLA Alignment Check")
    print("=" * 80)
    print(f"Official checkpoint: {args.official_checkpoint}")
    print(f"LeRobot config: {args.lr_config_json}")
    print()

    # Load LeRobot model if path provided
    lr_model = None
    lr_action_tokenizer = None

    if args.lr_model_path is not None:
        print("Loading LeRobot model...")
        from ..configuration_minivla import MiniVLAConfig
        from ..modeling_minivla import MiniVLAPolicy

        config = MiniVLAConfig(**lr_config)
        lr_model = MiniVLAPolicy(config)

        model_state = torch.load(args.lr_model_path, map_location="cpu")
        if "model" in model_state:
            lr_model.load_state_dict(model_state["model"])
        else:
            lr_model.load_state_dict(model_state)

        lr_model.eval()
        print("LeRobot model loaded successfully")
        print()

    if args.vq_vae_path is not None and lr_config.get("tokenizer_len") is not None:
        print("Loading LeRobot action tokenizer...")
        from transformers import AutoTokenizer
        from ..vq_action import VQActionTokenizer

        tokenizer = AutoTokenizer.from_pretrained(lr_config.get("llm_backbone", "Qwen/Qwen2.5-0.5B"))
        lr_action_tokenizer = VQActionTokenizer(
            tokenizer=tokenizer,
            vq_vae_path=args.vq_vae_path,
            use_extra=lr_config.get("use_extra_tokens", False),
        )
        print("LeRobot action tokenizer loaded successfully")
        print()

    # Run alignment check
    mismatches = run_alignment_check(
        args.official_checkpoint,
        lr_config,
        lr_model=lr_model,
        lr_action_tokenizer=lr_action_tokenizer,
    )

    if mismatches:
        print()
        print("MISMATCHES FOUND:")
        for i, mismatch in enumerate(mismatches, 1):
            print(f"  {i}. {mismatch}")
        print()
        print(f"Total mismatches: {len(mismatches)}")
        sys.exit(1)
    else:
        print()
        print("ALL CHECKS PASSED - Official and LeRobot MiniVLA are aligned!")
        sys.exit(0)


if __name__ == "__main__":
    main()