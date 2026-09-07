#!/usr/bin/env python
"""
MiniVLA Initialization Check Script

Prints detailed information about MiniVLA model initialization:
- VLM backbone source (HuggingFace vs official checkpoint)
- Whether full VLA checkpoint is loaded
- VQ tokenizer path
- Model parameter counts (total, trainable)
- Configuration details

Usage:
    python check_minivla_init.py [--policy-type minivla] [--vq-model-path PATH]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from lerobot.policies.factory import make_policy_config, get_policy_class
from lerobot.policies.minivla.configuration_minivla import MiniVLAConfig


def check_minivla_init(
    policy_type: str = "minivla",
    vq_model_path: str = "Stanford-ILIAD/pretrain_vq",
    action_tokenizer_type: str = "libero_vq_action_tokenizer",
    primary_image_key: str = "observation.images.corner",
    wrist_image_key: str = "observation.images.gripperPOV",
):
    """Check MiniVLA initialization and print details."""

    print("=" * 70)
    print("  MiniVLA Initialization Check")
    print("=" * 70)

    # Create config
    cfg = make_policy_config(
        policy_type,
        vq_model_path=vq_model_path,
        action_tokenizer_type=action_tokenizer_type,
        primary_image_key=primary_image_key,
        wrist_image_key=wrist_image_key,
    )

    # Print configuration
    print("\n[Configuration]")
    print(f"  Policy Type:          {policy_type}")
    print(f"  VLM Backbone:         {cfg.base_vlm_checkpoint}")
    print(f"  Vision Backbone:      {cfg.vision_backbone_id}")
    print(f"  LLM Backbone:         {cfg.llm_backbone_id}")
    print(f"  Image Size:           {cfg.image_size}")
    print(f"  Action Tokenizer:     {cfg.action_tokenizer_type}")
    print(f"  VQ Model Path:        {cfg.vq_model_path}")
    print(f"  Official VLA Checkpoint: {cfg.official_vla_checkpoint or 'None (from scratch)'}")
    print(f"  Primary Image Key:    {cfg.primary_image_key}")
    print(f"  Wrist Image Key:      {cfg.wrist_image_key}")
    print(f"  Chunk Size:           {cfg.chunk_size}")
    print(f"  VQ Mode:              {cfg.is_vq_mode}")

    # Check VQ tokenizer path
    vq_path = cfg.resolve_vq_model_path()
    print(f"\n[VQ Tokenizer]")
    print(f"  Resolved Path:        {vq_path or 'Not found'}")
    if vq_path:
        vq_dir = Path(vq_path)
        config_exists = (vq_dir / "config.json").exists()
        model_exists = (vq_dir / "checkpoints" / "model.pt").exists()
        print(f"  config.json:          {'Found' if config_exists else 'MISSING'}")
        print(f"  checkpoints/model.pt: {'Found' if model_exists else 'MISSING'}")

    # Check if loading full VLA checkpoint
    print(f"\n[Checkpoint Loading]")
    if cfg.official_vla_checkpoint:
        print(f"  Loading full VLA checkpoint: {cfg.official_vla_checkpoint}")
        print(f"  WARNING: This will load pretrained policy weights!")
    else:
        print(f"  NOT loading full VLA checkpoint")
        print(f"  VLM backbone will be initialized from HuggingFace pretrained weights")
        print(f"  Action head will be randomly initialized")

    # Create policy (without pretrained path)
    print(f"\n[Model Initialization]")
    policy_cls = get_policy_class(policy_type)

    # Create minimal config for initialization check
    cfg.device = "cpu"
    cfg.dtype = "float32"

    try:
        policy = policy_cls(cfg)
        print(f"  Policy created successfully: {policy_cls.__name__}")
    except Exception as e:
        print(f"  ERROR creating policy: {e}")
        import traceback
        traceback.print_exc()
        return

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n[Parameter Counts]")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  Frozen Parameters:    {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")

    # Check VQ-VAE is frozen
    if hasattr(policy.model, 'vq_vae') and policy.model.vq_vae is not None:
        vq_params = sum(p.numel() for p in policy.model.vq_vae.parameters())
        vq_trainable = sum(p.numel() for p in policy.model.vq_vae.parameters() if p.requires_grad)
        print(f"\n[VQ-VAE]")
        print(f"  Total VQ-VAE Params:  {vq_params:,}")
        print(f"  Trainable VQ Params:  {vq_trainable:,} (should be 0)")
        if vq_trainable > 0:
            print(f"  WARNING: VQ-VAE should be frozen!")

    # Check which components are trainable
    print(f"\n[Trainable Components]")
    for name, module in policy.model.named_children():
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in module.parameters())
        if trainable > 0:
            print(f"  {name:20s}: {trainable:,}/{total:,} trainable")

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)

    if cfg.official_vla_checkpoint:
        print("  STATUS: Loading FULL VLA checkpoint (NOT recommended for differentvlm experiments)")
        print("  This will load pretrained policy weights from Stanford-ILIAD checkpoint.")
    else:
        print("  STATUS: Initializing from VLM backbone only (CORRECT for differentvlm experiments)")
        print("  VLM backbone weights from HuggingFace, action head randomly initialized.")
        print("  This matches the differentvlm experimental setup.")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Check MiniVLA Initialization")
    parser.add_argument("--policy-type", type=str, default="minivla",
                       help="Policy type (minivla, minivla_t2, minivla_wrist)")
    parser.add_argument("--vq-model-path", type=str, default="Stanford-ILIAD/pretrain_vq",
                       help="Path to VQ-VAE model")
    parser.add_argument("--action-tokenizer-type", type=str, default="libero_vq_action_tokenizer",
                       help="Action tokenizer type")
    parser.add_argument("--primary-image-key", type=str, default="observation.images.corner",
                       help="Primary image observation key")
    parser.add_argument("--wrist-image-key", type=str, default="observation.images.gripperPOV",
                       help="Wrist image observation key")
    args = parser.parse_args()

    check_minivla_init(
        policy_type=args.policy_type,
        vq_model_path=args.vq_model_path,
        action_tokenizer_type=args.action_tokenizer_type,
        primary_image_key=args.primary_image_key,
        wrist_image_key=args.wrist_image_key,
    )


if __name__ == "__main__":
    main()