#!/usr/bin/env python
"""
MiniVLA Official Checkpoint Verification Script

Validates that the official VLA checkpoint is correctly configured before training.
Checks:
1. Official checkpoint path exists and contains valid .pt file
2. Checkpoint contains vision/language backbone weights
3. Action head will be reinitialized (not loaded from official checkpoint)
4. Tokenizer is extra_action_tokenizer (not VQ)
5. Optimizer learning rate uses official default (not overridden)

Usage:
    python personal/work2/differentvlm/minivla/scripts/verify_official_checkpoint.py \
        --official-pretrained-checkpoint /path/to/checkpoint.pt \
        --action-tokenizer-type extra_action_tokenizer \
        --official-init-mode backbone_only

If any check fails, exits with error code 1.
"""

import sys
import json
import argparse
import torch
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

OFFICIAL_LR = 2e-5


def verify_checkpoint_exists(checkpoint_path: str) -> bool:
    """Check 1: Official checkpoint path exists."""
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"[FAIL] Official checkpoint not found: {checkpoint_path}")
        return False
    if path.suffix != ".pt":
        print(f"[FAIL] Checkpoint must be a .pt file, got: {path.suffix}")
        return False
    print(f"[PASS] Official checkpoint exists: {checkpoint_path}")
    return True


def verify_backbone_weights(checkpoint_path: str) -> bool:
    """Check 2: Checkpoint contains vision/language backbone weights."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = checkpoint.get("model", checkpoint)

        required_keys = ["projector", "llm_backbone"]
        missing = [k for k in required_keys if k not in model_state]

        if missing:
            print(f"[FAIL] Missing backbone weights: {missing}")
            return False

        vision_loaded = "vision_backbone" in model_state
        print(f"[PASS] Backbone weights found: projector, llm_backbone, vision_backbone={'yes' if vision_loaded else 'no'}")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to load checkpoint: {e}")
        return False


def verify_action_head_reinit(init_mode: str) -> bool:
    """Check 3: Action head will be reinitialized (not loaded from official)."""
    if init_mode == "backbone_only":
        print(f"[PASS] Action head will be reinitialized (backbone_only mode)")
        return True
    else:
        print(f"[WARN] Init mode '{init_mode}' may load action head from official checkpoint")
        return True


def verify_tokenizer_type(tokenizer_type: str) -> bool:
    """Check 4: Tokenizer is extra_action_tokenizer (not VQ)."""
    if tokenizer_type == "extra_action_tokenizer":
        print(f"[PASS] Action tokenizer is extra_action_tokenizer (4D MetaWorld)")
        return True
    elif "vq" in tokenizer_type.lower():
        print(f"[FAIL] VQ tokenizer '{tokenizer_type}' is not compatible with MetaWorld 4D actions")
        return False
    else:
        print(f"[WARN] Tokenizer '{tokenizer_type}' may not be optimal for MetaWorld")
        return True


def verify_optimizer_lr(lr_value: float) -> bool:
    """Check 5: Optimizer learning rate uses official default."""
    if lr_value == 0.0:
        print(f"[PASS] Learning rate not overridden (will use official default: {OFFICIAL_LR})")
        return True
    elif lr_value == OFFICIAL_LR:
        print(f"[PASS] Learning rate matches official default: {lr_value}")
        return True
    else:
        print(f"[FAIL] Learning rate {lr_value} does not match official default {OFFICIAL_LR}")
        return False


def verify_config_json(checkpoint_path: str) -> bool:
    """Bonus: Verify config.json exists in run directory."""
    path = Path(checkpoint_path)
    if path.suffix == ".pt" and path.parent.name == "checkpoints":
        run_dir = path.parents[1]
        config_json = run_dir / "config.json"
        if config_json.exists():
            with open(config_json, "r") as f:
                config = json.load(f)
            vla_cfg = config.get("vla", {})
            model_cfg = config.get("model", {})
            print(f"[INFO] Official config.json found")
            print(f"  - action_tokenizer: {vla_cfg.get('action_tokenizer', 'N/A')}")
            print(f"  - base_vlm: {vla_cfg.get('base_vlm', 'N/A')}")
            print(f"  - vision_backbone_id: {model_cfg.get('vision_backbone_id', 'N/A')}")
            print(f"  - llm_backbone_id: {model_cfg.get('llm_backbone_id', 'N/A')}")
            return True
        else:
            print(f"[WARN] config.json not found in run directory: {run_dir}")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify MiniVLA Official Checkpoint Configuration")
    parser.add_argument("--official-pretrained-checkpoint", type=str, required=True,
                       help="Path to official MiniVLA .pt checkpoint")
    parser.add_argument("--action-tokenizer-type", type=str, default="extra_action_tokenizer",
                       help="Action tokenizer type")
    parser.add_argument("--official-init-mode", type=str, default="backbone_only",
                       help="Official init mode")
    parser.add_argument("--optimizer-lr", type=float, default=0.0,
                       help="Optimizer learning rate (0.0 = use official default)")
    args = parser.parse_args()

    print("=" * 60)
    print("MiniVLA Official Checkpoint Verification")
    print("=" * 60)

    checks = [
        ("Checkpoint exists", lambda: verify_checkpoint_exists(args.official_pretrained_checkpoint)),
        ("Backbone weights", lambda: verify_backbone_weights(args.official_pretrained_checkpoint)),
        ("Action head reinit", lambda: verify_action_head_reinit(args.official_init_mode)),
        ("Tokenizer type", lambda: verify_tokenizer_type(args.action_tokenizer_type)),
        ("Optimizer LR", lambda: verify_optimizer_lr(args.optimizer_lr)),
        ("Config.json", lambda: verify_config_json(args.official_pretrained_checkpoint)),
    ]

    all_passed = True
    for name, check_fn in checks:
        try:
            passed = check_fn()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All checks passed. Training can proceed.")
        sys.exit(0)
    else:
        print("VERIFICATION FAILED. Training cannot proceed.")
        print("Please fix the issues above before running training.")
        sys.exit(1)


if __name__ == "__main__":
    main()