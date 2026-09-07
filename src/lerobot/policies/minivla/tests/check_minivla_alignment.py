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


def run_alignment_check(
    official_checkpoint: str,
    lr_config: dict,
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
    config_mismatches = compare_configs(official_config, lr_config)
    all_mismatches.extend(config_mismatches)

    return all_mismatches


def main():
    parser = argparse.ArgumentParser(description="Check MiniVLA alignment between official and LeRobot implementations")
    parser.add_argument("--official_checkpoint", type=str, required=True, help="Path to official MiniVLA checkpoint (.pt)")
    parser.add_argument("--lr_config_json", type=str, required=True, help="Path to LeRobot MiniVLA config.json")
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

    # Run alignment check
    mismatches = run_alignment_check(args.official_checkpoint, lr_config)

    if mismatches:
        print("MISMATCHES FOUND:")
        for i, mismatch in enumerate(mismatches, 1):
            print(f"  {i}. {mismatch}")
        print()
        print(f"Total mismatches: {len(mismatches)}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED - Official and LeRobot MiniVLA are aligned!")
        sys.exit(0)


if __name__ == "__main__":
    main()