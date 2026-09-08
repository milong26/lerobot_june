#!/usr/bin/env python3
"""
Simple debug script to check TinyVLA evaluation.
Focuses on:
1. Task instruction passing
2. Action output statistics
3. Environment success conditions
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add lerobot to path
sys.path.insert(0, "/data/zhonglinye/jun/lerobot/src")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


def main():
    checkpoint_path = "/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/experiments/tinyvla_s_disassemble-v3_corner/checkpoints/tinyvla_tinyvla_s_disassemble-v3_corner/checkpoints/012000/pretrained_model"
    output_dir = Path("/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/debug_eval_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("TinyVLA Debug Script")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    
    # Load checkpoint config
    config_file = Path(checkpoint_path) / "config.json"
    with open(config_file) as f:
        ckpt_config = json.load(f)
    
    print(f"\nCheckpoint config:")
    print(f"  Policy type: {ckpt_config.get('type')}")
    print(f"  Device: {ckpt_config.get('device')}")
    print(f"  Input features: {list(ckpt_config.get('input_features', {}).keys())}")
    print(f"  Output features: {list(ckpt_config.get('output_features', {}).keys())}")
    print(f"  LoRA enabled: {ckpt_config.get('lora_enable')}")
    print(f"  Action dim: {ckpt_config.get('action_dim')}")
    print(f"  State dim: {ckpt_config.get('state_dim')}")
    
    # Check adapter config
    adapter_config_file = Path(checkpoint_path) / "adapter" / "adapter_config.json"
    if adapter_config_file.exists():
        with open(adapter_config_file) as f:
            adapter_config = json.load(f)
        print(f"\nAdapter config:")
        print(f"  LoRA r: {adapter_config.get('r')}")
        print(f"  LoRA alpha: {adapter_config.get('lora_alpha')}")
        print(f"  Target modules: {adapter_config.get('target_modules')}")
    
    # Check model files
    model_file = Path(checkpoint_path) / "model.safetensors"
    if model_file.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(model_file))
        print(f"\nModel state dict:")
        print(f"  Total keys: {len(state_dict)}")
        
        # Check for LoRA keys
        lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
        print(f"  LoRA keys: {len(lora_keys)}")
        
        # Check action head keys
        action_head_keys = [k for k in state_dict.keys() if 'action' in k.lower() or 'diffusion' in k.lower()]
        print(f"  Action head keys: {len(action_head_keys)}")
        
        # Show some key names
        print(f"\n  Sample keys (first 20):")
        for k in sorted(state_dict.keys())[:20]:
            print(f"    {k}: {state_dict[k].shape}")
    
    # Check policy preprocessor config
    preprocessor_file = Path(checkpoint_path) / "policy_preprocessor.json"
    if preprocessor_file.exists():
        with open(preprocessor_file) as f:
            preprocessor_config = json.load(f)
        print(f"\nPolicy preprocessor config:")
        print(f"  Keys: {list(preprocessor_config.keys())}")
    
    print("\n" + "="*60)
    print("Analysis:")
    print("="*60)
    
    # Check for potential issues
    issues = []
    
    # 1. Check if input features match environment output
    input_features = ckpt_config.get('input_features', {})
    expected_env_features = ['observation.images.top', 'observation.images.wrist', 'observation.state']
    for feat in expected_env_features:
        if feat not in input_features:
            issues.append(f"Missing input feature: {feat}")
    
    # 2. Check action dim
    action_dim = ckpt_config.get('action_dim')
    if action_dim != 4:
        issues.append(f"Action dim mismatch: expected 4 for MetaWorld, got {action_dim}")
    
    # 3. Check state dim
    state_dim = ckpt_config.get('state_dim')
    if state_dim != 4:
        issues.append(f"State dim mismatch: expected 4 for MetaWorld, got {state_dim}")
    
    # 4. Check LoRA keys
    if len(lora_keys) == 0:
        issues.append("No LoRA keys found in model state dict")
    
    if issues:
        print("\nPotential issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\nNo obvious issues found in checkpoint configuration.")
    
    # Save analysis
    analysis = {
        "checkpoint_path": checkpoint_path,
        "config": ckpt_config,
        "adapter_config": adapter_config if adapter_config_file.exists() else None,
        "model_stats": {
            "total_keys": len(state_dict),
            "lora_keys": len(lora_keys),
            "action_head_keys": len(action_head_keys),
        },
        "issues": issues,
    }
    
    output_file = output_dir / "checkpoint_analysis.json"
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\nAnalysis saved to: {output_file}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()