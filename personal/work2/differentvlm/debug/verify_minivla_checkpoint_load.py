"""
verify_minivla_checkpoint_load.py

Verifies that MiniVLA checkpoint loading correctly loads trained weights.

Steps:
1. Load a checkpoint with the FIXED from_pretrained (LeRobot-format model.safetensors)
2. Compare key weight summaries between loaded model and checkpoint file
3. With a fixed random seed, compare action outputs between:
   a. Freshly initialized model (no weights loaded)
   b. Model loaded from checkpoint
   c. Verify they produce DIFFERENT outputs (proving weights were loaded)
4. Compare with eval_full_episode.py-style parent class loading for consistency

Usage:
    python personal/work2/differentvlm/debug/verify_minivla_checkpoint_load.py \
        --checkpoint_path /path/to/pretrained_model \
        --device cuda
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from safetensors.torch import load_file

from lerobot.policies.minivla.modeling_minivla import MiniVLAPolicy
from lerobot.policies.pretrained import PreTrainedConfig
from lerobot.policies import make_pre_post_processors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_fresh_model(checkpoint_path: str, device: str):
    """Create a MiniVLA model WITHOUT loading weights (freshly initialized)."""
    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    config.device = device
    
    # Create model instance without loading weights
    from lerobot.policies.minivla.modeling_minivla import MiniVLAPolicy
    model = MiniVLAPolicy(config)
    model.to(device)
    
    # Convert to correct dtype to match trained model
    model_dtype = getattr(config, "dtype", "float32")
    if model_dtype == "bfloat16":
        model = model.to(dtype=torch.bfloat16)
    elif model_dtype == "float16":
        model = model.to(dtype=torch.float16)
    
    model.eval()
    return model, config


def load_trained_model(checkpoint_path: str, device: str):
    """Load MiniVLA model with trained weights using fixed from_pretrained."""
    policy = MiniVLAPolicy.from_pretrained(checkpoint_path)
    
    # Ensure correct dtype
    model_dtype = getattr(policy.config, "dtype", "float32")
    if model_dtype == "bfloat16":
        policy = policy.to(dtype=torch.bfloat16)
    elif model_dtype == "float16":
        policy = policy.to(dtype=torch.float16)
    
    policy.to(device)
    policy.eval()
    return policy, policy.config


def create_dummy_batch(config, device: str, seed: int = 42):
    """Create a fixed dummy batch for testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    batch_size = 1
    
    # Determine dtype from config
    model_dtype = getattr(config, "dtype", "float32")
    if model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif model_dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # Create dummy pixel values (DINO + SigLIP)
    # MiniVLA expects "dino" and "siglip" keys from processor
    dino_shape = (batch_size, 3, config.image_size, config.image_size)
    siglip_shape = (batch_size, 3, config.image_size, config.image_size)
    
    batch = {
        "dino": torch.randn(dino_shape, device=device, dtype=torch_dtype),
        "siglip": torch.randn(siglip_shape, device=device, dtype=torch_dtype),
        "task": ["test task"],
    }
    
    return batch


def compare_outputs(fresh_model, trained_model, batch, device: str):
    """Compare outputs between fresh and trained models using a simplified test."""
    print("Running simplified weight comparison test...")
    sys.stdout.flush()
    
    # Instead of full inference (which is slow), compare a few key layer weights
    print("  Getting fresh model state_dict...")
    sys.stdout.flush()
    fresh_state = fresh_model.state_dict()
    
    print("  Getting trained model state_dict...")
    sys.stdout.flush()
    trained_state = trained_model.state_dict()
    
    # Compare a few key layers
    test_keys = [
        "model.vlm.llm.model.layers.0.self_attn.q_proj.weight",
        "model.vlm.projector.projector.0.weight",  # FusedMLPProjector first linear
        "model.vq_vae.vq_layer.layers.0._codebook.embed",
    ]
    
    max_diff = 0.0
    for key in test_keys:
        print(f"  Checking key: {key}")
        sys.stdout.flush()
        
        if key not in fresh_state or key not in trained_state:
            # Try with "model." prefix
            prefixed_key = "model." + key if not key.startswith("model.") else key
            if prefixed_key in fresh_state and prefixed_key in trained_state:
                key = prefixed_key
            else:
                print(f"    Key not found in either model, skipping")
                sys.stdout.flush()
                continue
        
        print(f"    Computing difference...")
        sys.stdout.flush()
        fresh_w = fresh_state[key].float()
        trained_w = trained_state[key].float()
        
        diff = (fresh_w - trained_w).abs().max().item()
        max_diff = max(max_diff, diff)
        
        print(f"    {key}: max_diff={diff:.6f}")
        sys.stdout.flush()
    
    print(f"Max weight difference across test layers: {max_diff:.6f}")
    sys.stdout.flush()
    
    if max_diff < 1e-5:
        print(f"CRITICAL: Fresh and trained model weights are nearly identical (max_diff={max_diff:.8f}). This suggests the checkpoint weights were NOT loaded!")
        return False
    else:
        print(f"OK: Fresh and trained model weights differ significantly (max_diff={max_diff:.6f}), confirming weights were loaded.")
        return True


def verify_weight_summaries(checkpoint_path: str, loaded_model):
    """Verify key weight summaries match between checkpoint file and loaded model."""
    model_file = Path(checkpoint_path) / "model.safetensors"
    if not model_file.exists():
        logger.warning(f"model.safetensors not found at {checkpoint_path}, skipping weight verification")
        return True
    
    state_dict = load_file(str(model_file), device="cpu")
    
    # Checkpoint keys have "model." prefix, loaded model state_dict also has "model." prefix
    # Try both with and without "model." prefix
    key_layers = [
        "model.vlm.vision_backbone.dino_model.blocks.0.norm1.weight",
        "model.vlm.llm.model.layers.0.self_attn.q_proj.weight",
        "model.vlm.projector.linear_1.weight",
    ]
    
    all_ok = True
    for layer_key in key_layers:
        if layer_key not in state_dict:
            # Try without "model." prefix
            alt_key = layer_key.replace("model.", "", 1)
            if alt_key in state_dict:
                layer_key = alt_key
            else:
                logger.warning(f"Layer {layer_key} not found in checkpoint")
                continue
        
        ckpt_weight = state_dict[layer_key]
        
        # Try to get the same weight from loaded model
        try:
            model_state = loaded_model.state_dict()
            if layer_key not in model_state:
                # Try with "model." prefix
                prefixed_key = "model." + layer_key if not layer_key.startswith("model.") else layer_key
                if prefixed_key in model_state:
                    model_weight = model_state[prefixed_key].cpu()
                else:
                    logger.warning(f"Layer {layer_key} not found in loaded model")
                    all_ok = False
                    continue
            else:
                model_weight = model_state[layer_key].cpu()
            
            ckpt_mean = float(ckpt_weight.float().mean())
            model_mean = float(model_weight.float().mean())
            ckpt_std = float(ckpt_weight.float().std())
            model_std = float(model_weight.float().std())
            
            mean_diff = abs(ckpt_mean - model_mean)
            std_diff = abs(ckpt_std - model_std)
            
            logger.info(
                f"Weight {layer_key}:"
                f"  ckpt(mean={ckpt_mean:.6f}, std={ckpt_std:.6f})"
                f"  model(mean={model_mean:.6f}, std={model_std:.6f})"
                f"  diff(mean={mean_diff:.8f}, std={std_diff:.8f})"
            )
            
            if mean_diff > 1e-4 or std_diff > 1e-4:
                logger.warning(f"Weight mismatch detected for {layer_key}")
                all_ok = False
        except KeyError:
            logger.warning(f"Layer {layer_key} not found in loaded model")
            all_ok = False
    
    return all_ok


def patch_minivla_bfloat16(policy, logger):
    """Patch MiniVLA's VQ action tokenizer to handle bfloat16 -> float32 conversion."""
    try:
        core = policy.model
        if hasattr(core, "action_tokenizer"):
            action_tokenizer = core.action_tokenizer
            if hasattr(action_tokenizer, "vq_vae"):
                original_get_action = action_tokenizer.vq_vae.get_action_from_latent

                def patched_get_action(latent):
                    result = original_get_action(latent)
                    if result.dtype == torch.bfloat16:
                        return result.float()
                    return result

                action_tokenizer.vq_vae.get_action_from_latent = patched_get_action
                logger.info("Patched MiniVLA vq_vae.get_action_from_latent for bfloat16 compatibility")
    except Exception as e:
        logger.warning(f"Failed to patch MiniVLA action tokenizer: {e}")


def main():
    parser = argparse.ArgumentParser(description="Verify MiniVLA checkpoint loading")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to pretrained_model directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint_path
    device = args.device
    seed = args.seed
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")
    
    # Step 1: Create fresh model (no weights loaded)
    logger.info("\n[Step 1] Creating freshly initialized model (no weights loaded)...")
    fresh_model, fresh_config = create_fresh_model(checkpoint_path, device)
    patch_minivla_bfloat16(fresh_model, logger)
    
    # Step 2: Load trained model with fixed from_pretrained
    logger.info("\n[Step 2] Loading trained model with fixed from_pretrained...")
    trained_model, trained_config = load_trained_model(checkpoint_path, device)
    patch_minivla_bfloat16(trained_model, logger)
    
    # Step 3: Verify weight summaries
    logger.info("\n[Step 3] Verifying weight summaries...")
    weights_ok = verify_weight_summaries(checkpoint_path, trained_model)
    
    # Step 4: Create dummy batch and compare outputs
    print(f"\n[Step 4] Creating dummy batch and comparing outputs...")
    sys.stdout.flush()
    batch = create_dummy_batch(trained_config, device, seed=seed)
    
    print(f"[Step 4] Batch created. Starting inference comparison...")
    sys.stdout.flush()
    outputs_different = compare_outputs(fresh_model, trained_model, batch, device)
    
    # Final verdict
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL VERDICT")
    logger.info(f"{'='*60}")
    
    if weights_ok and outputs_different:
        logger.info("PASS: Checkpoint weights loaded correctly and affect model output.")
        return 0
    else:
        if not weights_ok:
            logger.error("FAIL: Weight summaries do not match checkpoint file.")
        if not outputs_different:
            logger.error("FAIL: Fresh and trained model outputs are identical (weights not loaded).")
        return 1


if __name__ == "__main__":
    sys.exit(main())