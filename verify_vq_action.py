"""
verify_vq_action.py

Verify VQ-VAE action tokenizer encode/decode consistency
using official MiniVLA VQ-VAE checkpoint from HuggingFace.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import torch
from transformers import AutoTokenizer

from lerobot.policies.minivla.configuration_minivla import MiniVLAConfig
from lerobot.policies.minivla.modeling_minivla import MiniVLAPolicy


def main():
    print("Loading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    vq_path = (
        "/data/zhonglinye/hfdata/hub/models--Stanford-ILIAD--pretrain_vq"
        "/snapshots/30ef2227f97dde1abb6d522ea80af85384235008"
        "/pretrain_vq+mx-libero_90+fach-7+ng-7+nemb-128+nlatent-512"
    )

    print("Building MiniVLA config...")
    config = MiniVLAConfig(
        action_tokenizer_type="libero_vq_action_tokenizer",
        vq_model_path=vq_path,
    )

    print("Loading MiniVLA policy (this loads VQ-VAE)...")
    policy = MiniVLAPolicy(config)

    vq_tokenizer = policy.model.action_tokenizer

    print("Testing encode/decode roundtrip...")
    np.random.seed(42)
    torch.manual_seed(42)

    # Test 1: Verify state_vq (quantized latent) matches latent reconstructed from codes
    print("\n=== Test 1: VQ latent consistency ===")
    action = np.random.uniform(-1, 1, (2, 8, 7)).astype(np.float32)
    print(f"Input action shape: {action.shape}")

    action_t = torch.from_numpy(action).to(vq_tokenizer.vq_vae.device)
    state_vq, vq_code = vq_tokenizer.vq_vae.get_code(action_t)
    print(f"state_vq shape: {state_vq.shape}")
    print(f"vq_code shape: {vq_code.shape}")

    latent = vq_tokenizer.vq_vae.draw_code_forward(vq_code)
    print(f"latent shape: {latent.shape}")

    latent_diff = torch.max(torch.abs(state_vq - latent)).item()
    print(f"state_vq vs latent max diff: {latent_diff:.8f}")

    # Also verify decode shape
    ret_action = vq_tokenizer.vq_vae.get_action_from_latent(latent)
    print(f"ret_action shape: {ret_action.shape}")

    shape_ok = ret_action.shape == (2, 8, 7)
    latent_ok = latent_diff < 1e-5

    if latent_ok and shape_ok:
        print("PASS: VQ latent consistency verified!")
    else:
        print("FAIL: VQ latent mismatch")

    # Test 2: encode_token_ids / decode_token_ids_to_actions roundtrip
    print("\n=== Test 2: Token ID encode/decode ===")
    token_ids = vq_tokenizer.encode_token_ids(action)
    print(f"Encoded token shape: {token_ids.shape}")

    decoded = vq_tokenizer.decode_token_ids_to_actions(token_ids)
    print(f"Decoded action shape: {decoded.shape}")

    if decoded.shape == (2, 8, 7):
        print("PASS: Decoded shape matches expected (2, 8, 7)!")
    else:
        print(f"FAIL: Expected shape (2, 8, 7), got {decoded.shape}")


if __name__ == "__main__":
    main()