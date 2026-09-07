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

    action = np.random.uniform(-1, 1, (2, 8, 7)).astype(np.float32)
    print(f"Input action shape: {action.shape}")

    encoded = vq_tokenizer.encode_token_ids(action)
    print(f"Encoded token shape: {encoded.shape}")

    decoded = vq_tokenizer.decode_token_ids_to_actions(encoded)
    print(f"Decoded action shape: {decoded.shape}")

    # VQ decode returns only the first time step, so compare action[:, 0, :] with decoded
    action_first_step = action[:, 0, :]
    max_diff = np.max(np.abs(action_first_step - decoded))
    print(f"Max absolute difference (first time step): {max_diff:.6f}")

    if max_diff < 0.01:
        print("PASS: VQ encode/decode is consistent!")
    else:
        print(f"FAIL: max_diff={max_diff:.6f} exceeds threshold 0.01")


if __name__ == "__main__":
    main()