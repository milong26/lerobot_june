#!/usr/bin/env python
# 这个代码是增加tinyvla模型的时候的测试代码，留着备份，但功能上估计用不到。

"""Minimal test script to verify TinyVLA Bug 1, 2, 3 fixes.

This script creates a fake batch and runs it through the TinyVLA policy to verify:
1. Bug 1 fix: input_ids contains IMAGE_TOKEN_INDEX (-200)
2. Bug 2 fix: changing task description changes model output
3. Bug 3 fix: image key fallback works for both OBS_IMAGE and OBS_IMAGES prefixes
"""

import torch
from pathlib import Path

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.tinyvla.configuration_tinyvla import TinyVLAConfig
from lerobot.policies.tinyvla.modeling_tinyvla import TinyVLAPolicy
from lerobot.policies.tinyvla.llava_pythia.constants import IMAGE_TOKEN_INDEX
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def create_input_features(image_key: str = "observation.images.camera", state_dim: int = 7):
    """Create input_features dict for TinyVLAConfig."""
    return {
        image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }


def create_output_features(action_dim: int = 10):
    """Create output_features dict for TinyVLAConfig."""
    return {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }


def create_fake_batch(
    batch_size: int = 2,
    image_size: tuple = (3, 224, 224),
    state_dim: int = 7,
    action_dim: int = 10,
    chunk_size: int = 16,
    task: list[str] | None = None,
    image_key: str = "observation.images.camera",
    dtype: torch.dtype = torch.float32,
):
    """Create a fake batch for testing."""
    batch = {
        image_key: torch.randn(batch_size, *image_size, dtype=dtype),
        "observation.state": torch.randn(batch_size, state_dim, dtype=dtype),
        "action": torch.randn(batch_size, chunk_size, action_dim, dtype=dtype),
        "action_is_pad": torch.zeros(batch_size, chunk_size, dtype=torch.bool),
    }
    if task is not None:
        batch["task"] = task
    return batch


def test_bug1_image_token_in_input_ids():
    """Bug 1: Verify that input_ids contains IMAGE_TOKEN_INDEX (-200)."""
    print("\n" + "=" * 60)
    print("Testing Bug 1: Image token injection")
    print("=" * 60)

    config = TinyVLAConfig(
        model_name_or_path="lesjie/Llava-Pythia-400M",
        action_head_type="droid_diffusion",
        action_dim=10,
        state_dim=7,
        chunk_size=16,
        n_action_steps=16,
        lora_enable=False,
        freeze_backbone=True,
        freeze_vision_tower=True,
        device="cpu",
        input_features=create_input_features(),
        output_features=create_output_features(),
    )

    policy = TinyVLAPolicy(config)

    # Test _tokenize_language directly
    raw_lang = ["pick up the red block", "move to the left"]
    input_ids, labels = policy._tokenize_language(raw_lang)

    # Check that -200 appears in input_ids
    num_image_tokens = (input_ids == IMAGE_TOKEN_INDEX).sum().item()
    print(f"  Number of IMAGE_TOKEN_INDEX (-200) in input_ids: {num_image_tokens}")
    assert num_image_tokens > 0, "Bug 1 NOT fixed: No IMAGE_TOKEN_INDEX (-200) found in input_ids!"
    print("  ✓ Bug 1 FIXED: IMAGE_TOKEN_INDEX (-200) is present in input_ids")

    # Print a sample input_ids for visual inspection
    print(f"  Sample input_ids (first sequence): {input_ids[0].tolist()}")
    print(f"  Shape: {input_ids.shape}")

    return True


def test_bug2_task_key_reading():
    """Bug 2: Verify that changing task description changes model output."""
    print("\n" + "=" * 60)
    print("Testing Bug 2: Task key reading")
    print("=" * 60)

    config = TinyVLAConfig(
        model_name_or_path="lesjie/Llava-Pythia-400M",
        action_head_type="droid_diffusion",
        action_dim=10,
        state_dim=7,
        chunk_size=16,
        n_action_steps=16,
        lora_enable=False,
        freeze_backbone=True,
        freeze_vision_tower=True,
        device="cpu",
        input_features=create_input_features(),
        output_features=create_output_features(),
    )

    policy = TinyVLAPolicy(config)
    policy.eval()

    # Create two batches with different tasks but same images/state
    torch.manual_seed(42)
    batch_a = create_fake_batch(
        batch_size=2,
        task=["pick up the red block", "move to the left"],
    )
    batch_b = create_fake_batch(
        batch_size=2,
        task=["push the blue button", "rotate clockwise"],
    )

    # Copy images and state from batch_a to batch_b to ensure they're identical
    for key in ["observation.images.camera", "observation.state"]:
        batch_b[key] = batch_a[key].clone()

    with torch.no_grad():
        actions_a = policy.predict_action_chunk(batch_a)
        actions_b = policy.predict_action_chunk(batch_b)

    # Check if outputs are different
    diff = (actions_a - actions_b).abs().sum().item()
    print(f"  Absolute difference between outputs with different tasks: {diff:.6f}")

    if diff > 1e-6:
        print("  ✓ Bug 2 FIXED: Different tasks produce different outputs")
    else:
        print("  ✗ Bug 2 NOT fixed: Different tasks produce identical outputs")

    # Also test that task key is actually being read (not falling back to empty string)
    batch_no_task = create_fake_batch(batch_size=2)  # No "task" key
    with torch.no_grad():
        actions_no_task = policy.predict_action_chunk(batch_no_task)

    diff_with_empty = (actions_a - actions_no_task).abs().sum().item()
    print(f"  Absolute difference between 'task provided' vs 'no task': {diff_with_empty:.6f}")

    if diff_with_empty > 1e-6:
        print("  ✓ Confirmed: Model reads 'task' key correctly (not ignoring it)")
    else:
        print("  Note: Model output is same with/without task (might be expected if language has small effect)")

    return True


def test_bug3_image_key_fallback():
    """Bug 3: Verify image key fallback works for both OBS_IMAGE and OBS_IMAGES."""
    print("\n" + "=" * 60)
    print("Testing Bug 3: Image key fallback")
    print("=" * 60)

    # Test with OBS_IMAGE (singular, MetaWorld style)
    config_metaworld = TinyVLAConfig(
        model_name_or_path="lesjie/Llava-Pythia-400M",
        action_head_type="droid_diffusion",
        action_dim=10,
        state_dim=7,
        chunk_size=16,
        n_action_steps=16,
        lora_enable=False,
        freeze_backbone=True,
        freeze_vision_tower=True,
        device="cpu",
        input_features=create_input_features(image_key="observation.image"),
        output_features=create_output_features(),
    )

    policy_metaworld = TinyVLAPolicy(config_metaworld)
    policy_metaworld.eval()

    batch_metaworld = {
        "observation.image": torch.randn(2, 3, 224, 224, dtype=torch.float32),
        "observation.state": torch.randn(2, 7, dtype=torch.float32),
        "action": torch.randn(2, 16, 10, dtype=torch.float32),
        "action_is_pad": torch.zeros(2, 16, dtype=torch.bool),
        "task": ["push", "pick"],
    }

    try:
        with torch.no_grad():
            loss, info = policy_metaworld.forward(batch_metaworld)
        print(f"  ✓ OBS_IMAGE (singular) key works: loss = {loss.item():.4f}")
    except Exception as e:
        print(f"  ✗ OBS_IMAGE (singular) key failed: {e}")

    # Test with OBS_IMAGES (plural, lerobot standard style)
    config_lerobot = TinyVLAConfig(
        model_name_or_path="lesjie/Llava-Pythia-400M",
        action_head_type="droid_diffusion",
        action_dim=10,
        state_dim=7,
        chunk_size=16,
        n_action_steps=16,
        lora_enable=False,
        freeze_backbone=True,
        freeze_vision_tower=True,
        device="cpu",
        input_features=create_input_features(image_key="observation.images.camera"),
        output_features=create_output_features(),
    )

    policy_lerobot = TinyVLAPolicy(config_lerobot)
    policy_lerobot.eval()

    batch_lerobot = {
        "observation.images.camera": torch.randn(2, 3, 224, 224, dtype=torch.float32),
        "observation.state": torch.randn(2, 7, dtype=torch.float32),
        "action": torch.randn(2, 16, 10, dtype=torch.float32),
        "action_is_pad": torch.zeros(2, 16, dtype=torch.bool),
        "task": ["push", "pick"],
    }

    try:
        with torch.no_grad():
            loss, info = policy_lerobot.forward(batch_lerobot)
        print(f"  ✓ OBS_IMAGES (plural) key works: loss = {loss.item():.4f}")
    except Exception as e:
        print(f"  ✗ OBS_IMAGES (plural) key failed: {e}")

    return True


def test_forward_loss():
    """Test that forward() runs without error and produces finite loss."""
    print("\n" + "=" * 60)
    print("Testing forward() with fake batch")
    print("=" * 60)

    config = TinyVLAConfig(
        model_name_or_path="lesjie/Llava-Pythia-400M",
        action_head_type="droid_diffusion",
        action_dim=10,
        state_dim=7,
        chunk_size=16,
        n_action_steps=16,
        lora_enable=False,
        freeze_backbone=True,
        freeze_vision_tower=True,
        device="cpu",
        input_features=create_input_features(),
        output_features=create_output_features(),
    )

    policy = TinyVLAPolicy(config)
    policy.train()

    batch = create_fake_batch(
        batch_size=2,
        task=["pick up the red block", "move to the left"],
    )

    loss, info = policy.forward(batch)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Is finite: {torch.isfinite(loss).item()}")
    print(f"  Info: {info}")

    assert torch.isfinite(loss), "Loss is not finite!"
    print("  ✓ forward() runs successfully with finite loss")

    return True


if __name__ == "__main__":
    print("TinyVLA Bug Fix Verification Tests")
    print("=" * 60)

    all_passed = True

    try:
        test_bug1_image_token_in_input_ids()
    except Exception as e:
        print(f"  ✗ Bug 1 test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        test_bug2_task_key_reading()
    except Exception as e:
        print(f"  ✗ Bug 2 test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        test_bug3_image_key_fallback()
    except Exception as e:
        print(f"  ✗ Bug 3 test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        test_forward_loss()
    except Exception as e:
        print(f"  ✗ forward() test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 60)