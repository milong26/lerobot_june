"""
本工具用于分析 policy-level attention / input utilization，不用于证明 embedding separability 或 SIC coverage。

目标：对已训练好的 SmolVLA checkpoint 做 attention 分析，研究 Random / Uniform / Ours 模型在
MetaWorld pick-place-v3 中到底关注了哪些输入，尤其关注为什么当前模型 grasp success 很高，
但最终 task success 明显更低的问题。

回答的研究问题：
1. 模型在 initial 阶段主要看 global camera 还是 wrist camera？
2. pre-grasp 阶段 wrist attention 是否上升？
3. grasp 后模型是否增加对 global scene / goal-related visual tokens 的关注？
4. Random / Uniform / Ours 在 post-grasp / pre-place 阶段的 attention pattern 是否不同？
5. Ours 是否因为训练数据选择导致模型过度关注某一路视觉输入？
6. attention pattern 是否和最终 success / failure 有关系？
"""

import os
import sys
import argparse
import json
import csv
import types
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_DEVICE_ID"] = "0"

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.envs.metaworld import MetaworldEnv
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE

# ============================================================
# Configuration constants
# ============================================================

TASK = "pick-place-v3"
DEFAULT_SEED = 10042
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "personal/work2/attention_fig/result"
DEFAULT_INFERENCE_TRACE_OUTPUT_DIR = PROJECT_ROOT / "personal/work2/attention_fig/result_inference_trace"

# Phase detection thresholds
GRIPPER_OBJECT_DIST_PRE_GRASP = 0.15
OBJECT_HEIGHT_POST_GRASP = 0.02
OBJECT_GOAL_DIST_PRE_PLACE = 0.12
MODEL_CONFIGS = [
    {
        "name": "random_corner_16k",
        "method": "random",
        "camera_name": "corner,gripperPOV",
        "path": "personal/work2/duibi/random_42_corner/random_112_seed42/checkpoints/016000/pretrained_model",
        "eval_task_success": 31.0,
    },
    {
        "name": "uniform_corner_16k",
        "method": "uniform",
        "camera_name": "corner,gripperPOV",
        "path": "personal/work2/duibi/uniform_42_corner/uniform_112_seed42/checkpoints/016000/pretrained_model",
        "eval_task_success": 28.0,
    },
    {
        "name": "ours_corner_16k",
        "method": "ours",
        "camera_name": "corner,gripperPOV",
        "path": "personal/work2/duibi/ours_112_seed42_corner/dynamicanchor_112_seed42/checkpoints/016000/pretrained_model",
        "eval_task_success": 24.5,
    },

    {
        "name": "random_corner2_16k",
        "method": "random",
        "camera_name": "corner2,gripperPOV",
        "path": "personal/work2/duibi/random_42_corner2/random_112_seed42/checkpoints/016000/pretrained_model",
        "eval_task_success": 42.5,
    },
    {
        "name": "uniform_corner2_16k",
        "method": "uniform",
        "camera_name": "corner2,gripperPOV",
        "path": "personal/work2/duibi/uniform_42_corner2/uniform_112_seed42/checkpoints/016000/pretrained_model",
        "eval_task_success": 26.0,
    },

    {
        "name": "random_corner3_16k",
        "method": "random",
        "camera_name": "corner3,gripperPOV",
        "path": "personal/work2/duibi/random_42_corner3/random_112_seed42/checkpoints/016000/pretrained_model",
        "eval_task_success": 34.5,
    },
    {
        "name": "uniform_corner3_16k",
        "method": "uniform",
        "camera_name": "corner3,gripperPOV",
        "path": "personal/work2/duibi/uniform_42_corner3/uniform_112_seed42/checkpoints/016000/pretrained_model",
        "eval_task_success": 37.0,
    },
    {
        "name": "ours_corner3_16k",
        "method": "ours",
        "camera_name": "corner3,gripperPOV",
        "path": "personal/work2/duibi/ours_112_seed42_corner3/dynamicanchor_112_seed42/checkpoints/016000/pretrained_model",
        "eval_task_success": 35.0,
    },
]

PHASES = ["initial", "pre_grasp", "post_grasp", "pre_place"]

DEFAULT_LAYERS = [0, 3, 7, 11]


# ============================================================
# Attention capture hook
# ============================================================

class AttentionExtractor:
    """Extract attention weights by wrapping eager_attention_forward with full metadata."""

    def __init__(self, vlm_with_expert):
        self.vlm_with_expert = vlm_with_expert
        self.original_method = vlm_with_expert.eager_attention_forward
        self.captured_attentions = []
        self.call_index = 0
        self._wrap_method()

    def _wrap_method(self):
        extractor = self

        def wrapped_attention_forward(attention_mask, batch_size, head_dim, query_states, key_states, value_states):
            if attention_mask.dtype != torch.bool:
                attention_mask_bool = attention_mask.bool()
            else:
                attention_mask_bool = attention_mask

            att_output = extractor.original_method(
                attention_mask_bool, batch_size, head_dim, query_states, key_states, value_states
            )

            num_att_heads = extractor.vlm_with_expert.num_attention_heads
            num_key_value_heads = extractor.vlm_with_expert.num_key_value_heads
            num_key_value_groups = num_att_heads // num_key_value_heads
            sequence_length = key_states.shape[1]

            key_states_exp = key_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            )
            key_states_exp = key_states_exp.reshape(
                batch_size, sequence_length, num_att_heads, head_dim
            )

            query_states_fp32 = query_states.to(dtype=torch.float32).transpose(1, 2)
            key_states_fp32 = key_states_exp.to(dtype=torch.float32).transpose(1, 2)

            att_weights = torch.matmul(query_states_fp32, key_states_fp32.transpose(2, 3))
            att_weights *= head_dim ** -0.5

            big_neg = torch.finfo(att_weights.dtype).min
            masked_att_weights = torch.where(attention_mask_bool[:, None, :, :], att_weights, big_neg)
            probs = torch.nn.functional.softmax(masked_att_weights, dim=-1)

            extractor.captured_attentions.append({
                "call_index": extractor.call_index,
                "layer_index": None,
                "source": "unknown",
                "probs": probs.detach().cpu(),
                "query_length": query_states.shape[1],
                "key_length": key_states.shape[1],
                "num_heads": num_att_heads,
            })
            extractor.call_index += 1

            return att_output

        extractor.vlm_with_expert.eager_attention_forward = types.MethodType(
            lambda self, *args, **kwargs: wrapped_attention_forward(*args, **kwargs),
            extractor.vlm_with_expert
        )

    def reset(self):
        self.captured_attentions = []
        self.call_index = 0

    def restore(self):
        self.vlm_with_expert.eager_attention_forward = self.original_method


def annotate_attention_calls(vlm_with_expert, captured, prefix_length, suffix_length):
    """Map captured attention calls to real model layers and attention sources.

    Replicates the logic from SmolVLMWithExpert.forward() to determine which
    layer uses forward_attn_layer (joint_self) vs forward_cross_attn_layer
    (prefix_self + expert_cross).

    Returns the same captured list with layer_index and source annotated.
    """
    num_layers = vlm_with_expert.num_vlm_layers
    attention_mode = vlm_with_expert.attention_mode
    self_attn_every_n = vlm_with_expert.self_attn_every_n_layers

    models = [vlm_with_expert.get_vlm_model().text_model, vlm_with_expert.lm_expert]
    model_layers = vlm_with_expert.get_model_layers(models)

    total_seq = prefix_length + suffix_length
    fill_kv_cache = False  # use_cache=False in our probe call

    cursor = 0

    for layer_idx in range(num_layers):
        use_joint = (
            fill_kv_cache
            or "cross" not in attention_mode
            or (self_attn_every_n > 0 and layer_idx % self_attn_every_n == 0)
        )

        if use_joint:
            # forward_attn_layer: joint self-attention over prefix+suffix
            if cursor >= len(captured):
                raise RuntimeError(
                    f"Expected joint_self call at layer {layer_idx} but only {len(captured)} "
                    f"calls captured (cursor={cursor}). num_layers={num_layers}, "
                    f"attention_mode={attention_mode}, self_attn_every_n={self_attn_every_n}"
                )
            call = captured[cursor]
            call["layer_index"] = layer_idx
            call["source"] = "joint_self"

            if call["query_length"] != total_seq or call["key_length"] != total_seq:
                raise RuntimeError(
                    f"joint_self shape mismatch at layer {layer_idx}: "
                    f"q={call['query_length']}, k={call['key_length']}, "
                    f"expected q=k={total_seq}"
                )
            cursor += 1
        else:
            # forward_cross_attn_layer: prefix_self + (optionally) expert_cross
            if cursor >= len(captured):
                raise RuntimeError(
                    f"Expected prefix_self call at layer {layer_idx} but only {len(captured)} "
                    f"calls captured (cursor={cursor})."
                )
            call = captured[cursor]
            call["layer_index"] = layer_idx
            call["source"] = "prefix_self"

            if call["query_length"] != prefix_length or call["key_length"] != prefix_length:
                raise RuntimeError(
                    f"prefix_self shape mismatch at layer {layer_idx}: "
                    f"q={call['query_length']}, k={call['key_length']}, "
                    f"expected q=k={prefix_length}"
                )
            cursor += 1

            # Check if expert layer exists for this model layer
            has_expert = model_layers[1][layer_idx] is not None
            if has_expert:
                if cursor >= len(captured):
                    raise RuntimeError(
                        f"Expected expert_cross call at layer {layer_idx} but only {len(captured)} "
                        f"calls captured (cursor={cursor})."
                    )
                call = captured[cursor]
                call["layer_index"] = layer_idx
                call["source"] = "expert_cross"

                if call["query_length"] != suffix_length or call["key_length"] != prefix_length:
                    raise RuntimeError(
                        f"expert_cross shape mismatch at layer {layer_idx}: "
                        f"q={call['query_length']}, k={call['key_length']}, "
                        f"expected q={suffix_length}, k={prefix_length}"
                    )
                cursor += 1

    if cursor != len(captured):
        raise RuntimeError(
            f"Call count mismatch: consumed {cursor} calls but captured {len(captured)}. "
            f"num_layers={num_layers}, attention_mode={attention_mode}, "
            f"self_attn_every_n={self_attn_every_n}, "
            f"captured_count={len(captured)}, consumed_count={cursor}"
        )

    return captured


def annotate_denoise_attention_calls(vlm_with_expert, captured, prefix_length, suffix_length):
    """Map captured attention calls to real model layers and sources during denoising.

    During denoising, the topology differs from probe:
    - inputs_embeds=[None, suffix_embs]: only suffix tokens are passed
    - KV cache is already filled with prefix from prefill
    - forward_attn_layer reads prefix+suffix KV cache (joint_self over prefix+suffix)
    - forward_cross_attn_layer: expert cross-attends to prefix (expert_cross)

    Args:
        vlm_with_expert: The SmolVLMWithExpertModel instance.
        captured: List of captured attention dicts from AttentionExtractor.
        prefix_length: Number of prefix (observation) tokens.
        suffix_length: Number of suffix (action) tokens.

    Returns:
        The same captured list with layer_index and source annotated.

    Raises:
        RuntimeError: If captured attention shapes do not match expected denoising topology.
    """
    num_layers = vlm_with_expert.num_vlm_layers
    attention_mode = vlm_with_expert.attention_mode
    self_attn_every_n = vlm_with_expert.self_attn_every_n_layers

    models = [vlm_with_expert.get_vlm_model().text_model, vlm_with_expert.lm_expert]
    model_layers = vlm_with_expert.get_model_layers(models)

    cursor = 0

    for layer_idx in range(num_layers):
        use_joint = (
            "cross" not in attention_mode
            or (self_attn_every_n > 0 and layer_idx % self_attn_every_n == 0)
        )

        if use_joint:
            if cursor >= len(captured):
                raise RuntimeError(
                    f"Expected denoise_joint_self call at layer {layer_idx} but only {len(captured)} "
                    f"calls captured (cursor={cursor}). num_layers={num_layers}, "
                    f"attention_mode={attention_mode}, self_attn_every_n={self_attn_every_n}"
                )
            call = captured[cursor]
            call["layer_index"] = layer_idx
            call["source"] = "denoise_joint_self"

            expected_key_len = prefix_length + suffix_length
            if call["key_length"] != expected_key_len:
                raise RuntimeError(
                    f"denoise_joint_self shape mismatch at layer {layer_idx}: "
                    f"q={call['query_length']}, k={call['key_length']}, "
                    f"expected k={expected_key_len}"
                )
            if call["query_length"] != suffix_length:
                raise RuntimeError(
                    f"denoise_joint_self query_length mismatch at layer {layer_idx}: "
                    f"q={call['query_length']}, expected q={suffix_length}"
                )
            cursor += 1
        else:
            has_expert = model_layers[1][layer_idx] is not None
            if has_expert:
                if cursor >= len(captured):
                    raise RuntimeError(
                        f"Expected expert_cross call at layer {layer_idx} but only {len(captured)} "
                        f"calls captured (cursor={cursor})."
                    )
                call = captured[cursor]
                call["layer_index"] = layer_idx
                call["source"] = "expert_cross"

                if call["query_length"] != suffix_length or call["key_length"] != prefix_length:
                    raise RuntimeError(
                        f"expert_cross shape mismatch at layer {layer_idx}: "
                        f"q={call['query_length']}, k={call['key_length']}, "
                        f"expected q={suffix_length}, k={prefix_length}"
                    )
                cursor += 1

    if cursor != len(captured):
        raise RuntimeError(
            f"Denoising call count mismatch: consumed {cursor} calls but captured {len(captured)}. "
            f"num_layers={num_layers}, attention_mode={attention_mode}, "
            f"self_attn_every_n={self_attn_every_n}, "
            f"captured_count={len(captured)}, consumed_count={cursor}"
        )

    return captured


# ============================================================
# Token span tracking
# ============================================================

def compute_token_spans(policy, images, img_masks, lang_tokens, lang_masks, state):
    """Compute exact token spans for each modality in the prefix sequence.

    Returns dict with keys: camera1, camera2, language, state, each with (start, end).
    Also returns total prefix length and image token counts per camera.
    """
    vlm = policy.model.vlm_with_expert
    bsize = state.shape[0]
    device = state.device

    spans = {}
    current_pos = 0
    image_token_counts = {}

    for img_idx, img_key in enumerate(policy.config.image_features):
        if img_idx >= len(images):
            break

        # When add_image_special_tokens=True, embed_prefix adds:
        #   [image_start_token] [image_tokens...] [image_end_token]
        # The image span should cover only the image tokens, not the special tokens.
        start_special = 1 if policy.model.add_image_special_tokens else 0
        end_special = 1 if policy.model.add_image_special_tokens else 0

        img_emb = policy.model.vlm_with_expert.embed_image(images[img_idx])
        num_img_tokens = img_emb.shape[1]
        image_token_counts[img_key] = num_img_tokens

        span_start = current_pos + start_special
        span_end = span_start + num_img_tokens

        spans[img_key] = (span_start, span_end)
        current_pos = span_start + num_img_tokens + end_special

    lang_emb = vlm.embed_language_tokens(lang_tokens)
    num_lang_tokens = lang_emb.shape[1]
    spans["language"] = (current_pos, current_pos + num_lang_tokens)
    current_pos += num_lang_tokens

    state_emb = policy.model.state_proj(state)
    if state_emb.ndim == 2:
        state_emb = state_emb[:, None, :]
    num_state_tokens = state_emb.shape[1]
    spans["state"] = (current_pos, current_pos + num_state_tokens)
    current_pos += num_state_tokens

    prefix_length = current_pos
    suffix_length = policy.config.chunk_size

    return spans, prefix_length, suffix_length, image_token_counts


def get_image_grid_from_model(vlm_with_expert, prepared_image, n_tokens):
    """Determine the image token grid layout from the VLM's vision encoder and connector.

    Uses the REAL prepared image (from policy.prepare_images) and the REAL n_tokens
    (from embed_image output) to derive the grid. Does NOT create a dummy image.

    The grid is derived from:
      1. prepared image H/W -> vision encoder patch grid (via patch_size)
      2. patch grid -> connector output grid (via connector scale_factor)
      3. verify: grid_h * grid_w == n_tokens
    """
    vlm_model = vlm_with_expert.get_vlm_model()
    h, w = prepared_image.shape[-2:]

    vision_cfg = vlm_model.vision_model.config
    patch_size = vision_cfg.patch_size
    if isinstance(patch_size, (list, tuple)):
        patch_h, patch_w = patch_size[0], patch_size[1]
    else:
        patch_h, patch_w = patch_size, patch_size

    patch_grid_h = h // patch_h
    patch_grid_w = w // patch_w

    connector = vlm_model.connector
    scale_factor = getattr(connector, "scale_factor", None)

    if scale_factor is not None:
        if patch_grid_h % scale_factor == 0 and patch_grid_w % scale_factor == 0:
            grid_h = patch_grid_h // scale_factor
            grid_w = patch_grid_w // scale_factor
            if grid_h * grid_w == n_tokens:
                return grid_h, grid_w, n_tokens, False

    # Strict fallback: only when n_tokens is a perfect square
    grid_size = int(round(n_tokens ** 0.5))
    if grid_size * grid_size == n_tokens:
        return grid_size, grid_size, n_tokens, True

    raise RuntimeError(
        f"Cannot determine image grid: prepared_image HxW={h}x{w}, "
        f"patch_size={patch_size}, n_tokens={n_tokens}, "
        f"connector scale_factor={scale_factor}, "
        f"patch_grid={patch_grid_h}x{patch_grid_w}"
    )


# ============================================================
# Environment and input creation
# ============================================================

def create_metaworld_input(task, seed, camera_name, device):
    """Create MetaWorld environment and get observation.

    Images are converted: uint8 [0,255] -> float32 / 255.0 -> [0,1]
    This matches the real eval pipeline where prepare_images() does img * 2 - 1.
    """
    env = MetaworldEnv(
        task=task,
        camera_name=camera_name,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=480,
        observation_height=480,
    )

    obs, info = env.reset(seed=seed)

    top_image_raw = obs["pixels/top"]
    wrist_image_raw = obs["pixels/wrist"] if "pixels/wrist" in obs else None
    state = obs["agent_pos"]

    top_image_np = top_image_raw.copy()
    wrist_image_np = wrist_image_raw.copy() if wrist_image_raw is not None else None

    top_img_chw = np.transpose(top_image_raw, (2, 0, 1))
    top_img_tensor = torch.from_numpy(top_img_chw).unsqueeze(0).unsqueeze(0).float() / 255.0

    if wrist_image_np is not None:
        wrist_img_chw = np.transpose(wrist_image_np, (2, 0, 1))
        wrist_img_tensor = torch.from_numpy(wrist_img_chw).unsqueeze(0).unsqueeze(0).float() / 255.0
    else:
        wrist_img_tensor = None

    state_tensor = torch.from_numpy(state.copy()).unsqueeze(0).float()

    print(f"  camera1 (top) shape={top_img_tensor.shape} dtype={top_img_tensor.dtype} "
          f"min={top_img_tensor.min().item():.4f} max={top_img_tensor.max().item():.4f}")
    if wrist_img_tensor is not None:
        print(f"  camera2 (wrist) shape={wrist_img_tensor.shape} dtype={wrist_img_tensor.dtype} "
              f"min={wrist_img_tensor.min().item():.4f} max={wrist_img_tensor.max().item():.4f}")

    return {
        "observation.images.camera1": top_img_tensor.to(device),
        "observation.images.camera2": wrist_img_tensor.to(device) if wrist_img_tensor is not None else None,
        "observation.state": state_tensor.to(device),
        "task": env.task_description,
        "top_image_np": top_image_np,
        "wrist_image_np": wrist_image_np,
        "state": state,
        "env": env,
        "info": info,
    }


def prepare_batch_for_model(model_data, policy, device):
    """Prepare batch in the format expected by the policy.

    Mirrors the real SmolVLA preprocessor pipeline from processor_smolvla.py:
      1. NewLineTaskProcessorStep: ensure task ends with '\n'
      2. TokenizerProcessorStep: tokenize with config.pad_language_to, padding_side='right',
         max_length=config.tokenizer_max_length
    """
    batch = {}

    for img_key in policy.config.image_features:
        if img_key in model_data and model_data[img_key] is not None:
            batch[img_key] = model_data[img_key]

    batch["observation.state"] = model_data["observation.state"]

    processor = policy.model.vlm_with_expert.processor
    task = model_data["task"]

    # NewLineTaskProcessorStep equivalent: ensure task ends with '\n'
    if not task.endswith("\n"):
        task = task + "\n"

    # TokenizerProcessorStep equivalent: use config values, NOT processor.tokenizer.model_max_length
    text_inputs = processor.tokenizer(
        task,
        return_tensors="pt",
        padding=policy.config.pad_language_to,
        padding_side="right",
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
    )

    batch["observation.language.tokens"] = text_inputs["input_ids"].to(device)
    batch["observation.language.attention_mask"] = text_inputs["attention_mask"].to(device)

    # Runtime sanity check
    lang_seq_len = text_inputs["input_ids"].shape[1]
    lang_non_pad = text_inputs["attention_mask"].sum().item()
    print(f"  Language tensor shape: {text_inputs['input_ids'].shape}")
    print(f"  Non-padding language tokens: {lang_non_pad}")
    print(f"  policy.config.tokenizer_max_length: {policy.config.tokenizer_max_length}")
    print(f"  policy.config.pad_language_to: {policy.config.pad_language_to}")

    return batch


# ============================================================
# Attention extraction
# ============================================================

def extract_attention_from_model(policy, batch, device):
    """Extract attention weights from SmolVLA model.

    Runs in eval mode (not training mode) to avoid dropout effects.
    """
    vlm_with_expert = policy.model.vlm_with_expert

    extractor = AttentionExtractor(vlm_with_expert)

    try:
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        spans, prefix_length, suffix_length, image_token_counts = compute_token_spans(
            policy, images, img_masks, lang_tokens, lang_masks, state
        )

        # Compute image grids using real prepared images and real n_tokens
        image_grids = {}
        for img_idx, img_key in enumerate(policy.config.image_features):
            if img_idx >= len(images):
                break
            n_tokens = image_token_counts[img_key]
            grid_h, grid_w, _, is_sqrt = get_image_grid_from_model(
                vlm_with_expert, images[img_idx], n_tokens
            )
            image_grids[img_key] = {
                "grid_h": grid_h,
                "grid_w": grid_w,
                "n_tokens": n_tokens,
                "prepared_height": int(images[img_idx].shape[-2]),
                "prepared_width": int(images[img_idx].shape[-1]),
                "is_sqrt_fallback": is_sqrt,
            }
            print(f"  Image grid {img_key}: {grid_h}x{grid_w}, tokens={n_tokens}, "
                  f"prepared={images[img_idx].shape[-2]}x{images[img_idx].shape[-1]}")

        # Token span sanity checks
        lang_span = spans.get("language")
        if lang_span:
            lang_span_len = lang_span[1] - lang_span[0]
            print(f"  Language span: {lang_span}, length={lang_span_len}")
            if lang_span_len > policy.config.tokenizer_max_length:
                raise RuntimeError(
                    f"Language span length {lang_span_len} exceeds config tokenizer_max_length "
                    f"{policy.config.tokenizer_max_length}. Tokenizer config mismatch."
                )

        expected_total = prefix_length + suffix_length
        print(f"  Prefix length: {prefix_length}")
        print(f"  Suffix length: {suffix_length}")
        print(f"  Expected total sequence: {expected_total}")
        print(f"  Token spans: {spans}")
        print(f"  Image token counts: {image_token_counts}")

        prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        bsize = state.shape[0]
        chunk_size = policy.config.chunk_size
        max_action_dim = policy.config.max_action_dim
        dummy_actions = torch.zeros(bsize, chunk_size, max_action_dim, device=device)
        dummy_timestep = torch.zeros(bsize, device=device)

        suffix_embs, suffix_pad_masks, suffix_att_masks = policy.model.embed_suffix(
            dummy_actions, dummy_timestep
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        from lerobot.policies.common.vla_utils import make_att_2d_masks
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        extractor.reset()

        with torch.no_grad():
            outputs, _ = vlm_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
            )

        captured = extractor.captured_attentions

        # Annotate captured calls with real model layer indices and attention sources
        captured = annotate_attention_calls(vlm_with_expert, captured, prefix_length, suffix_length)

        print(f"  Captured {len(captured)} attention calls")
        if captured:
            print(f"  Attention[0] probs shape: {captured[0]['probs'].shape}")
            print(f"  query_length={captured[0]['query_length']}, key_length={captured[0]['key_length']}")

            # Sanity check: captured attention shape vs expected total
            cap_key_len = captured[0]["key_length"]
            if cap_key_len != expected_total:
                print(f"  WARNING: captured key_length {cap_key_len} != expected total {expected_total}")

        # Print attention topology summary
        print(f"  Attention topology ({len(captured)} calls):")
        for call in captured:
            print(f"    call={call['call_index']} layer={call['layer_index']} "
                  f"source={call['source']} q={call['query_length']} k={call['key_length']}")

        return captured, spans, prefix_length, suffix_length, image_token_counts, image_grids

    except Exception as e:
        print(f"  Error during attention extraction: {e}")
        import traceback
        traceback.print_exc()
        return [], {}, 0, 0, {}, {}
    finally:
        extractor.restore()


# ============================================================
# Phase detection
# ============================================================

def detect_phase_from_env(env, step, max_steps):
    """Detect which phase the current step belongs to based on environment state.

    Uses gripper-object distance, object height, and object-goal distance.
    Returns phase name or None if no phase matches.
    """
    try:
        mjc = env._env
        if mjc is None:
            return None

        qpos = mjc.data.qpos.copy()

        gripper_pos = qpos[:3]
        object_pos = mjc.obj_init_pos.copy() if hasattr(mjc, 'obj_init_pos') and mjc.obj_init_pos is not None else None
        goal_pos = mjc.goal.copy() if hasattr(mjc, 'goal') and mjc.goal is not None else None

        if object_pos is None or goal_pos is None:
            return None

        gripper_object_dist = np.linalg.norm(gripper_pos - object_pos)
        object_goal_dist = np.linalg.norm(object_pos - goal_pos)

        object_z = object_pos[2] if len(object_pos) > 2 else 0.0

        if step == 0:
            return "initial"

        if gripper_object_dist < GRIPPER_OBJECT_DIST_PRE_GRASP and object_z < OBJECT_HEIGHT_POST_GRASP:
            return "pre_grasp"

        if object_z > OBJECT_HEIGHT_POST_GRASP:
            return "post_grasp"

        if object_z > OBJECT_HEIGHT_POST_GRASP and object_goal_dist < OBJECT_GOAL_DIST_PRE_PLACE:
            return "pre_place"

        return None

    except Exception as e:
        print(f"  Warning: phase detection failed: {e}")
        return None


def run_rollout_with_phases(policy, env, model_data, device, max_steps, mode):
    """Run a rollout and collect observations at different phases.

    Returns dict of phase -> observation data.
    """
    phase_data = {}
    phases_seen = set()
    episode_metrics = {
        "first_grasp_step": None,
        "grasp_reached": False,
        "max_object_height": 0.0,
        "min_object_goal_distance_after_grasp": None,
        "final_object_goal_distance": None,
        "success": False,
        "episode_length": 0,
        "post_grasp_drop": None,
        "release_detected": None,
    }

    obs = model_data
    step = 0
    grasp_detected = False
    prev_object_z = 0.0

    for step in range(max_steps):
        phase = detect_phase_from_env(env, step, max_steps)

        if phase and phase not in phases_seen:
            phases_seen.add(phase)

            batch = prepare_batch_for_model(obs, policy, device)
            captured, spans, prefix_len, suffix_len, img_token_counts, img_grids = extract_attention_from_model(
                policy, batch, device
            )

            phase_data[phase] = {
                "captured": captured,
                "spans": spans,
                "prefix_length": prefix_len,
                "suffix_length": suffix_len,
                "image_token_counts": img_token_counts,
                "image_grids": img_grids,
                "top_image": obs["top_image_np"].copy(),
                "wrist_image": obs["wrist_image_np"].copy() if obs["wrist_image_np"] is not None else None,
                "step": step,
                "state": obs["state"].copy(),
            }

            print(f"  Phase '{phase}' detected at step {step}")

        try:
            mjc = env._env
            if mjc is not None:
                qpos = mjc.data.qpos.copy()
                object_pos = mjc.obj_init_pos.copy() if hasattr(mjc, 'obj_init_pos') and mjc.obj_init_pos is not None else None
                goal_pos = mjc.goal.copy() if hasattr(mjc, 'goal') and mjc.goal is not None else None

                if object_pos is not None:
                    object_z = object_pos[2] if len(object_pos) > 2 else 0.0
                    episode_metrics["max_object_height"] = max(episode_metrics["max_object_height"], object_z)

                    if object_z > OBJECT_HEIGHT_POST_GRASP:
                        if not grasp_detected:
                            grasp_detected = True
                            episode_metrics["first_grasp_step"] = step
                            episode_metrics["grasp_reached"] = True

                    if grasp_detected and goal_pos is not None:
                        obj_goal_dist = np.linalg.norm(object_pos[:2] - goal_pos[:2])
                        if episode_metrics["min_object_goal_distance_after_grasp"] is None:
                            episode_metrics["min_object_goal_distance_after_grasp"] = obj_goal_dist
                        else:
                            episode_metrics["min_object_goal_distance_after_grasp"] = min(
                                episode_metrics["min_object_goal_distance_after_grasp"], obj_goal_dist
                            )

                    if goal_pos is not None:
                        episode_metrics["final_object_goal_distance"] = np.linalg.norm(
                            object_pos[:2] - goal_pos[:2]
                        )

                    prev_object_z = object_z
        except Exception:
            pass

        batch = prepare_batch_for_model(obs, policy, device)
        with torch.no_grad():
            actions = policy.predict_action_chunk(batch)
        action = actions[0, 0].cpu().numpy()

        obs_raw, reward, terminated, truncated, info = env.step(action)
        episode_metrics["episode_length"] = step + 1

        if info.get("grasp_success", 0):
            episode_metrics["grasp_reached"] = True
            if not grasp_detected:
                episode_metrics["first_grasp_step"] = step
                grasp_detected = True

        if info.get("is_success", False):
            episode_metrics["success"] = True

        obs = {
            "observation.images.camera1": torch.from_numpy(
                np.transpose(obs_raw["pixels/top"], (2, 0, 1))
            ).unsqueeze(0).unsqueeze(0).float() / 255.0,
            "observation.images.camera2": torch.from_numpy(
                np.transpose(obs_raw["pixels/wrist"], (2, 0, 1))
            ).unsqueeze(0).unsqueeze(0).float() / 255.0 if "pixels/wrist" in obs_raw else None,
            "observation.state": torch.from_numpy(obs_raw["agent_pos"]).unsqueeze(0).float(),
            "task": obs["task"],
            "top_image_np": obs_raw["pixels/top"].copy(),
            "wrist_image_np": obs_raw["pixels/wrist"].copy() if "pixels/wrist" in obs_raw else None,
            "state": obs_raw["agent_pos"].copy(),
            "env": env,
            "info": info,
        }

        if terminated or truncated:
            break

    for phase in PHASES:
        if phase not in phase_data:
            phase_data[phase] = {"reached": False}

    return phase_data, episode_metrics


# ============================================================
# Attention analysis
# ============================================================

def compute_attention_mass(attn_probs, spans, prefix_length, suffix_length, query_indices):
    """Compute attention mass for each token span.

    attn_probs: (batch, heads, query_len, key_len)
    spans: dict of token_name -> (start, end)
    query_indices: indices of query tokens to aggregate over
    """
    if len(attn_probs.shape) != 4:
        return {}

    probs = attn_probs[0]
    selected = probs[:, query_indices, :]

    masses = {}
    total = 0.0

    for name, (start, end) in spans.items():
        if start < selected.shape[-1] and end <= selected.shape[-1]:
            mass = selected[:, :, start:end].sum(dim=-1).mean().item()
            masses[f"{name}_mass"] = mass
            total += mass

    suffix_start = prefix_length
    suffix_end = prefix_length + suffix_length
    if suffix_start < selected.shape[-1] and suffix_end <= selected.shape[-1]:
        mass = selected[:, :, suffix_start:suffix_end].sum(dim=-1).mean().item()
        masses["suffix_mass"] = mass
        total += mass

    masses["total_accounted"] = total
    masses["other_mass"] = max(0.0, 1.0 - total)

    return masses


def get_query_indices_for_call(mode, attn_data, prefix_length, suffix_length):
    """Get query token indices based on aggregation strategy AND the current attention call.

    Query indices are in the coordinate system of THIS attention call's query tensor,
    NOT the full sequence. This is critical because:
    - joint_self: query = full sequence [0..prefix+suffix-1], suffix starts at prefix_length
    - expert_cross: query = only suffix tokens [0..suffix_length-1], index 0 = first suffix token
    - prefix_self: query = only prefix tokens [0..prefix_length-1], no suffix queries exist
    - denoise_joint_self: query = only suffix tokens [0..suffix_length-1] (during denoising)
    """
    source = attn_data.get("source", "unknown")
    query_length = attn_data["query_length"]

    if source == "joint_self":
        # Query is full sequence: [0 .. prefix+suffix-1]
        total_seq = prefix_length + suffix_length
        if mode == "last":
            return [query_length - 1]
        elif mode == "mean_suffix":
            start = prefix_length
            end = min(prefix_length + suffix_length, query_length)
            return list(range(start, end))
        elif mode == "mean_all":
            return list(range(query_length))
        else:
            return [query_length - 1]

    elif source == "expert_cross":
        # Query is only suffix tokens: [0 .. suffix_length-1]
        # Index 0 corresponds to the first action/suffix token
        if mode == "last":
            return [query_length - 1]
        elif mode == "mean_suffix":
            return list(range(query_length))
        elif mode == "mean_all":
            return list(range(query_length))
        else:
            return [query_length - 1]

    elif source == "denoise_joint_self":
        # During denoising, query is only suffix tokens: [0 .. suffix_length-1]
        # Key is prefix+suffix, but query is local to suffix
        if mode == "last":
            return [query_length - 1]
        elif mode == "mean_suffix":
            return list(range(query_length))
        elif mode == "mean_all":
            return list(range(query_length))
        else:
            return [query_length - 1]

    elif source == "prefix_self":
        # Query is only prefix tokens: no suffix/action queries exist
        # mean_suffix is unsupported for prefix_self
        if mode == "last":
            return [query_length - 1]
        elif mode == "mean_suffix":
            return []  # No suffix queries in prefix_self
        elif mode == "mean_all":
            return list(range(query_length))
        else:
            return [query_length - 1]

    else:
        # Unknown source: fallback to full query range
        if mode == "last":
            return [query_length - 1]
        elif mode == "mean_suffix":
            return list(range(query_length))
        elif mode == "mean_all":
            return list(range(query_length))
        else:
            return [query_length - 1]


def select_action_attention_for_layer(captured, layer_idx):
    """Select the best attention call for policy action visualization at a given model layer.

    Priority:
    1. expert_cross: query = action suffix, key = prefix observation (best for action attention)
    2. denoise_joint_self: query = action suffix, key = prefix+suffix (during denoising)
    3. joint_self: query = full sequence including suffix, key = full sequence
    4. prefix_self: NOT selected (no action/suffix queries)

    Returns the attention call dict, or None if no action-related call exists.
    """
    layer_calls = [c for c in captured if c.get("layer_index") == layer_idx]

    # Priority 1: expert_cross
    for call in layer_calls:
        if call.get("source") == "expert_cross":
            return call

    # Priority 2: denoise_joint_self
    for call in layer_calls:
        if call.get("source") == "denoise_joint_self":
            return call

    # Priority 3: joint_self
    for call in layer_calls:
        if call.get("source") == "joint_self":
            return call

    # No action-related attention call for this layer
    return None


# ============================================================
# Visualization
# ============================================================

def create_heatmap_from_attention(attn_1d, original_image, grid_h, grid_w):
    """Create a heatmap overlay from 1D attention vector using known grid layout.

    Strictly checks that len(attn_1d) == grid_h * grid_w. No silent truncation.
    """
    expected = grid_h * grid_w
    if len(attn_1d) != expected:
        raise ValueError(
            f"Image attention token count mismatch: got {len(attn_1d)}, "
            f"expected {expected} for grid {grid_h}x{grid_w}"
        )

    h_img, w_img = original_image.shape[:2]

    attn_2d = attn_1d.reshape(grid_h, grid_w)
    attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)

    attn_resized = cv2.resize(attn_2d, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
    attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
    heatmap = cv2.applyColorMap((attn_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

    return overlay, attn_2d


def generate_summary_figure(phase_data, model_name, method, phase_name, layer_indices,
                           average_heads, query_mode, output_path, image_features):
    """Generate a summary figure for one model/phase."""
    if "reached" in phase_data and not phase_data["reached"]:
        return None

    captured = phase_data.get("captured", [])
    if not captured:
        return None

    spans = phase_data.get("spans", {})
    prefix_length = phase_data.get("prefix_length", 0)
    suffix_length = phase_data.get("suffix_length", 0)
    top_image = phase_data.get("top_image")
    wrist_image = phase_data.get("wrist_image")
    image_token_counts = phase_data.get("image_token_counts", {})
    image_grids = phase_data.get("image_grids", {})

    # Determine camera keys from actual spans/image_grids (not from config order)
    image_feature_keys = list(image_features.keys()) if isinstance(image_features, dict) else list(image_features)
    available_cam_keys = [k for k in image_feature_keys if k in spans and k in image_grids]
    cam1_key = available_cam_keys[0] if len(available_cam_keys) > 0 else None
    cam2_key = available_cam_keys[1] if len(available_cam_keys) > 1 else None

    n_layers_viz = len(layer_indices)
    n_cols = 3
    n_rows = n_layers_viz + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].imshow(top_image)
    axes[0, 0].set_title("Camera 1 (Top)", fontsize=12, fontweight='bold')
    axes[0, 0].axis("off")

    if wrist_image is not None:
        axes[0, 1].imshow(wrist_image)
        axes[0, 1].set_title("Camera 2 (Wrist)", fontsize=12, fontweight='bold')
        axes[0, 1].axis("off")
    else:
        axes[0, 1].axis("off")

    axes[0, 2].axis("off")

    all_metrics = []

    for viz_idx, layer_idx in enumerate(layer_indices):
        # Select the best action-related attention call for this model layer
        attn_data = select_action_attention_for_layer(captured, layer_idx)

        if attn_data is None:
            # No action-related attention call for this layer
            axes[viz_idx + 1, 0].text(0.5, 0.5, f"No action-related\nattention call\nfor layer {layer_idx}",
                                      ha='center', va='center', transform=axes[viz_idx + 1, 0].transAxes,
                                      fontsize=10)
            axes[viz_idx + 1, 0].axis("off")
            axes[viz_idx + 1, 1].axis("off")
            axes[viz_idx + 1, 2].axis("off")
            all_metrics.append({
                "layer_index": layer_idx,
                "attention_source": "none",
                "query_length": None,
                "key_length": None,
                "query_count": 0,
                "masses": {},
            })
            continue

        attn_probs = attn_data["probs"]
        source = attn_data.get("source", "unknown")
        query_length = attn_data["query_length"]
        key_length = attn_data["key_length"]

        if attn_probs.ndim != 4:
            continue

        # Get query indices in THIS call's coordinate system
        query_indices = get_query_indices_for_call(query_mode, attn_data, prefix_length, suffix_length)

        if not query_indices:
            axes[viz_idx + 1, 0].text(0.5, 0.5, f"No valid queries\nfor layer {layer_idx}\n({source}, {query_mode})",
                                      ha='center', va='center', transform=axes[viz_idx + 1, 0].transAxes,
                                      fontsize=10)
            axes[viz_idx + 1, 0].axis("off")
            axes[viz_idx + 1, 1].axis("off")
            axes[viz_idx + 1, 2].axis("off")
            all_metrics.append({
                "layer_index": layer_idx,
                "attention_source": source,
                "query_length": query_length,
                "key_length": key_length,
                "query_count": 0,
                "masses": {},
            })
            continue

        # Validate query indices are within bounds
        if any(i < 0 or i >= query_length for i in query_indices):
            print(f"  WARNING: query indices out of bounds for layer {layer_idx} ({source})")
            query_indices = [i for i in query_indices if 0 <= i < query_length]
            if not query_indices:
                continue

        # Aggregate attention over selected queries
        if average_heads:
            avg_attn = attn_probs.mean(dim=1)  # [B, Q, K]
            attn_for_viz = avg_attn[0, query_indices, :]  # [Q_selected, K]
            if attn_for_viz.ndim == 2:
                attn_1d = attn_for_viz.mean(dim=0).cpu().numpy()  # [K]
            else:
                attn_1d = attn_for_viz.cpu().numpy()
        else:
            # Use head 0, but average over all selected query indices
            head0_attn = attn_probs[0, 0, query_indices, :]  # [Q_selected, K]
            attn_1d = head0_attn.mean(dim=0).cpu().numpy()  # [K]

        cam1_overlay = None
        cam2_overlay = None

        if cam1_key and cam1_key in spans and cam1_key in image_grids:
            c1_start, c1_end = spans[cam1_key]
            cam1_attn = attn_1d[c1_start:c1_end]
            grid_h = image_grids[cam1_key]["grid_h"]
            grid_w = image_grids[cam1_key]["grid_w"]
            if len(cam1_attn) == grid_h * grid_w:
                cam1_overlay, _ = create_heatmap_from_attention(cam1_attn, top_image, grid_h, grid_w)
            else:
                print(f"  WARNING: camera1 attention token count {len(cam1_attn)} != grid {grid_h}x{grid_w}={grid_h*grid_w}")

        if cam2_key and cam2_key in spans and cam2_key in image_grids and wrist_image is not None:
            c2_start, c2_end = spans[cam2_key]
            cam2_attn = attn_1d[c2_start:c2_end]
            grid_h = image_grids[cam2_key]["grid_h"]
            grid_w = image_grids[cam2_key]["grid_w"]
            if len(cam2_attn) == grid_h * grid_w:
                cam2_overlay, _ = create_heatmap_from_attention(cam2_attn, wrist_image, grid_h, grid_w)
            else:
                print(f"  WARNING: camera2 attention token count {len(cam2_attn)} != grid {grid_h}x{grid_w}={grid_h*grid_w}")

        if cam1_overlay is not None:
            axes[viz_idx + 1, 0].imshow(cam1_overlay)
            axes[viz_idx + 1, 0].set_title(f"Layer {layer_idx} ({source}) - Camera 1 Attention", fontsize=10)
            axes[viz_idx + 1, 0].axis("off")
        else:
            axes[viz_idx + 1, 0].axis("off")

        if cam2_overlay is not None:
            axes[viz_idx + 1, 1].imshow(cam2_overlay)
            axes[viz_idx + 1, 1].set_title(f"Layer {layer_idx} ({source}) - Camera 2 Attention", fontsize=10)
            axes[viz_idx + 1, 1].axis("off")
        else:
            axes[viz_idx + 1, 1].axis("off")

        masses = compute_attention_mass(attn_probs, spans, prefix_length, suffix_length, query_indices)
        all_metrics.append({
            "layer_index": layer_idx,
            "attention_source": source,
            "query_length": query_length,
            "key_length": key_length,
            "query_count": len(query_indices),
            "masses": masses,
        })

        mass_text = [f"source={source}", f"q={query_length}, k={key_length}", f"queries={len(query_indices)}"]
        for k, v in masses.items():
            if k.endswith("_mass"):
                mass_text.append(f"{k.replace('_mass', '')}: {v:.3f}")
        axes[viz_idx + 1, 2].text(0.05, 0.95, "\n".join(mass_text),
                                  transform=axes[viz_idx + 1, 2].transAxes,
                                  fontsize=8, verticalalignment='top',
                                  fontfamily='monospace')
        axes[viz_idx + 1, 2].axis("off")

    plt.suptitle(
        f"{model_name} ({method}) - {phase_name} phase\n"
        f"query_mode={query_mode}, layers={layer_indices}, avg_heads={average_heads}",
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return all_metrics


# ============================================================
# Inference trace helpers
# ============================================================

def create_shared_initial_noise(chunk_size, max_action_dim, noise_seed, device):
    """Create deterministic initial noise for fair comparison across models.

    Uses CPU torch.Generator to ensure reproducibility, then copies to target device.
    Distribution matches sample_noise() from flow_matching.py: standard normal float32.

    Args:
        chunk_size: Number of action steps in chunk.
        max_action_dim: Maximum action dimension.
        noise_seed: Random seed for noise generation.
        device: Target device for the noise tensor.

    Returns:
        Tensor of shape (1, chunk_size, max_action_dim) with standard normal noise.
    """
    shape = (1, chunk_size, max_action_dim)
    generator = torch.Generator(device="cpu").manual_seed(noise_seed)
    noise = torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, generator=generator)
    return noise.to(device)


def compute_tensor_checksum(tensor):
    """Compute a simple checksum for tensor verification."""
    return float(tensor.detach().float().cpu().sum().item())


def compute_image_checksum(tensor):
    """Compute checksum for image tensor verification."""
    return float(tensor.detach().float().cpu().sum().item())


def parse_heatmap_steps(heatmap_steps_str):
    """Parse heatmap step specification string.

    Args:
        heatmap_steps_str: Comma-separated string like "first,mid,last" or "0,5,9".

    Returns:
        List of step identifiers: "first", "mid", "last", or integer indices.
    """
    steps = []
    for s in heatmap_steps_str.split(","):
        s = s.strip()
        if s in ("first", "mid", "last"):
            steps.append(s)
        else:
            try:
                steps.append(int(s))
            except ValueError:
                print(f"  WARNING: Invalid heatmap step '{s}', ignoring.")
    return steps


def resolve_heatmap_step_indices(step_specs, num_steps):
    """Resolve step specifications to actual indices.

    Args:
        step_specs: List from parse_heatmap_steps().
        num_steps: Total number of denoising steps.

    Returns:
        Set of integer indices that should generate heatmaps.
    """
    indices = set()
    for spec in step_specs:
        if spec == "first":
            indices.add(0)
        elif spec == "mid":
            indices.add(num_steps // 2)
        elif spec == "last":
            indices.add(num_steps - 1)
        elif isinstance(spec, int):
            if 0 <= spec < num_steps:
                indices.add(spec)
            else:
                print(f"  WARNING: Heatmap step index {spec} out of range [0, {num_steps-1}], ignoring.")
    return indices


def extract_inference_trace_from_model(
    policy,
    batch,
    shared_noise,
    device,
    layer_indices,
    query_mode,
    heatmap_step_specs,
    output_dir,
    model_cfg,
):
    """Extract full inference trace from a SmolVLA model.

    Runs sample_actions_with_trace() with attention capturing at each denoising step.
    Produces attention mass CSV rows, step traces, heatmaps, and metadata.

    Args:
        policy: SmolVLAPolicy instance (eval mode).
        batch: Prepared input batch dict.
        shared_noise: Pre-generated initial noise tensor (1, chunk_size, max_action_dim).
        device: Target device.
        layer_indices: List of layer indices to analyze.
        query_mode: Query aggregation mode ("mean_suffix", "last", "mean_all").
        heatmap_step_specs: List of step specs for heatmap generation.
        output_dir: Path to save results for this model.
        model_cfg: Model configuration dict with name, method, camera_name, eval_task_success.

    Returns:
        Dict with:
            - attention_rows: List of dicts for CSV.
            - step_traces: List of per-step trace dicts.
            - final_action: Tensor of final action chunk.
            - metadata: Dict with model and run metadata.
            - initial_noise: The shared noise tensor used.
    """
    model_name = model_cfg["name"]
    method = model_cfg["method"]
    camera_name = model_cfg["camera_name"]
    eval_task_success = model_cfg.get("eval_task_success", None)

    vlm_with_expert = policy.model.vlm_with_expert
    extractor = AttentionExtractor(vlm_with_expert)

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

    spans, prefix_length, suffix_length, image_token_counts = compute_token_spans(
        policy, images, img_masks, lang_tokens, lang_masks, state
    )

    image_grids = {}
    for img_idx, img_key in enumerate(policy.config.image_features):
        if img_idx >= len(images):
            break
        n_tokens = image_token_counts[img_key]
        grid_h, grid_w, _, is_sqrt = get_image_grid_from_model(
            vlm_with_expert, images[img_idx], n_tokens
        )
        image_grids[img_key] = {
            "grid_h": grid_h,
            "grid_w": grid_w,
            "n_tokens": n_tokens,
            "prepared_height": int(images[img_idx].shape[-2]),
            "prepared_width": int(images[img_idx].shape[-1]),
            "is_sqrt_fallback": is_sqrt,
        }

    num_steps = policy.config.num_steps
    chunk_size = policy.config.chunk_size
    max_action_dim = policy.config.max_action_dim
    heatmap_indices = resolve_heatmap_step_indices(heatmap_step_specs, num_steps)

    available_cam_keys = [k for k in policy.config.image_features if k in spans and k in image_grids]
    cam1_key = available_cam_keys[0] if len(available_cam_keys) > 0 else None
    cam2_key = available_cam_keys[1] if len(available_cam_keys) > 1 else None

    attention_rows = []
    step_traces = []
    x_t_history = []
    v_t_history = []
    suffix_hidden_history = []
    timesteps_list = []

    heatmaps_dir = output_dir / "heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    denoise_counter = [0]

    def trace_callback(event_data):
        if event_data["event"] == "prefill_end":
            extractor.reset()
            return

        if event_data["event"] == "denoise_step":
            idx = denoise_counter[0]
            denoise_counter[0] += 1

            timestep_val = event_data["timestep"].detach().float().cpu().item()
            x_t = event_data["x_t"]
            suffix_hidden = event_data["suffix_hidden"]
            v_t = event_data["v_t"]
            prefix_len = event_data["prefix_length"]
            suffix_len = event_data["suffix_length"]

            x_t_history.append(x_t.detach().float().cpu())
            v_t_history.append(v_t.detach().float().cpu())
            suffix_hidden_history.append(suffix_hidden.detach().float().cpu())
            timesteps_list.append(timestep_val)

            captured = list(extractor.captured_attentions)

            try:
                annotated = annotate_denoise_attention_calls(
                    vlm_with_expert, captured, prefix_len, suffix_len
                )
            except RuntimeError as e:
                print(f"  WARNING: Attention annotation failed at step {idx}: {e}")
                extractor.reset()
                return

            for layer_idx in layer_indices:
                layer_calls = [c for c in annotated if c.get("layer_index") == layer_idx]
                if not layer_calls:
                    continue

                for call in layer_calls:
                    attn_probs = call["probs"]
                    source = call["source"]
                    q_len = call["query_length"]
                    k_len = call["key_length"]

                    query_indices = get_query_indices_for_call(
                        query_mode, call, prefix_len, suffix_len
                    )
                    if not query_indices:
                        continue

                    if any(i < 0 or i >= q_len for i in query_indices):
                        query_indices = [i for i in query_indices if 0 <= i < q_len]
                        if not query_indices:
                            continue

                    masses = compute_attention_mass(
                        attn_probs, spans, prefix_len, suffix_len, query_indices
                    )

                    row = {
                        "model_name": model_name,
                        "method": method,
                        "camera": camera_name,
                        "eval_task_success": eval_task_success,
                        "denoise_index": idx,
                        "timestep": timestep_val,
                        "layer": layer_idx,
                        "attention_source": source,
                        "query_length": q_len,
                        "key_length": k_len,
                        "query_count": len(query_indices),
                        "camera1_mass": masses.get(f"{cam1_key}_mass", 0.0) if cam1_key else 0.0,
                        "camera2_mass": masses.get(f"{cam2_key}_mass", 0.0) if cam2_key else 0.0,
                        "visual_total": (
                            (masses.get(f"{cam1_key}_mass", 0.0) if cam1_key else 0.0)
                            + (masses.get(f"{cam2_key}_mass", 0.0) if cam2_key else 0.0)
                        ),
                        "language_mass": masses.get("language_mass", 0.0),
                        "state_mass": masses.get("state_mass", 0.0),
                        "suffix_mass": masses.get("suffix_mass", 0.0),
                        "other_mass": masses.get("other_mass", 0.0),
                        "x_t_norm": float(torch.norm(x_t).item()),
                        "v_t_norm": float(torch.norm(v_t).item()),
                        "suffix_hidden_norm": float(torch.norm(suffix_hidden).item()),
                    }
                    attention_rows.append(row)

            if idx in heatmap_indices:
                for layer_idx in layer_indices:
                    layer_calls = [c for c in annotated if c.get("layer_index") == layer_idx]
                    if not layer_calls:
                        continue

                    attn_call = layer_calls[0]
                    attn_probs = attn_call["probs"]
                    source = attn_call["source"]

                    if attn_probs.ndim != 4:
                        continue

                    query_indices = get_query_indices_for_call(
                        query_mode, attn_call, prefix_len, suffix_len
                    )
                    if not query_indices:
                        continue

                    avg_attn = attn_probs.mean(dim=1)
                    attn_for_viz = avg_attn[0, query_indices, :]
                    if attn_for_viz.ndim == 2:
                        attn_1d = attn_for_viz.mean(dim=0).cpu().numpy()
                    else:
                        attn_1d = attn_for_viz.cpu().numpy()

                    for img_idx, img_key in enumerate(policy.config.image_features):
                        if img_idx >= len(images):
                            break
                        if img_key not in spans or img_key not in image_grids:
                            continue

                        img_start, img_end = spans[img_key]
                        img_attn = attn_1d[img_start:img_end]
                        grid_h = image_grids[img_key]["grid_h"]
                        grid_w = image_grids[img_key]["grid_w"]

                        if len(img_attn) != grid_h * grid_w:
                            print(f"  WARNING: heatmap attention count mismatch for {img_key}: "
                                  f"got {len(img_attn)}, expected {grid_h*grid_w}")
                            continue

                        orig_img = images[img_idx].detach().float().cpu()
                        orig_img = (orig_img + 1.0) / 2.0
                        orig_img = orig_img.squeeze().permute(1, 2, 0).numpy()
                        orig_img = (orig_img * 255).clip(0, 255).astype(np.uint8)

                        try:
                            overlay, _ = create_heatmap_from_attention(img_attn, orig_img, grid_h, grid_w)
                            step_label = f"step{idx}"
                            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                            ax.imshow(overlay)
                            ax.set_title(
                                f"{model_name}\nLayer {layer_idx} ({source})\n"
                                f"t={timestep_val:.3f}, {step_label}",
                                fontsize=10
                            )
                            ax.axis("off")
                            plt.tight_layout()
                            heatmap_path = heatmaps_dir / f"{img_key}_layer{layer_idx}_{step_label}.png"
                            plt.savefig(heatmap_path, dpi=120, bbox_inches="tight")
                            plt.close()
                        except Exception as e:
                            print(f"  WARNING: Heatmap generation failed for {img_key} layer {layer_idx}: {e}")

            extractor.reset()

    try:
        with torch.no_grad():
            final_action = policy.model.sample_actions_with_trace(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise=shared_noise,
                trace_callback=trace_callback,
            )
    finally:
        extractor.restore()

    initial_noise_cpu = shared_noise.detach().float().cpu()
    final_action_cpu = final_action.detach().float().cpu()

    x_t_tensor = torch.stack(x_t_history, dim=0) if x_t_history else torch.empty(0)
    v_t_tensor = torch.stack(v_t_history, dim=0) if v_t_history else torch.empty(0)
    suffix_hidden_tensor = torch.stack(suffix_hidden_history, dim=0) if suffix_hidden_history else torch.empty(0)
    timesteps_tensor = torch.tensor(timesteps_list, dtype=torch.float32)

    trace_pt = {
        "timesteps": timesteps_tensor,
        "x_t": x_t_tensor,
        "v_t": v_t_tensor,
        "suffix_hidden": suffix_hidden_tensor,
        "initial_noise": initial_noise_cpu,
        "final_action": final_action_cpu,
    }
    trace_path = output_dir / "trace.pt"
    torch.save(trace_pt, trace_path)

    input_checksums = {}
    for img_idx, img_key in enumerate(policy.config.image_features):
        if img_idx < len(images):
            input_checksums[img_key] = compute_image_checksum(images[img_idx])
    input_checksums["state"] = compute_tensor_checksum(state)
    input_checksums["noise"] = compute_tensor_checksum(initial_noise_cpu)

    metadata = {
        "model_name": model_name,
        "method": method,
        "camera": camera_name,
        "model_path": model_cfg["path"],
        "eval_task_success": eval_task_success,
        "seed": None,
        "noise_seed": None,
        "policy_config": {
            "chunk_size": chunk_size,
            "num_steps": num_steps,
            "max_action_dim": max_action_dim,
            "use_cache": policy.config.use_cache,
            "attention_mode": vlm_with_expert.attention_mode,
            "self_attn_every_n_layers": vlm_with_expert.self_attn_every_n_layers,
        },
        "image_token_counts": image_token_counts,
        "image_grids": {k: {kk: vv for kk, vv in v.items()} for k, v in image_grids.items()},
        "prefix_length": prefix_length,
        "suffix_length": suffix_length,
        "language_length": spans.get("language", (0, 0))[1] - spans.get("language", (0, 0))[0],
        "input_checksums": input_checksums,
        "noise_checksum": compute_tensor_checksum(initial_noise_cpu),
        "noise_shape": list(initial_noise_cpu.shape),
        "noise_mean": float(initial_noise_cpu.mean().item()),
        "noise_std": float(initial_noise_cpu.std().item()),
        "noise_norm": float(torch.norm(initial_noise_cpu).item()),
        "final_action_shape": list(final_action_cpu.shape),
    }

    for step_trace_idx in range(len(timesteps_list)):
        step_traces.append({
            "denoise_index": step_trace_idx,
            "timestep": timesteps_list[step_trace_idx],
            "x_t_norm": float(torch.norm(x_t_tensor[step_trace_idx]).item()),
            "v_t_norm": float(torch.norm(v_t_tensor[step_trace_idx]).item()),
            "suffix_hidden_norm": float(torch.norm(suffix_hidden_tensor[step_trace_idx]).item()),
        })

    return {
        "attention_rows": attention_rows,
        "step_traces": step_traces,
        "final_action": final_action_cpu,
        "metadata": metadata,
        "initial_noise": initial_noise_cpu,
        "x_t_tensor": x_t_tensor,
        "v_t_tensor": v_t_tensor,
        "suffix_hidden_tensor": suffix_hidden_tensor,
        "timesteps_tensor": timesteps_tensor,
    }


def run_sanity_check(policy, batch, shared_noise, device):
    """Verify that sample_actions_with_trace produces identical output to sample_actions.

    Args:
        policy: SmolVLAPolicy in eval mode.
        batch: Prepared input batch.
        shared_noise: Same initial noise for both calls.
        device: Target device.

    Returns:
        Dict with max_abs_diff and mean_abs_diff.
    """
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

    with torch.no_grad():
        action_baseline = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=shared_noise
        )
        action_trace = policy.model.sample_actions_with_trace(
            images, img_masks, lang_tokens, lang_masks, state, noise=shared_noise, trace_callback=None
        )

    diff = (action_baseline - action_trace).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def generate_inference_trace_summary_plots(
    all_attention_rows,
    all_model_traces,
    camera_groups,
    output_dir,
):
    """Generate summary comparison plots for inference trace analysis.

    Args:
        all_attention_rows: List of attention mass dicts from all models.
        all_model_traces: Dict mapping model_name -> trace result dict.
        camera_groups: Dict mapping camera_group_name -> list of model_names.
        output_dir: Directory to save plots.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for group_name, model_names in camera_groups.items():
        available_names = [n for n in model_names if n in all_model_traces]
        if not available_names:
            continue

        _plot_attention_over_denoising(
            all_attention_rows, available_names, all_model_traces, plots_dir, group_name
        )
        _plot_action_trajectory_statistics(
            available_names, all_model_traces, plots_dir, group_name
        )
        _plot_pairwise_action_divergence(
            available_names, all_model_traces, plots_dir, group_name
        )


def _plot_attention_over_denoising(
    all_attention_rows, model_names, all_model_traces, plots_dir, group_name,
):
    """Plot attention mass over denoising steps for selected layers."""
    layer_to_plot = 11
    layer_rows = [r for r in all_attention_rows if r["layer"] == layer_to_plot]
    if not layer_rows:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_to_plot = [
        ("visual_total", "Visual Attention Mass"),
        ("camera1_mass", "Camera 1 Attention Mass"),
        ("camera2_mass", "Camera 2 Attention Mass"),
        ("language_mass", "Language Attention Mass"),
    ]

    for ax_idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[ax_idx // 2, ax_idx % 2]
        for model_name in model_names:
            model_rows = [r for r in layer_rows if r["model_name"] == model_name]
            if not model_rows:
                continue
            model_rows_sorted = sorted(model_rows, key=lambda r: r["denoise_index"])
            xs = [r["denoise_index"] for r in model_rows_sorted]
            ys = [r[metric] for r in model_rows_sorted]
            success = all_model_traces[model_name]["metadata"].get("eval_task_success")
            method = all_model_traces[model_name]["metadata"].get("method", model_name.split("_")[0])
            label = f"{method.capitalize()} ({success}%)" if success else method.capitalize()
            ax.plot(xs, ys, marker="o", markersize=3, label=label)

        ax.set_xlabel("Denoising Step")
        ax.set_ylabel(title)
        ax.set_title(f"Layer {layer_to_plot} - {group_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Attention over Denoising Steps - {group_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{group_name}_attention_over_denoising.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_action_trajectory_statistics(model_names, all_model_traces, plots_dir, group_name):
    """Plot action trajectory statistics (norms) over denoising steps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ("v_t_norm", "v_t L2 Norm", axes[0]),
        ("x_t_norm", "x_t L2 Norm", axes[1]),
        ("suffix_hidden_norm", "Suffix Hidden L2 Norm", axes[2]),
    ]

    for metric, title, ax in metrics:
        for model_name in model_names:
            trace = all_model_traces.get(model_name)
            if trace is None:
                continue
            step_traces = trace.get("step_traces", [])
            if not step_traces:
                continue
            xs = [s["denoise_index"] for s in step_traces]
            ys = [s[metric] for s in step_traces]
            success = trace["metadata"].get("eval_task_success")
            method = trace["metadata"].get("method", model_name.split("_")[0])
            label = f"{method.capitalize()} ({success}%)" if success else method.capitalize()
            ax.plot(xs, ys, marker="o", markersize=3, label=label)

        ax.set_xlabel("Denoising Step")
        ax.set_ylabel(title)
        ax.set_title(f"{title} - {group_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Action Trajectory Statistics - {group_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{group_name}_action_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_pairwise_action_divergence(model_names, all_model_traces, plots_dir, group_name):
    """Plot pairwise action divergence between models."""
    if len(model_names) < 2:
        return

    pairwise_rows = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a = model_names[i]
            name_b = model_names[j]
            trace_a = all_model_traces.get(name_a)
            trace_b = all_model_traces.get(name_b)
            if trace_a is None or trace_b is None:
                continue

            v_t_a = trace_a.get("v_t_tensor")
            v_t_b = trace_b.get("v_t_tensor")
            x_t_a = trace_a.get("x_t_tensor")
            x_t_b = trace_b.get("x_t_tensor")
            hidden_a = trace_a.get("suffix_hidden_tensor")
            hidden_b = trace_b.get("suffix_hidden_tensor")

            if v_t_a is None or v_t_b is None:
                continue

            min_steps = min(v_t_a.shape[0], v_t_b.shape[0])
            if min_steps == 0:
                continue

            v_t_a_s = v_t_a[:min_steps]
            v_t_b_s = v_t_b[:min_steps]

            v_t_l2 = torch.norm(v_t_a_s - v_t_b_s, dim=(1, 2)).cpu().numpy()

            v_t_a_flat = v_t_a_s.reshape(min_steps, -1)
            v_t_b_flat = v_t_b_s.reshape(min_steps, -1)
            v_t_cos = torch.nn.functional.cosine_similarity(v_t_a_flat, v_t_b_flat, dim=1).cpu().numpy()

            method_a = trace_a["metadata"].get("method", name_a.split("_")[0])
            method_b = trace_b["metadata"].get("method", name_b.split("_")[0])
            pair_label = f"{method_a} vs {method_b}"

            xs = list(range(min_steps))
            axes[0].plot(xs, v_t_l2, marker="o", markersize=3, label=pair_label)
            axes[1].plot(xs, v_t_cos, marker="o", markersize=3, label=pair_label)

            for step_idx in range(min_steps):
                x_t_l2_val = 0.0
                hidden_cos_val = 0.0
                if x_t_a is not None and x_t_b is not None and step_idx < x_t_a.shape[0] and step_idx < x_t_b.shape[0]:
                    x_t_l2_val = float(torch.norm(x_t_a[step_idx] - x_t_b[step_idx]).item())
                if hidden_a is not None and hidden_b is not None and step_idx < hidden_a.shape[0] and step_idx < hidden_b.shape[0]:
                    h_a = hidden_a[step_idx].reshape(-1)
                    h_b = hidden_b[step_idx].reshape(-1)
                    hidden_cos_val = float(torch.nn.functional.cosine_similarity(h_a.unsqueeze(0), h_b.unsqueeze(0)).item())

                timestep_a = trace_a["timesteps_tensor"][step_idx].item() if step_idx < trace_a["timesteps_tensor"].shape[0] else 0.0

                pairwise_rows.append({
                    "camera": group_name,
                    "model_a": name_a,
                    "model_b": name_b,
                    "denoise_index": step_idx,
                    "timestep": timestep_a,
                    "v_t_l2": float(v_t_l2[step_idx]),
                    "v_t_cosine": float(v_t_cos[step_idx]),
                    "x_t_l2": x_t_l2_val,
                    "hidden_cosine": hidden_cos_val,
                })

    axes[0].set_xlabel("Denoising Step")
    axes[0].set_ylabel("v_t L2 Distance")
    axes[0].set_title(f"Pairwise v_t Divergence - {group_name}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Denoising Step")
    axes[1].set_ylabel("v_t Cosine Similarity")
    axes[1].set_title(f"Pairwise v_t Similarity - {group_name}")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Pairwise Action Divergence - {group_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{group_name}_pairwise_divergence.png", dpi=150, bbox_inches="tight")
    plt.close()

    if pairwise_rows:
        pairwise_csv = plots_dir / "pairwise_action_divergence.csv"
        fieldnames = [
            "camera", "model_a", "model_b", "denoise_index", "timestep",
            "v_t_l2", "v_t_cosine", "x_t_l2", "hidden_cosine",
        ]
        with open(pairwise_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pairwise_rows)


# ============================================================
# Main processing
# ============================================================

def process_single_model(model_cfg, device, seed, mode, query_mode, output_dir,
                        layer_indices, average_heads, max_steps,
                        noise_seed=None, heatmap_steps_str=None, shared_noise=None,
                        do_sanity_check=False):
    """Process a single model checkpoint."""
    model_name = model_cfg["name"]
    method = model_cfg["method"]
    camera_name = model_cfg["camera_name"]
    model_path = model_cfg["path"]

    p = Path(model_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p

    if not p.exists():
        print(f"  WARNING: Model path does not exist: {p}")
        return None

    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"Method: {method}")
    print(f"Camera: {camera_name}")
    print(f"Path: {p}")
    print(f"Seed: {seed}")
    print(f"Mode: {mode}")
    print(f"Query mode: {query_mode}")
    print(f"{'='*80}")

    policy = SmolVLAPolicy.from_pretrained(str(p))
    policy = policy.to(device)
    policy.eval()

    print(f"  policy.config.image_features: {policy.config.image_features}")
    print(f"  Task: {TASK}")

    model_dir = output_dir / model_name / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_name": model_name,
        "method": method,
        "camera_name": camera_name,
        "seed": seed,
        "mode": mode,
        "query_mode": query_mode,
        "layer_indices": layer_indices,
        "average_heads": average_heads,
        "max_steps": max_steps,
        "phase_detection_thresholds": {
            "gripper_object_dist_pre_grasp": GRIPPER_OBJECT_DIST_PRE_GRASP,
            "object_height_post_grasp": OBJECT_HEIGHT_POST_GRASP,
            "object_goal_dist_pre_place": OBJECT_GOAL_DIST_PRE_PLACE,
        },
        "phases": {},
        "episode_metrics": {},
    }

    all_metrics_rows = []
    env = None

    try:
        if mode == "inference_trace":
            model_data = create_metaworld_input(TASK, seed, camera_name, device)
            env = model_data["env"]
            metadata["obj_init_pos"] = model_data["info"].get("obj_init_pos", None)
            metadata["goal_pos"] = model_data["info"].get("goal_pos", None)
            metadata["state"] = model_data["state"].tolist()

            batch = prepare_batch_for_model(model_data, policy, device)

            if shared_noise is None:
                ns = noise_seed if noise_seed is not None else seed
                shared_noise = create_shared_initial_noise(
                    policy.config.chunk_size,
                    policy.config.max_action_dim,
                    ns,
                    device,
                )

            if do_sanity_check:
                print(f"  Running sanity check (sample_actions vs sample_actions_with_trace)...")
                sanity = run_sanity_check(policy, batch, shared_noise, device)
                print(f"  Sanity check: max_abs_diff={sanity['max_abs_diff']:.2e}, "
                      f"mean_abs_diff={sanity['mean_abs_diff']:.2e}")
                metadata["sanity_check"] = sanity

            heatmap_specs = parse_heatmap_steps(heatmap_steps_str) if heatmap_steps_str else ["first", "mid", "last"]

            trace_result = extract_inference_trace_from_model(
                policy=policy,
                batch=batch,
                shared_noise=shared_noise,
                device=device,
                layer_indices=layer_indices,
                query_mode=query_mode,
                heatmap_step_specs=heatmap_specs,
                output_dir=model_dir,
                model_cfg=model_cfg,
            )

            metadata["noise_seed"] = noise_seed if noise_seed is not None else seed
            metadata["noise_checksum"] = trace_result["metadata"]["noise_checksum"]
            metadata["input_checksums"] = trace_result["metadata"]["input_checksums"]

            all_metrics_rows = trace_result["attention_rows"]

            print(f"  Saved trace.pt to {model_dir / 'trace.pt'}")
            print(f"  Saved metadata.json to {model_dir / 'metadata.json'}")
            print(f"  Captured {len(all_metrics_rows)} attention rows")

            return all_metrics_rows, trace_result

        elif mode == "rollout":
            model_data = create_metaworld_input(TASK, seed, camera_name, device)
            env = model_data["env"]
            metadata["obj_init_pos"] = model_data["info"].get("obj_init_pos", None)
            metadata["goal_pos"] = model_data["info"].get("goal_pos", None)
            metadata["state"] = model_data["state"].tolist()

            phase_data, episode_metrics = run_rollout_with_phases(
                policy, model_data["env"], model_data, device, max_steps, mode
            )

            metadata["episode_metrics"] = episode_metrics

            for phase_name in PHASES:
                pdata = phase_data.get(phase_name, {})
                if "reached" in pdata and not pdata["reached"]:
                    metadata["phases"][phase_name] = {"reached": False}
                    print(f"  Phase '{phase_name}': not_reached")
                    continue

                if "captured" not in pdata or not pdata["captured"]:
                    metadata["phases"][phase_name] = {"reached": False}
                    continue

                metadata["phases"][phase_name] = {"reached": True, "step": pdata.get("step")}
                print(f"  Phase '{phase_name}': reached at step {pdata.get('step')}")

                output_path = model_dir / f"{phase_name}_summary.png"

                metrics = generate_summary_figure(
                    pdata, model_name, method, phase_name, layer_indices,
                    average_heads, query_mode, output_path, policy.config.image_features
                )

                if metrics:
                    image_feature_keys = list(policy.config.image_features.keys()) if isinstance(policy.config.image_features, dict) else list(policy.config.image_features)
                    cam1_key = image_feature_keys[0] if len(image_feature_keys) > 0 else None
                    cam2_key = image_feature_keys[1] if len(image_feature_keys) > 1 else None

                    for metric_record in metrics:
                        masses = metric_record.get("masses", {})
                        row = {
                            "model_name": model_name,
                            "method": method,
                            "camera": camera_name,
                            "seed": seed,
                            "phase": phase_name,
                            "success": episode_metrics.get("success", False),
                            "grasp_reached": episode_metrics.get("grasp_reached", False),
                            "layer": metric_record["layer_index"],
                            "attention_source": metric_record.get("attention_source", "unknown"),
                            "query_length": metric_record.get("query_length"),
                            "key_length": metric_record.get("key_length"),
                            "query_count": metric_record.get("query_count", 0),
                            "camera1_mass": masses.get(f"{cam1_key}_mass", 0.0) if cam1_key else 0.0,
                            "camera2_mass": masses.get(f"{cam2_key}_mass", 0.0) if cam2_key else 0.0,
                            "language_mass": masses.get("language_mass", 0),
                            "state_mass": masses.get("state_mass", 0),
                            "suffix_mass": masses.get("suffix_mass", 0),
                            "other_mass": masses.get("other_mass", 0),
                        }
                        all_metrics_rows.append(row)

        else:
            model_data = create_metaworld_input(TASK, seed, camera_name, device)
            env = model_data["env"]
            metadata["obj_init_pos"] = model_data["info"].get("obj_init_pos", None)
            metadata["goal_pos"] = model_data["info"].get("goal_pos", None)
            metadata["state"] = model_data["state"].tolist()

            batch = prepare_batch_for_model(model_data, policy, device)
            captured, spans, prefix_len, suffix_len, img_token_counts, img_grids = extract_attention_from_model(
                policy, batch, device
            )

            phase_data = {
                "captured": captured,
                "spans": spans,
                "prefix_length": prefix_len,
                "suffix_length": suffix_len,
                "image_token_counts": img_token_counts,
                "image_grids": img_grids,
                "top_image": model_data["top_image_np"].copy(),
                "wrist_image": model_data["wrist_image_np"].copy() if model_data["wrist_image_np"] is not None else None,
                "step": 0,
                "state": model_data["state"].copy(),
            }

            output_path = model_dir / "initial_summary.png"
            metrics = generate_summary_figure(
                phase_data, model_name, method, "initial (probe)", layer_indices,
                average_heads, query_mode, output_path, policy.config.image_features
            )

            if metrics:
                image_feature_keys = list(policy.config.image_features.keys()) if isinstance(policy.config.image_features, dict) else list(policy.config.image_features)
                cam1_key = image_feature_keys[0] if len(image_feature_keys) > 0 else None
                cam2_key = image_feature_keys[1] if len(image_feature_keys) > 1 else None

                for metric_record in metrics:
                    masses = metric_record.get("masses", {})
                    row = {
                        "model_name": model_name,
                        "method": method,
                        "camera": camera_name,
                        "seed": seed,
                        "phase": "initial",
                        "success": None,
                        "grasp_reached": None,
                        "layer": metric_record["layer_index"],
                        "attention_source": metric_record.get("attention_source", "unknown"),
                        "query_length": metric_record.get("query_length"),
                        "key_length": metric_record.get("key_length"),
                        "query_count": metric_record.get("query_count", 0),
                        "camera1_mass": masses.get(f"{cam1_key}_mass", 0.0) if cam1_key else 0.0,
                        "camera2_mass": masses.get(f"{cam2_key}_mass", 0.0) if cam2_key else 0.0,
                        "language_mass": masses.get("language_mass", 0),
                        "state_mass": masses.get("state_mass", 0),
                        "suffix_mass": masses.get("suffix_mass", 0),
                        "other_mass": masses.get("other_mass", 0),
                    }
                    all_metrics_rows.append(row)

    finally:
        if env is not None:
            try:
                env.close()
                print(f"  Environment closed successfully")
            except Exception as e:
                print(f"  WARNING: Failed to close environment: {e}")

    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved metadata to {metadata_path}")

    del policy
    torch.cuda.empty_cache()

    if mode == "inference_trace":
        return all_metrics_rows, None
    return all_metrics_rows


def main():
    parser = argparse.ArgumentParser(
        description="SmolVLA attention analysis tool for MetaWorld pick-place-v3"
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mode", type=str, default="probe", choices=["probe", "rollout", "inference_trace"])
    parser.add_argument("--query-mode", type=str, default="mean_suffix",
                        choices=["last", "mean_suffix", "mean_all"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Single model path. If not provided, uses MODEL_CONFIGS.")
    parser.add_argument("--camera-name", type=str, default=None,
                        help="Override camera name for single model.")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices, e.g. 0,3,7,11")
    parser.add_argument("--heads", type=str, default="all",
                        help="'all' or comma-separated head indices")
    parser.add_argument("--average-heads", action="store_true", default=True,
                        help="Average attention across heads (default: True)")
    parser.add_argument("--no-average-heads", action="store_true",
                        help="Do not average heads, show individual heads")
    parser.add_argument("--noise-seed", type=int, default=None,
                        help="Seed for deterministic initial noise generation (inference_trace mode).")
    parser.add_argument("--trace-heatmap-steps", type=str, default="first,mid,last",
                        help="Comma-separated denoising steps for heatmap generation, e.g. first,mid,last or 0,5,9.")
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run sanity check comparing sample_actions vs sample_actions_with_trace.")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    if args.mode == "inference_trace":
        output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_INFERENCE_TRACE_OUTPUT_DIR
    else:
        output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.layers:
        layer_indices = [int(x) for x in args.layers.split(",")]
    else:
        layer_indices = DEFAULT_LAYERS

    average_heads = args.average_heads and not args.no_average_heads

    if args.model_path:
        model_cfg = {
            "name": Path(args.model_path).parent.parent.name,
            "method": "custom",
            "camera_name": args.camera_name or "corner2,gripperPOV",
            "path": args.model_path,
        }
        models_to_process = [model_cfg]
    else:
        models_to_process = MODEL_CONFIGS

    all_rows = []
    all_model_traces = {}

    if args.mode == "inference_trace":
        noise_seed = args.noise_seed if args.noise_seed is not None else args.seed

        camera_groups = {}
        for mc in models_to_process:
            cam = mc["camera_name"]
            if cam not in camera_groups:
                camera_groups[cam] = []
            camera_groups[cam].append(mc["name"])

        for cam_group, model_names_in_group in camera_groups.items():
            print(f"\n{'#'*80}")
            print(f"# Processing camera group: {cam_group}")
            print(f"# Models: {model_names_in_group}")
            print(f"{'#'*80}")

            first_model_cfg = None
            for mc in models_to_process:
                if mc["name"] in model_names_in_group:
                    first_model_cfg = mc
                    break

            shared_noise = None
            shared_model_data = None
            shared_batch = None
            first_policy = None

            if first_model_cfg is not None:
                p = Path(first_model_cfg["path"])
                if not p.is_absolute():
                    p = PROJECT_ROOT / p
                if p.exists():
                    first_policy = SmolVLAPolicy.from_pretrained(str(p))
                    first_policy = first_policy.to(device)
                    first_policy.eval()

                    shared_model_data = create_metaworld_input(TASK, args.seed, cam_group, device)
                    shared_batch = prepare_batch_for_model(shared_model_data, first_policy, device)

                    shared_noise = create_shared_initial_noise(
                        first_policy.config.chunk_size,
                        first_policy.config.max_action_dim,
                        noise_seed,
                        device,
                    )

                    print(f"  Shared noise shape: {shared_noise.shape}")
                    print(f"  Shared noise checksum: {compute_tensor_checksum(shared_noise):.4f}")
                    print(f"  Image checksums:")
                    for img_idx, img_key in enumerate(first_policy.config.image_features):
                        images, _ = first_policy.prepare_images(shared_batch)
                        if img_idx < len(images):
                            print(f"    {img_key}: {compute_image_checksum(images[img_idx]):.4f}")

                    del first_policy
                    torch.cuda.empty_cache()

            for model_cfg in models_to_process:
                if model_cfg["name"] not in model_names_in_group:
                    continue

                result = process_single_model(
                    model_cfg, device, args.seed, args.mode, args.query_mode,
                    output_dir, layer_indices, average_heads, args.max_steps,
                    noise_seed=noise_seed,
                    heatmap_steps_str=args.trace_heatmap_steps,
                    shared_noise=shared_noise,
                    do_sanity_check=args.sanity_check,
                )

                if result is not None:
                    rows, trace_result = result
                    if rows:
                        all_rows.extend(rows)
                    if trace_result is not None:
                        all_model_traces[model_cfg["name"]] = trace_result

                del shared_model_data
                del shared_batch
                torch.cuda.empty_cache()

        if all_model_traces:
            generate_inference_trace_summary_plots(
                all_rows, all_model_traces, camera_groups, output_dir
            )

        inference_csv = output_dir / "inference_attention_trace.csv"
        if all_rows:
            fieldnames = [
                "model_name", "method", "camera", "eval_task_success",
                "denoise_index", "timestep", "layer", "attention_source",
                "query_length", "key_length", "query_count",
                "camera1_mass", "camera2_mass", "visual_total",
                "language_mass", "state_mass", "suffix_mass", "other_mass",
                "x_t_norm", "v_t_norm", "suffix_hidden_norm",
            ]
            with open(inference_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"\nSaved inference trace CSV to {inference_csv}")

            inference_json = output_dir / "inference_attention_trace.json"
            with open(inference_json, "w") as f:
                json.dump(all_rows, f, indent=2)
            print(f"Saved inference trace JSON to {inference_json}")

    else:
        for model_cfg in models_to_process:
            rows = process_single_model(
                model_cfg, device, args.seed, args.mode, args.query_mode,
                output_dir, layer_indices, average_heads, args.max_steps
            )
            if rows:
                all_rows.extend(rows)

        if all_rows:
            summary_csv = output_dir / "attention_summary.csv"
            fieldnames = [
                "model_name", "method", "camera", "seed", "phase",
                "success", "grasp_reached", "layer",
                "attention_source", "query_length", "key_length", "query_count",
                "camera1_mass", "camera2_mass", "language_mass", "state_mass",
                "suffix_mass", "other_mass"
            ]
            with open(summary_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"\nSaved summary to {summary_csv}")

            summary_json = output_dir / "attention_metrics.json"
            with open(summary_json, "w") as f:
                json.dump(all_rows, f, indent=2)
            print(f"Saved metrics JSON to {summary_json}")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()