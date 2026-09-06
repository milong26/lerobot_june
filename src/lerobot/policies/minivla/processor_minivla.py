"""
processor_minivla.py

LeRobot processor pipeline for MiniVLA.
Mirrors teach_code/MiniVLA/prismatic/models/backbones/vision/dinosiglip_vit.py,
prismatic/vla/datasets/datasets.py (RLDSBatchTransform), and
prismatic/util/data_utils.py (PaddedCollatorForActionPrediction).

Key design:
  - Adds MiniVLAImageProcessorStep to convert observation.images.* -> {"dino", "siglip"}
  - LeRobot processor handles batch, device, ACTION QUANTILES normalization
  - Visual tensors are NOT pre-normalized (model applies DINO/SigLIP transforms)
  - base: single primary image
  - T2: [-1, 0] primary frames, temporal order
  - wrist: current primary + current wrist, primary first
  - task field preserved fully
  - postprocessor denormalizes actions from [-1, 1] back to original scale
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torchvision.transforms.functional as F

from lerobot.configs import NormalizationMode
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    make_default_pre_post_processors,
)

from .configuration_minivla import MiniVLAConfig


# ---------------------------------------------------------------------------
# DINO and SigLIP normalization constants (official TIMM defaults)
# ---------------------------------------------------------------------------
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)
DINO_SIZE = 224
SIGLIP_SIZE = 224


@dataclass
class MiniVLAImageProcessorStep(ProcessorStep):
    """
    Processor step that converts observation.images.* tensors to DINO/SigLIP format.
    Handles single-frame (base) and multi-frame (T2, wrist) variants.
    """
    config: MiniVLAConfig

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        pixel_values = _extract_pixel_values_for_variant(data, self.config)
        data["dino"] = pixel_values["dino"]
        data["siglip"] = pixel_values["siglip"]
        return data

    def __repr__(self) -> str:
        return (
            f"MiniVLAImageProcessorStep("
            f"seq_len={self.config.image_sequence_len}, "
            f"wrist={self.config.use_wrist_image})"
        )


def _normalize_image_tensor(
    img: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    size: int = 224,
) -> torch.Tensor:
    """
    Normalize a batch of images [B, C, H, W] for DINO/SigLIP.
    Skips re-normalization if values are already in normalized range.
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)

    img_min = img.min().item()
    img_max = img.max().item()

    if img_min < -0.5 or img_max > 3.0:
        return img

    if img.shape[-1] != size or img.shape[-2] != size:
        img = F.resize(img, size=size, interpolation=F.InterpolationMode.BICUBIC)

    img = F.normalize(img, mean=mean, std=std)
    return img


def _process_image_batch(
    images: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    size: int = 224,
) -> torch.Tensor:
    """
    Process a batch of images [B, C, H, W] or [B, T, C, H, W] for DINO/SigLIP.
    """
    if images.dim() == 4:
        return _normalize_image_tensor(images, mean, std, size)
    elif images.dim() == 5:
        b, t = images.shape[:2]
        images = images.view(b * t, *images.shape[2:])
        images = _normalize_image_tensor(images, mean, std, size)
        return images.view(b, t, *images.shape[1:])
    else:
        raise ValueError(f"Unexpected image shape: {images.shape}")


def _find_primary_image_keys(batch: dict[str, Any]) -> list[str]:
    """Find observation.images.* keys that are NOT wrist cameras."""
    keys = []
    for key in sorted(batch.keys()):
        if key.startswith("observation.images"):
            if "wrist" not in key.lower():
                keys.append(key)
    return keys


def _find_wrist_image_key(batch: dict[str, Any]) -> str | None:
    """Find observation.images.* key that contains 'wrist'."""
    for key in sorted(batch.keys()):
        if key.startswith("observation.images") and "wrist" in key.lower():
            return key
    return None


def _extract_pixel_values_for_variant(
    batch: dict[str, Any],
    config: MiniVLAConfig,
) -> dict[str, torch.Tensor]:
    """
    Extract and process images for the specific MiniVLA variant.
    Returns {"dino": tensor, "siglip": tensor}.
    """
    device = config.device

    if config.image_sequence_len == 1 and not config.use_wrist_image:
        # Base: single primary image
        primary_keys = _find_primary_image_keys(batch)
        if not primary_keys:
            raise ValueError("No observation.images.* key found in batch")

        primary_key = primary_keys[0]
        primary_img = batch[primary_key]
        if primary_img.dim() == 5:
            primary_img = primary_img.squeeze(1)

        dino = _process_image_batch(primary_img, DINO_MEAN, DINO_STD, DINO_SIZE).to(device)
        siglip = _process_image_batch(primary_img, SIGLIP_MEAN, SIGLIP_STD, SIGLIP_SIZE).to(device)

    elif config.image_sequence_len == 2 and not config.use_wrist_image:
        # T2: two primary images, temporal order old -> current
        primary_keys = _find_primary_image_keys(batch)
        if len(primary_keys) < 2:
            raise ValueError(f"T2 variant requires at least 2 primary image keys, found {len(primary_keys)}")

        old_img = batch[primary_keys[0]]
        current_img = batch[primary_keys[1]]

        if old_img.dim() == 5:
            old_img = old_img.squeeze(1)
        if current_img.dim() == 5:
            current_img = current_img.squeeze(1)

        combined = torch.stack([old_img, current_img], dim=1)

        dino = _process_image_batch(combined, DINO_MEAN, DINO_STD, DINO_SIZE).to(device)
        siglip = _process_image_batch(combined, SIGLIP_MEAN, SIGLIP_STD, SIGLIP_SIZE).to(device)

    elif config.image_sequence_len == 2 and config.use_wrist_image:
        # Wrist: primary -> wrist, fixed order primary first
        primary_keys = _find_primary_image_keys(batch)
        wrist_key = _find_wrist_image_key(batch)

        if not primary_keys or wrist_key is None:
            raise ValueError("Wrist variant requires both primary and wrist image keys")

        primary_img = batch[primary_keys[0]]
        wrist_img = batch[wrist_key]

        if primary_img.dim() == 5:
            primary_img = primary_img.squeeze(1)
        if wrist_img.dim() == 5:
            wrist_img = wrist_img.squeeze(1)

        combined = torch.stack([primary_img, wrist_img], dim=1)

        dino = _process_image_batch(combined, DINO_MEAN, DINO_STD, DINO_SIZE).to(device)
        siglip = _process_image_batch(combined, SIGLIP_MEAN, SIGLIP_STD, SIGLIP_SIZE).to(device)

    else:
        raise ValueError(
            f"Unsupported variant config: seq_len={config.image_sequence_len}, wrist={config.use_wrist_image}"
        )

    return {"dino": dino, "siglip": siglip}


def make_minivla_pre_post_processors(
    config: MiniVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Factory for base MiniVLA (single primary image)."""
    preprocessor, postprocessor = make_default_pre_post_processors(
        config, dataset_stats, normalizer_device=config.device
    )
    image_step = MiniVLAImageProcessorStep(config)
    preprocessor.steps.insert(0, image_step)
    return preprocessor, postprocessor


def make_minivla_t2_pre_post_processors(
    config: MiniVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Factory for minivla_t2 (two primary images, temporal order old -> current)."""
    preprocessor, postprocessor = make_default_pre_post_processors(
        config, dataset_stats, normalizer_device=config.device
    )
    image_step = MiniVLAImageProcessorStep(config)
    preprocessor.steps.insert(0, image_step)
    return preprocessor, postprocessor


def make_minivla_wrist_pre_post_processors(
    config: MiniVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Factory for minivla_wrist (primary + wrist, primary first)."""
    preprocessor, postprocessor = make_default_pre_post_processors(
        config, dataset_stats, normalizer_device=config.device
    )
    image_step = MiniVLAImageProcessorStep(config)
    preprocessor.steps.insert(0, image_step)
    return preprocessor, postprocessor