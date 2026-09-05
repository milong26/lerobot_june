"""
processor_minivla.py

LeRobot processor pipeline for MiniVLA.
Mirrors teach_code/MiniVLA/prismatic/models/backbones/vision/dinosiglip_vit.py,
prismatic/vla/datasets/datasets.py (RLDSBatchTransform), and
prismatic/util/data_utils.py (PaddedCollatorForActionPrediction).

Key design:
  - LeRobot processor handles batch, device, ACTION QUANTILES normalization
  - Visual tensors are NOT pre-normalized (model applies DINO/SigLIP transforms)
  - base: single primary image
  - T2: [-1, 0] primary frames, temporal order
  - wrist: current primary + current wrist, primary first
  - task field preserved fully
  - postprocessor denormalizes actions from [-1, 1] back to original scale
"""

from __future__ import annotations

from typing import Any

import torch

from lerobot.configs import NormalizationMode
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    make_default_pre_post_processors,
)

from .configuration_minivla import MiniVLAConfig


def make_minivla_pre_post_processors(
    config: MiniVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Factory for base MiniVLA (single primary image).
    ACTION uses QUANTILES normalization; VISUAL uses IDENTITY.
    """
    return make_default_pre_post_processors(
        config, dataset_stats, normalizer_device=config.device
    )


def make_minivla_t2_pre_post_processors(
    config: MiniVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Factory for minivla_t2 (two primary images, temporal order old -> current).
    """
    return make_default_pre_post_processors(
        config, dataset_stats, normalizer_device=config.device
    )


def make_minivla_wrist_pre_post_processors(
    config: MiniVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Factory for minivla_wrist (primary + wrist, primary first).
    """
    return make_default_pre_post_processors(
        config, dataset_stats, normalizer_device=config.device
    )