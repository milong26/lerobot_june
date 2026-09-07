"""
processor_minivla.py

LeRobot processor pipeline for MiniVLA.
Mirrors teach_code/MiniVLA/prismatic/models/backbones/vision/dinosiglip_vit.py,
prismatic/models/backbones/vision/base_vision.py (VisionBackbone.get_image_transform),
prismatic/vla/datasets/datasets.py (RLDSBatchTransform), and
prismatic/util/data_utils.py (PaddedCollatorForActionPrediction).

Key design:
  - MiniVLAImageProcessorStep registered via @ProcessorStepRegistry
  - Uses make_default_policy_processor_steps() and make_policy_processor_pipelines()
    following src/lerobot/policies/smolvla/processor_smolvla.py pattern
  - Processor order: rename_observations -> add_batch_dim -> MiniVLAImageProcessorStep -> to_device -> normalize
  - Image transform on CPU BEFORE normalize; normalize only handles LeRobot action/state
  - Base: single primary image via explicit primary_image_key
  - T2: [-1, 0] frames from the same primary camera time dimension
  - Wrist: primary -> wrist via explicit wrist_image_key (compatible with gripperPOV naming)
  - REUSES encoders.py build_dinosiglip_image_transform() (no model weights)
  - uint8->float/255 conversion handled before official transform
  - Tensor path uses apply_tensor_image() (no ToTensor double-application)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import NormalizationMode
from lerobot.processor import (
    EnvTransition,
    PipelineFeatureType,
    PolicyAction,
    PolicyFeature,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_minivla import MiniVLAConfig
from .encoders import build_dinosiglip_image_transform, DinoSigLIPImageTransform


TARGET_SIZE = 224


def _load_official_stats(config: MiniVLAConfig) -> dict | None:
    """Load official dataset_statistics.json if available."""
    stats_path = config.official_dataset_statistics_path
    if stats_path and Path(stats_path).exists():
        with open(stats_path, "r") as f:
            return json.load(f)

    if config.official_vla_checkpoint:
        ckpt_path = Path(config.official_vla_checkpoint)
        if ckpt_path.suffix == ".pt" and ckpt_path.parent.name == "checkpoints":
            run_dir = ckpt_path.parents[1]
            stats_json = run_dir / "dataset_statistics.json"
            if stats_json.exists():
                with open(stats_json, "r") as f:
                    return json.load(f)
    return None


def _convert_official_stats_to_leformat(
    official_stats: dict, action_unnorm_key: str
) -> dict[str, dict[str, torch.Tensor]] | None:
    """
    Convert official dataset_statistics.json to LeRobot ACTION QUANTILES format.
    Only postprocessor uses this for unnormalization.
    """
    if not official_stats:
        return None

    if action_unnorm_key:
        if action_unnorm_key not in official_stats:
            raise ValueError(
                f"action_unnorm_key '{action_unnorm_key}' not found in official statistics. "
                f"Available keys: {list(official_stats.keys())}"
            )
        keys_to_use = [action_unnorm_key]
    elif len(official_stats) == 1:
        keys_to_use = list(official_stats.keys())
    else:
        raise ValueError(
            f"Official statistics contain multiple dataset keys {list(official_stats.keys())}. "
            f"Please set action_unnorm_key explicitly to choose which statistics to use."
        )

    result = {}
    for key in keys_to_use:
        stats = official_stats[key]
        if "action" in stats and "q01" in stats["action"]:
            action_stats = stats["action"]
            result[key] = {
                "action": {
                    "q01": torch.tensor(action_stats["q01"], dtype=torch.float32),
                    "q99": torch.tensor(action_stats["q99"], dtype=torch.float32),
                    "mask": torch.tensor(
                        action_stats.get("mask", [True] * len(action_stats["q01"])),
                        dtype=torch.bool,
                    ),
                }
            }
    return result if result else None


@ProcessorStepRegistry.register("minivla_image_processor")
@dataclass
class MiniVLAImageProcessorStep(ProcessorStep):
    """
    Processor step that converts observation images to DINO/SigLIP format.
    Handles single-frame (base), multi-frame (T2), and wrist variants.
    REUSES encoders.py build_dinosiglip_image_transform() (no model weights).
    """

    primary_image_key: str = ""
    wrist_image_key: str = ""
    image_sequence_len: int = 1
    use_wrist_image: bool = False
    vision_backbone_id: str = "dinosiglip-vit-so-224px"
    image_resize_strategy: str = "resize-naive"
    image_size: int = 224

    def __post_init__(self):
        self._image_transform: DinoSigLIPImageTransform = build_dinosiglip_image_transform(
            vision_backbone_id=self.vision_backbone_id,
            image_resize_strategy=self.image_resize_strategy,
            image_size=self.image_size,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION)
        if obs is None:
            return transition

        pixel_values = self._extract_pixel_values(obs)

        new_obs = dict(obs)
        new_obs["dino"] = pixel_values["dino"]
        new_obs["siglip"] = pixel_values["siglip"]

        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = new_obs
        return new_transition

    def _extract_pixel_values(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self.use_wrist_image:
            return self._process_wrist(obs)
        elif self.image_sequence_len == 2:
            return self._process_t2(obs)
        else:
            return self._process_base(obs)

    def _process_base(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        primary_key = self.primary_image_key
        if primary_key not in obs:
            raise ValueError(
                f"Primary image key '{primary_key}' not found in observation. "
                f"Available keys: {[k for k in obs if k.startswith('observation.images')]}"
            )
        img = obs[primary_key]
        img = self._ensure_float_01(img)
        img = self._to_4d(img)

        dino = self._apply_tensor_transform(img, "dino")
        siglip = self._apply_tensor_transform(img, "siglip")
        return {"dino": dino, "siglip": siglip}

    def _process_t2(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        primary_key = self.primary_image_key
        if primary_key not in obs:
            raise ValueError(
                f"T2 variant: primary image key '{primary_key}' not found in observation."
            )
        img = obs[primary_key]
        img = self._ensure_float_01(img)

        if img.dim() == 5:
            b, t, c, h, w = img.shape
            if t < 2:
                raise ValueError(
                    f"T2 variant requires at least 2 time frames, got {t} for key '{primary_key}'"
                )
            old_frame = img[:, -2]
            current_frame = img[:, -1]
        elif img.dim() == 4:
            raise ValueError(
                f"T2 variant requires [B, T, C, H, W] with T>=2, got [B, C, H, W]. "
                f"Cannot copy the same frame to fake temporal dimension."
            )
        else:
            raise ValueError(f"Unexpected image shape for T2: {img.shape}")

        combined = torch.stack([old_frame, current_frame], dim=1)

        dino = self._apply_tensor_transform_multi(combined, "dino")
        siglip = self._apply_tensor_transform_multi(combined, "siglip")
        return {"dino": dino, "siglip": siglip}

    def _process_wrist(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        primary_key = self.primary_image_key
        wrist_key = self.wrist_image_key

        if primary_key not in obs:
            raise ValueError(
                f"Wrist variant: primary image key '{primary_key}' not found in observation."
            )
        if wrist_key not in obs:
            raise ValueError(
                f"Wrist variant: wrist image key '{wrist_key}' not found in observation. "
                f"Available keys: {[k for k in obs if k.startswith('observation.images')]}"
            )

        primary_img = obs[primary_key]
        wrist_img = obs[wrist_key]

        primary_img = self._ensure_float_01(primary_img)
        wrist_img = self._ensure_float_01(wrist_img)

        primary_img = self._to_4d(primary_img)
        wrist_img = self._to_4d(wrist_img)

        combined = torch.stack([primary_img, wrist_img], dim=1)

        dino = self._apply_tensor_transform_multi(combined, "dino")
        siglip = self._apply_tensor_transform_multi(combined, "siglip")
        return {"dino": dino, "siglip": siglip}

    def _ensure_float_01(self, img: torch.Tensor) -> torch.Tensor:
        """Convert uint8 to float [0,1], validate float input is in [0,1]."""
        if img.dtype == torch.uint8:
            return img.float() / 255.0
        if img.is_floating_point():
            if img.min() < -0.01 or img.max() > 1.01:
                raise ValueError(
                    f"Input image tensor is already floating-point but has range [{img.min():.4f}, {img.max():.4f}]. "
                    f"Expected [0, 1] for raw images. If the image has already been DINO/SigLIP normalized, "
                    f"do not pass it through the MiniVLAImageProcessorStep again."
                )
            return img
        return img.float()

    def _to_4d(self, img: torch.Tensor) -> torch.Tensor:
        """Ensure image is 4D: (B, C, H, W)."""
        if img.dim() == 3:
            return img.unsqueeze(0)
        if img.dim() == 5:
            return img.squeeze(1)
        return img

    def _apply_tensor_transform(
        self,
        img: torch.Tensor,
        branch: str,
    ) -> torch.Tensor:
        """
        Apply tensor-safe transform from encoders.py DinoSigLIPImageTransform.apply_tensor_image().
        Does NOT pass Tensor through ToTensor (no-op for tensors in TIMM Compose).
        Handles batch dimension by iterating.
        """
        if img.dim() == 4:
            b, c, h, w = img.shape
            results = []
            for i in range(b):
                single_img = img[i]
                transform_obj = self._image_transform
                transformed = transform_obj.apply_tensor_image(single_img)
                results.append(transformed[branch])
            return torch.stack(results, dim=0)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

    def _apply_tensor_transform_multi(
        self,
        img: torch.Tensor,
        branch: str,
    ) -> torch.Tensor:
        """Apply tensor-safe transform to multi-frame images (B, T, C, H, W)."""
        if img.dim() == 5:
            b, t, c, h, w = img.shape
            img_flat = img.view(b * t, c, h, w)
            results = []
            for i in range(b * t):
                single_img = img_flat[i]
                transform_obj = self._image_transform
                transformed = transform_obj.apply_tensor_image(single_img)
                results.append(transformed[branch])
            stacked = torch.stack(results, dim=0)
            return stacked.view(b, t, results[0].shape[0], self.image_size, self.image_size)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

    def get_config(self) -> dict[str, Any]:
        return {
            "primary_image_key": self.primary_image_key,
            "wrist_image_key": self.wrist_image_key,
            "image_sequence_len": self.image_sequence_len,
            "use_wrist_image": self.use_wrist_image,
            "vision_backbone_id": self.vision_backbone_id,
            "image_resize_strategy": self.image_resize_strategy,
            "image_size": self.image_size,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Define output features for dino and siglip tensors.
        Shape matches actual pixel tensor dimensions:
        - Base: (B, C, 224, 224) -> shape (C, 224, 224)
        - T2/Wrist: (B, T, C, 224, 224) -> shape (T, C, 224, 224)
        """
        new_features = {ft: dict(feats) for ft, feats in features.items()}
        obs_features = new_features.setdefault(PipelineFeatureType.OBSERVATION, {})

        if self.image_sequence_len == 1:
            shape = (3, self.image_size, self.image_size)
        else:
            shape = (self.image_sequence_len, 3, self.image_size, self.image_size)

        obs_features["dino"] = PolicyFeature(
            type=NormalizationMode.IDENTITY,
            shape=shape,
        )
        obs_features["siglip"] = PolicyFeature(
            type=NormalizationMode.IDENTITY,
            shape=shape,
        )
        return new_features


def _build_dataset_stats_for_processor(
    config: MiniVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, dict[str, torch.Tensor]] | None:
    """
    Determine which stats to use for normalization.
    Priority: LeRobot dataset_stats > official stats via config.official_dataset_statistics_path.
    """
    if dataset_stats:
        return dataset_stats

    official = _load_official_stats(config)
    if official:
        unnorm_key = config.action_unnorm_key or ""
        return _convert_official_stats_to_leformat(official, unnorm_key)

    return None


def make_minivla_pre_post_processors(
    config: MiniVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    config.resolve_camera_keys(dataset_meta)

    proc_stats = _build_dataset_stats_for_processor(config, dataset_stats)
    steps = make_default_policy_processor_steps(config, proc_stats)

    image_step = MiniVLAImageProcessorStep(
        primary_image_key=config.primary_image_key,
        wrist_image_key=config.wrist_image_key,
        image_sequence_len=config.image_sequence_len,
        use_wrist_image=config.use_wrist_image,
        vision_backbone_id=config.vision_backbone_id,
        image_resize_strategy=config.image_resize_strategy,
        image_size=config.image_size,
    )

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        image_step,
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


def make_minivla_t2_pre_post_processors(
    config: MiniVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    config.resolve_camera_keys(dataset_meta)

    proc_stats = _build_dataset_stats_for_processor(config, dataset_stats)
    steps = make_default_policy_processor_steps(config, proc_stats)

    image_step = MiniVLAImageProcessorStep(
        primary_image_key=config.primary_image_key,
        wrist_image_key=config.wrist_image_key,
        image_sequence_len=config.image_sequence_len,
        use_wrist_image=config.use_wrist_image,
        vision_backbone_id=config.vision_backbone_id,
        image_resize_strategy=config.image_resize_strategy,
        image_size=config.image_size,
    )

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        image_step,
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


def make_minivla_wrist_pre_post_processors(
    config: MiniVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    config.resolve_camera_keys(dataset_meta)

    proc_stats = _build_dataset_stats_for_processor(config, dataset_stats)
    steps = make_default_policy_processor_steps(config, proc_stats)

    image_step = MiniVLAImageProcessorStep(
        primary_image_key=config.primary_image_key,
        wrist_image_key=config.wrist_image_key,
        image_sequence_len=config.image_sequence_len,
        use_wrist_image=config.use_wrist_image,
        vision_backbone_id=config.vision_backbone_id,
        image_resize_strategy=config.image_resize_strategy,
        image_size=config.image_size,
    )

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        image_step,
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)