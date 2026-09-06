"""
processor_minivla.py

LeRobot processor pipeline for MiniVLA.
Mirrors teach_code/MiniVLA/prismatic/models/backbones/vision/dinosiglip_vit.py,
prismatic/vla/datasets/datasets.py (RLDSBatchTransform), and
prismatic/util/data_utils.py (PaddedCollatorForActionPrediction).

Key design:
  - MiniVLAImageProcessorStep registered via @ProcessorStepRegistry
  - Uses make_default_policy_processor_steps() and make_policy_processor_pipelines()
    following src/lerobot/policies/smolvla/processor_smolvla.py pattern
  - Processor order: rename_observations -> add_batch_dim -> to_device -> normalize -> MiniVLAImageProcessorStep
  - Base: single primary image via explicit primary_image_key
  - T2: [-1, 0] frames from the same primary camera time dimension
  - Wrist: primary -> wrist via explicit wrist_image_key (compatible with gripperPOV naming)
  - No guessing of normalization state; always uint8->float/255 then DINO/SigLIP norm
  - F.resize to [224,224] with TIMM interpolation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF

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


# ---------------------------------------------------------------------------
# DINO and SigLIP normalization constants (official TIMM defaults)
# ---------------------------------------------------------------------------
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)
DINO_SIZE = 224
SIGLIP_SIZE = 224


@ProcessorStepRegistry.register("minivla_image_processor")
@dataclass
class MiniVLAImageProcessorStep(ProcessorStep):
    """
    Processor step that converts observation images to DINO/SigLIP format.
    Handles single-frame (base), multi-frame (T2), and wrist variants.
    """

    primary_image_key: str = ""
    wrist_image_key: str = ""
    image_sequence_len: int = 1
    use_wrist_image: bool = False

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

        dino = self._normalize_and_resize(img, DINO_MEAN, DINO_STD, DINO_SIZE)
        siglip = self._normalize_and_resize(img, SIGLIP_MEAN, SIGLIP_STD, SIGLIP_SIZE)
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
            b, c, h, w = img.shape
            old_frame = img
            current_frame = img
        else:
            raise ValueError(f"Unexpected image shape for T2: {img.shape}")

        combined = torch.stack([old_frame, current_frame], dim=1)

        dino = self._normalize_and_resize(combined, DINO_MEAN, DINO_STD, DINO_SIZE)
        siglip = self._normalize_and_resize(combined, SIGLIP_MEAN, SIGLIP_STD, SIGLIP_SIZE)
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

        dino = self._normalize_and_resize(combined, DINO_MEAN, DINO_STD, DINO_SIZE)
        siglip = self._normalize_and_resize(combined, SIGLIP_MEAN, SIGLIP_STD, SIGLIP_SIZE)
        return {"dino": dino, "siglip": siglip}

    def _ensure_float_01(self, img: torch.Tensor) -> torch.Tensor:
        if img.dtype == torch.uint8:
            return img.float() / 255.0
        if img.is_floating_point():
            vmin = img.min().item()
            vmax = img.max().item()
            if vmin < 0 or vmax > 1.0:
                raise ValueError(
                    f"Float image values outside [0,1] range: min={vmin}, max={vmax}. "
                    f"Expected uint8 or float [0,1] input."
                )
            return img
        return img.float()

    def _to_4d(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 3:
            return img.unsqueeze(0)
        if img.dim() == 5:
            return img.squeeze(1)
        return img

    def _normalize_and_resize(
        self,
        img: torch.Tensor,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        size: int,
    ) -> torch.Tensor:
        if img.dim() == 4:
            b, c, h, w = img.shape
            if h != size or w != size:
                img = F.interpolate(img, size=[size, size], mode="bilinear", align_corners=False)
            img = TVF.normalize(img, mean=mean, std=std)
            return img
        elif img.dim() == 5:
            b, t, c, h, w = img.shape
            img = img.view(b * t, c, h, w)
            if h != size or w != size:
                img = F.interpolate(img, size=[size, size], mode="bilinear", align_corners=False)
            img = TVF.normalize(img, mean=mean, std=std)
            return img.view(b, t, c, size, size)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

    def get_config(self) -> dict[str, Any]:
        return {
            "primary_image_key": self.primary_image_key,
            "wrist_image_key": self.wrist_image_key,
            "image_sequence_len": self.image_sequence_len,
            "use_wrist_image": self.use_wrist_image,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features = {ft: dict(feats) for ft, feats in features.items()}
        obs_features = new_features.setdefault(PipelineFeatureType.OBSERVATION, {})
        obs_features["dino"] = PolicyFeature(
            type=NormalizationMode.IDENTITY,
            shape=(self.image_sequence_len * 1024,),
        )
        obs_features["siglip"] = PolicyFeature(
            type=NormalizationMode.IDENTITY,
            shape=(self.image_sequence_len * 1024,),
        )
        return new_features


def make_minivla_pre_post_processors(
    config: MiniVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    steps = make_default_policy_processor_steps(config, dataset_stats)

    image_step = MiniVLAImageProcessorStep(
        primary_image_key=config.primary_image_key,
        wrist_image_key=config.wrist_image_key,
        image_sequence_len=config.image_sequence_len,
        use_wrist_image=config.use_wrist_image,
    )

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.to_device,
        steps.normalize,
        image_step,
    ]
    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


def make_minivla_t2_pre_post_processors(
    config: MiniVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    steps = make_default_policy_processor_steps(config, dataset_stats)

    image_step = MiniVLAImageProcessorStep(
        primary_image_key=config.primary_image_key,
        wrist_image_key=config.wrist_image_key,
        image_sequence_len=config.image_sequence_len,
        use_wrist_image=config.use_wrist_image,
    )

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.to_device,
        steps.normalize,
        image_step,
    ]
    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


def make_minivla_wrist_pre_post_processors(
    config: MiniVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    steps = make_default_policy_processor_steps(config, dataset_stats)

    image_step = MiniVLAImageProcessorStep(
        primary_image_key=config.primary_image_key,
        wrist_image_key=config.wrist_image_key,
        image_sequence_len=config.image_sequence_len,
        use_wrist_image=config.use_wrist_image,
    )

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.to_device,
        steps.normalize,
        image_step,
    ]
    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)