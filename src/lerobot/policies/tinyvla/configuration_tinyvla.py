# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("tinyvla")
@dataclass
class TinyVLAConfig(PreTrainedConfig):
    """
    Configuration for the TinyVLA (Vision-Language-Action) policy.

    TinyVLA uses a LLaVA-Pythia VLM backbone with a DROID UNet Diffusion or DETR VAE action head.
    It supports multi-camera visual input, robot state, and language instructions.

    Args:
        n_obs_steps: Number of observation steps (currently only 1 is supported).
        chunk_size: Size of the action prediction chunk.
        n_action_steps: Number of action steps to execute per policy invocation.
        model_name_or_path: Path to the LLaVA-Pythia base model checkpoint.
        action_head_type: Type of action head, either "droid_diffusion" or "act".
        action_dim: Dimension of the action space.
        state_dim: Dimension of the robot state space.
        lora_enable: Whether to use LoRA for efficient fine-tuning.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        freeze_vision_tower: Whether to freeze the vision encoder.
        freeze_backbone: Whether to freeze the language model backbone.
        tokenizer_max_length: Maximum tokenizer length for language input.
        num_inference_timesteps: Number of DDIM inference steps for diffusion head.
        pretrain_image_size: Image size for pretrained checkpoint.
        optimizer_lr: Learning rate.
        optimizer_weight_decay: Weight decay.
        scheduler_warmup_steps: Number of warmup steps for scheduler.
        scheduler_decay_steps: Number of decay steps for scheduler.
        scheduler_decay_lr: Final decayed learning rate.
    """

    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Model architecture
    model_name_or_path: str = "lesjie/Llava-Pythia-400M"
    action_head_type: str = "droid_diffusion"
    action_dim: int = 10
    state_dim: int = 7

    # LoRA settings
    lora_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 256
    lora_dropout: float = 0.05

    # Freezing
    freeze_vision_tower: bool = False
    freeze_backbone: bool = False

    # Tokenizer
    tokenizer_max_length: int = 2048

    # Diffusion inference
    num_inference_timesteps: int = 10

    # Image
    pretrain_image_size: int = 320

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-10
    optimizer_betas: tuple[float, float] = (0.9, 0.98)
    optimizer_eps: float = 1e-7
    optimizer_grad_clip_norm: float = 10.0

    scheduler_warmup_steps: int = 100
    scheduler_decay_steps: int = 10000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. "
                f"Got {self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.action_head_type not in ["droid_diffusion", "act"]:
            raise ValueError(
                f"`action_head_type` must be 'droid_diffusion' or 'act'. Got '{self.action_head_type}'."
            )

    def validate_features(self) -> None:
        """Validate and infer features from dataset/environment.

        This method is called by make_policy after input_features and output_features
        have been populated from the dataset metadata or environment config.
        It infers action_dim and state_dim from these features.
        """
        if not self.image_features:
            raise ValueError("At least one image input is required for TinyVLA.")
        if not self.action_feature:
            raise ValueError("An action output is required for TinyVLA.")

        # Infer action_dim from output_features
        action_shape = self.action_feature.shape
        self.action_dim = action_shape[0] if action_shape else self.action_dim

        # Infer state_dim from input_features if available
        if OBS_STATE in self.input_features:
            state_shape = self.input_features[OBS_STATE].shape
            self.state_dim = state_shape[0] if state_shape else self.state_dim
        else:
            # No state input, create a dummy feature with state_dim=0
            self.state_dim = 0

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None