"""
configuration_minivla.py

MiniVLA configuration classes mirroring the official MiniVLA (PrismaticVLM + OpenVLA) setup.
Three variants are registered:
  - minivla:      single primary image, official base config
  - minivla_t2:   two primary images (temporal: old -> current), official T2 config
  - minivla_wrist: primary + wrist images, official wrist config

Reference files in teach_code/MiniVLA:
  - prismatic/conf/models.py  (Prism_Qwen25_0_5B_Extra_DINOSigLIP_224px)
  - prismatic/conf/vla.py     (Exp_Qwen25_DinoSigLIP_224px_0_5B_LIBERO_90, T2, wrist variants)
  - vq/pretrain_vq+mx-libero_90+.../config.json
"""

import logging
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig

logger = logging.getLogger(__name__)

_OFFICIAL_QWEN_BASE = "Qwen/Qwen2.5-0.5B"
_OFFICIAL_VISION_BACKBONE = "dinosiglip-vit-so-224px"
_OFFICIAL_LLM_BACKBONE = "qwen25-0_5b-extra"
_OFFICIAL_IMAGE_SIZE = 224
_OFFICIAL_IMAGE_RESIZE = "resize-naive"
_OFFICIAL_ARCH_SPECIFIER = "no-align+fused-gelu-mlp"
_OFFICIAL_NUM_EXTRA_TOKENS = 256
_OFFICIAL_CHUNK_SIZE = 8
_OFFICIAL_N_ACTION_STEPS = 1
_OFFICIAL_VQVAE_N_EMBED = 128
_OFFICIAL_VQVAE_GROUPS = 7
_OFFICIAL_N_LATENT_DIMS = 512
_OFFICIAL_VQ_ACTION_DIM = 7
_OFFICIAL_LR = 2e-5
_OFFICIAL_WEIGHT_DECAY = 0.0
_OFFICIAL_GRAD_CLIP_NORM = 1.0


def _default_normalization_mapping() -> dict[str, NormalizationMode]:
    return {
        "VISUAL": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.QUANTILES,
    }


@dataclass
class _MiniVLAConfigBase(PreTrainedConfig):
    """Shared base for all MiniVLA variants. Subclasses set variant-specific deltas."""

    # === Official backbone identifiers ===
    vision_backbone_id: str = _OFFICIAL_VISION_BACKBONE
    llm_backbone_id: str = _OFFICIAL_LLM_BACKBONE
    official_vla_checkpoint: str = ""

    # === Qwen / Tokenizer ===
    base_vlm_checkpoint: str = _OFFICIAL_QWEN_BASE
    num_extra_tokens: int = _OFFICIAL_NUM_EXTRA_TOKENS

    # === Vision ===
    image_size: int = _OFFICIAL_IMAGE_SIZE
    image_resize_strategy: str = _OFFICIAL_IMAGE_RESIZE
    arch_specifier: str = _OFFICIAL_ARCH_SPECIFIER
    image_sequence_len: int = 1
    use_wrist_image: bool = False

    # === Action / VQ ===
    chunk_size: int = _OFFICIAL_CHUNK_SIZE
    n_action_steps: int = _OFFICIAL_N_ACTION_STEPS
    vqvae_n_embed: int = _OFFICIAL_VQVAE_N_EMBED
    vqvae_groups: int = _OFFICIAL_VQVAE_GROUPS
    n_latent_dims: int = _OFFICIAL_N_LATENT_DIMS
    vq_action_dim: int = _OFFICIAL_VQ_ACTION_DIM
    vq_model_path: str = ""

    # === Training defaults (vla-full-train) ===
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision_training: bool = True
    reduce_in_full_precision: bool = True
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False

    # === Optimizer (AdamW defaults) ===
    optimizer_lr: float = _OFFICIAL_LR
    optimizer_weight_decay: float = _OFFICIAL_WEIGHT_DECAY
    optimizer_grad_clip_norm: float = _OFFICIAL_GRAD_CLIP_NORM
    optimizer_betas: tuple[float, float] = field(default_factory=lambda: (0.9, 0.999))
    optimizer_eps: float = 1e-8

    # === Scheduler ===
    scheduler_type: str = "constant"
    scheduler_warmup_ratio: float = 0.0

    # === Normalization ===
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=_default_normalization_mapping
    )

    def __post_init__(self):
        super().__post_init__()

    def validate_features(self) -> None:
        image_features = self.image_features
        if not image_features:
            raise ValueError("At least one visual input is required for MiniVLA.")
        if not self.action_feature:
            raise ValueError("action output is required for MiniVLA.")

        action_dim = self.action_feature.shape[0]
        if action_dim != self.vq_action_dim:
            raise ValueError(
                f"LeRobot action dimension ({action_dim}) does not match the VQ configuration "
                f"input_dim_w ({self.vq_action_dim}). You must pre-train a VQ model for this "
                f"dataset's action dimension. Set vq_model_path to a compatible VQ checkpoint."
            )

    def validate_vla_config(self) -> None:
        pass

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> list:
        raise NotImplementedError("Subclasses must define observation_delta_indices.")

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


# ---------------------------------------------------------------------------
# minivla (base): single current primary image
# ---------------------------------------------------------------------------
@PreTrainedConfig.register_subclass("minivla")
@dataclass
class MiniVLAConfig(_MiniVLAConfigBase):
    """Official base MiniVLA: one primary image at the current timestep."""

    image_sequence_len: int = 1
    use_wrist_image: bool = False

    @property
    def observation_delta_indices(self) -> list:
        return [0]


# ---------------------------------------------------------------------------
# minivla_t2: two primary images, temporal order old -> current
# ---------------------------------------------------------------------------
@PreTrainedConfig.register_subclass("minivla_t2")
@dataclass
class MiniVLAT2Config(_MiniVLAConfigBase):
    """Official T2 MiniVLA: two primary images ordered from old to current."""

    image_sequence_len: int = 2
    use_wrist_image: bool = False

    @property
    def observation_delta_indices(self) -> list:
        return [-1, 0]


# ---------------------------------------------------------------------------
# minivla_wrist: primary + wrist, fixed order primary first
# ---------------------------------------------------------------------------
@PreTrainedConfig.register_subclass("minivla_wrist")
@dataclass
class MiniVLAWristConfig(_MiniVLAConfigBase):
    """Official wrist MiniVLA: current primary followed by current wrist."""

    image_sequence_len: int = 2
    use_wrist_image: bool = True

    @property
    def observation_delta_indices(self) -> list:
        return [0]