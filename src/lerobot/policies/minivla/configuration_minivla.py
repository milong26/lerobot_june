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
from pathlib import Path
from typing import Optional

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

    # === Camera keys (explicit, no guessing) ===
    primary_image_key: str = ""
    wrist_image_key: str = ""

    # === Action / VQ ===
    action_tokenizer_type: str = "extra_action_tokenizer"
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

    # === dtype for training (matches official Qwen BF16) ===
    dtype: str = "bfloat16"

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

    @property
    def is_vq_mode(self) -> bool:
        return self.action_tokenizer_type in (
            "libero_vq_extra_action_tokenizer",
            "libero_vq_action_tokenizer",
            "libero_vq_h0_extra_action_tokenizer",
            "bridge_vq_extra_action_tokenizer",
            "vq_action_tokenizer",
        )

    @property
    def action_delta_indices(self) -> list:
        if self.is_vq_mode:
            return list(range(self.chunk_size))
        return [0]

    def validate_features(self) -> None:
        image_features = self.image_features
        if not image_features:
            raise ValueError("At least one visual input is required for MiniVLA.")
        if not self.action_feature:
            raise ValueError("action output is required for MiniVLA.")

        if not self.primary_image_key:
            raise ValueError(
                "primary_image_key must be set in config to identify the primary camera. "
                "e.g. 'observation.images.cam_high'."
            )

        if self.use_wrist_image and not self.wrist_image_key:
            raise ValueError(
                "wrist_image_key must be set when use_wrist_image=True. "
                "e.g. 'observation.images.wrist' or 'observation.images.gripperPOV'."
            )

        if self.is_vq_mode:
            if not self.vq_model_path and not self.official_vla_checkpoint:
                raise ValueError(
                    "VQ mode requires either vq_model_path or official_vla_checkpoint "
                    "pointing to a directory containing a 'vq' subdirectory."
                )
            vq_path = self.resolve_vq_model_path()
            if vq_path:
                vq_dir = Path(vq_path)
                config_json = vq_dir / "config.json"
                model_pt = vq_dir / "checkpoints" / "model.pt"
                if not config_json.exists():
                    raise ValueError(f"VQ config.json not found at {config_json}")
                if not model_pt.exists():
                    raise ValueError(f"VQ checkpoint model.pt not found at {model_pt}")

            action_dim = self.action_feature.shape[0]
            if action_dim != self.vq_action_dim:
                raise ValueError(
                    f"LeRobot action dimension ({action_dim}) does not match the VQ configuration "
                    f"vq_action_dim ({self.vq_action_dim}). You must pre-train a VQ model for this "
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
    def reward_delta_indices(self) -> None:
        return None

    def resolve_vq_model_path(self) -> str:
        """Resolve VQ model path from config or official checkpoint directory."""
        if self.vq_model_path:
            return self.vq_model_path
        if self.official_vla_checkpoint:
            ckpt_dir = Path(self.official_vla_checkpoint).parent
            default_vq = ckpt_dir / "vq"
            if default_vq.exists():
                return str(default_vq)
        return ""


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