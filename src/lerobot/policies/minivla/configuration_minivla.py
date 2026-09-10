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

import json
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
_OFFICIAL_LR = 2e-5
_OFFICIAL_WEIGHT_DECAY = 0.0
_OFFICIAL_GRAD_CLIP_NORM = 1.0

ACTION_TOKENIZERS = {
    "action_tokenizer",
    "extra_action_tokenizer",
    "libero_vq_action_tokenizer",
    "libero_vq_extra_action_tokenizer",
    "libero_vq_h0_extra_action_tokenizer",
    "bridge_vq_extra_action_tokenizer",
    "vq_action_tokenizer",
}

VQ_TOKENIZER_TYPES = {
    "libero_vq_action_tokenizer",
    "libero_vq_extra_action_tokenizer",
    "libero_vq_h0_extra_action_tokenizer",
    "bridge_vq_extra_action_tokenizer",
    "vq_action_tokenizer",
}


def _default_normalization_mapping() -> dict[str, NormalizationMode]:
    """
    Official MiniVLA normalization mapping.
    ACTION uses QUANTILE normalization (q01/q99) matching official dataset_statistics.json.
    VISUAL uses IDENTITY (no normalization, only DINO/SigLIP transform).
    """
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
    vq_model_path: str = ""

    # === Official statistics & unnorm key ===
    official_dataset_statistics_path: str = ""
    action_unnorm_key: str = ""

    # === Runtime-resolved tokenizer type (set by modeling after reading official config) ===
    _resolved_action_tokenizer_type: str = ""

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

    # === Per-component learning rates ===
    # projector_lr: if > 0, overrides optimizer_lr for projector params
    # backbone_lr: if > 0, overrides optimizer_lr for vision/LLM backbone params
    projector_lr: float = 0.0
    backbone_lr: float = 0.0

    # === Scheduler ===
    scheduler_type: str = "constant"
    scheduler_warmup_ratio: float = 0.0
    # Absolute warmup steps (overrides warmup_ratio when > 0)
    scheduler_warmup_steps: int = 0

    # === Normalization ===
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=_default_normalization_mapping
    )

    def __post_init__(self):
        super().__post_init__()

    @property
    def is_vq_mode(self) -> bool:
        tok_type = self._resolved_action_tokenizer_type or self.action_tokenizer_type
        return tok_type in VQ_TOKENIZER_TYPES

    @property
    def action_delta_indices(self) -> list:
        if self.is_vq_mode:
            vq_path = self.resolve_vq_model_path()
            if vq_path:
                vq_config_json = Path(vq_path) / "config.json"
                if vq_config_json.exists():
                    with open(vq_config_json, "r") as f:
                        vq_cfg = json.load(f)
                    input_dim_h = vq_cfg.get("input_dim_h", self.chunk_size)
                    return list(range(input_dim_h))
            return list(range(self.chunk_size))
        return [0]

    def resolve_camera_keys(self, dataset_meta=None) -> None:
        """
        Resolve primary_image_key and wrist_image_key.
        Strictly follows MiniVLA official camera mapping:
        - Primary image: must be explicitly set or match known primary camera patterns
        - Wrist image: must be explicitly set when use_wrist_image=True
        NO automatic guessing of camera order. If dataset_meta exists, validate keys
        against available observation.images.* keys with explicit error messages.
        """
        image_keys = []
        if dataset_meta is not None:
            for k, v in dataset_meta.features.items():
                if k.startswith("observation.images."):
                    image_keys.append(k)

        if not self.primary_image_key:
            if dataset_meta is not None and image_keys:
                raise ValueError(
                    f"primary_image_key is not set. Available image keys in dataset: {image_keys}. "
                    f"MiniVLA requires explicit primary_image_key configuration. "
                    f"Please set primary_image_key to one of the available keys (e.g., 'observation.images.cam_high')."
                )
            else:
                raise ValueError(
                    "primary_image_key must be set explicitly. "
                    f"Available image keys: {image_keys}. "
                    f"e.g. 'observation.images.cam_high'."
                )

        if self.use_wrist_image and not self.wrist_image_key:
            if dataset_meta is not None:
                raise ValueError(
                    f"use_wrist_image=True but wrist_image_key is not set. "
                    f"Available image keys: {image_keys}. "
                    f"Please set wrist_image_key explicitly (e.g., 'observation.images.wrist' or 'observation.images.gripperPOV')."
                )
            else:
                raise ValueError(
                    "wrist_image_key must be set when use_wrist_image=True. "
                    f"Available image keys: {image_keys}. "
                    "e.g. 'observation.images.wrist' or 'observation.images.gripperPOV'."
                )

    def validate_features(self) -> None:
        """
        Validate feature configuration against official MiniVLA T2/wrist variant requirements.
        Checks image_sequence_len, use_wrist_image, observation_delta_indices consistency.
        """
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

        if self.image_sequence_len == 2 and not self.use_wrist_image:
            delta = self.observation_delta_indices
            if delta != [-1, 0]:
                raise ValueError(
                    f"T2 variant (image_sequence_len=2, use_wrist_image=False) requires "
                    f"observation_delta_indices=[-1, 0], got {delta}."
                )

        if self.image_sequence_len == 2 and self.use_wrist_image:
            delta = self.observation_delta_indices
            if delta != [0]:
                raise ValueError(
                    f"Wrist variant (image_sequence_len=2, use_wrist_image=True) requires "
                    f"observation_delta_indices=[0], got {delta}."
                )

        if self.image_sequence_len == 1 and not self.use_wrist_image:
            delta = self.observation_delta_indices
            if delta != [0]:
                raise ValueError(
                    f"Base variant (image_sequence_len=1, use_wrist_image=False) requires "
                    f"observation_delta_indices=[0], got {delta}."
                )

        if self.is_vq_mode:
            vq_path = self.resolve_vq_model_path()
            if not vq_path:
                raise ValueError(
                    "VQ mode requires either vq_model_path or official_vla_checkpoint "
                    "pointing to a directory containing a valid VQ config.json and checkpoints/model.pt."
                )
            vq_dir = Path(vq_path)
            config_json = vq_dir / "config.json"
            model_pt = vq_dir / "checkpoints" / "model.pt"
            if not config_json.exists():
                raise ValueError(f"VQ config.json not found at {config_json}")
            if not model_pt.exists():
                raise ValueError(f"VQ checkpoint model.pt not found at {model_pt}")

            with open(config_json, "r") as f:
                vq_cfg = json.load(f)

            vq_action_dim = vq_cfg.get("input_dim_w", None)
            vq_input_dim_h = vq_cfg.get("input_dim_h", None)
            vq_vqvae_groups = vq_cfg.get("vqvae_groups", None)
            vq_vqvae_n_embed = vq_cfg.get("vqvae_n_embed", None)

            if vq_action_dim is None:
                raise ValueError(f"VQ config.json missing 'input_dim_w'")

            action_dim = self.action_feature.shape[0]
            if action_dim != vq_action_dim:
                raise ValueError(
                    f"LeRobot action dimension ({action_dim}) does not match the VQ configuration "
                    f"input_dim_w ({vq_action_dim}). You must pre-train a VQ model for this "
                    f"dataset's action dimension. Set vq_model_path to a compatible VQ checkpoint."
                )

            if self.vqvae_n_embed != vq_vqvae_n_embed:
                logger.warning(
                    f"Config vqvae_n_embed ({self.vqvae_n_embed}) != VQ config vqvae_n_embed ({vq_vqvae_n_embed}). "
                    f"Using VQ config value."
                )
            if self.vqvae_groups != vq_vqvae_groups:
                logger.warning(
                    f"Config vqvae_groups ({self.vqvae_groups}) != VQ config vqvae_groups ({vq_vqvae_groups}). "
                    f"Using VQ config value."
                )

            delta_indices = self.action_delta_indices
            if vq_input_dim_h is not None and len(delta_indices) != vq_input_dim_h:
                raise ValueError(
                    f"VQ input_dim_h ({vq_input_dim_h}) != len(action_delta_indices) ({len(delta_indices)}). "
                    f"This indicates a mismatch between the VQ model and the config. "
                    f"Ensure chunk_size and observation_delta_indices match the VQ model's temporal horizon."
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
        """
        Resolve VQ model path with strict priority:
        1. Explicit vq_model_path
        2. Official config.json action_tokenizer -> ACTION_TOKENIZERS path
        3. official_vla_checkpoint所在run目录
        4. teach_code/MiniVLA根目录
        Must return a directory containing config.json and checkpoints/model.pt.
        """
        if self.vq_model_path:
            candidate = Path(self.vq_model_path)
            if candidate.is_absolute() and (candidate / "config.json").exists():
                return str(candidate)

        if self.official_vla_checkpoint:
            ckpt_path = Path(self.official_vla_checkpoint)
            if ckpt_path.suffix == ".pt" and ckpt_path.parent.name == "checkpoints":
                run_dir = ckpt_path.parents[1]
                config_json = run_dir / "config.json"
                if config_json.exists():
                    with open(config_json, "r") as f:
                        vla_cfg = json.load(f).get("vla", {})
                    atype = vla_cfg.get("action_tokenizer", "")
                    if atype in ("libero_vq_action_tokenizer", "libero_vq_extra_action_tokenizer",
                                 "libero_vq_h0_extra_action_tokenizer", "bridge_vq_extra_action_tokenizer"):
                        from .vq_action import ACTION_TOKENIZERS as _AT
                        if atype in _AT:
                            partial_fn = _AT[atype]
                            rel_path = partial_fn.keywords.get("vq_vae_path", "")
                            if rel_path:
                                teach_root = Path(__file__).parent.parent.parent.parent.parent / "teach_code" / "MiniVLA"
                                candidate = teach_root / rel_path
                                if (candidate / "config.json").exists():
                                    return str(candidate)

                vq_dir = run_dir / "vq"
                if (vq_dir / "config.json").exists() and (vq_dir / "checkpoints" / "model.pt").exists():
                    return str(vq_dir)

        teach_root = Path(__file__).parent.parent.parent.parent.parent / "teach_code" / "MiniVLA"
        if teach_root.exists():
            for vq_sub in teach_root.glob("vq/*/"):
                if (vq_sub / "config.json").exists() and (vq_sub / "checkpoints" / "model.pt").exists():
                    return str(vq_sub)

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