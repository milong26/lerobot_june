import logging
import warnings
from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig

logger = logging.getLogger(__name__)


@PreTrainedConfig.register_subclass("minivla")
@dataclass
class MiniVLAConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    d_model: int = 128

    vision_encoder_path: str = ""
    language_model_path: str = ""
    tokenizer_path: str = ""
    vision_hidden_dim: int = 768
    language_hidden_dim: int = 768

    image_size: int = 224
    main_image_key: str = "observation.images.top"
    wrist_image_keys: list = field(default_factory=list)
    history_image_keys: list = field(default_factory=list)
    num_image_history: int = 2

    action_chunk_size: int = 8
    action_vocab_size: int = 256
    vq_codebook_size: int = 256
    vq_num_layers: int = 2

    state_dim: int = 0
    action_dim: int = 0

    freeze_vision_encoder: bool = True
    freeze_language_model: bool = False

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.action_chunk_size <= 0:
            raise ValueError(f"action_chunk_size must be positive, got {self.action_chunk_size}")
        if self.vq_codebook_size <= 0:
            raise ValueError(f"vq_codebook_size must be positive, got {self.vq_codebook_size}")

    def get_image_keys(self) -> list:
        keys = []
        if self.main_image_key:
            keys.append(self.main_image_key)
        for k in self.history_image_keys:
            if k and k not in keys:
                keys.append(k)
        for k in self.wrist_image_keys:
            if k and k not in keys:
                keys.append(k)
        return keys

    def validate_features(self) -> None:
        image_features = self.image_features
        if not image_features:
            raise ValueError("At least one visual input is required for MiniVLA.")
        if not self.robot_state_feature:
            raise ValueError("observation.state is required for MiniVLA.")
        if not self.action_feature:
            raise ValueError("action output is required for MiniVLA.")

        self.action_dim = self.action_feature.shape[0]
        self.state_dim = self.robot_state_feature.shape[0]

        if self.main_image_key not in image_features:
            available = [k for k in image_features if k.startswith("observation.images")]
            if available:
                self.main_image_key = sorted(available)[0]
                logger.warning(
                    f"main_image_key '{self.main_image_key}' not in config, "
                    f"selected from available: {self.main_image_key}"
                )
            else:
                raise ValueError("No observation.images.* features found in dataset.")

        actual_history = []
        for k in self.history_image_keys:
            if k in image_features:
                actual_history.append(k)
        self.history_image_keys = actual_history

        actual_wrist = []
        for k in self.wrist_image_keys:
            if k in image_features:
                actual_wrist.append(k)
        self.wrist_image_keys = actual_wrist

    def validate_vla_config(self) -> None:
        if not self.language_model_path:
            warnings.warn(
                "language_model_path is empty. A dummy backbone will be used. "
                "Set this path to load a real language model.",
                UserWarning,
            )
        if not self.vision_encoder_path:
            warnings.warn(
                "vision_encoder_path is empty. A dummy vision encoder will be used. "
                "Set this path to load a real vision encoder.",
                UserWarning,
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=1e-4,
            weight_decay=0,
            betas=(0.9, 0.999),
            eps=1e-8,
            grad_clip_norm=10.0,
        )

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> list:
        if self.num_image_history > 0:
            return list(range(-self.num_image_history, 1))
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.action_chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None