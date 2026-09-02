from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("minivla")
@dataclass
class MiniVLAConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    d_model: int = 128
    d_word: int = 64
    diffusion_T: int = 16
    beta_start: float = 1e-4
    beta_end: float = 1e-2
    time_emb_dim: int = 32
    diffusion_hidden_dim: int = 128
    image_size: int = 64
    image_key: str = "observation.images.top"
    text_key: str = "task"
    max_text_length: int = 32
    state_dim: int = 0
    action_dim: int = 0
    vocab: dict = field(default_factory=lambda: {"<pad>": 0, "<unk>": 1})
    task_texts: list = field(default_factory=list)
    default_instruction: str = ""
    optimizer_lr: float = 1e-4

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.diffusion_T <= 0:
            raise ValueError(f"diffusion_T must be positive, got {self.diffusion_T}")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.max_text_length <= 0:
            raise ValueError(f"max_text_length must be positive, got {self.max_text_length}")

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("At least one visual input is required for MiniVLA.")
        if not self.robot_state_feature:
            raise ValueError("observation.state is required for MiniVLA.")
        if not self.action_feature:
            raise ValueError("action output is required for MiniVLA.")

        self.action_dim = self.action_feature.shape[0]
        self.state_dim = self.robot_state_feature.shape[0]

        image_features = self.image_features
        if self.image_key not in image_features:
            if OBS_IMAGES in image_features and "observation.images.top" in image_features:
                self.image_key = "observation.images.top"
            else:
                keys = sorted(image_features.keys())
                self.image_key = keys[0]

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=0,
            betas=(0.9, 0.999),
            eps=1e-8,
            grad_clip_norm=10.0,
        )

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return [0]

    @property
    def reward_delta_indices(self) -> None:
        return None