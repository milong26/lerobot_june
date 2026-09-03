import logging
import warnings

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VisionEncoderWrapper(nn.Module):
    def __init__(self, vision_encoder_path: str = "", image_size: int = 224,
                 hidden_dim: int = 1024, freeze: bool = True):
        super().__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.freeze = freeze

        if vision_encoder_path:
            try:
                from transformers import AutoModel
                self.encoder = AutoModel.from_pretrained(
                    vision_encoder_path, local_files_only=True
                )
                if hasattr(self.encoder, "vision_model"):
                    self.encoder = self.encoder.vision_model
                if hasattr(self.encoder, "encoder"):
                    self.encoder = self.encoder.encoder
                logger.info(f"Loaded vision encoder from {vision_encoder_path}")
            except Exception as e:
                warnings.warn(
                    f"Failed to load vision encoder from {vision_encoder_path}: {e}. "
                    "Falling back to dummy encoder."
                )
                self.encoder = None
        else:
            self.encoder = None

        if self.encoder is None:
            self._build_dummy_encoder()

        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _build_dummy_encoder(self):
        self.dummy_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
            nn.LayerNorm([64, self.image_size // 4, self.image_size // 4]),
            nn.GELU(),
            nn.Conv2d(64, self.hidden_dim, kernel_size=3, stride=2, padding=1),
        )
        self.encoder = self.dummy_conv
        self._is_dummy = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_is_dummy", False) or self.encoder is self.dummy_conv:
            x = self.dummy_conv(x)
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            return x
        else:
            outputs = self.encoder(x)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            if isinstance(outputs, torch.Tensor):
                if outputs.dim() == 3:
                    return outputs
                b, c, h, w = outputs.shape
                return outputs.permute(0, 2, 3, 1).reshape(b, h * w, c)
            if isinstance(outputs, (list, tuple)):
                last = outputs[-1]
                if last.dim() == 3:
                    return last
                b, c, h, w = last.shape
                return last.permute(0, 2, 3, 1).reshape(b, h * w, c)
            raise ValueError(f"Unexpected encoder output type: {type(outputs)}")


class MultiImageVisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = VisionEncoderWrapper(
            vision_encoder_path=config.vision_encoder_path,
            image_size=config.image_size,
            hidden_dim=config.vision_hidden_dim,
            freeze=config.freeze_vision_encoder,
        )
        self.config = config

    def forward(self, images: list[torch.Tensor]) -> torch.Tensor:
        all_tokens = []
        for img in images:
            if img.dim() == 5:
                b, t, c, h, w = img.shape
                img = img.view(b * t, c, h, w)
            tokens = self.encoder(img)
            if img.dim() == 5:
                b, t, c, h, w = img.shape
                n_tokens = tokens.shape[1]
                tokens = tokens.view(b, t * n_tokens, -1)
            all_tokens.append(tokens)

        if len(all_tokens) == 1:
            return all_tokens[0]

        return torch.cat(all_tokens, dim=1)


class ImageEncoderTinyCNN(nn.Module):
    """Legacy encoder kept for backward compatibility. Not used by MiniVLAPolicy."""

    def __init__(self, d_model=128):
        super().__init__()
        warnings.warn(
            "ImageEncoderTinyCNN is legacy and not used by MiniVLAPolicy.",
            DeprecationWarning,
        )
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.proj = nn.Linear(128, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.mean(dim=[2, 3])
        x = self.proj(x)
        x = self.ln(x)
        return x


class TextEncoderTinyGRU(nn.Module):
    """Legacy encoder kept for backward compatibility. Not used by MiniVLAPolicy."""

    def __init__(self, vocab_size, d_word=64, d_model=128):
        super().__init__()
        warnings.warn(
            "TextEncoderTinyGRU is legacy and not used by MiniVLAPolicy.",
            DeprecationWarning,
        )
        self.embedding = nn.Embedding(vocab_size, d_word)
        self.gru = nn.GRU(d_word, d_model, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        _, h_last = self.gru(x)
        x = h_last[0]
        x = self.ln(x)
        return x


class StateEncoderMLP(nn.Module):
    """Legacy encoder kept for backward compatibility. Not used by MiniVLAPolicy."""

    def __init__(self, state_dim, d_model=128):
        super().__init__()
        warnings.warn(
            "StateEncoderMLP is legacy and not used by MiniVLAPolicy.",
            DeprecationWarning,
        )
        self.fc1 = nn.Linear(state_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, s):
        x = self.fc1(s)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.ln(x)
        return x