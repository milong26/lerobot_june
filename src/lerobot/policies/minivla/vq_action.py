"""
vq_action.py

Official VQ-VAE action tokenizer for MiniVLA.
Mirrors teach_code/MiniVLA/prismatic/vla/action_tokenizer.py (VQActionTokenizer),
teach_code/MiniVLA/vqvae/vqvae/vqvae.py (EncoderMLP, VqVae), and
teach_code/MiniVLA/vq/pretrain_vq+mx-libero_90+fach-7+ng-7+nemb-128+nlatent-512/config.json.

Key design:
  - EncoderMLP: Linear -> GELU -> Linear -> GELU -> Linear
  - VqVae: encoder + decoder + ResidualVQ
  - VQActionTokenizer: preprocess -> encode -> decode
  - Frozen VQ-VAE (eval, requires_grad=False)
  - Token mapping: token_id = tokenizer_len - 1 - code
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from lerobot.utils.import_utils import require_package


# ---------------------------------------------------------------------------
# ResidualVQ (minimal self-contained implementation, MIT license)
# ---------------------------------------------------------------------------
class ResidualVQ(nn.Module):
    """
    Residual Vector Quantizer.
    Mirrors the official ResidualVQ from vector-quantize-pytorch / MiniVLA.
    """

    def __init__(
        self,
        dim: int,
        num_quantizers: int,
        codebook_size: int,
        **kwargs,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList([
            VectorQuantize(dim=dim, codebook_size=codebook_size, **kwargs)
            for _ in range(num_quantizers)
        ])

    @property
    def codebooks(self):
        return torch.stack([layer.codebook for layer in self.layers])

    def get_code(self, x: torch.Tensor) -> torch.Tensor:
        """Get codes from all quantizers."""
        all_codes = []
        residual = x
        for layer in self.layers:
            codes = layer.get_code(residual)
            all_codes.append(codes)
            quantized = layer.get_output_from_indices(codes)
            residual = residual - quantized
        return torch.stack(all_codes, dim=-1)  # (B, num_quantizers)

    def draw_code_forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct from codes."""
        out = None
        for i, layer in enumerate(self.layers):
            quantized = layer.get_output_from_indices(codes[..., i])
            out = quantized if out is None else out + quantized
        return out


class VectorQuantize(nn.Module):
    """
    Minimal VectorQuantize implementation matching official behavior.
    """

    def __init__(self, dim: int, codebook_size: int, commitment_weight: float = 1.0):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_weight = commitment_weight

        # Codebook
        embed = torch.randn(codebook_size, dim)
        self.register_buffer("codebook", embed)

    @property
    def codebook(self):
        return self._codebook

    @codebook.setter
    def codebook(self, value):
        self.register_buffer("_codebook", value)

    def get_code(self, x: torch.Tensor) -> torch.Tensor:
        """Get nearest codebook indices."""
        # x: (B, dim), codebook: (K, dim)
        dist = torch.cdist(x, self.codebook)
        return torch.argmin(dist, dim=-1)

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Lookup codebook vectors by indices."""
        return self.codebook[indices]


# ---------------------------------------------------------------------------
# EncoderMLP (official MiniVLA structure)
# ---------------------------------------------------------------------------
class EncoderMLP(nn.Module):
    """
    Official EncoderMLP from MiniVLA VQ-VAE.
    Linear -> GELU -> Linear -> GELU -> Linear
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        layer_num: int = 2,
        last_activation: str = "none",
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(layer_num - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        if last_activation == "tanh":
            layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# VqVae (official structure)
# ---------------------------------------------------------------------------
class VqVae(nn.Module):
    """
    Official VQ-VAE from MiniVLA.
    encoder + decoder + vq_layer (ResidualVQ)
    """

    def __init__(
        self,
        input_dim_h: int = 8,
        input_dim_w: int = 7,
        n_latent_dims: int = 512,
        vqvae_n_embed: int = 128,
        vqvae_groups: int = 7,
        encoder_loss_multiplier: float = 1.0,
        act_scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim_h = input_dim_h
        self.input_dim_w = input_dim_w
        self.n_latent_dims = n_latent_dims
        self.vqvae_n_embed = vqvae_n_embed
        self.vqvae_groups = vqvae_groups
        self.encoder_loss_multiplier = encoder_loss_multiplier
        self.act_scale = act_scale

        # Encoder
        self.encoder = EncoderMLP(
            input_dim=input_dim_h * input_dim_w,
            output_dim=n_latent_dims,
            hidden_dim=256,
            layer_num=2,
        )
        # Decoder
        self.decoder = EncoderMLP(
            input_dim=n_latent_dims,
            output_dim=input_dim_h * input_dim_w,
            hidden_dim=256,
            layer_num=2,
        )
        # Residual VQ
        self.vq_layer = ResidualVQ(
            dim=n_latent_dims,
            num_quantizers=vqvae_groups,
            codebook_size=vqvae_n_embed,
        )

    def preprocess(self, actions: torch.Tensor) -> torch.Tensor:
        """Official preprocess: scale and flatten."""
        # actions: (B, H, W) -> (B, H*W)
        return actions * self.act_scale

    def forward_encoder(self, actions: torch.Tensor) -> torch.Tensor:
        preprocessed = self.preprocess(actions)
        return self.encoder(preprocessed)

    def get_code(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode actions to VQ codes: (B, H, W) -> (B, groups)."""
        latent = self.forward_encoder(actions)
        return self.vq_layer.get_code(latent)

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codes back to actions: (B, groups) -> (B, H*W)."""
        quantized = self.vq_layer.draw_code_forward(codes)
        return self.decoder(quantized) / self.act_scale


# ---------------------------------------------------------------------------
# VQActionTokenizer (official interface)
# ---------------------------------------------------------------------------
class VQActionTokenizer:
    """
    Official VQActionTokenizer for MiniVLA.
    Wraps the frozen VQ-VAE and handles token <-> code mapping.
    """

    def __init__(
        self,
        vq_model_path: Optional[str] = None,
        input_dim_h: int = 8,
        input_dim_w: int = 7,
        n_latent_dims: int = 512,
        vqvae_n_embed: int = 128,
        vqvae_groups: int = 7,
        encoder_loss_multiplier: float = 1.0,
        act_scale: float = 1.0,
        tokenizer_len: int = 0,
    ):
        self.input_dim_h = input_dim_h
        self.input_dim_w = input_dim_w
        self.n_latent_dims = n_latent_dims
        self.vqvae_n_embed = vqvae_n_embed
        self.vqvae_groups = vqvae_groups
        self.encoder_loss_multiplier = encoder_loss_multiplier
        self.act_scale = act_scale
        self.tokenizer_len = tokenizer_len

        self.vqvae = VqVae(
            input_dim_h=input_dim_h,
            input_dim_w=input_dim_w,
            n_latent_dims=n_latent_dims,
            vqvae_n_embed=vqvae_n_embed,
            vqvae_groups=vqvae_groups,
            encoder_loss_multiplier=encoder_loss_multiplier,
            act_scale=act_scale,
        )

        if vq_model_path is not None:
            self._load_vq_checkpoint(vq_model_path)

        self.vqvae.eval()
        for param in self.vqvae.parameters():
            param.requires_grad = False

    def _load_vq_checkpoint(self, vq_model_path: str) -> None:
        """Load official VQ checkpoint."""
        path = Path(vq_model_path)
        if path.is_dir():
            path = path / "model.pt"
        if not path.exists():
            raise FileNotFoundError(f"VQ checkpoint not found at {vq_model_path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("model", ckpt)
        self.vqvae.load_state_dict(state_dict, strict=True)

    def encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode action chunk to VQ codes.
        actions: (B, chunk_size, action_dim) -> codes: (B, vqvae_groups)
        """
        # actions: (B, H, W) where H=chunk_size, W=action_dim
        codes = self.vqvae.get_code(actions)
        return codes

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode VQ codes to action chunk.
        codes: (B, vqvae_groups) -> actions: (B, chunk_size, action_dim)
        """
        return self.vqvae.decode_codes(codes)

    def encode_token_ids(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode actions to token IDs (for Qwen input).
        token_id = tokenizer_len - 1 - code
        """
        codes = self.encode_actions(actions)
        return self.tokenizer_len - 1 - codes

    def decode_token_ids_to_actions(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Decode token IDs back to actions.
        code = tokenizer_len - 1 - token_id
        """
        codes = self.tokenizer_len - 1 - token_ids
        return self.decode_codes(codes)

    @property
    def vq_action_dim(self) -> int:
        return self.input_dim_w