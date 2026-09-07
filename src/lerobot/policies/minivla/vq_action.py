"""
vq_action.py

Official VQ-VAE action tokenizer for MiniVLA.
Mirrors teach_code/MiniVLA/vqvae/vqvae/vqvae.py (EncoderMLP, VqVae),
teach_code/MiniVLA/prismatic/vla/action_tokenizer.py (ActionTokenizer, VQActionTokenizer),
and teach_code/MiniVLA/vq/pretrain_vq+mx-libero_90+fach-7+ng-7+nemb-128+nlatent-512/config.json.

Key design:
  - EncoderMLP: Linear -> ReLU -> (Linear -> ReLU) * layer_num -> Linear -> (optional activation)
  - hidden_dim=128 (official), ReLU (not GELU)
  - VqVae: nn.Module with encoder + decoder + ResidualVQ
  - VQActionTokenizer: preprocess -> encode -> decode with official token mapping
  - Frozen VQ-VAE (eval, requires_grad=False)
  - Token mapping: token_id = tokenizer_len - 1 - code
  - Official checkpoint loading via load_official_checkpoint() (not overriding state_dict)
  - [B,T,A] -> [B,T*A] flatten before encoding, [B,T*A] -> [B,T,A] after decoding
  - action/act_scale processing as in official code
  - Dynamic device via next(self.parameters()).device
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast


# ---------------------------------------------------------------------------
# weights_init_encoder (official initialization)
# ---------------------------------------------------------------------------
def weights_init_encoder(m):
    """Official weight initialization for VQ-VAE encoder/decoder."""
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


# ---------------------------------------------------------------------------
# EncoderMLP (official MiniVLA structure)
# ---------------------------------------------------------------------------
class EncoderMLP(nn.Module):
    """
    Official EncoderMLP from MiniVLA VQ-VAE.
    Linear -> ReLU -> (Linear -> ReLU) * layer_num -> Linear -> (optional last_activation)
    Uses hidden_dim=128 and ReLU (NOT GELU).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        layer_num: int = 1,
        last_activation=None,
    ):
        super().__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(layer_num):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

        if last_activation is not None:
            self.last_layer = last_activation
        else:
            self.last_layer = None

        self.apply(weights_init_encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        state = self.fc(h)
        if self.last_layer:
            state = self.last_layer(state)
        return state


# ---------------------------------------------------------------------------
# VqVae (official structure as nn.Module)
# ---------------------------------------------------------------------------
class VqVae(nn.Module):
    """
    Official VQ-VAE from MiniVLA.
    encoder + decoder + vq_layer (ResidualVQ from vector_quantize_pytorch)
    Properly inherits nn.Module for standard state_dict/load_state_dict.
    """

    def __init__(
        self,
        obs_dim: int = 60,
        input_dim_h: int = 10,
        input_dim_w: int = 9,
        n_latent_dims: int = 512,
        vqvae_n_embed: int = 32,
        vqvae_groups: int = 4,
        eval_mode: bool = True,
        encoder_loss_multiplier: float = 1.0,
        act_scale: float = 1.0,
    ):
        super().__init__()
        self.n_latent_dims = n_latent_dims
        self.input_dim_h = input_dim_h
        self.input_dim_w = input_dim_w
        self.rep_dim = self.n_latent_dims
        self.vqvae_n_embed = vqvae_n_embed
        self.vqvae_lr = 1e-3
        self.vqvae_groups = vqvae_groups
        self.encoder_loss_multiplier = encoder_loss_multiplier
        self.act_scale = act_scale

        # Import official ResidualVQ
        try:
            from vector_quantize_pytorch import ResidualVQ as OfficialResidualVQ
        except ImportError:
            raise ImportError(
                "vector_quantize_pytorch is required for VQ-VAE. "
                "Install with: pip install vector-quantize-pytorch"
            )

        self.vq_layer = OfficialResidualVQ(
            dim=self.n_latent_dims,
            num_quantizers=self.vqvae_groups,
            codebook_size=self.vqvae_n_embed,
        )
        self.embedding_dim = self.n_latent_dims

        if self.input_dim_h == 1:
            self.encoder = EncoderMLP(
                input_dim=input_dim_w, output_dim=n_latent_dims
            )
            self.decoder = EncoderMLP(
                input_dim=n_latent_dims, output_dim=input_dim_w
            )
        else:
            self.encoder = EncoderMLP(
                input_dim=input_dim_w * self.input_dim_h, output_dim=n_latent_dims
            )
            self.decoder = EncoderMLP(
                input_dim=n_latent_dims, output_dim=input_dim_w * self.input_dim_h
            )

        if eval_mode:
            self._freeze()

    @property
    def device(self):
        return next(self.parameters()).device

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def draw_code_forward(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        """Decode from codes using official get_codes_from_indices().sum(dim=0)."""
        with torch.no_grad():
            z_embed = self.vq_layer.get_codes_from_indices(encoding_indices)
            z_embed = z_embed.sum(dim=0)
        return z_embed

    def get_action_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to action chunk [B, T, A]."""
        output = self.decoder(latent) * self.act_scale
        return rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)

    def preprocess(self, state: torch.Tensor) -> torch.Tensor:
        """Official preprocess: divide by act_scale and flatten."""
        if not torch.is_tensor(state):
            state = torch.tensor(state, device=self.device)
        if self.input_dim_h == 1:
            state = state.squeeze(-2)
        else:
            state = rearrange(state, "N T A -> N (T A)")
        return state.to(self.device)

    def get_code(self, state: torch.Tensor, required_recon: bool = False):
        """Encode state to VQ codes."""
        state = state / self.act_scale
        state = self.preprocess(state)
        with torch.no_grad():
            state_rep = self.encoder(state)
            state_rep_shape = state_rep.shape[:-1]
            state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
            state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
            state_vq = state_rep_flat.view(*state_rep_shape, -1)
            vq_code = vq_code.view(*state_rep_shape, -1)
            vq_loss_state = torch.sum(vq_loss_state)
            if required_recon:
                recon_state = self.decoder(state_vq) * self.act_scale
                recon_state_ae = self.decoder(state_rep) * self.act_scale
                if self.input_dim_h == 1:
                    return state_vq, vq_code, recon_state, recon_state_ae
                else:
                    return (
                        state_vq,
                        vq_code,
                        torch.swapaxes(recon_state, -2, -1),
                        torch.swapaxes(recon_state_ae, -2, -1),
                    )
            else:
                return state_vq, vq_code

    def load_official_checkpoint(self, load_dir: str):
        """
        Load from official checkpoint format.
        Official state_dict has keys: encoder, decoder, optimizer, vq_embedding.
        Does NOT override standard nn.Module.load_state_dict.
        """
        state_dict = torch.load(load_dir, map_location="cpu")

        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.vq_layer.load_state_dict(state_dict["vq_embedding"])

        self._freeze()


# ---------------------------------------------------------------------------
# ActionTokenizer (official non-VQ action tokenizer)
# ---------------------------------------------------------------------------
class ActionTokenizer:
    """
    Official ActionTokenizer from teach_code/MiniVLA/prismatic/vla/action_tokenizer.py.
    Discretizes continuous robot actions into N bins per dimension.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: int = -1,
        max_action: int = 1,
        use_extra: bool = False,
    ):
        self.tokenizer, self.n_bins, self.min_action, self.max_action = (
            tokenizer,
            bins,
            min_action,
            max_action,
        )

        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        self.tokenizer_len = self.tokenizer.vocab_size
        if isinstance(tokenizer, Qwen2TokenizerFast) and use_extra:
            self.tokenizer_len = len(self.tokenizer)
        elif use_extra:
            raise NotImplementedError("Cannot use extra tokens for this tokenizer!")

        self.action_token_begin_idx: int = int(self.tokenizer_len - (self.n_bins + 1))
        self.action_token_end_idx: int = int(self.tokenizer_len)

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        if len(discretized_action.shape) <= 1:
            return self.tokenizer.decode(list(self.tokenizer_len - discretized_action))
        else:
            return self.tokenizer.batch_decode((self.tokenizer_len - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        discretized_actions = self.tokenizer_len - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins

    @property
    def required_future_horizon(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# VQActionTokenizer (official VQ action tokenizer)
# ---------------------------------------------------------------------------
class VQActionTokenizer(ActionTokenizer):
    """
    Official VQActionTokenizer from teach_code/MiniVLA/prismatic/vla/action_tokenizer.py.
    Loads VqVae and handles token <-> code mapping.
    Accepts both NumPy and Torch inputs.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        vq_vae_path: str = "",
        device: str = "cpu",
        use_extra: bool = False,
    ):
        self.tokenizer = tokenizer
        self._init_device = device

        # Require caller to pass an absolute VQ directory path (resolved by configuration_minivla.py)
        vq_path = Path(vq_vae_path)
        if not vq_path.is_absolute():
            raise ValueError(
                f"VQActionTokenizer requires an absolute vq_vae_path, got relative: {vq_vae_path}. "
                f"Use config.resolve_vq_model_path() to resolve the path before calling this constructor."
            )
        if not vq_path.exists():
            raise FileNotFoundError(f"VQ model directory not found: {vq_path}")

        self.vq_path = vq_path
        vq_model_path = self.vq_path / "checkpoints" / "model.pt"
        vq_config_path = self.vq_path / "config.json"
        assert vq_model_path.exists(), f"Missing VQ checkpoint path: {vq_model_path}"
        assert vq_config_path.exists(), f"Missing VQ config path: {vq_config_path}"

        with open(vq_config_path, "r") as f:
            vq_config = dict(json.load(f))

        vq_config["eval_mode"] = True

        self.vq_vae = VqVae(**vq_config)
        self.vq_vae.load_official_checkpoint(str(vq_model_path))

        self.n_bins = self.vq_vae.vqvae_n_embed

        self.tokenizer_len = self.tokenizer.vocab_size
        if isinstance(tokenizer, Qwen2TokenizerFast) and use_extra:
            self.tokenizer_len = len(self.tokenizer)
        elif use_extra:
            raise NotImplementedError("Cannot use extra tokens for this tokenizer!")

        self.action_token_begin_idx: int = int(self.tokenizer_len - (self.n_bins + 1))
        self.action_token_end_idx: int = int(self.tokenizer_len)

    def __call__(self, action) -> Union[str, List[str]]:
        """
        Encode action to text tokens.
        Accepts both numpy arrays and torch tensors.
        Mirrors teach_code/MiniVLA/prismatic/vla/action_tokenizer.py::VQActionTokenizer.__call__.
        """
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.array(action)

        # Validate input shape: must match VQ config input_dim_h and input_dim_w
        if action.ndim == 1:
            # [A] -> [1, 1, A]
            action = action[np.newaxis, np.newaxis, :]
        elif action.ndim == 2:
            # [T, A] -> [1, T, A]
            action = action[np.newaxis, :]
        elif action.ndim == 3:
            # [B, T, A] - keep as is
            pass
        else:
            raise ValueError(f"Unexpected action shape: {action.shape}, expected [T,A], [1,T,A] or [B,T,A]")

        # Validate T and A dimensions
        if action.shape[-2] != self.vq_vae.input_dim_h:
            raise ValueError(
                f"Action time dimension {action.shape[-2]} does not match VQ input_dim_h {self.vq_vae.input_dim_h}"
            )
        if action.shape[-1] != self.vq_vae.input_dim_w:
            raise ValueError(
                f"Action feature dimension {action.shape[-1]} does not match VQ input_dim_w {self.vq_vae.input_dim_w}"
            )

        action = torch.from_numpy(action).to(self.vq_vae.device)
        _, vq_code = self.vq_vae.get_code(action)
        assert torch.all(vq_code >= 0) and torch.all(vq_code < self.n_bins)

        return self.tokenizer.decode(list(self.tokenizer_len - 1 - vq_code[0].detach().cpu().tolist()))

    def decode_token_ids_to_actions(self, action_token_ids) -> np.ndarray:
        """
        Decode action token IDs to continuous actions.
        Accepts both numpy arrays and torch tensors.
        Mirrors teach_code/MiniVLA/prismatic/vla/action_tokenizer.py::VQActionTokenizer.decode_token_ids_to_actions.
        """
        if isinstance(action_token_ids, torch.Tensor):
            action_token_ids = action_token_ids.detach().cpu().numpy()
        action_token_ids = np.array(action_token_ids)

        action_token_ids = self.tokenizer_len - 1 - action_token_ids
        initial_shape = action_token_ids.shape
        action_token_ids = np.clip(action_token_ids, 0, self.n_bins - 1)

        # Validate last dimension equals vqvae_groups
        if action_token_ids.shape[-1] != self.vq_vae.vqvae_groups:
            raise ValueError(
                f"Action token IDs last dimension {action_token_ids.shape[-1]} "
                f"does not match VQ vqvae_groups {self.vq_vae.vqvae_groups}"
            )

        action_token_ids = torch.from_numpy(action_token_ids).to(self.vq_vae.device).reshape(
            -1, self.vq_vae.vqvae_groups
        )
        assert torch.all(action_token_ids >= 0) and torch.all(action_token_ids < self.n_bins)

        latent = self.vq_vae.draw_code_forward(action_token_ids)
        ret_action = self.vq_vae.get_action_from_latent(latent)

        # Return only the first horizon action as per official behavior
        if action_token_ids.shape[0] == 1 and len(initial_shape) == 1:
            return ret_action[0, 0].detach().cpu().numpy()

        return ret_action[:, 0].detach().cpu().numpy()

    @property
    def required_future_horizon(self) -> int:
        return self.vq_vae.input_dim_h - 1


# ---------------------------------------------------------------------------
# Action tokenizer registry (matching official ACTION_TOKENIZERS)
# ---------------------------------------------------------------------------
ACTION_TOKENIZERS = {
    "action_tokenizer": ActionTokenizer,
    "extra_action_tokenizer": partial(ActionTokenizer, use_extra=True),
    "libero_vq_action_tokenizer": partial(
        VQActionTokenizer, vq_vae_path="vq/pretrain_vq+mx-libero_90+fach-7+ng-7+nemb-128+nlatent-512"
    ),
    "libero_vq_extra_action_tokenizer": partial(
        VQActionTokenizer, vq_vae_path="vq/pretrain_vq+mx-libero_90+fach-7+ng-7+nemb-128+nlatent-512", use_extra=True
    ),
    "libero_vq_h0_extra_action_tokenizer": partial(
        VQActionTokenizer, vq_vae_path="vq/pretrain_vq+mx-libero_90+fach-0+ng-7+nemb-128+nlatent-512", use_extra=True
    ),
    "bridge_vq_extra_action_tokenizer": partial(
        VQActionTokenizer,
        vq_vae_path="vq/pretrain_modvq+mx-bridge_dataset+fach-7+ng-7+nemb-256+nlatent-512",
        use_extra=True,
    ),
}