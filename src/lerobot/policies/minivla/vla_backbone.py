"""
vla_backbone.py

Official MiniVLA backbone: DINO-SigLIP patch -> FusedMLPProjector -> Qwen2.5 CausalLM.
Mirrors teach_code/MiniVLA/prismatic/models/vlms/prismatic.py (PrismaticVLM) and
prismatic/models/backbones/llm/qwen25.py.

Key design:
  - Vision patches inserted after first token of each sequence
  - attention_mask extended with True for vision tokens
  - labels set to IGNORE_INDEX (-100) for vision tokens
  - generation cache support (past_key_values)
  - No state tokens, no dummy layers, no full-zero text embeddings
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .encoders import DINOSigLIPViTBackbone
from .fusion import FusedMLPProjector
from .tokenizer import VLATokenizerWrapper

IGNORE_INDEX = -100


class MiniVLAVLBackbone(nn.Module):
    """
    Official MiniVLA VLM backbone.
    Combines DINO-SigLIP vision encoder, FusedMLP projector, and Qwen2.5 CausalLM.
    """

    def __init__(
        self,
        vision_backbone_id: str = "dinosiglip-vit-so-224px",
        llm_backbone_id: str = "qwen25-0_5b-extra",
        base_vlm_checkpoint: str = "Qwen/Qwen2.5-0.5B",
        image_size: int = 224,
        image_resize_strategy: str = "resize-naive",
        arch_specifier: str = "no-align+fused-gelu-mlp",
        image_sequence_len: int = 1,
        num_extra_tokens: int = 256,
        enable_gradient_checkpointing: bool = True,
        freeze_vision_backbone: bool = False,
        freeze_llm_backbone: bool = False,
        unfreeze_last_llm_layer: bool = False,
    ):
        super().__init__()
        self.image_sequence_len = image_sequence_len
        self.num_extra_tokens = num_extra_tokens

        # === Vision backbone ===
        self.vision_backbone = DINOSigLIPViTBackbone(
            vision_backbone_id=vision_backbone_id,
            image_resize_strategy=image_resize_strategy,
            default_image_size=image_size,
            image_sequence_len=image_sequence_len,
        )
        if freeze_vision_backbone:
            for param in self.vision_backbone.parameters():
                param.requires_grad = False

        # === Projector ===
        fused_vision_dim = self.vision_backbone.embed_dim
        # Get LLM hidden size from config
        self._llm_config = AutoModelForCausalLM.from_pretrained(
            base_vlm_checkpoint, trust_remote_code=True
        ).config
        llm_dim = self._llm_config.hidden_size

        self.projector = FusedMLPProjector(
            fused_vision_dim=fused_vision_dim,
            llm_dim=llm_dim,
            mlp_type="fused-gelu-mlp",
        )

        # === LLM ===
        self.llm = AutoModelForCausalLM.from_pretrained(
            base_vlm_checkpoint, trust_remote_code=True
        )
        # Resize embeddings for extra tokens
        self.llm.resize_token_embeddings(
            self.llm.config.vocab_size + num_extra_tokens,
            pad_to_multiple_of=64,
        )
        self.llm.config.pad_token_id = self.llm.config.eos_token_id

        if freeze_llm_backbone:
            for param in self.llm.parameters():
                param.requires_grad = False

        if unfreeze_last_llm_layer:
            for param in self.llm.layers[-1].parameters():
                param.requires_grad = True

        # === Gradient checkpointing ===
        if enable_gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()

    @property
    def num_patches(self) -> int:
        return self.vision_backbone.num_patches

    def forward(
        self,
        pixel_values: dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ):
        """
        Training forward: inserts vision patches into LLM embeddings.
        Generation forward: uses past_key_values cache to skip vision backbone.
        """
        if past_key_values is not None:
            # Generation mode: only process the last token
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        # === Get vision patches ===
        patch_embeddings = self.vision_backbone(pixel_values)  # (B, num_patches, fused_dim)
        projected_patches = self.projector(patch_embeddings)  # (B, num_patches, llm_dim)

        # === Get LLM embeddings ===
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, seq_len, llm_dim)

        # === Insert vision patches after first token ===
        # inputs_embeds: (B, seq_len, llm_dim)
        # projected_patches: (B, num_patches, llm_dim)
        # Result: (B, 1 + num_patches + seq_len - 1, llm_dim)
        before = inputs_embeds[:, :1, :]  # (B, 1, llm_dim)
        after = inputs_embeds[:, 1:, :]   # (B, seq_len - 1, llm_dim)
        inputs_embeds = torch.cat([before, projected_patches, after], dim=1)

        # === Extend attention_mask ===
        # Original: (B, seq_len)
        # Vision mask: (B, num_patches) all True
        vision_mask = torch.ones(
            inputs_embeds.shape[0], self.num_patches,
            dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask = torch.cat(
            [attention_mask[:, :1], vision_mask, attention_mask[:, 1:]], dim=1
        )

        # === Set labels for vision tokens to IGNORE_INDEX ===
        if labels is not None:
            vision_labels = torch.full(
                (labels.shape[0], self.num_patches),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([labels[:, :1], vision_labels, labels[:, 1:]], dim=1)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Official prepare_inputs_for_generation matching PrismaticVLM."""
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
        }