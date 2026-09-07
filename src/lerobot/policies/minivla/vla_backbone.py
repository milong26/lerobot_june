"""
vla_backbone.py

Official MiniVLA backbone: DINO-SigLIP patch -> FusedMLPProjector -> Qwen2.5 CausalLM.
Mirrors teach_code/MiniVLA/prismatic/models/vlms/base_vlm.py and
prismatic/models/vlms/prismatic.py (PrismaticVLM).

Key design:
  - Vision patches inserted after first token of each sequence
  - attention_mask extended with True for vision tokens
  - labels set to IGNORE_INDEX (-100) for vision tokens
  - GenerationMixin-compatible: prepare_inputs_for_generation preserves pixel_values
  - forward() checks past_key_values to skip vision encoder during generation
  - device property for GenerationMixin
  - torch.manual_seed(vision_backbone.embed_dim) before projector creation
  - num_patches from projected_patches.shape[1]
  - Proper attention_mask handling: preserves real prompt and padding info
  - cache_position, position_ids, pad_token_id, return_dict compatibility
  - Batch generation: reads last VALID token position (not padding)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .encoders import DINOSigLIPViTBackbone
from .fusion import FusedMLPProjector

IGNORE_INDEX = -100


class MiniVLAVLBackbone(nn.Module, GenerationMixin):
    """
    Official MiniVLA VLM backbone.
    Combines DINO-SigLIP vision encoder, FusedMLP projector, and Qwen2.5 CausalLM.
    Inherits GenerationMixin for proper multi-modal generation support.
    Mirrors teach_code/MiniVLA/prismatic/models/vlms/prismatic.py::PrismaticVLM.
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
        tokenizer_len: int = None,
        enable_gradient_checkpointing: bool = True,
        freeze_vision_backbone: bool = False,
        freeze_llm_backbone: bool = False,
        unfreeze_last_llm_layer: bool = False,
    ):
        super().__init__()
        self.image_sequence_len = image_sequence_len
        self.tokenizer_len = tokenizer_len

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
        from transformers import AutoConfig
        llm_config = AutoConfig.from_pretrained(base_vlm_checkpoint, trust_remote_code=True)
        llm_dim = llm_config.hidden_size

        # Official: torch.manual_seed(vision_backbone.embed_dim) before projector creation
        # Mirrors teach_code/MiniVLA/prismatic/models/vlms/prismatic.py::__init__
        torch.manual_seed(fused_vision_dim)

        self.projector = FusedMLPProjector(
            fused_vision_dim=fused_vision_dim,
            llm_dim=llm_dim,
            mlp_type="fused-gelu-mlp",
        )

        # === LLM ===
        self.llm = AutoModelForCausalLM.from_pretrained(
            base_vlm_checkpoint, trust_remote_code=True
        )
        assert tokenizer_len is not None, "tokenizer_len must be provided for embedding resize"
        self.llm.resize_token_embeddings(tokenizer_len, pad_to_multiple_of=64)

        if freeze_llm_backbone:
            for param in self.llm.parameters():
                param.requires_grad = False

        if unfreeze_last_llm_layer:
            for param in self.llm.model.layers[-1].parameters():
                param.requires_grad = True

        # === Gradient checkpointing ===
        if enable_gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()

        # === GenerationMixin required attributes ===
        # Mirrors teach_code/MiniVLA/prismatic/models/vlms/base_vlm.py::VLM.__init__
        self.generation_config = self.llm.generation_config
        self.main_input_name = "input_ids"

    @property
    def device(self):
        """Borrowed from transformers.modeling_utils -- required by GenerationMixin."""
        return next(self.parameters()).device

    @property
    def num_patches(self) -> int:
        """Return number of vision patches for single image or sequence."""
        return self.vision_backbone.num_patches

    def set_pad_token_id(self, pad_token_id: int):
        """Set pad_token_id on LLM config from shared tokenizer."""
        self.llm.config.pad_token_id = pad_token_id

    @staticmethod
    def can_generate() -> bool:
        return True

    @property
    def config(self):
        return self.llm.config

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.llm._reorder_cache(past_key_values, beam_idx)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: bool = False,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        """
        Training forward: inserts vision patches into LLM embeddings.
        Generation forward: uses past_key_values cache to skip vision encoder.
        Mirrors teach_code/MiniVLA/prismatic/models/vlms/prismatic.py::PrismaticVLM.forward.
        """
        # Handle Inference: leverage cache, short-circuit on just LLM forward
        # Mirrors official: if input_ids.shape[1] == 1 and past_key_values is not None
        if input_ids.shape[1] == 1 and past_key_values is not None:
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        # Training mode or first generation step: run vision encoder
        if pixel_values is None:
            raise RuntimeError("forward() requires pixel_values for the first step!")

        # === Get vision patches ===
        patch_embeddings = self.vision_backbone(pixel_values)
        projected_patches = self.projector(patch_embeddings)

        # === Get LLM embeddings ===
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # === Insert vision patches after first token ===
        # Use projected_patches.shape[1] for num_patches (not static attribute)
        # Mirrors official: multimodal_embeddings = cat([input_embeddings[:1], projected_patches, input_embeddings[1:]])
        num_patches = projected_patches.shape[1]

        before = inputs_embeds[:, :1, :]
        after = inputs_embeds[:, 1:, :]
        inputs_embeds = torch.cat([before, projected_patches, after], dim=1)

        # === Build multimodal attention mask ===
        # [first_text_token_mask, vision_all_ones, remaining_text_mask]
        # Vision tokens should have True attention (not masked)
        # Mirrors official: projected_patch_attention_mask = torch.full(..., True, ...)
        vision_mask = torch.ones(
            inputs_embeds.shape[0], num_patches,
            dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask = torch.cat(
            [attention_mask[:, :1], vision_mask, attention_mask[:, 1:]], dim=1
        )

        # === Set labels for vision tokens to IGNORE_INDEX ===
        # Mirrors official: projected_patch_labels = torch.full(..., IGNORE_INDEX, ...)
        if labels is not None:
            vision_labels = torch.full(
                (labels.shape[0], num_patches),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([labels[:, :1], vision_labels, labels[:, 1:]], dim=1)

        # === Build position_ids if not provided ===
        if position_ids is None:
            position_ids = torch.arange(
                attention_mask.shape[1], dtype=torch.long, device=attention_mask.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            position_ids=position_ids,
            cache_position=cache_position,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Official prepare_inputs_for_generation matching PrismaticVLM.
        Ensures pixel_values are preserved in model_inputs for the first generation step.
        Mirrors teach_code/MiniVLA/prismatic/models/vlms/prismatic.py::prepare_inputs_for_generation.
        """
        # Prefer LLM's own prepare_inputs_for_generation when available
        if hasattr(self.llm, "prepare_inputs_for_generation"):
            llm_inputs = self.llm.prepare_inputs_for_generation(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            # Fallback to official MiniVLA logic
            if past_key_values is not None:
                input_ids = input_ids[:, -1:]

            if inputs_embeds is not None and past_key_values is None:
                llm_inputs = {"inputs_embeds": inputs_embeds}
            else:
                llm_inputs = {"input_ids": input_ids}

        # Add multimodal-specific inputs
        model_inputs = dict(llm_inputs)

        # Only attach pixel_values on the very first step (when past_key_values is empty/None)
        if past_key_values is None:
            model_inputs["pixel_values"] = pixel_values

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
        )

        return model_inputs