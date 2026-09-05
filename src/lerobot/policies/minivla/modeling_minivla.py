"""
modeling_minivla.py

Official MiniVLA policy for LeRobot.
Mirrors teach_code/MiniVLA/prismatic/models/vlas/openvla.py,
prismatic/models/vlms/prismatic.py, prismatic/vla/datasets/datasets.py,
and vla-scripts/train.py.

Key design:
  - MiniVLACore: vision_backbone + projector + Qwen CausalLM + frozen VQ tokenizer
  - No independent action_logits_head
  - Training: VQActionTokenizer -> 7 action tokens -> Qwen input_ids/labels -> CausalLM loss
  - Inference: autoregressive generate -> decode to [B,8,A]
  - predict_action_chunk: returns [B, chunk_size, action_dim]
  - select_action: returns [B, action_dim] (chunk[:, 0])
  - get_optim_params: only trainable vision, projector, Qwen params
"""

from __future__ import annotations

from typing import Optional, Unpack

import torch
import torch.nn as nn

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy

from .configuration_minivla import MiniVLAConfig
from .tokenizer import VLATokenizerWrapper
from .vq_action import VQActionTokenizer
from .vla_backbone import MiniVLAVLBackbone, IGNORE_INDEX


class MiniVLACore(nn.Module):
    """
    Official MiniVLA core model.
    vision_backbone + FusedMLPProjector + Qwen2.5 CausalLM + frozen VQ tokenizer.
    """

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.config = config

        # === VLM Backbone ===
        self.vlm = MiniVLAVLBackbone(
            vision_backbone_id=config.vision_backbone_id,
            llm_backbone_id=config.llm_backbone_id,
            base_vlm_checkpoint=config.base_vlm_checkpoint,
            image_size=config.image_size,
            image_resize_strategy=config.image_resize_strategy,
            arch_specifier=config.arch_specifier,
            image_sequence_len=config.image_sequence_len,
            num_extra_tokens=config.num_extra_tokens,
            enable_gradient_checkpointing=config.enable_gradient_checkpointing,
            freeze_vision_backbone=config.freeze_vision_backbone,
            freeze_llm_backbone=config.freeze_llm_backbone,
            unfreeze_last_llm_layer=config.unfreeze_last_llm_layer,
        )

        # === Tokenizer ===
        self.tokenizer = VLATokenizerWrapper(
            base_vlm_checkpoint=config.base_vlm_checkpoint,
            num_extra_tokens=config.num_extra_tokens,
        )

        # === VQ Action Tokenizer ===
        self.vq_tokenizer = VQActionTokenizer(
            vq_model_path=config.vq_model_path if config.vq_model_path else None,
            input_dim_h=config.chunk_size,
            input_dim_w=config.vq_action_dim,
            n_latent_dims=config.n_latent_dims,
            vqvae_n_embed=config.vqvae_n_embed,
            vqvae_groups=config.vqvae_groups,
            encoder_loss_multiplier=1.0,
            act_scale=1.0,
            tokenizer_len=self.tokenizer.tokenizer_len,
        )

    def forward(
        self,
        pixel_values: dict[str, torch.Tensor],
        instruction: list[str],
        action: Optional[torch.Tensor] = None,
    ):
        """
        Training forward pass.
        pixel_values: {"dino": tensor, "siglip": tensor}
        instruction: list of task strings
        action: [B, chunk_size, action_dim] normalized actions
        Returns: CausalLM loss
        """
        batch_size = len(instruction)

        if action is not None:
            # === Encode actions to VQ token IDs ===
            action_token_ids = self.vq_tokenizer.encode_token_ids(action)  # (B, vqvae_groups)

            # === Build prompts ===
            input_ids_list = []
            labels_list = []
            attention_mask_list = []

            for i in range(batch_size):
                # Build full prompt with action tokens
                action_text = ""
                for j in range(self.config.vqvae_groups):
                    token_id = action_token_ids[i, j].item()
                    action_text += self.tokenizer.tokenizer.decode([token_id])

                prompt = self.tokenizer.build_prompt(instruction[i], action_text)
                encoded = self.tokenizer.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                )
                input_ids = encoded["input_ids"].squeeze(0)  # (seq_len,)
                seq_len = len(input_ids)

                # === Build labels ===
                # Only action tokens + <|im_end|> + <|endoftext|> are kept
                # Rest is IGNORE_INDEX
                labels = torch.full_like(input_ids, IGNORE_INDEX)

                # Find action token positions (last vqvae_groups + 2 tokens)
                # The action tokens are at the end of the assistant response
                # We keep the last vqvae_groups + 2 tokens as labels
                num_keep = self.config.vqvae_groups + 2  # action tokens + im_end + endoftext
                labels[-num_keep:] = input_ids[-num_keep:]

                input_ids_list.append(input_ids)
                labels_list.append(labels)
                attention_mask_list.append(torch.ones(seq_len, dtype=torch.long))

            # === Pad sequences ===
            input_ids = nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = nn.utils.rnn.pad_sequence(
                labels_list, batch_first=True, padding_value=IGNORE_INDEX
            )
            attention_mask = nn.utils.rnn.pad_sequence(
                attention_mask_list, batch_first=True, padding_value=0
            )
        else:
            # Inference: build prompt without action
            input_ids_list = []
            attention_mask_list = []
            for i in range(batch_size):
                prompt = self.tokenizer.build_inference_prompt(instruction[i])
                encoded = self.tokenizer.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                )
                input_ids = encoded["input_ids"].squeeze(0)
                input_ids_list.append(input_ids)
                attention_mask_list.append(torch.ones(len(input_ids), dtype=torch.long))

            input_ids = nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            attention_mask = nn.utils.rnn.pad_sequence(
                attention_mask_list, batch_first=True, padding_value=0
            )
            labels = None

        input_ids = input_ids.to(pixel_values["dino"].device)
        attention_mask = attention_mask.to(pixel_values["dino"].device)
        if labels is not None:
            labels = labels.to(pixel_values["dino"].device)

        # === Forward through VLM ===
        outputs = self.vlm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs

    @torch.no_grad()
    def predict_action_chunk(
        self,
        pixel_values: dict[str, torch.Tensor],
        instruction: list[str],
        do_sample: bool = False,
        temperature: float = 0.0,
        max_new_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Official autoregressive action prediction.
        Returns: [B, chunk_size, action_dim]
        """
        batch_size = len(instruction)
        if max_new_tokens is None:
            max_new_tokens = self.config.vqvae_groups

        # === Build inference prompt ===
        input_ids_list = []
        for i in range(batch_size):
            prompt = self.tokenizer.build_inference_prompt(instruction[i])
            encoded = self.tokenizer.tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
            input_ids_list.append(encoded["input_ids"].squeeze(0))

        input_ids = nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(pixel_values["dino"].device)

        # === Run VLM forward to get initial outputs ===
        outputs = self.vlm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
        )

        # === Autoregressive generation ===
        generated = self.vlm.llm.generate(
            inputs_embeds=None,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # === Extract action token IDs ===
        # The generated tokens after the prompt are the action tokens
        prompt_len = input_ids.shape[1]
        action_token_ids = generated[:, prompt_len:prompt_len + self.config.vqvae_groups]

        # === Validate token IDs are in VQ range ===
        vq_token_start = self.tokenizer.tokenizer_len - self.config.vqvae_n_embed
        vq_token_end = self.tokenizer.tokenizer_len - 1
        assert (action_token_ids >= vq_token_start).all() and (action_token_ids <= vq_token_end).all(), (
            f"Generated action tokens out of VQ range [{vq_token_start}, {vq_token_end}]"
        )

        # === Decode to actions ===
        actions = self.vq_tokenizer.decode_token_ids_to_actions(action_token_ids)
        return actions

    def get_optim_params(self) -> dict:
        """
        Returns only trainable parameters (vision, projector, Qwen).
        VQ-VAE is excluded.
        """
        params = []
        # Vision backbone
        if self.config.freeze_vision_backbone:
            pass
        else:
            params.extend(self.vlm.vision_backbone.parameters())

        # Projector
        params.extend(self.vlm.projector.parameters())

        # LLM
        if self.config.freeze_llm_backbone:
            if self.config.unfreeze_last_llm_layer:
                params.extend(self.vlm.llm.layers[-1].parameters())
        else:
            params.extend(self.vlm.llm.parameters())

        return {"params": [p for p in params if p.requires_grad]}


class MiniVLAPolicy(PreTrainedPolicy):
    """
    LeRobot-compatible MiniVLA policy.
    """

    config_class = MiniVLAConfig
    name = "minivla"

    def __init__(self, config: MiniVLAConfig):
        super().__init__(config)
        self.model = MiniVLACore(config)
        self._action_queue: Optional[torch.Tensor] = None

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        """
        Training forward.
        batch contains:
          - observation.images.*: image tensors
          - task: list of instruction strings
          - action: [B, chunk_size, action_dim] normalized actions
        """
        # === Extract pixel values ===
        pixel_values = self._extract_pixel_values(batch)

        # === Extract instruction ===
        instruction = batch.get("task", [""] * batch["action"].shape[0])
        if isinstance(instruction, torch.Tensor):
            instruction = ["" for _ in range(len(instruction))]

        # === Forward ===
        outputs = self.model(
            pixel_values=pixel_values,
            instruction=instruction,
            action=batch["action"],
        )

        return outputs.loss, None

    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> torch.Tensor:
        """
        Returns [B, chunk_size, action_dim].
        """
        pixel_values = self._extract_pixel_values(batch)
        instruction = batch.get("task", [""] * batch["action"].shape[0] if "action" in batch else 1)
        if isinstance(instruction, torch.Tensor):
            instruction = ["" for _ in range(len(instruction))]

        return self.model.predict_action_chunk(
            pixel_values=pixel_values,
            instruction=instruction,
            **kwargs,
        )

    def select_action(
        self, batch: dict[str, torch.Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> torch.Tensor:
        """
        Returns [B, action_dim] (chunk[:, 0]).
        Only executes the first step of the chunk.
        """
        if self._action_queue is None or self._action_queue.shape[1] == 0:
            # Predict new chunk
            chunk = self.predict_action_chunk(batch, **kwargs)
            self._action_queue = chunk

        action = self._action_queue[:, 0]
        # Remove first step from queue
        self._action_queue = self._action_queue[:, 1:]
        return action

    def reset(self):
        """Clear action queue and generation cache."""
        self._action_queue = None
        if hasattr(self.model.vlm.llm, "generation_config"):
            pass  # Cache is handled by transformers

    def get_optim_params(self) -> dict:
        return self.model.get_optim_params()

    def _extract_pixel_values(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Extract DINO and SigLIP pixel values from LeRobot batch.
        Expects batch to already contain "dino" and "siglip" keys from processor.
        """
        if "dino" in batch and "siglip" in batch:
            return {"dino": batch["dino"], "siglip": batch["siglip"]}

        # Fallback: look for observation.images.* keys
        pixel_values = {}
        for key in batch:
            if key.startswith("observation.images"):
                pixel_values[key] = batch[key]
        return pixel_values


# ---------------------------------------------------------------------------
# Policy aliases for LeRobot config -> policy class name resolution
# ---------------------------------------------------------------------------
class MiniVLAT2Policy(MiniVLAPolicy):
    """Alias for minivla_t2 variant."""
    config_class = MiniVLAConfig  # Will be overridden by config type
    name = "minivla_t2"


class MiniVLAWristPolicy(MiniVLAPolicy):
    """Alias for minivla_wrist variant."""
    config_class = MiniVLAConfig  # Will be overridden by config type
    name = "minivla_wrist"