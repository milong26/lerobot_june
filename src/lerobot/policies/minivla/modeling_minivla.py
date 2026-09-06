"""
modeling_minivla.py

Official MiniVLA policy for LeRobot.
Mirrors teach_code/MiniVLA/prismatic/models/vlas/openvla.py,
prismatic/models/vlms/prismatic.py, prismatic/vla/datasets/datasets.py,
and vla-scripts/train.py.

Key design:
  - MiniVLACore: vision_backbone + projector + Qwen CausalLM + frozen action tokenizer
  - No independent action_logits_head
  - Training: action -> action tokenizer -> action token text -> Qwen prompt -> input_ids/labels -> CausalLM loss
  - Inference: super().generate with pixel_values -> decode to [B, chunk_size, A]
  - predict_action_chunk: returns [B, chunk_size, action_dim]
  - select_action: returns [B, action_dim] (chunk[:, 0])
  - get_optim_params: only trainable vision, projector, Qwen params
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Unpack

import numpy as np
import torch
import torch.nn as nn

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy

from .configuration_minivla import MiniVLAConfig, MiniVLAT2Config, MiniVLAWristConfig
from .tokenizer import VLATokenizerWrapper
from .vq_action import VQActionTokenizer, ActionTokenizer
from .vla_backbone import MiniVLAVLBackbone, IGNORE_INDEX


class MiniVLACore(nn.Module):
    """
    Official MiniVLA core model.
    vision_backbone + FusedMLPProjector + Qwen2.5 CausalLM + frozen action tokenizer.
    """

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.config = config

        # === Tokenizer (shared between VLM and action tokenizer) ===
        self.tokenizer = VLATokenizerWrapper(
            base_vlm_checkpoint=config.base_vlm_checkpoint,
            num_extra_tokens=config.num_extra_tokens,
        )

        # === VLM Backbone ===
        # Pass tokenizer_len from shared tokenizer for embedding resize
        self.vlm = MiniVLAVLBackbone(
            vision_backbone_id=config.vision_backbone_id,
            llm_backbone_id=config.llm_backbone_id,
            base_vlm_checkpoint=config.base_vlm_checkpoint,
            image_size=config.image_size,
            image_resize_strategy=config.image_resize_strategy,
            arch_specifier=config.arch_specifier,
            image_sequence_len=config.image_sequence_len,
            tokenizer_len=self.tokenizer.tokenizer_len,
            enable_gradient_checkpointing=config.enable_gradient_checkpointing,
            freeze_vision_backbone=config.freeze_vision_backbone,
            freeze_llm_backbone=config.freeze_llm_backbone,
            unfreeze_last_llm_layer=config.unfreeze_last_llm_layer,
        )

        # Sync pad_token_id from shared tokenizer to LLM config
        self.vlm.set_pad_token_id(self.tokenizer.pad_token_id)

        # === Action Tokenizer ===
        # Sync the tokenizer_len from shared tokenizer
        tokenizer_len = self.tokenizer.tokenizer_len

        if config.is_vq_mode:
            self.action_tokenizer = VQActionTokenizer(
                tokenizer=self.tokenizer.tokenizer,
                vq_vae_path=config.vq_model_path,
                device="cpu",  # Will be moved by .to(device)
                use_extra=True,
            )
            # Register VQ-VAE as nn.Module so it moves with .to(device)
            self.vq_vae = self.action_tokenizer.vq_vae
            # Ensure VQ is frozen and in eval mode
            for param in self.vq_vae.parameters():
                param.requires_grad = False
            self.vq_vae.eval()
        else:
            self.action_tokenizer = ActionTokenizer(
                tokenizer=self.tokenizer.tokenizer,
                bins=256,
                min_action=-1,
                max_action=1,
                use_extra=True,
            )
            self.vq_vae = None

        # === Load official checkpoint if specified ===
        if config.official_vla_checkpoint:
            self._load_official_checkpoint(config.official_vla_checkpoint)

    def _load_official_checkpoint(self, checkpoint_path: str):
        """
        Load official MiniVLA checkpoint following teach_code/MiniVLA/prismatic/models/vlms/prismatic.py::from_pretrained.
        Expects checkpoint["model"] with keys: projector, llm_backbone, optional vision_backbone.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Official checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint.get("model", checkpoint)

        # Load projector
        if "projector" in model_state:
            missing, unexpected = self.vlm.projector.load_state_dict(model_state["projector"], strict=False)
            if missing:
                print(f"[Official Checkpoint] Missing projector keys: {missing}")
            if unexpected:
                print(f"[Official Checkpoint] Unexpected projector keys: {unexpected}")

        # Load LLM backbone
        if "llm_backbone" in model_state:
            llm_state = model_state["llm_backbone"]
            # Filter out keys that don't belong to the LLM
            filtered_llm_state = {}
            for k, v in llm_state.items():
                # Remove any prefix if present
                clean_key = k.replace("llm.", "") if k.startswith("llm.") else k
                filtered_llm_state[clean_key] = v
            missing, unexpected = self.vlm.llm.load_state_dict(filtered_llm_state, strict=False)
            if missing:
                print(f"[Official Checkpoint] Missing LLM keys: {missing}")
            if unexpected:
                print(f"[Official Checkpoint] Unexpected LLM keys: {unexpected}")

        # Load vision backbone (optional)
        if "vision_backbone" in model_state:
            missing, unexpected = self.vlm.vision_backbone.load_state_dict(model_state["vision_backbone"], strict=False)
            if missing:
                print(f"[Official Checkpoint] Missing vision keys: {missing}")
            if unexpected:
                print(f"[Official Checkpoint] Unexpected vision keys: {unexpected}")

        # Load adjacent config.json for action tokenizer type
        config_dir = checkpoint_path.parent
        config_json = config_dir / "config.json"
        if config_json.exists():
            with open(config_json, "r") as f:
                vla_config = json.load(f)
            if "action_tokenizer" in vla_config:
                print(f"[Official Checkpoint] Action tokenizer type: {vla_config['action_tokenizer']}")

        # Load dataset_statistics.json for action denormalization
        dataset_stats = config_dir / "dataset_statistics.json"
        if dataset_stats.exists():
            with open(dataset_stats, "r") as f:
                self.dataset_statistics = json.load(f)
            print(f"[Official Checkpoint] Loaded dataset statistics from {dataset_stats}")

        print(f"[Official Checkpoint] Loaded from {checkpoint_path}")

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
        action: [B, chunk_size, action_dim] normalized actions (in [-1, 1] range)
        Returns: CausalLM loss
        """
        batch_size = len(instruction)

        # Infer batch size from pixel_values if action is None
        if action is None and batch_size == 0:
            batch_size = pixel_values["dino"].shape[0]
            instruction = [""] * batch_size

        if action is not None:
            # === Encode actions to text ===
            action_texts = []
            for i in range(batch_size):
                single_action = action[i].cpu().numpy()  # [chunk_size, action_dim]

                if self.config.is_vq_mode:
                    # VQ mode: action is [chunk_size, action_dim] -> encode to VQ codes
                    action_tensor = torch.from_numpy(single_action).unsqueeze(0)  # [1, chunk_size, action_dim]
                    action_text = self.action_tokenizer(action_tensor)
                else:
                    # Non-VQ mode: use last action step (single-step discretization)
                    action_text = self.action_tokenizer(single_action[-1])  # Last step only

                action_texts.append(action_text)

            # === Build prompts ===
            input_ids_list = []
            labels_list = []
            attention_mask_list = []

            for i in range(batch_size):
                # Build full prompt with action tokens
                prompt = self.tokenizer.build_prompt(instruction[i], action_texts[i])
                encoded = self.tokenizer.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                )
                input_ids = encoded["input_ids"].squeeze(0)  # (seq_len,)
                seq_len = len(input_ids)

                # === Build labels matching official RLDSBatchTransform ===
                # Tokenize the action text to find num_answer_tokens
                action_tokens = self.tokenizer.tokenizer(action_texts[i])["input_ids"]
                num_answer_tokens = len(action_tokens)

                # Qwen has 2 end tokens: <|im_end|><|endoftext|>
                num_end_tokens = 2

                # Create labels: only action tokens and end tokens participate in loss
                labels = input_ids.clone()
                labels[: -(num_answer_tokens + num_end_tokens)] = IGNORE_INDEX

                input_ids_list.append(input_ids)
                labels_list.append(labels)
                # Create attention mask: 1 for valid tokens, 0 for padding
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
        Official multi-modal autoregressive action prediction.
        Uses super().generate with pixel_values to leverage KV cache.
        Returns: [B, chunk_size, action_dim]
        """
        batch_size = len(instruction)
        if batch_size == 0:
            batch_size = pixel_values["dino"].shape[0]
            instruction = [""] * batch_size

        # Determine number of tokens to generate
        if self.config.is_vq_mode:
            num_action_tokens = self.config.vqvae_groups
        else:
            num_action_tokens = self.config.vq_action_dim

        if max_new_tokens is None:
            max_new_tokens = num_action_tokens

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

        # === Multi-modal generation using GenerationMixin ===
        # This calls super().generate which uses prepare_inputs_for_generation
        # to properly handle pixel_values in the first step and KV cache thereafter
        generated_ids = self.vlm.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # === Extract action token IDs ===
        prompt_len = input_ids.shape[1]
        action_token_ids = generated_ids[:, prompt_len:prompt_len + num_action_tokens]

        # === Decode to actions ===
        action_token_ids_np = action_token_ids.cpu().numpy()

        if self.config.is_vq_mode:
            # VQ mode: decode VQ token IDs to [B, chunk_size, action_dim]
            actions = self.action_tokenizer.decode_token_ids_to_actions(action_token_ids_np)
            # Ensure output shape is [B, chunk_size, action_dim]
            if actions.ndim == 2:
                actions = actions.unsqueeze(1)  # [B, 1, A] -> need to expand
            # VQ returns [B, 0] (first horizon element), reshape to [B, 1, A]
            if actions.ndim == 2:
                actions = actions.unsqueeze(1)
        else:
            # Non-VQ mode: decode to [B, action_dim]
            actions = self.action_tokenizer.decode_token_ids_to_actions(action_token_ids_np)
            # Expand to [B, 1, action_dim] for consistency
            actions = actions.reshape(batch_size, 1, -1)

        return actions

    def get_optim_params(self) -> dict:
        """
        Returns only trainable parameters (vision, projector, Qwen).
        VQ-VAE is excluded.
        """
        params = []
        # Vision backbone
        if not self.config.freeze_vision_backbone:
            params.extend(self.vlm.vision_backbone.parameters())

        # Projector
        params.extend(self.vlm.projector.parameters())

        # LLM
        if self.config.freeze_llm_backbone:
            if self.config.unfreeze_last_llm_layer:
                # Use correct Qwen path: self.llm.model.layers[-1]
                params.extend(self.vlm.llm.model.layers[-1].parameters())
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
        if "task" in batch:
            instruction = batch["task"]
            if isinstance(instruction, torch.Tensor):
                instruction = ["" for _ in range(len(instruction))]
        else:
            # Infer batch size from action or pixel_values
            if "action" in batch:
                batch_size = batch["action"].shape[0]
            else:
                batch_size = pixel_values["dino"].shape[0]
            instruction = [""] * batch_size

        # === Forward ===
        action = batch.get("action", None)
        outputs = self.model(
            pixel_values=pixel_values,
            instruction=instruction,
            action=action,
        )

        return outputs.loss, None

    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> torch.Tensor:
        """
        Returns [B, chunk_size, action_dim].
        """
        pixel_values = self._extract_pixel_values(batch)

        # === Extract instruction ===
        if "task" in batch:
            instruction = batch["task"]
            if isinstance(instruction, torch.Tensor):
                instruction = ["" for _ in range(len(instruction))]
        else:
            # Infer batch size from pixel_values
            batch_size = pixel_values["dino"].shape[0]
            instruction = [""] * batch_size

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
        Follows official: decode_token_ids_to_actions(...) -> ret_action[:, 0]
        """
        if self._action_queue is None or self._action_queue.shape[1] == 0:
            # Predict new chunk
            chunk = self.predict_action_chunk(batch, **kwargs)
            # Cache only the first n_action_steps
            n_steps = self.config.n_action_steps
            self._action_queue = chunk[:, :n_steps]

        action = self._action_queue[:, 0]
        # Remove first step from queue
        self._action_queue = self._action_queue[:, 1:]
        return action

    def reset(self):
        """Clear action queue and generation cache."""
        self._action_queue = None

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
    config_class = MiniVLAT2Config
    name = "minivla_t2"


class MiniVLAWristPolicy(MiniVLAPolicy):
    """Alias for minivla_wrist variant."""
    config_class = MiniVLAWristConfig
    name = "minivla_wrist"