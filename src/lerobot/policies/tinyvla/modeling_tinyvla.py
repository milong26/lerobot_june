#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..pretrained import PreTrainedPolicy
from .configuration_tinyvla import TinyVLAConfig
from .llava_pythia.model.language_model.pythia.llava_pythia import (
    LlavaPythiaConfig,
    LlavaPythiaForCausalLM,
)
from .llava_pythia.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX
from .llava_pythia.llava_pythia_utils import find_all_linear_names

logger = logging.getLogger(__name__)


class TinyVLAPolicy(PreTrainedPolicy):
    """
    TinyVLA (Vision-Language-Action) Policy.

    TinyVLA uses a LLaVA-Pythia VLM backbone with a DROID UNet Diffusion or DETR VAE action head.
    It processes multi-camera images, robot state, and language instructions to predict action chunks.
    """

    config_class = TinyVLAConfig
    name = "tinyvla"

    def __init__(
        self,
        config: TinyVLAConfig,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._build_model()
        self._setup_tokenizer()
        self._setup_image_processor()

        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self.reset()

    def _build_model(self):
        """Build the LLaVA-Pythia model with action head."""
        llava_config = LlavaPythiaConfig.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
        )

        # Set action head parameters
        llava_config.action_head_type = self.config.action_head_type
        llava_config.action_dim = self.config.action_dim
        llava_config.state_dim = self.config.state_dim
        llava_config.chunk_size = self.config.chunk_size
        llava_config.concat = "token_cat"

        # Build the model
        self.model = LlavaPythiaForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            config=llava_config,
            trust_remote_code=True,
            _fast_init=False,
        )

        self.model.config.use_cache = False

        # Apply freezing
        if self.config.freeze_backbone:
            self.model.get_model().requires_grad_(False)
        else:
            self.model.get_model().requires_grad_(True)

        if self.config.freeze_vision_tower:
            for p in self.model.get_model().vision_tower.parameters():
                p.requires_grad = False

        # Always train action head
        self.model.proj_to_action.requires_grad_(True)
        
        # For diffusion head, initialize it now so we can set requires_grad
        if self.config.action_head_type == 'droid_diffusion':
            self.model._init_diffusion_head()
        
        if self.model.embed_out is not None:
            self.model.embed_out.requires_grad_(True)

        # Apply LoRA if enabled
        if self.config.lora_enable:
            self._apply_lora()

        self.model.to(self.config.device)
        
        # Move diffusion head to device if it exists
        if self.config.action_head_type == 'droid_diffusion' and hasattr(self.model, 'embed_out') and self.model.embed_out is not None:
            self.model.embed_out.to(self.config.device)

    def _apply_lora(self):
        """Apply LoRA to the model."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "LoRA is enabled but 'peft' is not installed. "
                "Install it with: pip install peft, "
                "or disable LoRA by setting --policy.lora_enable=false"
            )

        def rank0_print(*args):
            print(*args)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=find_all_linear_names(
                self.model, rank0_print, "vit llm"
            ),
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _setup_tokenizer(self):
        """Set up the tokenizer for language processing."""
        import transformers

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            model_max_length=self.config.tokenizer_max_length,
            padding_side="right",
            trust_remote_code=True,
        )
        self.tokenizer.pad_token_id = 1

    def _setup_image_processor(self):
        """Set up the image processor (CLIP or SigLIP)."""
        vision_tower = self.model.get_model().vision_tower
        vision_config = vision_tower.config

        if hasattr(vision_config, "image_size"):
            self.image_size = vision_config.image_size
        else:
            self.image_size = 224

        # Get the image processor from the vision tower
        self.image_mean = getattr(vision_tower, "image_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]))
        self.image_std = getattr(vision_tower, "image_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]))

    def _process_images(self, images: list[Tensor]) -> Tensor:
        """Process images for the vision encoder.

        Args:
            images: List of image tensors, each of shape (B, C, H, W).

        Returns:
            Processed image tensor.
        """
        device = self.config.device

        # Stack all camera views along batch dimension for processing
        all_images = torch.cat(images, dim=0).to(device)  # (num_cam * B, C, H, W)

        # Resize to expected size
        if all_images.shape[-2:] != (self.image_size, self.image_size):
            all_images = F.interpolate(
                all_images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Normalize
        mean = self.image_mean.to(device).view(1, 3, 1, 1)
        std = self.image_std.to(device).view(1, 3, 1, 1)
        all_images = (all_images - mean) / std

        return all_images

    def _tokenize_language(self, raw_lang: list[str]) -> tuple[Tensor, Tensor]:
        """Tokenize language instructions.

        Args:
            raw_lang: List of language instruction strings.

        Returns:
            Tuple of (input_ids, labels).
        """
        batch_input_ids = []
        batch_labels = []

        for lang in raw_lang:
            prompt = f"{DEFAULT_IMAGE_TOKEN}\n{lang}"

            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                max_length=self.config.tokenizer_max_length,
                truncation=True,
            )
            input_ids = encoded.input_ids[0]
            labels = input_ids.clone()

            labels[:] = IGNORE_INDEX

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)

        # Pad to max length
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_input_ids = torch.full((len(batch_input_ids), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((len(batch_labels), max_len), IGNORE_INDEX, dtype=torch.long)

        for i, (ids, lbls) in enumerate(zip(batch_input_ids, batch_labels)):
            padded_input_ids[i, :len(ids)] = ids
            padded_labels[i, :len(lbls)] = lbls

        return padded_input_ids.to(self.config.device), padded_labels.to(self.config.device)

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if p.requires_grad
                ],
            }
        ]

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save the policy model and configuration to a directory."""
        super()._save_pretrained(save_directory)
        logger.info(f"Saved TinyVLA policy to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: TinyVLAConfig | None = None,
        **kwargs,
    ) -> "TinyVLAPolicy":
        """Load a TinyVLA policy from a pretrained checkpoint."""
        policy = super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            **kwargs,
        )
        return policy

    def reset(self):
        """Reset the action queue. Called whenever the environment is reset."""
        self._action_queue.clear()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action to execute in the environment.

        Uses action chunking: predicts a full chunk and caches it, then pops one action at a time.
        """
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        states = batch[OBS_STATE].to(self.config.device) if OBS_STATE in batch else None
        image_keys = sorted([k for k in self.config.image_features if k in batch])
        if not image_keys:
            image_keys = sorted([k for k in batch if k.startswith(OBS_IMAGES)])
        images = [batch[k].to(self.config.device) for k in image_keys]

        if len(images) == 0:
            raise ValueError("No image observations found in batch. TinyVLA requires at least one camera.")

        # Process images
        processed_images = self._process_images(images)

        # Get language
        batch_size = states.shape[0] if states is not None else images[0].shape[0]
        raw_lang = batch.get("language", [""] * batch_size)

        if isinstance(raw_lang, Tensor):
            raw_lang = [""] * batch_size

        if isinstance(raw_lang, str):
            raw_lang = [raw_lang] * batch_size

        input_ids, labels = self._tokenize_language(raw_lang)

        # Split images for multi-camera
        num_cams = len(images)
        image_chunks = torch.chunk(processed_images, num_cams, dim=0)

        kwargs = {
            "input_ids": input_ids,
            "labels": labels,
            "images": image_chunks[0],
            "states": states,
            "eval": True,
        }
        if num_cams > 1:
            kwargs["images_r"] = image_chunks[1]
        if num_cams > 2:
            kwargs["images_top"] = image_chunks[2]

        actions = self.model(**kwargs)
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Dictionary containing:
                - observation.images.*: Image tensors (B, C, H, W)
                - observation.state: Robot state (B, state_dim)
                - action: Action tensor (B, chunk_size, action_dim)
                - action_is_pad: Padding mask (B, chunk_size)
                - language: Language instruction strings (optional)

        Returns:
            Tuple of (loss, info_dict).
        """
        # Prepare inputs
        states = batch[OBS_STATE].to(self.config.device) if OBS_STATE in batch else None
        actions = batch[ACTION].to(self.config.device)
        is_pad = batch.get("action_is_pad", torch.zeros(actions.shape[:2], dtype=torch.bool, device=self.config.device))
        is_pad = is_pad.to(self.config.device)

        image_keys = sorted([k for k in self.config.image_features if k in batch])
        if not image_keys:
            image_keys = sorted([k for k in batch if k.startswith(OBS_IMAGES)])
        images = [batch[k].to(self.config.device) for k in image_keys]

        # Process images
        processed_images = self._process_images(images)

        # Get language
        batch_size = actions.shape[0]
        raw_lang = batch.get("language", None)
        if raw_lang is None or (isinstance(raw_lang, Tensor) and raw_lang.numel() == 0):
            raw_lang = [""] * batch_size
        elif isinstance(raw_lang, str):
            raw_lang = [raw_lang] * batch_size
        elif isinstance(raw_lang, (list, np.ndarray)):
            raw_lang = list(raw_lang)
        else:
            raw_lang = [""] * batch_size

        input_ids, labels = self._tokenize_language(raw_lang)

        # Split images for multi-camera
        num_cams = len(images)
        image_chunks = torch.chunk(processed_images, num_cams, dim=0)

        # Run model forward
        kwargs = {
            "input_ids": input_ids,
            "labels": labels,
            "images": image_chunks[0],
            "actions": actions,
            "states": states,
            "is_pad": is_pad,
            "eval": False,
        }
        if num_cams > 1:
            kwargs["images_r"] = image_chunks[1]
        if num_cams > 2:
            kwargs["images_top"] = image_chunks[2]

        output = self.model(**kwargs)

        if isinstance(output, dict):
            loss = output["loss"]
            info = {k: v.item() if isinstance(v, Tensor) else v for k, v in output.items() if k != "loss"}
        else:
            loss = output[0] if isinstance(output, tuple) else output
            info = {}

        return loss, info