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
  - Inference: batch-safe autoregressive action token generation with KV cache
  - predict_action_chunk: returns [B, chunk_size, action_dim]
  - select_action: returns [B, action_dim] (chunk[:, 0])
  - get_optim_params: only trainable vision, projector, Qwen params
  - Official checkpoint loading following prismatic/models/load.py and prismatic.py::from_pretrained
  - LeRobot checkpoint save/restore via standard PreTrainedPolicy
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

        self.vlm.set_pad_token_id(self.tokenizer.pad_token_id)

        # === Action Tokenizer ===
        tokenizer_len = self.tokenizer.tokenizer_len

        if config.is_vq_mode:
            vq_path = config.resolve_vq_model_path()
            if not vq_path:
                raise ValueError(
                    "VQ mode requires vq_model_path or official_vla_checkpoint "
                    "pointing to a directory with a 'vq' subdirectory."
                )
            self.action_tokenizer = VQActionTokenizer(
                tokenizer=self.tokenizer.tokenizer,
                vq_vae_path=vq_path,
                device="cpu",
                use_extra=True,
            )
            self.vq_vae = self.action_tokenizer.vq_vae
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
        Load official MiniVLA checkpoint following
        teach_code/MiniVLA/prismatic/models/vlms/prismatic.py::from_pretrained
        and teach_code/MiniVLA/prismatic/models/load.py::load_vla.

        Expects checkpoint["model"] with keys: projector, llm_backbone, optional vision_backbone.
        Does NOT use strict=False silently; raises on unexpected missing/unexpected keys
        except for known wrapper/prefix differences.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Official checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint.get("model", checkpoint)

        # Load projector
        if "projector" not in model_state:
            raise ValueError(
                "Official checkpoint missing 'projector' key. "
                "Expected keys: ['projector', 'llm_backbone', optional 'vision_backbone']."
            )
        missing, unexpected = self.vlm.projector.load_state_dict(model_state["projector"], strict=True)
        if missing:
            raise ValueError(f"[Official Checkpoint] Missing projector keys: {missing}")
        if unexpected:
            raise ValueError(f"[Official Checkpoint] Unexpected projector keys: {unexpected}")

        # Load LLM backbone
        if "llm_backbone" not in model_state:
            raise ValueError("Official checkpoint missing 'llm_backbone' key.")
        llm_state = model_state["llm_backbone"]

        # Filter: remove any 'llm.' prefix that might exist in official checkpoints
        filtered_llm_state = {}
        for k, v in llm_state.items():
            clean_key = k.replace("llm.", "") if k.startswith("llm.") else k
            filtered_llm_state[clean_key] = v

        # Handle embedding padding differences (official may have different vocab size)
        official_embed_shape = filtered_llm_state.get("model.embed_tokens.weight", None)
        if official_embed_shape is not None:
            official_vocab_size = official_embed_shape.shape[0]
            current_vocab_size = self.vlm.llm.model.embed_tokens.weight.shape[0]
            if official_vocab_size != current_vocab_size:
                # Resize to official size, then our extra tokens will be re-added
                self.vlm.llm.resize_token_embeddings(official_vocab_size)

        missing, unexpected = self.vlm.llm.load_state_dict(filtered_llm_state, strict=True)
        if missing:
            raise ValueError(f"[Official Checkpoint] Missing LLM keys: {missing}")
        if unexpected:
            raise ValueError(f"[Official Checkpoint] Unexpected LLM keys: {unexpected}")

        # Load vision backbone (optional)
        if "vision_backbone" in model_state:
            missing, unexpected = self.vlm.vision_backbone.load_state_dict(
                model_state["vision_backbone"], strict=True
            )
            if missing:
                raise ValueError(f"[Official Checkpoint] Missing vision keys: {missing}")
            if unexpected:
                raise ValueError(f"[Official Checkpoint] Unexpected vision keys: {unexpected}")

        # Load adjacent config.json for action tokenizer type
        config_dir = checkpoint_path.parents[1]
        config_json = config_dir / "config.json"
        if config_json.exists():
            with open(config_json, "r") as f:
                vla_config = json.load(f)
            if "vla" in vla_config and "action_tokenizer" in vla_config["vla"]:
                print(f"[Official Checkpoint] Action tokenizer type: {vla_config['vla']['action_tokenizer']}")

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

        if action is None and batch_size == 0:
            batch_size = pixel_values["dino"].shape[0]
            instruction = [""] * batch_size

        if action is not None:
            action_texts = []
            for i in range(batch_size):
                single_action = action[i].cpu().numpy()

                if self.config.is_vq_mode:
                    action_tensor = single_action
                    action_text = self.action_tokenizer(action_tensor)
                else:
                    action_text = self.action_tokenizer(single_action[0])

                action_texts.append(action_text)

            input_ids_list = []
            labels_list = []
            attention_mask_list = []

            for i in range(batch_size):
                prompt = self.tokenizer.build_prompt(instruction[i], action_texts[i])
                encoded = self.tokenizer.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                )
                input_ids = encoded["input_ids"].squeeze(0)
                seq_len = len(input_ids)

                action_tokens = self.tokenizer.tokenizer(action_texts[i])["input_ids"]
                num_answer_tokens = len(action_tokens)
                num_end_tokens = 2

                labels = input_ids.clone()
                labels[: -(num_answer_tokens + num_end_tokens)] = IGNORE_INDEX

                input_ids_list.append(input_ids)
                labels_list.append(labels)
                attention_mask_list.append(torch.ones(seq_len, dtype=torch.long))

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
                input_ids_list.append(encoded["input_ids"].squeeze(0))
                attention_mask_list.append(torch.ones(len(input_ids_list[-1]), dtype=torch.long))

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
        Batch-safe: each sample executes image+prompt forward(use_cache=True) ->
        read last position logits -> greedy/sample -> send single token + past_key_values
        back to VLM -> repeat for required token count.
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
            num_action_tokens = self.config.action_feature.shape[0]

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

        # === Batch-safe autoregressive generation with KV cache ===
        # First forward pass: image + full prompt -> get past_key_values and last logits
        outputs = self.vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=torch.ones_like(input_ids, device=input_ids.device),
            use_cache=True,
        )

        # Get the logits from the last position
        logits = outputs.logits[:, -1, :]  # [B, vocab_size]
        past_key_values = outputs.past_key_values

        # Build attention mask for generation (right padding)
        attention_mask = torch.ones(
            input_ids.shape[0], input_ids.shape[1],
            dtype=torch.long, device=input_ids.device
        )

        generated_token_ids = []
        current_input_ids = input_ids

        for step in range(max_new_tokens):
            if do_sample and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated_token_ids.append(next_token)

            if step < max_new_tokens - 1:
                # Feed single token + past_key_values back to VLM
                outputs = self.vlm(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                # Update attention mask
                new_mask = torch.ones(
                    next_token.shape[0], 1,
                    dtype=torch.long, device=next_token.device
                )
                attention_mask = torch.cat([attention_mask, new_mask], dim=1)

        # === Extract action token IDs ===
        action_token_ids = torch.cat(generated_token_ids, dim=1)  # [B, num_tokens]

        # === Decode to actions ===
        action_token_ids_np = action_token_ids.cpu().numpy()

        if self.config.is_vq_mode:
            actions = self.action_tokenizer.decode_token_ids_to_actions(action_token_ids_np)
            actions = torch.from_numpy(actions).float()
            if actions.ndim == 1:
                actions = actions.unsqueeze(0).unsqueeze(0)
            elif actions.ndim == 2:
                actions = actions.unsqueeze(1)
        else:
            actions = self.action_tokenizer.decode_token_ids_to_actions(action_token_ids_np)
            actions = torch.from_numpy(actions).float()
            if actions.ndim == 1:
                actions = actions.unsqueeze(0).unsqueeze(0)
            elif actions.ndim == 2:
                actions = actions.unsqueeze(1)

        # Ensure output is on same device as pixel_values
        actions = actions.to(pixel_values["dino"].device)
        return actions

    def get_optim_params(self) -> dict:
        """
        Returns only trainable parameters (vision, projector, Qwen).
        VQ-VAE is excluded.
        """
        params = []
        if not self.config.freeze_vision_backbone:
            params.extend(self.vlm.vision_backbone.parameters())

        params.extend(self.vlm.projector.parameters())

        if self.config.freeze_llm_backbone:
            if self.config.unfreeze_last_llm_layer:
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
        pixel_values = self._extract_pixel_values(batch)

        if "task" in batch:
            instruction = batch["task"]
            if isinstance(instruction, torch.Tensor):
                instruction = ["" for _ in range(len(instruction))]
        else:
            if "action" in batch:
                batch_size = batch["action"].shape[0]
            else:
                batch_size = pixel_values["dino"].shape[0]
            instruction = [""] * batch_size

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
        pixel_values = self._extract_pixel_values(batch)

        if "task" in batch:
            instruction = batch["task"]
            if isinstance(instruction, torch.Tensor):
                instruction = ["" for _ in range(len(instruction))]
        else:
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
        if self._action_queue is None or self._action_queue.shape[1] == 0:
            chunk = self.predict_action_chunk(batch, **kwargs)
            n_steps = self.config.n_action_steps
            self._action_queue = chunk[:, :n_steps]

        action = self._action_queue[:, 0]
        self._action_queue = self._action_queue[:, 1:]
        return action

    def reset(self):
        self._action_queue = None

    def get_optim_params(self) -> dict:
        return self.model.get_optim_params()

    def _extract_pixel_values(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Extract DINO and SigLIP pixel values from LeRobot batch.
        Expects batch to already contain "dino" and "siglip" keys from processor.
        Raises clear error if missing.
        """
        if "dino" in batch and "siglip" in batch:
            return {"dino": batch["dino"], "siglip": batch["siglip"]}

        raise ValueError(
            "Batch missing 'dino' and 'siglip' keys. "
            "The MiniVLA processor must run first to convert observation.images.* "
            "into DINO/SigLIP normalized tensors. "
            "Ensure make_minivla_pre_post_processors() is configured and applied."
        )


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