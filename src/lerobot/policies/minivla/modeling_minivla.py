"""
modeling_minivla.py

Official MiniVLA policy for LeRobot.
Mirrors teach_code/MiniVLA/prismatic/models/vlas/openvla.py,
prismatic/models/vlms/prismatic.py, prismatic/models/load.py,
and prismatic/vla/action_tokenizer.py.

Key design:
  - MiniVLACore: vision_backbone + projector + Qwen CausalLM + frozen action tokenizer
  - No independent action_logits_head
  - Training: action -> action tokenizer -> action token text -> Qwen prompt -> input_ids/labels -> CausalLM loss
  - Inference: batch-safe autoregressive action token generation with KV cache
  - predict_action_chunk: returns [B, 1, A] for non-VQ, [B, 1, A] for VQ (first horizon)
  - select_action: returns [B, action_dim] (chunk[:, 0])
  - get_optim_params: only trainable vision, projector, Qwen params
  - Official checkpoint loading following prismatic/models/load.py::load_vla and prismatic.py::from_pretrained
  - LeRobot checkpoint save/restore via standard PreTrainedPolicy
  - Action dimension dynamically from config.action_feature or official dataset statistics
  - Action denormalization using official q01/q99 quantile stats
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Unpack

import numpy as np
import torch
import torch.nn as nn

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy

from .configuration_minivla import MiniVLAConfig, MiniVLAT2Config, MiniVLAWristConfig, VQ_TOKENIZER_TYPES
from .tokenizer import VLATokenizerWrapper
from .vq_action import VQActionTokenizer, ActionTokenizer
from .vla_backbone import MiniVLAVLBackbone, IGNORE_INDEX


class MiniVLACore(nn.Module):
    """
    Official MiniVLA core model.
    vision_backbone + FusedMLPProjector + Qwen2.5 CausalLM + frozen action tokenizer.
    Mirrors teach_code/MiniVLA/prismatic/models/vlas/openvla.py::OpenVLA.
    """

    def __init__(self, config: MiniVLAConfig, **kwargs):
        super().__init__()
        # Store config as a plain Python attribute (not nn.Module parameter)
        # Using object.__setattr__ to bypass PyTorch's __setattr__ which would
        # register it as a submodule and include its fields in parameters()
        self.__dict__['config'] = config
        self.dataset_statistics = None

        # === Read official metadata BEFORE creating VLM and action tokenizer ===
        # Mirrors teach_code/MiniVLA/prismatic/models/load.py::load_vla
        self._read_official_metadata()

        # === Tokenizer (shared between VLM and action tokenizer) ===
        self.tokenizer = VLATokenizerWrapper(
            base_vlm_checkpoint=config.base_vlm_checkpoint,
            num_extra_tokens=config.num_extra_tokens,
        )

        # === Build action tokenizer based on resolved config ===
        self._build_action_tokenizer()

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

        # === Load official checkpoint if specified ===
        if config.official_vla_checkpoint:
            self._load_official_checkpoint(config.official_vla_checkpoint)

    def _read_official_metadata(self):
        """
        Read official checkpoint metadata and update config BEFORE creating VLM and action tokenizer.
        Mirrors teach_code/MiniVLA/prismatic/models/load.py::load_vla.
        If official_vla_checkpoint is specified, reads config.json to get:
        - vla.action_tokenizer (final tokenizer type)
        - vla.base_vlm (base VLM checkpoint)
        - vla.image_sequence_len, vla.use_wrist_image
        - model.vision_backbone_id, model.llm_backbone_id, model.arch_specifier, model.image_size
        These parameters take priority over local default configuration.
        """
        if not self.config.official_vla_checkpoint:
            return

        checkpoint_path = Path(self.config.official_vla_checkpoint)
        if not checkpoint_path.exists() or checkpoint_path.suffix != ".pt":
            return

        run_dir = checkpoint_path.parents[1]
        config_json = run_dir / "config.json"
        if not config_json.exists():
            return

        with open(config_json, "r") as f:
            full_config = json.load(f)

        vla_cfg = full_config.get("vla", {})
        model_cfg = full_config.get("model", {})

        # Parse final action tokenizer type (priority over local config)
        action_tokenizer_type = vla_cfg.get("action_tokenizer", self.config.action_tokenizer_type)
        self.config._resolved_action_tokenizer_type = action_tokenizer_type

        # NOTE: Skip base_vlm override from official config. The official base_vlm field
        # contains a local path name like "prism-qwen25-extra-dinosiglip-224px+0_5b" which
        # is not a valid HuggingFace repo ID. We use the default Qwen/Qwen2.5-0.5B instead.
        # base_vlm = vla_cfg.get("base_vlm", self.config.base_vlm_checkpoint)
        # self.config.base_vlm_checkpoint = base_vlm

        # Parse vision/LLM backbone identifiers from model config (priority over local config)
        vision_backbone_id = model_cfg.get("vision_backbone_id", self.config.vision_backbone_id)
        llm_backbone_id = model_cfg.get("llm_backbone_id", self.config.llm_backbone_id)
        arch_specifier = model_cfg.get("arch_specifier", self.config.arch_specifier)
        image_size = model_cfg.get("image_size", self.config.image_size)

        self.config.vision_backbone_id = vision_backbone_id
        self.config.llm_backbone_id = llm_backbone_id
        self.config.arch_specifier = arch_specifier
        self.config.image_size = image_size

        # Parse image_sequence_len and use_wrist_image (priority over local config)
        image_sequence_len = vla_cfg.get("image_sequence_len", self.config.image_sequence_len)
        use_wrist_image = vla_cfg.get("use_wrist_image", self.config.use_wrist_image)
        self.config.image_sequence_len = image_sequence_len
        self.config.use_wrist_image = use_wrist_image

        # Load dataset statistics for action denormalization
        dataset_statistics_json = run_dir / "dataset_statistics.json"
        if dataset_statistics_json.exists():
            with open(dataset_statistics_json, "r") as f:
                self.dataset_statistics = json.load(f)

    def _build_action_tokenizer(self):
        """
        Build action tokenizer based on resolved config.
        Uses config._resolved_action_tokenizer_type (set by _read_official_metadata)
        or falls back to config.action_tokenizer_type.
        VQ tokenizer behavior must match MiniVLA official action_tokenizer.py exactly.
        """
        action_tokenizer_type = self.config._resolved_action_tokenizer_type or self.config.action_tokenizer_type

        if action_tokenizer_type in VQ_TOKENIZER_TYPES:
            # VQ mode
            vq_path = self.config.resolve_vq_model_path()
            if not vq_path:
                raise ValueError(
                    f"VQ mode requires vq_model_path or official_vla_checkpoint "
                    f"pointing to a directory with a valid VQ config.json and checkpoints/model.pt. "
                    f"Got action_tokenizer_type={action_tokenizer_type}"
                )
            self.action_tokenizer = VQActionTokenizer(
                tokenizer=self.tokenizer.tokenizer,
                vq_vae_path=vq_path,
                device="cuda:0",
                use_extra=True,
            )
            self.vq_vae = self.action_tokenizer.vq_vae
            for param in self.vq_vae.parameters():
                param.requires_grad = False
            self.vq_vae.eval()
        else:
            # Non-VQ mode (extra_action_tokenizer or action_tokenizer)
            self.action_tokenizer = ActionTokenizer(
                tokenizer=self.tokenizer.tokenizer,
                bins=256,
                min_action=-1,
                max_action=1,
                use_extra=True,
            )
            self.vq_vae = None

    def _load_official_checkpoint(self, checkpoint_path: str):
        """
        Load official MiniVLA checkpoint following
        teach_code/MiniVLA/prismatic/models/load.py::load_vla
        and teach_code/MiniVLA/prismatic/models/vlms/prismatic.py::from_pretrained.

        Steps:
        1. Read config.json from checkpoint_path.parents[1] (run directory)
        2. Parse vla.action_tokenizer, base_vlm, and model config
        3. Read dataset_statistics.json for action denormalization
        4. Load projector, llm_backbone, vision_backbone weights
        5. Handle embedding vocab size mismatches
        6. Load VQ-VAE weights if in VQ mode
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Official checkpoint not found: {checkpoint_path}")

        # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
        assert (checkpoint_path.suffix == ".pt") and (checkpoint_path.parent.name == "checkpoints"), (
            "Invalid checkpoint path! Expected path like '<run_dir>/checkpoints/<name>.pt'"
        )
        run_dir = checkpoint_path.parents[1]

        # Get paths for config.json and dataset_statistics.json
        config_json = run_dir / "config.json"
        dataset_statistics_json = run_dir / "dataset_statistics.json"

        if not config_json.exists():
            raise FileNotFoundError(f"Missing config.json for run_dir={run_dir}")

        # Load VLA config
        with open(config_json, "r") as f:
            full_config = json.load(f)

        vla_cfg = full_config.get("vla", {})
        action_tokenizer_type = vla_cfg.get("action_tokenizer", "action_tokenizer")
        base_vlm = vla_cfg.get("base_vlm", self.config.base_vlm_checkpoint)

        # Load dataset statistics for action denormalization
        if dataset_statistics_json.exists():
            with open(dataset_statistics_json, "r") as f:
                self.dataset_statistics = json.load(f)

        # Load checkpoint weights
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

        # Handle embedding vocab size differences
        official_embed_shape = filtered_llm_state.get("model.embed_tokens.weight", None)
        if official_embed_shape is not None:
            official_vocab_size = official_embed_shape.shape[0]
            current_vocab_size = self.vlm.llm.model.embed_tokens.weight.shape[0]
            if official_vocab_size != current_vocab_size:
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

        # If VQ mode, load VQ-VAE weights from official checkpoint directory
        if self.config.is_vq_mode:
            vq_path = self.config.resolve_vq_model_path()
            if vq_path and hasattr(self, "vq_vae") and self.vq_vae is not None:
                vq_model_path = Path(vq_path) / "checkpoints" / "model.pt"
                if vq_model_path.exists():
                    self.vq_vae.load_official_checkpoint(str(vq_model_path))

    def _get_action_dim(self) -> int:
        """
        Get action dimension from config.action_feature or official dataset statistics.
        Mirrors teach_code/MiniVLA/prismatic/models/vlas/openvla.py::get_action_dim.
        """
        if hasattr(self.config, "action_feature") and self.config.action_feature is not None:
            return self.config.action_feature.shape[0]

        if self.dataset_statistics:
            for key, stats in self.dataset_statistics.items():
                if "action" in stats and "q01" in stats["action"]:
                    return len(stats["action"]["q01"])

        if self.config.is_vq_mode:
            return self.config.vq_action_dim

        raise ValueError(
            "Cannot determine action dimension. "
            "Set config.action_feature or provide dataset_statistics."
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
        action: [B, chunk_size, action_dim] normalized actions (in [-1, 1] range)
        Returns: CausalLM loss
        Mirrors teach_code/MiniVLA/prismatic/vla/datasets/datasets.py (RLDSBatchTransform).

        Action token generation logic:
        - VQ mode: action[i, :required_horizon] where required_horizon = required_future_horizon + 1
          This matches official action[-required_future_horizon-1:] slice
        - Non-VQ mode: action[i, delta_idx] where delta_idx = action_delta_indices[0]
          For base config, delta_idx=0 means action[i, 0] (current action)
        """
        batch_size = len(instruction)

        if action is None and batch_size == 0:
            batch_size = pixel_values["dino"].shape[0]
            instruction = [""] * batch_size

        if action is not None:
            # Validate input shape
            assert action.dim() == 3, f"Expected action shape [B, T, A], got {action.shape}"

            action_texts = []
            vq_expected_dim = self.action_tokenizer.vq_vae.input_dim_w
            
            for i in range(batch_size):
                if self.config.is_vq_mode:
                    # VQ mode: use full action chunk based on required_future_horizon
                    # Official: action[-required_future_horizon-1:]
                    # LeRobot action shape: [B, chunk_size, action_dim]
                    # required_horizon = required_future_horizon + 1 (includes current step)
                    required_horizon = self.action_tokenizer.required_future_horizon + 1
                    action_tensor = action[i, :required_horizon].cpu().numpy()
                    
                    # Pad action dimension to match VQ tokenizer expected dimension
                    current_dim = action_tensor.shape[-1]
                    if current_dim < vq_expected_dim:
                        pad_width = [(0, 0)] * (action_tensor.ndim - 1) + [(0, vq_expected_dim - current_dim)]
                        action_tensor = np.pad(action_tensor, pad_width, mode='constant', constant_values=0.0)
                    
                    action_text = self.action_tokenizer(action_tensor)
                else:
                    # Non-VQ mode: use current action based on action_delta_indices
                    # action_delta_indices = [0] means action[i, 0]
                    delta_idx = self.config.action_delta_indices[0]
                    action_text = self.action_tokenizer(action[i, delta_idx].cpu().numpy())

                action_texts.append(action_text)

            input_ids_list = []
            labels_list = []
            attention_mask_list = []

            for i in range(batch_size):
                # Build prompt with official template: "What action should the robot take to {instruction}?"
                prompt = self.tokenizer.build_prompt(instruction[i], action_texts[i])
                encoded = self.tokenizer.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                )
                input_ids = encoded["input_ids"].squeeze(0)
                seq_len = len(input_ids)

                # Official: labels[: -(num_answer_tokens + 2)] = IGNORE_INDEX
                # num_answer_tokens = len(action_tokens), 2 = eos tokens
                # This masks out the instruction portion, only training on action tokens
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
        read last VALID multimodal position logits -> greedy/sample -> send single token + past_key_values
        back to VLM -> repeat for required token count.
        Returns: [B, 1, A] for non-VQ, [B, 1, A] for VQ (first horizon only)
        Mirrors teach_code/MiniVLA/prismatic/models/vlas/openvla.py::predict_action.
        """
        batch_size = len(instruction)
        if batch_size == 0:
            batch_size = pixel_values["dino"].shape[0]
            instruction = [""] * batch_size

        # Determine number of tokens to generate
        action_dim = self._get_action_dim()

        if self.config.is_vq_mode:
            num_action_tokens = self.config.vqvae_groups
        else:
            # Non-VQ: number of tokens = action dimension
            num_action_tokens = action_dim

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

        # === Build proper attention_mask (not all ones) ===
        # Right padding: 1 for real tokens, 0 for padding
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # === Batch-safe autoregressive generation with KV cache ===
        # First forward pass: image + full prompt -> get past_key_values and last logits
        outputs = self.vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=True,
        )

        # Get the logits from the last VALID multimodal position (not padding)
        # After vision token insertion, the sequence is: [BOS, vision_patches, text_tokens]
        # last_valid_text_index = attention_mask.sum(dim=1) - 1  [B]
        # num_patches from VLM backbone
        # multimodal_last_index = last_valid_text_index + num_patches
        num_patches = self.vlm.num_patches
        last_valid_text_index = attention_mask.sum(dim=1) - 1  # [B]
        multimodal_last_index = last_valid_text_index + num_patches  # [B]

        batch_indices = torch.arange(batch_size, device=input_ids.device)
        logits = outputs.logits[batch_indices, multimodal_last_index]  # [B, vocab_size]
        past_key_values = outputs.past_key_values

        generated_token_ids = []

        for step in range(max_new_tokens):
            if do_sample and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated_token_ids.append(next_token)

            if step < max_new_tokens - 1:
                # Feed single token + past_key_values back to VLM with cached forward
                # Build multimodal attention mask: append 1 for each generated token
                outputs = self.vlm(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                # Extend attention_mask for the new token
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)],
                    dim=1,
                )

        # === Extract action token IDs ===
        action_token_ids = torch.cat(generated_token_ids, dim=1)  # [B, num_tokens]

        # === Decode to actions ===
        action_token_ids_np = action_token_ids.cpu().numpy()

        if self.config.is_vq_mode:
            actions = self.action_tokenizer.decode_token_ids_to_actions(action_token_ids_np)
            actions = torch.from_numpy(actions).float()
            # VQ decode returns [B, T, A] or [T, A]
            if actions.ndim == 1:
                # Single action [A] -> [B, 1, A]
                actions = actions.unsqueeze(0).unsqueeze(0)
            elif actions.ndim == 2:
                # [B, A] or [T, A] -> [B, 1, A] (take first horizon)
                actions = actions.unsqueeze(1)
        else:
            actions = self.action_tokenizer.decode_token_ids_to_actions(action_token_ids_np)
            actions = torch.from_numpy(actions).float()
            # Non-VQ decode returns [B, A] or [A]
            if actions.ndim == 1:
                # Single action [A] -> [B, 1, A]
                actions = actions.unsqueeze(0).unsqueeze(0)
            elif actions.ndim == 2:
                # [B, A] -> [B, 1, A]
                actions = actions.unsqueeze(1)

        # NOTE: No internal denormalization here. LeRobot postprocessor handles action unnormalization.
        # Ensure output is on same device as pixel_values
        actions = actions.to(pixel_values["dino"].device)
        return actions

    def _denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Denormalize actions using official dataset statistics (q01/q99).
        Mirrors teach_code/MiniVLA/prismatic/models/vlas/openvla.py::predict_action.
        actions: [B, T, A] normalized actions in [-1, 1] range
        Returns: [B, T, A] denormalized actions
        """
        if not self.dataset_statistics:
            return actions

        # Find the first dataset statistics entry with action data
        action_stats = None
        for key, stats in self.dataset_statistics.items():
            if "action" in stats and "q01" in stats["action"]:
                action_stats = stats["action"]
                break

        if action_stats is None:
            return actions

        q01 = torch.tensor(action_stats["q01"], dtype=actions.dtype, device=actions.device)
        q99 = torch.tensor(action_stats["q99"], dtype=actions.dtype, device=actions.device)

        # Official denormalization: 0.5 * (normalized + 1) * (q99 - q01) + q01
        # mask indicates which dimensions should be denormalized (default: all True)
        mask = action_stats.get("mask", [True] * len(q01))
        mask = torch.tensor(mask, dtype=torch.bool, device=actions.device)

        # Expand mask for broadcasting: [A] -> [1, 1, A]
        while mask.dim() < actions.dim():
            mask = mask.unsqueeze(0)

        denormalized = torch.where(
            mask,
            0.5 * (actions + 1) * (q99 - q01) + q01,
            actions,
        )
        return denormalized

    def get_optim_params(self) -> dict:
        """
        Returns only trainable parameters (vision, projector, Qwen).
        VQ-VAE is excluded.
        """
        params = []
        
        if not self.config.freeze_vision_backbone:
            for p in self.vlm.vision_backbone.parameters():
                if isinstance(p, torch.Tensor) and p.requires_grad:
                    params.append(p)
                elif not isinstance(p, torch.Tensor):
                    print(f"[WARNING] vision_backbone non-Tensor: {type(p)} = {repr(p)[:100]}")

        for p in self.vlm.projector.parameters():
            if isinstance(p, torch.Tensor) and p.requires_grad:
                params.append(p)
            elif not isinstance(p, torch.Tensor):
                print(f"[WARNING] projector non-Tensor: {type(p)} = {repr(p)[:100]}")

        if self.config.freeze_llm_backbone:
            if self.config.unfreeze_last_llm_layer:
                for p in self.vlm.llm.model.layers[-1].parameters():
                    if isinstance(p, torch.Tensor) and p.requires_grad:
                        params.append(p)
                    elif not isinstance(p, torch.Tensor):
                        print(f"[WARNING] llm last_layer non-Tensor: {type(p)} = {repr(p)[:100]}")
        else:
            for p in self.vlm.llm.parameters():
                if isinstance(p, torch.Tensor) and p.requires_grad:
                    params.append(p)
                elif not isinstance(p, torch.Tensor):
                    print(f"[WARNING] llm non-Tensor: {type(p)} = {repr(p)[:100]}")

        # Final validation
        for i, p in enumerate(params):
            if not isinstance(p, torch.Tensor):
                print(f"[ERROR] Parameter {i} is not a Tensor: type={type(p)}, value={repr(p)[:200]}")
            elif not p.requires_grad:
                print(f"[WARNING] Parameter {i} does not require grad: shape={p.shape}")

        print(f"[DEBUG] Total trainable parameters: {len(params)}")
        print(f"[DEBUG] First param type: {type(params[0]) if params else 'empty'}")
        print(f"[DEBUG] Last param type: {type(params[-1]) if params else 'empty'}")
        
        return {"params": params}


class MiniVLAPolicy(PreTrainedPolicy):
    """
    LeRobot-compatible MiniVLA policy.
    """

    config_class = MiniVLAConfig
    name = "minivla"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        config: MiniVLAConfig | None = None,
        **kwargs,
    ):
        """
        Override PreTrainedPolicy.from_pretrained to handle .pt checkpoints
        instead of safetensors.
        """
        import os
        from pathlib import Path

        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
            )

        # Find .pt checkpoint file in the checkpoints/ subdirectory
        pretrained_path = Path(pretrained_name_or_path)
        checkpoints_dir = pretrained_path / "checkpoints"

        if checkpoints_dir.exists():
            pt_files = list(checkpoints_dir.glob("*.pt"))
            if pt_files:
                # Use the checkpoint with the lowest loss (usually the last one alphabetically)
                pt_files.sort()
                official_checkpoint = str(pt_files[-1])
                print(f"[MiniVLA] Loading official checkpoint: {official_checkpoint}")
                config.official_vla_checkpoint = official_checkpoint
            else:
                print(f"[MiniVLA] No .pt files found in {checkpoints_dir}, using random initialization")
        else:
            print(f"[MiniVLA] No checkpoints directory found at {checkpoints_dir}, using random initialization")

        # Create the policy instance (MiniVLACore.__init__ will load the checkpoint if set)
        instance = cls(config, **kwargs)
        instance.to(config.device)
        instance.eval()
        return instance

    def __init__(self, config: MiniVLAConfig, **kwargs):
        super().__init__(config)
        self.model = MiniVLACore(config)
        self._action_queue: Optional[torch.Tensor] = None

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        pixel_values = self._extract_pixel_values(batch)

        if "action" in batch:
            batch_size = batch["action"].shape[0]
        else:
            batch_size = pixel_values["dino"].shape[0]

        if "task" in batch:
            instruction = batch["task"]
            if isinstance(instruction, torch.Tensor):
                instruction = [""] * batch_size
        else:
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
                instruction = [""] * batch_size
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