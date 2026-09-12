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
import logging
from pathlib import Path
from typing import Optional, Unpack

import numpy as np
import torch
import torch.nn as nn

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy

from .configuration_minivla import MiniVLAConfig, MiniVLAT2Config, MiniVLAWristConfig, MiniVLAWristPretrainedConfig, VQ_TOKENIZER_TYPES
from .tokenizer import VLATokenizerWrapper
from .vq_action import VQActionTokenizer, ActionTokenizer
from .vla_backbone import MiniVLAVLBackbone, IGNORE_INDEX

logger = logging.getLogger(__name__)


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
        # Skip in backbone_only mode to avoid importing official 7D action config
        if config.official_init_mode != "backbone_only":
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

        # === Load official checkpoint if specified (full policy loading, old path) ===
        if config.official_vla_checkpoint:
            self._load_official_checkpoint(config.official_vla_checkpoint)

        # === Backbone-only initialization from official pretrained checkpoint ===
        if config.official_init_mode == "backbone_only" and config.official_pretrained_checkpoint:
            self._load_official_backbone_only_checkpoint(config.official_pretrained_checkpoint)

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
                device=self.config.device,
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

    def _read_official_backbone_metadata(self, checkpoint_path: str) -> dict:
        """
        Read backbone architecture metadata from official run directory config.json.
        Only reads vision_backbone_id, llm_backbone_id, arch_specifier, image_size,
        and LLM vocab/extra token configuration.
        
        Does NOT read: vla.action_tokenizer, dataset statistics, action dimension,
        VQ configuration, wrist/base action config, or any embodiment-specific settings.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists() or checkpoint_path.suffix != ".pt":
            return {}

        run_dir = checkpoint_path.parents[1]
        config_json = run_dir / "config.json"
        if not config_json.exists():
            return {}

        with open(config_json, "r") as f:
            full_config = json.load(f)

        model_cfg = full_config.get("model", {})
        metadata = {}

        backbone_keys = [
            "vision_backbone_id",
            "llm_backbone_id",
            "arch_specifier",
            "image_size",
            "image_resize_strategy",
        ]
        for key in backbone_keys:
            if key in model_cfg:
                metadata[key] = model_cfg[key]

        return metadata

    def _load_official_backbone_only_checkpoint(self, checkpoint_path: str):
        """
        Load only vision_backbone, projector, and llm_backbone weights from an
        official MiniVLA checkpoint. Used by minivla_wrist_pretrained to initialize
        from Stanford-ILIAD/minivla-libero90-prismatic pretrained weights.

        This function:
        - Loads projector, llm_backbone, vision_backbone weights (when available)
        - Does NOT load VQ-VAE weights
        - Does NOT load action tokenizer objects
        - Does NOT read or apply official 7D action statistics
        - Does NOT override normalization_mapping
        - Does NOT override action_tokenizer_type
        - Does NOT set _resolved_action_tokenizer_type
        - Does NOT override chunk_size, n_action_steps, image_sequence_len,
          use_wrist_image, primary_image_key, wrist_image_key, action_unnorm_key
        - Ensures official 7D embodiment info does NOT enter MetaWorld 4D pipeline
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Official backbone checkpoint not found: {checkpoint_path}")

        assert (checkpoint_path.suffix == ".pt") and (checkpoint_path.parent.name == "checkpoints"), (
            "Invalid checkpoint path! Expected path like '<run_dir>/checkpoints/<name>.pt'"
        )

        # Read backbone metadata only (no action/VQ config)
        backbone_meta = self._read_official_backbone_metadata(checkpoint_path)
        if backbone_meta:
            logger.info(f"[Backbone-Only Init] Official backbone metadata: {backbone_meta}")

        # Load checkpoint weights
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint.get("model", checkpoint)

        # Track real load_state_dict results for logging
        all_loaded_modules = []
        all_skipped_modules = []
        all_missing_keys = []
        all_unexpected_keys = []

        # Load projector
        if "projector" in model_state:
            missing, unexpected = self.vlm.projector.load_state_dict(model_state["projector"], strict=True)
            if missing:
                raise ValueError(f"[Backbone-Only Init] Missing projector keys: {missing}")
            if unexpected:
                raise ValueError(f"[Backbone-Only Init] Unexpected projector keys: {unexpected}")
            all_loaded_modules.append("projector")
            logger.info("[Backbone-Only Init] Loaded projector weights from official checkpoint.")
        else:
            all_skipped_modules.append("projector (not in checkpoint)")
            logger.warning("[Backbone-Only Init] No projector weights in official checkpoint, using pretrained init.")

        # Load LLM backbone
        if "llm_backbone" in model_state:
            llm_state = model_state["llm_backbone"]
            filtered_llm_state = {}
            for k, v in llm_state.items():
                clean_key = k.replace("llm.", "") if k.startswith("llm.") else k
                filtered_llm_state[clean_key] = v

            official_embed_shape = filtered_llm_state.get("model.embed_tokens.weight", None)
            if official_embed_shape is not None:
                official_vocab_size = official_embed_shape.shape[0]
                current_vocab_size = self.vlm.llm.model.embed_tokens.weight.shape[0]
                if official_vocab_size != current_vocab_size:
                    self.vlm.llm.resize_token_embeddings(official_vocab_size)

            missing, unexpected = self.vlm.llm.load_state_dict(filtered_llm_state, strict=True)
            all_missing_keys.extend(missing)
            all_unexpected_keys.extend(unexpected)
            all_loaded_modules.append("llm_backbone")
            logger.info("[Backbone-Only Init] Loaded LLM backbone weights from official checkpoint.")
        else:
            all_skipped_modules.append("llm_backbone (not in checkpoint)")
            logger.warning("[Backbone-Only Init] No LLM backbone weights in official checkpoint, using pretrained init.")

        # Load vision backbone (optional in official checkpoint)
        if "vision_backbone" in model_state:
            missing, unexpected = self.vlm.vision_backbone.load_state_dict(
                model_state["vision_backbone"], strict=True
            )
            all_missing_keys.extend(missing)
            all_unexpected_keys.extend(unexpected)
            all_loaded_modules.append("vision_backbone")
            logger.info("[Backbone-Only Init] Loaded vision backbone weights from official checkpoint.")
        else:
            all_skipped_modules.append("vision_backbone (not in checkpoint)")
            logger.info("[Backbone-Only Init] No vision backbone weights in official checkpoint, using pretrained init.")

        # Print real load_state_dict results
        print(f"\n{'='*60}")
        print(f"[OFFICIAL CHECKPOINT LOAD RESULTS]")
        print(f"  Loaded modules: {all_loaded_modules}")
        print(f"  Skipped modules: {all_skipped_modules}")
        print(f"  Missing keys: {len(all_missing_keys)}")
        if all_missing_keys:
            print(f"    {all_missing_keys[:10]}{'...' if len(all_missing_keys) > 10 else ''}")
        print(f"  Unexpected keys: {len(all_unexpected_keys)}")
        if all_unexpected_keys:
            print(f"    {all_unexpected_keys[:10]}{'...' if len(all_unexpected_keys) > 10 else ''}")
        print(f"{'='*60}\n")

        # Explicitly do NOT:
        # - Load VQ-VAE weights
        # - Load action tokenizer
        # - Read dataset_statistics.json
        # - Override config._resolved_action_tokenizer_type
        # - Override any MetaWorld action pipeline configuration
        logger.info("[Backbone-Only Init] Backbone initialization complete. MetaWorld 4D action pipeline unchanged.")

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
        action_is_pad: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ):
        """
        Training forward pass.
        pixel_values: {"dino": tensor, "siglip": tensor}
        instruction: list of task strings
        action: [B, chunk_size, action_dim] normalized actions (in [-1, 1] range)
        action_is_pad: [B, chunk_size] boolean mask of padded actions
        state: [B, state_dim] normalized proprioceptive state (in [-1, 1] range)
        Returns: dict with loss and additional metrics
        Mirrors teach_code/MiniVLA/prismatic/vla/datasets/datasets.py (RLDSBatchTransform).
        State handling mirrors official OpenVLA: action_proprio_normalization_type=BOUNDS_Q99.

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

            action_dim = action.shape[-1]

            if self.config.is_vq_mode:
                # Strict VQ dimension validation: no silent zero-padding
                vq_expected_dim = self.action_tokenizer.vq_vae.input_dim_w
                vq_expected_h = self.action_tokenizer.vq_vae.input_dim_h
                required_horizon = self.action_tokenizer.required_future_horizon + 1

                if action_dim != vq_expected_dim:
                    raise ValueError(
                        f"VQ action dimension mismatch: dataset action_dim={action_dim}, "
                        f"VQ input_dim_w={vq_expected_dim}, chunk_size={self.config.chunk_size}, "
                        f"VQ input_dim_h={vq_expected_h}. "
                        f"Cannot silently pad or truncate actions. "
                        f"Use a VQ tokenizer with input_dim_w matching the dataset action_dim, "
                        f"or switch to a non-VQ action tokenizer (e.g., extra_action_tokenizer)."
                    )

            # Official MiniVLA does NOT include state text in prompts.
            # State (proprioception) is processed separately, not as text tokens.
            # Keep state parameter for API compatibility but do not format as text.
            state_texts = [None] * batch_size

            action_texts = []
            valid_action_mask = []

            for i in range(batch_size):
                if self.config.is_vq_mode:
                    # VQ mode: use full action chunk based on required_future_horizon
                    required_horizon = self.action_tokenizer.required_future_horizon + 1
                    action_slice = action[i, :required_horizon].cpu().numpy()

                    # Check if this action chunk has any padding
                    if action_is_pad is not None:
                        chunk_is_pad = action_is_pad[i, :required_horizon].cpu().numpy()
                        is_valid = not chunk_is_pad.any()
                    else:
                        is_valid = True

                    valid_action_mask.append(is_valid)

                    if is_valid:
                        action_text = self.action_tokenizer(action_slice)
                    else:
                        # For padded chunks, generate dummy text that will be fully masked
                        action_text = ""
                else:
                    # Non-VQ mode: use current action based on action_delta_indices
                    delta_idx = self.config.action_delta_indices[0]
                    action_slice = action[i, delta_idx].cpu().numpy()

                    # Check if this action is padded
                    if action_is_pad is not None:
                        is_valid = not action_is_pad[i, delta_idx].item()
                    else:
                        is_valid = True

                    valid_action_mask.append(is_valid)

                    if is_valid:
                        action_text = self.action_tokenizer(action_slice)
                    else:
                        action_text = ""

                action_texts.append(action_text)

            input_ids_list = []
            labels_list = []
            attention_mask_list = []

            for i in range(batch_size):
                if not valid_action_mask[i] or not action_texts[i]:
                    # Skip padded actions: mask entire sequence
                    prompt = self.tokenizer.build_prompt(instruction[i], "", state_texts[i])
                    encoded = self.tokenizer.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=False,
                        truncation=False,
                    )
                    input_ids = encoded["input_ids"].squeeze(0)
                    labels = torch.full_like(input_ids, IGNORE_INDEX)
                else:
                    # Build prompt with official template: "What action should the robot take to {instruction}?"
                    prompt = self.tokenizer.build_prompt(instruction[i], action_texts[i], state_texts[i])
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
                attention_mask_list.append(torch.ones(len(input_ids), dtype=torch.long))

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
            # Inference mode: official MiniVLA does NOT include state text in prompts.
            # Keep state parameter for API compatibility but do not format as text.
            inference_state_texts = [None] * batch_size

            input_ids_list = []
            attention_mask_list = []
            for i in range(batch_size):
                prompt = self.tokenizer.build_inference_prompt(instruction[i], inference_state_texts[i])
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

        result = {"loss": outputs.loss}

        # Compute additional metrics when labels are available
        # NOTE: outputs.logits shape is [B, seq_len_with_vision, vocab_size]
        # The labels were modified by VLM to include IGNORE_INDEX for vision tokens
        # We need to use the logits sequence length to align with the modified labels
        if labels is not None and outputs.logits is not None and outputs.loss is not None:
            # Use the logits sequence to determine valid positions
            # outputs.logits shape: [B, seq_len_with_vision, vocab_size]
            seq_len = outputs.logits.shape[1]
            
            # Reconstruct the full labels with vision token positions set to IGNORE_INDEX
            # This matches what VLM forward does: [labels[:, :1], vision_labels, labels[:, 1:]]
            num_patches = self.vlm.num_patches
            full_labels = torch.full(
                (labels.shape[0], seq_len),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            # Copy original labels: [first_token, ..., remaining]
            # VLM inserts vision patches after first token, so:
            full_labels[:, :1] = labels[:, :1]  # First token
            full_labels[:, 1 + num_patches:] = labels[:, 1:]  # Remaining text tokens
            
            valid_token_count = (full_labels != IGNORE_INDEX).sum().item()
            if valid_token_count > 0:
                # Token cross entropy (already computed as loss)
                result["token_cross_entropy"] = outputs.loss.item()
                result["valid_token_count"] = valid_token_count

                # Top-1 token accuracy
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = full_labels[..., 1:].contiguous()
                valid_mask = shift_labels != IGNORE_INDEX
                if valid_mask.any():
                    top1_preds = shift_logits.argmax(dim=-1)
                    top1_correct = (top1_preds == shift_labels) & valid_mask
                    result["action_token_top1_accuracy"] = top1_correct.sum().item() / valid_mask.sum().item()

                    # Exact action token sequence accuracy (per-sample)
                    batch_size_actual = full_labels.shape[0]
                    exact_correct = 0
                    total_valid_samples = 0
                    for b in range(batch_size_actual):
                        sample_mask = valid_mask[b]
                        if sample_mask.any():
                            total_valid_samples += 1
                            if top1_correct[b][sample_mask].all():
                                exact_correct += 1
                    if total_valid_samples > 0:
                        result["exact_action_token_sequence_accuracy"] = exact_correct / total_valid_samples

        return result

    @torch.no_grad()
    def predict_action_chunk(
        self,
        pixel_values: dict[str, torch.Tensor],
        instruction: list[str],
        state: Optional[torch.Tensor] = None,
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

        # Format state as text for prompt (mirrors official OpenVLA proprio handling)
        state_texts = []
        if state is not None:
            if state.dim() == 3:
                state = state.squeeze(1)
            for i in range(batch_size):
                state_vals = state[i].cpu().numpy()
                state_texts.append(", ".join([f"{v:.3f}" for v in state_vals]))
        else:
            state_texts = [None] * batch_size

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
            prompt = self.tokenizer.build_inference_prompt(instruction[i], state_texts[i])
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

    def get_optim_params(self) -> list[dict]:
        """
        Returns parameter groups following official MiniVLA optimizer configuration.
        Mirrors teach_code/MiniVLA/prismatic/training/strategies/fsdp.py::run_setup:
          - decay group: params with ndim > 1 and not ending in ".bias" (with weight_decay)
          - no_decay group: params with ndim <= 1 or ending in ".bias" (weight_decay=0.0)
        VQ-VAE parameters are excluded (frozen).
        Returns a list of dicts: [{params: [...], weight_decay: ...}, ...]
        """
        base_lr = self.config.optimizer_lr
        weight_decay = self.config.optimizer_weight_decay

        decay = []
        no_decay = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Skip VQ-VAE parameters (frozen)
            if "vq_vae" in name:
                continue

            # Official grouping: bias and low-dim params get no weight decay
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        param_groups = [
            {"params": decay, "weight_decay": weight_decay, "lr": base_lr, "name": "decay"},
            {"params": no_decay, "weight_decay": 0.0, "lr": base_lr, "name": "no_decay"},
        ]

        # Print parameter group details for verification
        print(f"\n{'='*60}")
        print(f"[OPTIM PARAM GROUPS - Official MiniVLA decay/no_decay grouping]")
        total_params = 0
        for i, group in enumerate(param_groups):
            num_params = sum(p.numel() for p in group["params"])
            total_params += num_params
            print(
                f"  Group {i}: {group['name']}, "
                f"lr={group['lr']:.2e}, "
                f"weight_decay={group['weight_decay']}, "
                f"num_tensors={len(group['params'])}, "
                f"total_elements={num_params:,}"
            )
        print(f"  Total trainable parameters: {total_params:,}")
        print(f"  Official LR: {base_lr:.2e}")
        print(f"  Official weight_decay: {weight_decay}")
        print(f"{'='*60}\n")

        return param_groups


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
        Override PreTrainedPolicy.from_pretrained to handle both .pt official checkpoints
        and LeRobot-format model.safetensors checkpoints.
        
        Loading strategy:
        1. If LeRobot-format checkpoint (model.safetensors) exists, use parent class
           PreTrainedPolicy.from_pretrained() which properly loads via _load_as_safetensor.
        2. If .pt official checkpoint exists in checkpoints/, load via MiniVLACore.
        3. If neither exists, raise an error (no silent random initialization).
        """
        import os
        from pathlib import Path
        from safetensors.torch import load_file

        pretrained_path = Path(pretrained_name_or_path)
        checkpoints_dir = pretrained_path / "checkpoints"
        is_lerobot_format = (pretrained_path / "model.safetensors").exists()

        # Priority 1: LeRobot-format checkpoint (model.safetensors + config.json)
        if is_lerobot_format:
            if config is None:
                config = cls.config_class.from_pretrained(
                    pretrained_name_or_path=pretrained_name_or_path,
                )
            
            # When reloading a LeRobot checkpoint, disable official backbone-only init.
            # The LeRobot safetensors already contains the fully trained model weights,
            # so we must NOT re-initialize from the official .pt checkpoint.
            if hasattr(config, "official_init_mode"):
                config.official_init_mode = "none"
            if hasattr(config, "official_pretrained_checkpoint"):
                config.official_pretrained_checkpoint = ""
            if hasattr(config, "official_vla_checkpoint"):
                config.official_vla_checkpoint = ""
            
            # Create the policy instance
            instance = cls(config, **kwargs)
            
            # Load model.safetensors using the parent class mechanism
            model_file = str(pretrained_path / "model.safetensors")
            logger.info(f"[MiniVLA] Loading LeRobot-format checkpoint: {model_file}")
            
            state_dict = load_file(model_file, device="cpu")
            missing_keys, unexpected_keys = instance.load_state_dict(state_dict, strict=False)
            
            # Log loading diagnostics
            total_loaded = len(state_dict)
            logger.info(f"[MiniVLA] Checkpoint absolute path: {os.path.abspath(model_file)}")
            logger.info(f"[MiniVLA] Loaded parameters: {total_loaded}")
            logger.info(f"[MiniVLA] Missing keys: {len(missing_keys)}")
            logger.info(f"[MiniVLA] Unexpected keys: {len(unexpected_keys)}")
            
            if missing_keys:
                # Filter out non-parameter buffers (e.g., running_mean, running_var in BatchNorm)
                param_missing = [k for k in missing_keys if k in dict(instance.named_parameters())]
                buffer_missing = [k for k in missing_keys if k not in dict(instance.named_parameters())]
                
                # Handle tied embedding weights: embed_tokens is often tied to lm_head
                # and may be saved under a different key or omitted from the checkpoint
                known_tied_keys = {
                    "model.vlm.llm.model.embed_tokens.weight",
                    "vlm.llm.model.embed_tokens.weight",
                    "llm.model.embed_tokens.weight",
                }
                actual_param_missing = [
                    k for k in param_missing
                    if k not in known_tied_keys
                ]
                tied_missing = [
                    k for k in param_missing
                    if k in known_tied_keys
                ]
                
                if tied_missing:
                    logger.warning(
                        f"[MiniVLA] {len(tied_missing)} tied embedding key(s) missing from checkpoint "
                        f"(expected if weights are tied to lm_head): {tied_missing}"
                    )
                    # Copy from lm_head if available
                    for tied_key in tied_missing:
                        # Try to find corresponding lm_head key
                        lm_head_key = tied_key.replace("embed_tokens.weight", "lm_head.weight")
                        if lm_head_key in state_dict:
                            logger.info(
                                f"[MiniVLA] Copying {lm_head_key} -> {tied_key} (tied weights)"
                            )
                            # Find the actual parameter in the model
                            for name, param in instance.named_parameters():
                                if name == tied_key:
                                    param.data.copy_(state_dict[lm_head_key])
                                    break
                
                if actual_param_missing:
                    logger.error(
                        f"[MiniVLA] CRITICAL: {len(actual_param_missing)} missing parameter keys: "
                        f"{actual_param_missing[:20]}{'...' if len(actual_param_missing) > 20 else ''}"
                    )
                    raise RuntimeError(
                        f"MiniVLA checkpoint loading failed: {len(actual_param_missing)} parameter keys "
                        f"are missing from the loaded state dict. The checkpoint may be incomplete."
                    )
                if buffer_missing:
                    logger.warning(
                        f"[MiniVLA] {len(buffer_missing)} missing buffer keys (non-trainable, may be OK): "
                        f"{buffer_missing[:10]}"
                    )
            
            if unexpected_keys:
                logger.warning(
                    f"[MiniVLA] {len(unexpected_keys)} unexpected keys in checkpoint: "
                    f"{unexpected_keys[:10]}{'...' if len(unexpected_keys) > 20 else ''}"
                )
            
            # Check for shape mismatches
            shape_mismatches = []
            for key, param in state_dict.items():
                if key in dict(instance.named_parameters()):
                    model_param = dict(instance.named_parameters())[key]
                    if param.shape != model_param.shape:
                        shape_mismatches.append((key, tuple(param.shape), tuple(model_param.shape)))
            
            if shape_mismatches:
                logger.error(
                    f"[MiniVLA] CRITICAL: {len(shape_mismatches)} shape mismatches detected: "
                    f"{shape_mismatches[:10]}"
                )
                raise RuntimeError(
                    f"MiniVLA checkpoint loading failed: {len(shape_mismatches)} shape mismatches. "
                    f"First few: {shape_mismatches[:5]}"
                )
            
            # Log key weight summaries for verification
            key_layers = [
                "vlm.vision_backbone.dino_model.blocks.0.norm1.weight",
                "vlm.llm.model.layers.0.self_attn.q_proj.weight",
                "vlm.projector.linear_1.weight",
            ]
            for layer_key in key_layers:
                if layer_key in state_dict:
                    w = state_dict[layer_key]
                    logger.info(
                        f"[MiniVLA] Weight summary: {layer_key} -> "
                        f"shape={tuple(w.shape)}, mean={w.float().mean().item():.6f}, "
                        f"std={w.float().std().item():.6f}"
                    )
            
            if not missing_keys and not unexpected_keys and not shape_mismatches:
                logger.info("[MiniVLA] All keys matched successfully")
            
            # Convert model to float32 for maximum compatibility during inference
            instance = instance.float()
            instance.to(config.device)
            instance.eval()
            return instance

        # Priority 2: Official .pt checkpoint in checkpoints/ directory
        if checkpoints_dir.exists():
            pt_files = list(checkpoints_dir.glob("*.pt"))
            if pt_files:
                pt_files.sort()
                official_checkpoint = str(pt_files[-1])
                logger.info(f"[MiniVLA] Loading official .pt checkpoint: {official_checkpoint}")
                config.official_vla_checkpoint = official_checkpoint
                
                # Create the policy instance (MiniVLACore.__init__ will load the .pt checkpoint)
                instance = cls(config, **kwargs)
                instance.to(config.device)
                instance.eval()
                return instance
        
        # No valid checkpoint found - raise error instead of silent random initialization
        raise FileNotFoundError(
            f"No valid MiniVLA checkpoint found at {pretrained_name_or_path}. "
            f"Expected either: "
            f"1) LeRobot-format: model.safetensors + config.json in {pretrained_path}, or "
            f"2) Official .pt checkpoint in {checkpoints_dir}"
        )

    def __init__(self, config: MiniVLAConfig, **kwargs):
        super().__init__(config)
        self.model = MiniVLACore(config)
        self._action_queue: Optional[torch.Tensor] = None

        # === Startup validation: action dimension and tokenizer ===
        self._validate_action_config()

    def _validate_action_config(self):
        """
        Validate action configuration before training starts.
        Checks:
        - Action dimension matches expected (not 7D LIBERO VQ)
        - Action tokenizer is not a VQ tokenizer when using non-VQ mode
        - Policy output shape is correct
        """
        action_dim = self.config.action_feature.shape[0] if self.config.action_feature else None
        tokenizer_type = self.config._resolved_action_tokenizer_type or self.config.action_tokenizer_type

        print(f"\n{'='*60}")
        print(f"[ACTION CONFIG VALIDATION]")
        print(f"  action dimension: {action_dim}")
        print(f"  action tokenizer: {tokenizer_type}")
        print(f"  is_vq_mode: {self.config.is_vq_mode}")

        # Reject 7D VQ tokenizer for MetaWorld 4D actions
        if action_dim == 7 and self.config.is_vq_mode:
            raise ValueError(
                f"CRITICAL: Detected 7D VQ action tokenizer (LIBERO) with 7D actions! "
                f"MetaWorld requires 4D actions with extra_action_tokenizer. "
                f"action_tokenizer_type={tokenizer_type}, action_dim={action_dim}. "
                f"Please set action_tokenizer_type='extra_action_tokenizer' and ensure action_dim=4."
            )

        # Reject any VQ tokenizer when action_dim != VQ input_dim_w
        if self.config.is_vq_mode and hasattr(self.model, "action_tokenizer") and self.model.action_tokenizer is not None:
            if hasattr(self.model.action_tokenizer, "vq_vae") and self.model.action_tokenizer.vq_vae is not None:
                vq_input_dim_w = self.model.action_tokenizer.vq_vae.input_dim_w
                if action_dim != vq_input_dim_w:
                    raise ValueError(
                        f"CRITICAL: VQ action tokenizer input_dim_w ({vq_input_dim_w}) "
                        f"does not match action dimension ({action_dim}). "
                        f"Use a VQ tokenizer compatible with {action_dim}D actions, "
                        f"or switch to extra_action_tokenizer for non-VQ mode."
                    )

        print(f"  policy output shape: [B, chunk_size={self.config.chunk_size}, action_dim={action_dim}]")
        print(f"  Validation PASSED")
        print(f"{'='*60}\n")

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
        action_is_pad = batch.get("action_is_pad", None)
        state = batch.get("observation.state", None)
        outputs = self.model(
            pixel_values=pixel_values,
            instruction=instruction,
            action=action,
            action_is_pad=action_is_pad,
            state=state,
        )

        # MiniVLACore.forward now returns a dict with loss and metrics
        if isinstance(outputs, dict):
            loss = outputs["loss"]
            metrics = {k: v for k, v in outputs.items() if k != "loss"}
            return loss, metrics if metrics else None

        return outputs.loss, None

    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> torch.Tensor:
        pixel_values = self._extract_pixel_values(batch)

        if "task" in batch:
            instruction = batch["task"]
            if isinstance(instruction, torch.Tensor):
                batch_size = pixel_values["dino"].shape[0]
                instruction = [""] * batch_size
        else:
            batch_size = pixel_values["dino"].shape[0]
            instruction = [""] * batch_size

        state = batch.get("observation.state", None)
        return self.model.predict_action_chunk(
            pixel_values=pixel_values,
            instruction=instruction,
            state=state,
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


class MiniVLAWristPretrainedPolicy(MiniVLAPolicy):
    """
    Alias for minivla_wrist_pretrained variant.
    
    Uses official MiniVLA pretrained vision/projector/Qwen weights for initialization,
    then fine-tunes with MetaWorld 4D action pipeline using extra_action_tokenizer.
    
    All forward(), predict_action_chunk(), select_action(), processor usage,
    LeRobot checkpoint save/restore, and from_pretrained() logic are inherited
    from MiniVLAPolicy without modification.
    """
    config_class = MiniVLAWristPretrainedConfig
    name = "minivla_wrist_pretrained"