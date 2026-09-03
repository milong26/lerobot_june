import logging
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.utils.constants import ACTION, OBS_STATE

from ..pretrained import PreTrainedPolicy
from .configuration_minivla import MiniVLAConfig
from .encoders import MultiImageVisionEncoder
from .fusion import StateProjector
from .tokenizer import VLATokenizerWrapper
from .vla_backbone import MiniVLABackbone
from .vq_action import ResidualVQActionHead

logger = logging.getLogger(__name__)


class MiniVLACore(nn.Module):
    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.config = config

        self.vision_encoder = MultiImageVisionEncoder(config)

        self.vla_backbone = MiniVLABackbone(config)

        self.action_decoder = ResidualVQActionHead(config)

        if config.use_state_projection:
            self.state_projector = StateProjector(
                state_dim=config.state_dim,
                llm_hidden_dim=config.d_model,
            )
        else:
            self.state_projector = None

    def encode_observation(self, images: list[torch.Tensor],
                           text_dict: dict,
                           state: torch.Tensor) -> torch.Tensor:
        image_tokens = self.vision_encoder(images)

        state_embedding = None
        if state is not None and self.state_projector is not None:
            state_embedding = self.state_projector(state)

        input_ids = text_dict["input_ids"]
        attention_mask = text_dict.get("attention_mask", None)

        hidden_states = self.vla_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_tokens=image_tokens,
            state_embedding=state_embedding,
        )
        return hidden_states

    def forward(self, images: list[torch.Tensor],
                text_dict: dict,
                state: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.encode_observation(images, text_dict, state)

        b = hidden_states.shape[0]
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)

        if actions.shape[1] < self.config.action_chunk_size:
            pad_size = self.config.action_chunk_size - actions.shape[1]
            pad = actions[:, -1:].repeat(1, pad_size, 1)
            actions = torch.cat([actions, pad], dim=1)

        actions = actions[:, :self.config.action_chunk_size]

        loss, logits = self.action_decoder(hidden_states, actions)
        return loss

    def predict_action_chunk(self, images: list[torch.Tensor],
                             text_dict: dict,
                             state: torch.Tensor) -> torch.Tensor:
        hidden_states = self.encode_observation(images, text_dict, state)

        _, logits = self.action_decoder(hidden_states)

        actions = self.action_decoder.decode_action(logits)

        b = actions.shape[0]
        actions = actions.view(b, self.config.action_chunk_size, self.config.action_dim)
        return actions


class MiniVLAPolicy(PreTrainedPolicy):
    config_class = MiniVLAConfig
    name = "minivla"

    def __init__(self, config, dataset_stats=None, dataset_meta=None, **kwargs):
        super().__init__(config)
        self.config = config
        config.validate_features()
        config.validate_vla_config()

        self.tokenizer = VLATokenizerWrapper(tokenizer_path=config.tokenizer_path)

        self.model = MiniVLACore(config)

        self._action_queue: dict[int, torch.Tensor] = {}

    def _get_batch_size(self, batch):
        for key in [OBS_STATE, ACTION]:
            if key in batch:
                v = batch[key]
                if isinstance(v, torch.Tensor):
                    return v.shape[0]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                return v.shape[0]
        return 1

    def _prepare_images(self, batch) -> list[torch.Tensor]:
        image_keys = self.config.get_image_keys()
        images = []
        for key in image_keys:
            if key in batch:
                img = batch[key]
                if img.dim() == 5:
                    b, t, c, h, w = img.shape
                    img = img.view(b, t, c, h, w)
                elif img.dim() == 4:
                    img = img.unsqueeze(1)
                else:
                    raise ValueError(
                        f"Expected image with 4 or 5 dimensions, got {img.dim()}."
                    )

                if img.dtype == torch.uint8:
                    img = img.float() / 255.0

                if img.shape[-2:] != (self.config.image_size, self.config.image_size):
                    img = F.interpolate(
                        img,
                        size=(self.config.image_size, self.config.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                images.append(img.to(self.config.device))

        if not images:
            raise ValueError("No image data found in batch.")

        return images

    def _prepare_text(self, batch) -> dict:
        batch_size = self._get_batch_size(batch)
        texts = self._extract_texts(batch, batch_size)
        return self.tokenizer.batch_encode(
            texts, max_length=64, device=self.config.device
        )

    def _prepare_state(self, batch) -> torch.Tensor:
        state = batch[OBS_STATE]
        if state.dim() == 3:
            if state.shape[1] != 1:
                state = state[:, -1:]
            state = state.squeeze(1)
        if state.dim() != 2:
            raise ValueError(
                f"Expected state with shape (B, state_dim), got {tuple(state.shape)}."
            )
        return state.float().to(self.config.device)

    def _extract_texts(self, batch, batch_size):
        text_key = "task"
        if text_key in batch:
            val = batch[text_key]
            if isinstance(val, str):
                return [val] * batch_size
            if isinstance(val, (list, tuple)) and len(val) == batch_size:
                return [str(v) for v in val]

        task_index_key = "task_index"
        if task_index_key in batch:
            idx = batch[task_index_key]
            if isinstance(idx, torch.Tensor):
                if idx.dim() == 2 and idx.shape[1] == 1:
                    idx = idx.squeeze(1)
                idx_list = idx.tolist()
            elif isinstance(idx, (list, tuple)):
                idx_list = [int(i) for i in idx]
            else:
                idx_list = [int(idx)] * batch_size

            if self.config.task_texts:
                texts = []
                for i in idx_list:
                    if 0 <= i < len(self.config.task_texts):
                        texts.append(self.config.task_texts[i])
                    else:
                        texts.append(self.config.default_instruction)
                return texts

        if self.config.default_instruction:
            return [self.config.default_instruction] * batch_size

        return [""] * batch_size

    def get_optim_params(self):
        return self.model.parameters()

    def reset(self, batch_idx: int = 0):
        if batch_idx in self._action_queue:
            del self._action_queue[batch_idx]

    def forward(self, batch):
        images = self._prepare_images(batch)
        state = self._prepare_state(batch)
        text_dict = self._prepare_text(batch)

        action = batch[ACTION]
        if action.dim() == 3:
            if action.shape[1] != 1:
                action = action[:, :self.config.action_chunk_size]
            else:
                pad_size = self.config.action_chunk_size - 1
                pad = action.repeat(1, pad_size, 1)
                action = torch.cat([action, pad], dim=1)
        elif action.dim() == 2:
            action = action.unsqueeze(1)
            pad_size = self.config.action_chunk_size - 1
            pad = action.repeat(1, pad_size, 1)
            action = torch.cat([action, pad], dim=1)

        action = action.float().to(self.config.device)

        loss = self.model(images, text_dict, state, action)
        return loss, None

    def predict_action_chunk(self, batch, **kwargs):
        images = self._prepare_images(batch)
        state = self._prepare_state(batch)
        text_dict = self._prepare_text(batch)

        actions = self.model.predict_action_chunk(images, text_dict, state)
        return actions

    def select_action(self, batch, **kwargs):
        batch_size = self._get_batch_size(batch)

        actions_list = []
        for i in range(batch_size):
            if i in self._action_queue and len(self._action_queue[i]) > 0:
                action = self._action_queue[i].popleft()
                actions_list.append(action)
            else:
                images = self._prepare_images(batch)
                state = self._prepare_state(batch)
                text_dict = self._prepare_text(batch)

                chunk = self.model.predict_action_chunk(images, text_dict, state)
                chunk_i = chunk[i]

                queue = deque()
                for step in range(self.config.action_chunk_size):
                    queue.append(chunk_i[step])
                self._action_queue[i] = queue

                action = queue.popleft()
                actions_list.append(action)

        actions = torch.stack(actions_list, dim=0)
        return actions