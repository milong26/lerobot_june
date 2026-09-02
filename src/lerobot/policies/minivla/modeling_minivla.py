import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.utils.constants import ACTION, OBS_STATE

from ..pretrained import PreTrainedPolicy
from .configuration_minivla import MiniVLAConfig
from .diffusion_head import DiffusionConfig, DiffusionPolicyHead
from .encoders import ImageEncoderTinyCNN, StateEncoderMLP, TextEncoderTinyGRU
from .fusion import FusionMLP
from .tokenizer import SimpleTokenizer


class MiniVLACore(nn.Module):
    def __init__(
        self,
        vocab_size,
        state_dim,
        action_dim,
        d_model=128,
        d_word=64,
        diffusion_T=16,
        beta_start=1e-4,
        beta_end=1e-2,
        time_emb_dim=32,
        diffusion_hidden_dim=128,
    ):
        super().__init__()
        self.image_encoder = ImageEncoderTinyCNN(d_model=d_model)
        self.text_encoder = TextEncoderTinyGRU(vocab_size, d_word=d_word, d_model=d_model)
        self.state_encoder = StateEncoderMLP(state_dim, d_model=d_model)
        self.fusion = FusionMLP(d_model=d_model)

        diffusion_cfg = DiffusionConfig(
            T=diffusion_T,
            beta_start=beta_start,
            beta_end=beta_end,
            action_dim=action_dim,
            cond_dim=d_model,
        )
        self.diffusion_head = DiffusionPolicyHead(
            diffusion_cfg,
            time_emb_dim=time_emb_dim,
            diffusion_hidden_dim=diffusion_hidden_dim,
        )

    def encode_obs(self, image, text_tokens, state):
        img_token = self.image_encoder(image)
        txt_token = self.text_encoder(text_tokens)
        state_token = self.state_encoder(state)
        fused_context = self.fusion(img_token, txt_token, state_token)
        return fused_context

    def loss(self, image, text_tokens, state, actions):
        cond = self.encode_obs(image, text_tokens, state)
        return self.diffusion_head.loss(actions, cond)

    def act(self, image, text_tokens, state):
        cond = self.encode_obs(image, text_tokens, state)
        return self.diffusion_head.sample(cond)


class MiniVLAPolicy(PreTrainedPolicy):
    config_class = MiniVLAConfig
    name = "minivla"

    def __init__(self, config, dataset_stats=None, dataset_meta=None, **kwargs):
        super().__init__(config)
        self.config = config
        config.validate_features()

        self.tokenizer = SimpleTokenizer(vocab=config.vocab if config.vocab else None)

        if dataset_meta is not None and hasattr(dataset_meta, "tasks"):
            task_list = [str(t) for t in dataset_meta.tasks.index]
            if not config.task_texts:
                config.task_texts = task_list
            if len(config.vocab) <= 2:
                self.tokenizer.build_from_texts(config.task_texts)
                config.vocab = dict(self.tokenizer.vocab)

        if not config.default_instruction and config.task_texts:
            config.default_instruction = config.task_texts[0]

        self.model = MiniVLACore(
            vocab_size=len(config.vocab),
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            d_model=config.d_model,
            d_word=config.d_word,
            diffusion_T=config.diffusion_T,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            time_emb_dim=config.time_emb_dim,
            diffusion_hidden_dim=config.diffusion_hidden_dim,
        )

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

    def _prepare_image(self, batch):
        image_key = self.config.image_key
        if image_key not in batch:
            available = [k for k in batch if k.startswith("observation.images")]
            if available:
                image_key = sorted(available)[0]
            else:
                raise ValueError(f"No image key found in batch. Expected '{self.config.image_key}'.")

        image = batch[image_key]
        if image.dim() == 5:
            if image.shape[1] != 1:
                raise ValueError(
                    f"Expected time dimension of 1, got {image.shape[1]}. "
                    "MiniVLA only supports single-step observation."
                )
            image = image.squeeze(1)

        if image.dim() != 4:
            raise ValueError(
                f"Expected image with shape (B, C, H, W), got {tuple(image.shape)}."
            )

        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        if image.shape[-2:] != (self.config.image_size, self.config.image_size):
            image = F.interpolate(
                image,
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            )

        if image.shape[1] != 3:
            raise ValueError(f"Expected 3-channel image, got {image.shape[1]} channels.")

        return image.to(self.config.device)

    def _prepare_state(self, batch):
        state = batch[OBS_STATE]
        if state.dim() == 3:
            if state.shape[1] != 1:
                raise ValueError(
                    f"MiniVLA only supports single-step observation, but received time length {state.shape[1]}. "
                    f"Expected shape (B, 1, state_dim), got {tuple(state.shape)}."
                )
            state = state.squeeze(1)
        if state.dim() != 2:
            raise ValueError(
                f"Expected state with shape (B, state_dim), got {tuple(state.shape)}."
            )
        state = state.float().to(self.config.device)
        if state.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"Expected state dim {self.config.state_dim}, got {state.shape[-1]}"
            )
        return state

    def _prepare_action(self, batch):
        action = batch[ACTION]
        if action.dim() == 3:
            if action.shape[1] != 1:
                raise ValueError(
                    f"MiniVLA only supports single-step action, but received time length {action.shape[1]}. "
                    f"Expected shape (B, 1, action_dim), got {tuple(action.shape)}."
                )
            action = action.squeeze(1)
        if action.dim() != 2:
            raise ValueError(
                f"Expected action with shape (B, action_dim), got {tuple(action.shape)}."
            )
        action = action.float().to(self.config.device)
        if action.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"Expected action dim {self.config.action_dim}, got {action.shape[-1]}"
            )
        return action

    def _extract_texts(self, batch, batch_size):
        text_key = self.config.text_key
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

        return [self.config.default_instruction] * batch_size

    def _prepare_text_tokens(self, batch, batch_size):
        texts = self._extract_texts(batch, batch_size)
        return self.tokenizer.batch_encode(
            texts, max_length=self.config.max_text_length, device=self.config.device
        )

    def get_optim_params(self):
        return self.model.parameters()

    def reset(self):
        pass

    def forward(self, batch):
        batch_size = self._get_batch_size(batch)
        image = self._prepare_image(batch)
        state = self._prepare_state(batch)
        action = self._prepare_action(batch)
        tokens = self._prepare_text_tokens(batch, batch_size)

        loss = self.model.loss(image, tokens, state, action)
        return loss, None

    def predict_action_chunk(self, batch, **kwargs):
        image = self._prepare_image(batch)
        state = self._prepare_state(batch)
        batch_size = self._get_batch_size(batch)
        tokens = self._prepare_text_tokens(batch, batch_size)

        actions = self.model.act(image, tokens, state)
        return actions.unsqueeze(1)

    def select_action(self, batch, **kwargs):
        actions = self.predict_action_chunk(batch, **kwargs)
        return actions[:, 0]