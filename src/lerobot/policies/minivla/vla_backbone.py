import logging
import warnings

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MiniVLABackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_hidden_dim = config.vision_hidden_dim
        self.language_hidden_dim = config.language_hidden_dim
        self.language_model_path = config.language_model_path
        self.freeze_language_model = config.freeze_language_model

        self.language_model = None
        if self.language_model_path:
            try:
                from transformers import AutoModelForCausalLM
                self.language_model = AutoModelForCausalLM.from_pretrained(
                    self.language_model_path, local_files_only=True
                )
                if hasattr(self.language_model, "model"):
                    self.language_model = self.language_model.model
                if hasattr(self.language_model, "embed_tokens"):
                    self.language_hidden_dim = self.language_model.embed_tokens.embedding_dim
                logger.info(f"Loaded language model from {self.language_model_path}")
            except Exception as e:
                warnings.warn(
                    f"Failed to load language model from {self.language_model_path}: {e}. "
                    "Falling back to dummy backbone."
                )
                self.language_model = None

        if self.language_model is None:
            self._build_dummy_backbone()

        if self.freeze_language_model and self.language_model is not None:
            for param in self.language_model.parameters():
                param.requires_grad = False

        self.vision_projector = nn.Linear(self.vision_hidden_dim, self.language_hidden_dim)
        self.state_projector = nn.Linear(config.state_dim, self.language_hidden_dim)

    def _build_dummy_backbone(self):
        self.dummy_layers = nn.Sequential(
            nn.Linear(self.language_hidden_dim, self.language_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.language_hidden_dim),
            nn.Linear(self.language_hidden_dim, self.language_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.language_hidden_dim),
        )
        self._is_dummy = True

    def prepare_inputs_embeds(self, image_tokens: torch.Tensor,
                              input_ids: torch.Tensor,
                              state_embedding: torch.Tensor = None):
        b = image_tokens.shape[0]

        image_embeds = self.vision_projector(image_tokens)

        text_embeds = self.language_model.embed_tokens(input_ids) if not getattr(self, "_is_dummy", False) \
            else torch.zeros(b, input_ids.shape[1], self.language_hidden_dim, device=image_tokens.device)

        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        if state_embedding is not None:
            state_token = self.state_projector(state_embedding).unsqueeze(1)
            inputs_embeds = torch.cat([inputs_embeds, state_token], dim=1)

        return inputs_embeds

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                image_tokens: torch.Tensor = None,
                state_embedding: torch.Tensor = None,
                action_labels: torch.Tensor = None):

        inputs_embeds = self.prepare_inputs_embeds(image_tokens, input_ids, state_embedding)

        b, seq_len, d = inputs_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones(b, seq_len, device=inputs_embeds.device)

        if getattr(self, "_is_dummy", False):
            hidden_states = self.dummy_layers(inputs_embeds)
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                hidden_states = outputs.hidden_states[-1]
            elif hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs[0]

        return hidden_states