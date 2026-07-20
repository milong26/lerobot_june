import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, GPTNeoXModel, GPTNeoXPreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

from lerobot.policies.tinyvla.llava_pythia.model.language_model.pythia.configuration_llava_pythia import LlavaPythiaConfig
from lerobot.policies.tinyvla.llava_pythia.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

logger = logging.get_logger(__name__)


class LLavaPythiaModel(LlavaMetaModel, GPTNeoXModel):
    config_class = LlavaPythiaConfig

    def __init__(self, config):
        super(LLavaPythiaModel, self).__init__(config)


class LlavaPythiaForCausalLM(GPTNeoXPreTrainedModel, LlavaMetaForCausalLM):
    config_class = LlavaPythiaConfig

    def __init__(self, config):
        super(GPTNeoXPreTrainedModel, self).__init__(config)
        self.gpt_neox = LLavaPythiaModel(config)

        self.head_type = config.action_head_type
        self.visual_concat = config.concat
        self.action_dim = config.action_dim

        if config.action_head_type == 'act':
            from lerobot.policies.tinyvla.policy_heads.models import build_ACT_head
            self.embed_out = build_ACT_head(config.act['act'])
            middle_dim = int(max(config.hidden_size, config.act['act']['hidden_dim']) / 2)
            self.proj_to_action = nn.Sequential(
                nn.Linear(config.hidden_size, middle_dim),
                nn.LayerNorm(middle_dim),
                nn.ReLU(),
                nn.Linear(middle_dim, config.act['act']['hidden_dim']),
                nn.LayerNorm(config.act['act']['hidden_dim']),
            )

        elif config.action_head_type == 'droid_diffusion':
            self.proj_to_action = nn.Identity()
            self.embed_out = None  # Will be initialized lazily
            self.num_queries = config.chunk_size
            self.noise_samples = 1
            self.num_inference_timesteps = 10
            self._diffusion_initialized = False

        self.post_init()

    def get_channel_proj(self, x):
        return self.channel_proj(x)

    def encode_images(self, images, proj=True):
        image_features = self.get_model().get_vision_tower()(images)
        if proj:
            image_features = self.get_model().mm_projector(image_features)
        return image_features

    def get_mm_projector(self, image_features):
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def get_image_fusion_embedding(self, visual_concat=None, images=None, images_r=None, images_top=None, states=None):
        if "channel_cat" not in visual_concat:
            image_features = self.encode_images(images)
        if images_top is not None:
            image_features_top = self.encode_images(images_top)
        if images_r is not None:
            if visual_concat == 'token_cat':
                image_features_r = self.encode_images(images_r)
                image_features = torch.cat([image_features, image_features_r], dim=1)
                if images_top is not None:
                    image_features = torch.cat([image_features, image_features_top], dim=1)
            else:
                raise ValueError(f"Unimplemented concat style:{visual_concat}")
        return image_features

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def get_model(self):
        return self.gpt_neox

    def _init_diffusion_head(self):
        """Lazily initialize diffusion head components to avoid requiring diffusers at import time."""
        if self._diffusion_initialized:
            return
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from lerobot.policies.tinyvla.policy_heads.models import ConditionalUnet1D
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )
        self.embed_out = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.config.hidden_size,
            state_dim=self.config.state_dim
        )
        self._diffusion_initialized = True

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            actions=None,
            states=None,
            images_r=None,
            images_top=None,
            is_pad=None,
            eval=False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images, images_r=images_r, images_top=images_top, visual_concat=self.visual_concat, states=states)

        outputs = self.get_model()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]

        if self.head_type == 'fc':
            loss, logits = self.forward_fc_head(labels, actions, hidden_states, states)

        elif self.head_type == 'act':
            if not eval:
                loss = self.forward_act_head(actions, hidden_states, states, is_pad)
                logits = None
            else:
                action = self.forward_act_head(actions, hidden_states, states, is_pad)
                return action
        elif self.head_type == 'droid_diffusion':
            if not eval:
                loss = self.forward_diffusion_head(actions, hidden_states, states, is_pad)
                logits = None
            else:
                action = self.forward_diffusion_head(actions, hidden_states, states, is_pad)
                return action

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward_fc_head(self, labels, actions, hidden_states, states):
        logits = self.embed_out(input_feature=hidden_states, state_tensor=states)

        loss = None
        if labels is not None and actions is None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if actions is not None:
            loss = torch.nn.functional.huber_loss(logits, actions)
        return loss, logits

    def kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld

    def forward_act_head(self, actions, hidden_states, states, is_pad=None, vq_sample=None):
        env_state = None

        hidden_states = self.proj_to_action(hidden_states)
        if actions is not None:
            actions = actions[:, :self.embed_out.num_queries]
            is_pad = is_pad[:, :self.embed_out.num_queries]

            loss_dict = dict()
            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.embed_out(
                qpos=states, hidden_states=hidden_states, env_state=env_state, actions=actions, is_pad=is_pad,
                vq_sample=vq_sample
            )

            total_kld, dim_wise_kld, mean_kld = self.kl_divergence(mu, logvar)

            all_l1 = torch.nn.functional.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.config.act['act']['kl_weight']
            return loss_dict
        else:
            a_hat, _, (_, _), _, _ = self.embed_out(
                qpos=states, hidden_states=hidden_states, env_state=env_state, vq_sample=vq_sample
            )
            return a_hat

    def forward_diffusion_head(self, actions, hidden_states, states, is_pad):
        self._init_diffusion_head()  # Ensure diffusion head is initialized
        if actions is not None:
            B = actions.size(0)
            actions = actions[:, :self.num_queries]
            is_pad = is_pad[:, :self.num_queries]
            num_noise_samples = self.noise_samples

            noise = torch.randn([num_noise_samples] + list(actions.shape), device=actions.device, dtype=actions.dtype)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=actions.device
            ).long()

            timesteps, noise = timesteps.to(actions.device), noise.to(actions.device)

            noisy_actions = torch.cat([self.noise_scheduler.add_noise(
                actions, noise[i], timesteps)
                for i in range(len(noise))], dim=0)

            noisy_actions = noisy_actions.to(dtype=actions.dtype)
            assert hidden_states.ndim == 3

            hidden_states = hidden_states.repeat(num_noise_samples, 1, 1)
            timesteps = timesteps.repeat(num_noise_samples)
            is_pad = is_pad.repeat(num_noise_samples, 1)
            states = states.repeat(num_noise_samples, 1)

            noise_pred = self.embed_out(noisy_actions, timesteps, global_cond=hidden_states, states=states)
            noise = noise.view(noise.size(0) * noise.size(1), *noise.size()[2:])
            loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='none')
            loss = (loss * ~is_pad.unsqueeze(-1)).mean()
            return {'loss': loss}
        else:
            B = 1
            Tp = self.num_queries
            action_dim = self.action_dim

            noisy_action = torch.randn((B, Tp, action_dim), device=hidden_states.device)
            naction = noisy_action.to(dtype=hidden_states.dtype)
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                noise_pred = self.embed_out(naction, k, global_cond=hidden_states, states=states)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction


AutoConfig.register("llava_pythia", LlavaPythiaConfig)
AutoModelForCausalLM.register(LlavaPythiaConfig, LlavaPythiaForCausalLM)