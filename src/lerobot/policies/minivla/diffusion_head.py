from dataclasses import dataclass

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    T: int = 16
    beta_start: float = 1e-4
    beta_end: float = 1e-2
    action_dim: int = 4
    cond_dim: int = 128


def make_beta_schedule(cfg):
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half_dim, device=t.device)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=emb.device)], dim=-1)
        return emb


class ActionDenoiseModel(nn.Module):
    def __init__(self, cfg, time_emb_dim=32, hidden_dim=128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        in_dim = cfg.action_dim + time_emb_dim + cfg.cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cfg.action_dim),
        )

    def forward(self, x_t, t, cond):
        t_emb = self.time_embed(t)
        x = torch.cat([x_t, t_emb, cond], dim=-1)
        eps_pred = self.mlp(x)
        return eps_pred


class DiffusionPolicyHead(nn.Module):
    def __init__(self, cfg, time_emb_dim=32, diffusion_hidden_dim=128):
        super().__init__()
        self.cfg = cfg
        self.denoise_model = ActionDenoiseModel(cfg, time_emb_dim=time_emb_dim, hidden_dim=diffusion_hidden_dim)

        betas, alphas, alpha_bar = make_beta_schedule(cfg)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0, t, noise):
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

    def loss(self, actions, cond):
        actions = actions.view(actions.shape[0], -1)
        B = actions.shape[0]
        t = torch.randint(0, self.cfg.T, (B,), device=actions.device)
        noise = torch.randn_like(actions)
        x_t = self.q_sample(actions, t, noise)
        eps_pred = self.denoise_model(x_t, t, cond)
        return F.mse_loss(eps_pred, noise)

    def sample(self, cond, n_samples=None):
        if n_samples is None:
            B = cond.size(0)
        else:
            B = n_samples
            cond = cond.expand(B, -1)
        action_dim = self.cfg.action_dim
        x_t = torch.randn(B, action_dim, device=cond.device)

        for t_step in range(self.cfg.T - 1, -1, -1):
            t = torch.full((B,), t_step, device=cond.device, dtype=torch.long)
            eps_pred = self.denoise_model(x_t, t, cond)

            beta_t = self.betas[t_step]
            alpha_t = self.alphas[t_step]
            alpha_bar_t = self.alpha_bar[t_step]

            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

            if t_step > 0:
                noise = torch.randn_like(x_t)
                x_t = torch.sqrt(alpha_t) * x0_pred + torch.sqrt(beta_t) * noise
            else:
                x_t = x0_pred

        return x_t