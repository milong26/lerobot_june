import torch
import torch.nn as nn


class VisionProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(vision_dim, llm_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class StateProjector(nn.Module):
    def __init__(self, state_dim: int, llm_hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(state_dim, llm_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class VLATokenFusion(nn.Module):
    def __init__(self, vision_dim: int, state_dim: int, llm_hidden_dim: int):
        super().__init__()
        self.vision_projector = VisionProjector(vision_dim, llm_hidden_dim)
        self.state_projector = StateProjector(state_dim, llm_hidden_dim)

    def forward(self, vision_tokens: torch.Tensor,
                state_embedding: torch.Tensor = None) -> tuple:
        projected_vision = self.vision_projector(vision_tokens)
        if state_embedding is not None:
            state_token = self.state_projector(state_embedding).unsqueeze(1)
            return projected_vision, state_token
        return projected_vision, None


class FusionMLP(nn.Module):
    """Legacy fusion module kept for backward compatibility. Not used by MiniVLAPolicy."""

    def __init__(self, d_model=128):
        super().__init__()
        import warnings
        warnings.warn(
            "FusionMLP is legacy and not used by MiniVLAPolicy.",
            DeprecationWarning,
        )
        self.fc1 = nn.Linear(3 * d_model, d_model)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, img_token, txt_token, state_token):
        x = torch.cat([img_token, txt_token, state_token], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.ln(x)
        return x