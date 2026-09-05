"""
fusion.py

Official FusedMLPProjector for MiniVLA.
Mirrors teach_code/MiniVLA/prismatic/util/nn_utils.py FusedMLPProjector.

Architecture:
  Linear(fused_vision_dim, 4*fused_vision_dim, bias=True)
  -> GELU
  -> Linear(4*fused_vision_dim, llm_dim, bias=True)
  -> GELU
  -> Linear(llm_dim, llm_dim, bias=True)
"""

import torch
import torch.nn as nn


class FusedMLPProjector(nn.Module):
    """
    Official FusedMLPProjector from MiniVLA.
    Maps concatenated DINO+SigLIP patch features to LLM embedding dimension.
    """

    def __init__(self, fused_vision_dim: int, llm_dim: int, mlp_type: str = "fused-gelu-mlp"):
        super().__init__()
        self.initial_projection_dim = fused_vision_dim * 4
        if mlp_type == "fused-gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"FusedMLPProjector with mlp_type='{mlp_type}' is not supported!")

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(fused_img_patches)