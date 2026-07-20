import math
import torch
from torch import nn


def position_encoding_1d(tensor, temperature=10000):
    """
    Generates 1D positional encoding for transformer input.

    Args:
        tensor: Input tensor of shape (B, seq_len, dim).
        temperature: Temperature parameter for scaling.

    Returns:
        Positional encoding tensor of the same shape as input.
    """
    dim_t = torch.arange(tensor.shape[-1], dtype=torch.float32, device=tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / tensor.shape[-1])

    pos = torch.arange(tensor.shape[1], dtype=torch.float32, device=tensor.device)
    pos = pos.unsqueeze(1) / dim_t.unsqueeze(0)

    pos_x = pos[..., 0::2].sin()
    pos_y = pos[..., 1::2].cos()
    pos_enc = torch.stack((pos_x, pos_y), dim=-1).flatten(-2)

    return pos_enc.unsqueeze(0).repeat(tensor.shape[0], 1, 1)