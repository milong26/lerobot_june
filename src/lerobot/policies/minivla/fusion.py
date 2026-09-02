import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
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