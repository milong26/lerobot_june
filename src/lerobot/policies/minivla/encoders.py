import torch.nn as nn


class ImageEncoderTinyCNN(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.proj = nn.Linear(128, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.mean(dim=[2, 3])
        x = self.proj(x)
        x = self.ln(x)
        return x


class TextEncoderTinyGRU(nn.Module):
    def __init__(self, vocab_size, d_word=64, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_word)
        self.gru = nn.GRU(d_word, d_model, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        _, h_last = self.gru(x)
        x = h_last[0]
        x = self.ln(x)
        return x


class StateEncoderMLP(nn.Module):
    def __init__(self, state_dim, d_model=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, s):
        x = self.fc1(s)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.ln(x)
        return x