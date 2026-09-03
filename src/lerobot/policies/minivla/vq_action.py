import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionTokenizer:
    def __init__(self, action_dim: int, chunk_size: int = 8,
                 vocab_size: int = 256, action_min: float = -1.0,
                 action_max: float = 1.0):
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.vocab_size = vocab_size
        self.action_min = action_min
        self.action_max = action_max

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        actions = torch.clamp(actions, self.action_min, self.action_max)
        normalized = (actions - self.action_min) / (self.action_max - self.action_min)
        codes = (normalized * (self.vocab_size - 1)).round().long()
        codes = torch.clamp(codes, 0, self.vocab_size - 1)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.float()
        normalized = codes / (self.vocab_size - 1)
        actions = normalized * (self.action_max - self.action_min) + self.action_min
        return actions


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int = 256, num_stages: int = 1):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_stages = num_stages

        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, dim))
            for _ in range(num_stages)
        ])

    def forward(self, x: torch.Tensor):
        residual = x.clone()
        all_codes = []
        all_quantized = torch.zeros_like(x)

        for stage in range(self.num_stages):
            codebook = self.codebooks[stage]
            distances = torch.cdist(residual, codebook)
            codes = torch.argmin(distances, dim=-1)
            quantized = F.embedding(codes, codebook)

            all_codes.append(codes)
            all_quantized = all_quantized + quantized
            residual = residual - quantized

        all_codes = torch.stack(all_codes, dim=-1)
        return all_quantized, all_codes


class ResidualVQActionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        self.chunk_size = config.action_chunk_size
        self.vocab_size = config.action_vocab_size
        self.codebook_size = config.vq_codebook_size
        self.hidden_dim = config.d_model

        self.action_tokenizer = ActionTokenizer(
            action_dim=self.action_dim,
            chunk_size=self.chunk_size,
            vocab_size=self.vocab_size,
        )

        self.total_action_tokens = self.chunk_size * self.action_dim

        self.vq = ResidualVectorQuantizer(
            dim=self.hidden_dim,
            codebook_size=self.codebook_size,
            num_stages=1,
        )

        self.action_logits_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.vocab_size),
        )

    def forward(self, hidden_states: torch.Tensor, action_labels: torch.Tensor = None):
        b = hidden_states.shape[0]

        pooled = hidden_states.mean(dim=1)

        action_logits = self.action_logits_head(pooled)

        if action_labels is not None:
            action_codes = self.action_tokenizer.encode(action_labels)
            action_codes = action_codes.view(b, -1)

            code_embeddings = F.embedding(action_codes, self.vq.codebooks[0])
            code_embeddings = code_embeddings.view(b, self.total_action_tokens, -1)
            code_embeddings = code_embeddings.mean(dim=1)

            quantized, codes = self.vq(pooled)

            recon_loss = F.mse_loss(quantized, code_embeddings.detach())
            commit_loss = F.mse_loss(pooled.detach(), quantized)

            pred_loss = F.cross_entropy(
                action_logits.view(-1, self.vocab_size),
                action_codes.view(-1),
                ignore_index=-100,
            )

            total_loss = pred_loss + recon_loss + commit_loss * 0.25
            return total_loss, action_logits
        else:
            return None, action_logits

    def decode_action(self, logits: torch.Tensor) -> torch.Tensor:
        b = logits.shape[0]
        probs = F.softmax(logits, dim=-1)
        codes = torch.argmax(probs, dim=-1)

        codes = codes.view(b, self.chunk_size, self.action_dim)
        actions = self.action_tokenizer.decode(codes)
        return actions