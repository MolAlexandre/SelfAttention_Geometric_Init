import math

import torch
import torch.nn as nn


class BERTEmbeddings(nn.Module):
    """Token + sinusoidal positional embeddings for BERT-style encoder."""

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        pe = self._create_positional_encoding(max_len, d_model)
        self.register_buffer('positional_encoding', pe)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        embeddings = self.token_embedding(input_ids) + self.positional_encoding[:, :seq_len, :]
        return self.dropout(self.layer_norm(embeddings))


class VITEmbeddings(nn.Module):
    """Patch + learnable positional embeddings for ViT."""

    def __init__(self, d_embedding: int, patch_size: int, img_size: int = 32, in_channels: int = 3):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(
            in_channels, d_embedding, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_embedding))
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_embedding) * 0.02
        )
        self.layer_norm = nn.LayerNorm(d_embedding, eps=1e-6)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B = x.shape[0]
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)          # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)                           # [B, 1, D]
        x = torch.cat([cls, x], dim=1) + self.position_embedding        # [B, N+1, D]
        return self.dropout(self.layer_norm(x))
