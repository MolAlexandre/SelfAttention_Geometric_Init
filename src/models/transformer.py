"""
Core Transformer building blocks shared by BERT and ViT.
"""

import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with optional symmetric W_QK initialization.

    When ``symmetric_init=True``, W_Q and W_K are initialised identically so
    that W_QK = W_Q @ W_K^T is symmetric at t=0 (as in Section 3 of the paper).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 symmetric_init: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = math.sqrt(self.d_head)
        self.symmetric_init = symmetric_init

        self.query = nn.Linear(d_model, d_model, bias=True)
        self.key = nn.Linear(d_model, d_model, bias=True)
        self.value = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.zeros_(self.query.bias)

        if self.symmetric_init:
            # W_K = W_Q  →  W_QK = W_Q @ W_Q^T  (symmetric)
            self.key.weight.copy_(self.query.weight)
            self.key.bias.copy_(self.query.bias)
        else:
            nn.init.xavier_uniform_(self.key.weight)
            nn.init.zeros_(self.key.bias)

        nn.init.xavier_uniform_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.zeros_(self.W_o.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:    [B, T, D]
            mask: [B, 1, 1, T]  (0 = masked position)
        Returns:
            [B, T, D]
        """
        B, T, _ = x.shape

        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        Q, K, V = reshape(self.query(x)), reshape(self.key(x)), reshape(self.value(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    """Position-wise two-layer FFN with GELU activation."""

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    """Pre-LN Transformer encoder block (attention + FFN, both with residual)."""

    def __init__(self, d_model: int, num_heads: int, d_hidden: int,
                 dropout: float = 0.1, symmetric_init: bool = False):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout, symmetric_init)
        self.feed_forward = FeedForward(d_model, d_hidden, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm1(x + self.drop1(self.attention(x, mask)))
        x = self.norm2(x + self.drop2(self.feed_forward(x)))
        return x
