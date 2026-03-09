"""
BERT-style encoder-only Transformer for Masked Language Modeling.
"""

import torch
import torch.nn as nn

from .embeddings import BERTEmbeddings
from .transformer import EncoderBlock


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int,
                 d_hidden: int, max_len: int = 512, dropout: float = 0.1,
                 symmetric_init: bool = False):
        super().__init__()
        self.embeddings = BERTEmbeddings(vocab_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_hidden, dropout, symmetric_init)
            for _ in range(num_layers)
        ])

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embeddings(input_ids)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class MLMHead(nn.Module):
    """Projects encoder output to vocabulary logits."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.decoder = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.layer_norm(self.activation(self.dense(x))))


class BERTForMLM(nn.Module):
    """Full BERT model for Masked Language Modeling."""

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int,
                 d_hidden: int, max_len: int = 512, dropout: float = 0.1,
                 symmetric_init: bool = False):
        super().__init__()
        self.encoder  = TransformerEncoder(
            vocab_size, d_model, num_heads, num_layers,
            d_hidden, max_len, dropout, symmetric_init,
        )
        self.mlm_head = MLMHead(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.mlm_head(self.encoder(input_ids, attention_mask))

    def compute_loss(self, input_ids: torch.Tensor, labels: torch.Tensor,
                     attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss(ignore_index=-100)(
            logits.view(-1, self.vocab_size), labels.view(-1)
        )
        return loss
