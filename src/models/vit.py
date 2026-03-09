"""
Vision Transformer (ViT) for image classification.
Reuses the shared EncoderBlock from transformer.py.
"""

import torch
import torch.nn as nn

from .embeddings import VITEmbeddings
from .transformer import EncoderBlock


class VITHead(nn.Module):
    """MLP classification head applied to the [CLS] token."""

    def __init__(self, num_classes: int, d_model: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.layer_norm(x[:, 0]))  # CLS token


class VITForClassification(nn.Module):
    """Vision Transformer for image classification."""

    def __init__(self, num_classes: int = 10, d_model: int = 256, num_heads: int = 4,
                 num_layers: int = 6, d_hidden: int = 1024, img_size: int = 32,
                 patch_size: int = 4, in_channels: int = 3, dropout: float = 0.1,
                 symmetric_init: bool = False):
        super().__init__()
        assert img_size % patch_size == 0
        assert d_model % num_heads == 0

        self.embeddings = VITEmbeddings(d_model, patch_size, img_size, in_channels)
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_hidden, dropout, symmetric_init)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.head = VITHead(num_classes, d_model, d_hidden, dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(images)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.head(self.norm(x))

    def compute_loss(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.CrossEntropyLoss()(self.forward(images), labels)
