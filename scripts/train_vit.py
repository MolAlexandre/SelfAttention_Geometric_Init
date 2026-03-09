#!/usr/bin/env python
"""
Entry point: train a ViT for image classification on CIFAR-10.

Usage:
    python scripts/train_vit.py --model standard
    python scripts/train_vit.py --model symmetric
    python scripts/train_vit.py --model symmetric --resume checkpoints/.../best.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import ViT6LayerCIFAR10, VitTrainingConfig
from src.data.cifar import CifarDatasetManager
from src.models.vit import VITForClassification
from src.training.vit_trainer import VITTrainer


def train(model_name: str, resume_checkpoint: str | None = None) -> float:
    print("=" * 80)
    print(f"ViT CIFAR-10 — {model_name.upper()}")
    print("=" * 80)

    symmetric = model_name == "symmetric"
    model_cfg = ViT6LayerCIFAR10(symmetric_init=symmetric)
    train_cfg = VitTrainingConfig()
    train_cfg.symmetric_init = symmetric

    # Data
    dm = CifarDatasetManager()
    train_loader, val_loader = dm.create_dataloaders(
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
    )

    # Model
    model = VITForClassification(
        num_classes=model_cfg.num_classes, d_model=model_cfg.d_model,
        num_heads=model_cfg.num_heads, num_layers=model_cfg.num_layers,
        d_hidden=model_cfg.d_hidden, img_size=model_cfg.img_size,
        patch_size=model_cfg.patch_size, in_channels=model_cfg.in_channels,
        dropout=model_cfg.dropout, symmetric_init=model_cfg.symmetric_init,
    ).to(train_cfg.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Paramètres: {n_params:,}")

    trainer = VITTrainer(model, train_loader, val_loader, train_cfg)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        trainer.load_checkpoint(resume_checkpoint)

    return trainer.train(
        num_epochs=train_cfg.num_epochs,
        num_layers=model_cfg.num_layers,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['standard', 'symmetric'], default='standard')
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    train(args.model, args.resume)


if __name__ == "__main__":
    main()
