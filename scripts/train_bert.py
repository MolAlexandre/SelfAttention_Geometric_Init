#!/usr/bin/env python
"""
Entry point: train a BERT-style model for Masked Language Modeling.

Usage:
    python scripts/train_bert.py --model standard
    python scripts/train_bert.py --model symmetric
    python scripts/train_bert.py --model symmetric --resume checkpoints/symmetric_best.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import BERTMiniConfig, TrainingConfig
from src.data.wikipedia import WikipediaDatasetManager
from src.models.bert import BERTForMLM
from src.training.bert_trainer import BERTTrainer


def train(model_name: str, resume_checkpoint: str | None = None) -> float:
    print("=" * 80)
    print(f"BERT MLM — {model_name.upper()}")
    print("=" * 80)

    symmetric   = model_name == "symmetric"
    model_cfg   = BERTMiniConfig(symmetric_init=symmetric)
    train_cfg   = TrainingConfig()

    # Data
    dm = WikipediaDatasetManager()
    ds = dm.load_and_split()
    train_loader, val_loader = dm.create_dataloaders(
        ds, batch_size=train_cfg.batch_size,
        max_length=model_cfg.max_len,
        mlm_probability=train_cfg.mlm_probability,
        num_workers=train_cfg.num_workers,
    )

    # Model
    model = BERTForMLM(
        vocab_size=model_cfg.vocab_size, d_model=model_cfg.d_model,
        num_heads=model_cfg.num_heads, num_layers=model_cfg.num_layers,
        d_hidden=model_cfg.d_hidden, max_len=model_cfg.max_len,
        dropout=model_cfg.dropout, symmetric_init=model_cfg.symmetric_init,
    ).to(train_cfg.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Paramètres: {n_params:,}")

    trainer = BERTTrainer(model, train_loader, val_loader, train_cfg)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        trainer.load_checkpoint(resume_checkpoint)

    return trainer.train(
        num_epochs=train_cfg.num_epochs,
        checkpoint_prefix=model_name,
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
