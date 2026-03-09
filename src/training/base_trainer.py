"""
Abstract base trainer with shared checkpoint / scheduler / scaler logic.
"""

import csv
import os
from abc import ABC, abstractmethod

import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


class BaseTrainer(ABC):
    def __init__(self, model, train_loader, val_loader, config):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config

        steps_per_epoch   = len(train_loader) // config.gradient_accumulation_steps
        self.total_steps  = steps_per_epoch * config.num_epochs
        self.warmup_steps = int(self.total_steps * config.warmup_ratio)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.scheduler = self._make_scheduler()
        self.scaler    = GradScaler(enabled=config.mixed_precision)

        self.start_epoch     = 0
        self.global_step     = 0
        self.metrics_history = []

    def _make_scheduler(self) -> LambdaLR:
        ws, ts = self.warmup_steps, self.total_steps

        def lr_lambda(step):
            if step < ws:
                return step / max(1, ws)
            return max(0.0, (ts - step) / max(1, ts - ws))

        return LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _base_checkpoint(self, epoch: int, extra: dict) -> dict:
        return {
            'epoch':                epoch,
            'global_step':          self.global_step,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict':    self.scaler.state_dict(),
            'metrics_history':      self.metrics_history,
            **extra,
        }

    def _load_base_checkpoint(self, checkpoint: dict) -> None:
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch     = checkpoint['epoch'] + 1
        self.global_step     = checkpoint['global_step']
        self.metrics_history = checkpoint.get('metrics_history', [])

    def save_metrics_csv(self, csv_path: str, fieldnames: list) -> None:
        if not self.metrics_history:
            return
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics_history)
        print(f"✓ Métriques sauvegardées: {csv_path}")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def train_epoch(self, epoch: int): ...

    @abstractmethod
    def validate(self): ...

    @abstractmethod
    def train(self, **kwargs): ...
