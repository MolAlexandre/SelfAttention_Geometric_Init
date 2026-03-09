"""
Trainer for VITForClassification (image classification).
"""

import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm
from src.metrics.symmetry import compute_model_symmetry
from .base_trainer import BaseTrainer


class VITTrainer(BaseTrainer):
    """Train and evaluate a VITForClassification model."""

    def __init__(self, model, train_loader, val_loader, config):
        super().__init__(model, train_loader, val_loader, config)
        self.best_val_acc = 0.0

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = "symmetric" if getattr(config, 'symmetric_init', False) else "standard"
        self.save_dir = f"checkpoints/{timestamp}_{model_type}"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Checkpoints: {self.save_dir}")

        label_smoothing = getattr(config, 'label_smoothing', 0.0)
        self.criterion  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ------------------------------------------------------------------

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            return batch['image'].to(self.config.device), batch['label'].to(self.config.device)
        images, labels = batch
        return images.to(self.config.device), labels.to(self.config.device)

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss, correct, samples, num_batches, accum = 0.0, 0, 0, 0, 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", dynamic_ncols=True)
        for batch in pbar:
            images, labels = self._unpack_batch(batch)

            if self.config.mixed_precision:
                with autocast(device_type="cuda"):
                    logits = self.model(images)
                    loss   = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            (self.scaler.scale(loss / self.config.gradient_accumulation_steps)
             if self.config.mixed_precision
             else (loss / self.config.gradient_accumulation_steps)).backward()

            accum += 1
            if accum % self.config.gradient_accumulation_steps == 0:
                if self.config.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                accum = 0
                self.global_step += 1

            with torch.no_grad():
                correct += (logits.argmax(-1) == labels).sum().item()
                samples += labels.size(0)

            total_loss  += loss.item()
            num_batches += 1
            pbar.set_postfix({
                'avg_loss': f'{total_loss / num_batches:.4f}',
                'acc': f'{100. * correct / samples:.2f}%',
            })

        return total_loss / num_batches, 100. * correct / samples

    def validate(self):
        self.model.eval()
        total_loss, correct, samples, num_batches = 0.0, 0, 0, 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", dynamic_ncols=True)
            for batch in pbar:
                images, labels = self._unpack_batch(batch)
                if self.config.mixed_precision:
                    with autocast(device_type='cuda'):
                        logits = self.model(images)
                        loss   = self.criterion(logits, labels)
                else:
                    logits = self.model(images)
                    loss   = self.criterion(logits, labels)

                correct     += (logits.argmax(-1) == labels).sum().item()
                samples     += labels.size(0)
                total_loss  += loss.item()
                num_batches += 1
                pbar.set_postfix({'val_acc': f'{100. * correct / samples:.2f}%'})

        return total_loss / num_batches, 100. * correct / samples

    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_acc: float, path: str) -> None:
        torch.save(self._base_checkpoint(epoch, {
            'val_acc':      val_acc,
            'best_val_acc': self.best_val_acc,
        }), path)
        print(f"✓ Checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self._load_base_checkpoint(ckpt)
        self.best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f"✓ Reprise epoch {self.start_epoch} | best_acc={self.best_val_acc:.2f}%")

    # ------------------------------------------------------------------

    def train(self, num_epochs: int, num_layers: int) -> float:
        print(f"\nDébut entraînement (epoch {self.start_epoch + 1} → {num_epochs})")
        print("=" * 80)

        csv_fieldnames = (
            ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'symmetry_avg']
            + [f'layer_{i}' for i in range(num_layers)]
        )
        csv_path = os.path.join(self.save_dir, "metrics.csv")

        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'=' * 80}\nEPOCH {epoch + 1}/{num_epochs}\n{'=' * 80}")

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc     = self.validate()
            sym = compute_model_symmetry(self.model)

            print(f"[{epoch + 1}] train={train_loss:.4f}/{train_acc:.2f}% | "
                  f"val={val_loss:.4f}/{val_acc:.2f}% | sym={sym['average']:+.4f}")

            self.metrics_history.append({
                'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc,
                'val_loss': val_loss, 'val_acc': val_acc,
                'symmetry_avg': sym['average'],
                **{k: v for k, v in sym.items() if k != 'average'},
            })

            if (epoch + 1) % 25 == 0:
                self.save_checkpoint(
                    epoch, val_acc,
                    os.path.join(self.save_dir, f"epoch_{epoch + 1}.pt")
                )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(
                    epoch, val_acc, os.path.join(self.save_dir, "best.pt")
                )
                print(f"✓ Nouveau meilleur modèle (val_acc={val_acc:.2f}%)")

            self.save_metrics_csv(csv_path, csv_fieldnames)

        print(f"\n{'=' * 80}\nENTRAÎNEMENT TERMINÉ | best_val_acc={self.best_val_acc:.2f}%")
        return self.best_val_acc
