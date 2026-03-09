"""
Trainer for BERTForMLM (Masked Language Modeling).
"""

import os
import torch
from torch.amp import autocast
from tqdm import tqdm
from src.metrics.symmetry import compute_model_symmetry, log_symmetry_scores
from .base_trainer import BaseTrainer


class BERTTrainer(BaseTrainer):
    """Train and evaluate a BERTForMLM model."""

    def __init__(self, model, train_loader, val_loader, config):
        super().__init__(model, train_loader, val_loader, config)
        self.best_val_loss = float('inf')
        os.makedirs("checkpoints", exist_ok=True)

    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, num_batches, accum = 0.0, 0, 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", dynamic_ncols=True)
        for batch in pbar:
            ids   = batch['input_ids'].to(self.config.device)
            lbls  = batch['labels'].to(self.config.device)
            mask  = batch['attention_mask'].to(self.config.device)

            if self.config.mixed_precision:
                with autocast(device_type="cuda"):
                    loss = self.model.compute_loss(ids, lbls, mask)
            else:
                loss = self.model.compute_loss(ids, lbls, mask)

            loss = loss / self.config.gradient_accumulation_steps
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

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

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            self.global_step += 1
            pbar.set_postfix({
                'avg_loss': f'{total_loss / num_batches:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
            })

        return total_loss / num_batches

    def validate(self):
        self.model.eval()
        total_loss, num_batches = 0.0, 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", dynamic_ncols=True)
            for batch in pbar:
                ids  = batch['input_ids'].to(self.config.device)
                lbls = batch['labels'].to(self.config.device)
                mask = batch['attention_mask'].to(self.config.device)

                if self.config.mixed_precision:
                    with autocast(device_type='cuda'):
                        loss = self.model.compute_loss(ids, lbls, mask)
                else:
                    loss = self.model.compute_loss(ids, lbls, mask)

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'val_loss': f'{total_loss / num_batches:.4f}'})

        avg_loss   = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return avg_loss, perplexity

    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self._base_checkpoint(epoch, {
            'val_loss':      val_loss,
            'best_val_loss': self.best_val_loss,
        }), path)
        print(f"✓ Checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self._load_base_checkpoint(ckpt)
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"✓ Reprise epoch {self.start_epoch} | best_loss={self.best_val_loss:.4f}")

    # ------------------------------------------------------------------

    def train(self, num_epochs: int, checkpoint_prefix: str, num_layers: int) -> float:
        print(f"\nDébut entraînement (epoch {self.start_epoch + 1} → {num_epochs})")
        print("=" * 80)

        csv_fieldnames = (
            ['epoch', 'train_loss', 'val_loss', 'perplexity', 'symmetry_avg']
            + [f'layer_{i}' for i in range(num_layers)]
        )

        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n{'=' * 80}\nEPOCH {epoch + 1}/{num_epochs}\n{'=' * 80}")

            train_loss = self.train_epoch(epoch)
            val_loss, perplexity = self.validate()
            sym = compute_model_symmetry(self.model)
            log_symmetry_scores(sym, epoch)

            print(f"[{epoch + 1}] train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | ppl={perplexity:.2f} | "
                  f"sym={sym['average']:+.4f}")

            self.metrics_history.append({
                'epoch': epoch + 1, 'train_loss': train_loss,
                'val_loss': val_loss, 'perplexity': perplexity,
                'symmetry_avg': sym['average'],
                **{k: v for k, v in sym.items() if k != 'average'},
            })

            # Periodic checkpoint
            self.save_checkpoint(
                epoch, val_loss,
                f"checkpoints/{checkpoint_prefix}_epoch_{epoch + 1}.pt"
            )

            # Best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    epoch, val_loss, f"checkpoints/{checkpoint_prefix}_best.pt"
                )
                print(f"Nouveau meilleur modèle (val_loss={val_loss:.4f})")

            self.save_metrics_csv(f"metrics_{checkpoint_prefix}.csv", csv_fieldnames)

        print(f"\n{'=' * 80}\nENTRAÎNEMENT TERMINÉ | best_val_loss={self.best_val_loss:.4f}")
        return self.best_val_loss
