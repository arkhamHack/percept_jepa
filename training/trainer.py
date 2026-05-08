"""Training loop for multimodal perception and legacy JEPA models."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .losses import combined_loss, multimodal_combined_loss


class Trainer:
    """Trains multimodal perception or legacy JEPA model.

    Supports staged training: stage 1 freezes V-JEPA, stage 2 unfreezes
    last N blocks with a lower learning rate.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        config: Any,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.is_multimodal = hasattr(model, "set_training_stage")

        if torch.cuda.device_count() > 1:
            print(f"[Trainer] Using DataParallel on {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.epochs: int = _cfg(config, "epochs", 50)
        self.grad_accum_steps: int = _cfg(config, "grad_accum_steps", 1)
        self.amp_enabled: bool = _cfg(config, "amp_enabled", True)
        self.ema_momentum: float = _cfg(config, "ema_momentum", 0.996)
        self.checkpoint_dir: str = _cfg(config, "checkpoint_dir", "checkpoints")
        self.log_interval: int = _cfg(config, "log_interval", 50)
        self.loss_weights: dict[str, float] | None = _cfg(config, "loss_weights", None)

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train(self, num_epochs: int | None = None) -> None:
        num_epochs = num_epochs or self.epochs
        best_val_loss = float("inf")

        print(f"[Trainer] Starting training for {num_epochs} epochs "
              f"(device={self.device}, multimodal={self.is_multimodal})")

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            train_losses = self.train_one_epoch(epoch)
            val_losses = self.validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"[Epoch {epoch}/{num_epochs}] "
                f"train_loss={train_losses['total']:.4f}  "
                f"val_loss={val_losses['total']:.4f}  "
                f"lr={lr:.2e}  time={elapsed:.1f}s"
            )

            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                path = os.path.join(self.checkpoint_dir, "best.pt")
                self.save_checkpoint(path, epoch, best_val_loss)
                print(f"  -> saved best checkpoint (val_loss={best_val_loss:.4f})")

        print(f"[Trainer] Training complete. Best val_loss={best_val_loss:.4f}")

    def train_one_epoch(self, epoch: int = 0) -> dict[str, float]:
        self.model.train()
        running: dict[str, float] = {}
        n_batches = len(self.train_loader)

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader, 1):
            batch = _to_device(batch, self.device)

            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                outputs = self._forward(batch)
                losses = self._compute_loss(outputs, batch)
                loss = losses["total"] / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if step % self.grad_accum_steps == 0 or step == n_batches:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if not self.is_multimodal:
                    self._update_target_encoder()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()

            if step % self.log_interval == 0:
                avg = {k: v / step for k, v in running.items()}
                print(
                    f"  [epoch {epoch} step {step}/{n_batches}] "
                    + "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
                )

        return {k: v / n_batches for k, v in running.items()}

    @torch.no_grad()
    def validate(self, epoch: int = 0) -> dict[str, float]:
        self.model.eval()
        running: dict[str, float] = {}
        n_batches = len(self.val_loader)

        for batch in self.val_loader:
            batch = _to_device(batch, self.device)
            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                outputs = self._forward(batch)
                losses = self._compute_loss(outputs, batch)
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        print(f"  [val epoch {epoch}] " + "  ".join(f"{k}={v:.4f}" for k, v in avg.items()))
        return avg

    def save_checkpoint(self, path: str, epoch: int, best_val_loss: float) -> None:
        model = self.model
        if isinstance(model, nn.DataParallel):
            model = model.module
        state = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model = self.model
        if isinstance(model, nn.DataParallel):
            model = model.module
        model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"[Trainer] Resumed from checkpoint (epoch {ckpt['epoch']})")
        return ckpt["epoch"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.is_multimodal:
            return self.model(
                image=batch["image"],
                radar_points=batch["radar_points"],
                radar_mask=batch["radar_mask"],
            )
        else:
            return self.model(
                image=batch["image"],
                radar_points=batch["radar_points"],
                radar_mask=batch["radar_mask"],
                future_image=batch.get("future_image"),
                agent_states=batch.get("agent_states"),
            )

    def _compute_loss(self, outputs, batch):
        if self.is_multimodal:
            return multimodal_combined_loss(outputs, batch, self.loss_weights)
        else:
            return combined_loss(outputs, batch, self.loss_weights)

    def _update_target_encoder(self) -> None:
        model = self.model
        if isinstance(model, nn.DataParallel):
            model = model.module
        if hasattr(model, "update_target_encoder"):
            model.update_target_encoder(self.ema_momentum)


# -----------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------

def _cfg(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out
