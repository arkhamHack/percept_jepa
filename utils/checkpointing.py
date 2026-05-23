"""
Checkpoint Management
======================
Save and load model/optimizer/scheduler state dicts.

Supports:
  - Saving with metadata (epoch, metrics, config)
  - Resuming training from any checkpoint
  - Best model tracking
  - Automatic cleanup of old checkpoints
"""

import os
import json
import glob
import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages saving and loading of training checkpoints.

    Directory layout:
        checkpoint_dir/
            epoch_010.pt
            epoch_020.pt
            best.pt          ← copy of best validation checkpoint
            latest.pt        ← symlink to most recent

    Args:
        checkpoint_dir: path to checkpoint directory
        keep_last:      number of recent checkpoints to keep
    """

    def __init__(self, checkpoint_dir: str, keep_last: int = 3):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.ckpt_dir = checkpoint_dir
        self.keep_last = keep_last
        self._best_metric = float('inf')

    def save(
        self,
        epoch:      int,
        model:      torch.nn.Module,
        optimizer:  torch.optim.Optimizer,
        scheduler:  Any,
        metrics:    Dict[str, float],
        cfg_dict:   Optional[Dict] = None,
    ) -> str:
        """
        Save a checkpoint.

        Returns:
            path to saved file
        """
        ckpt = {
            'epoch':          epoch,
            'model':          model.state_dict(),
            'optimizer':      optimizer.state_dict(),
            'scheduler':      scheduler.state_dict() if scheduler else None,
            'metrics':        metrics,
            'config':         cfg_dict or {},
        }
        filename = os.path.join(self.ckpt_dir, f'epoch_{epoch:04d}.pt')
        torch.save(ckpt, filename)
        logger.info(f"Checkpoint saved: {filename}")

        # Update 'latest' symlink
        latest = os.path.join(self.ckpt_dir, 'latest.pt')
        if os.path.islink(latest):
            os.remove(latest)
        os.symlink(os.path.abspath(filename), latest)

        # Save best checkpoint
        val_loss = metrics.get('val_loss', float('inf'))
        if val_loss < self._best_metric:
            self._best_metric = val_loss
            best_path = os.path.join(self.ckpt_dir, 'best.pt')
            torch.save(ckpt, best_path)
            logger.info(f"  → New best checkpoint (val_loss={val_loss:.4f})")

        # Clean up old checkpoints
        self._cleanup()

        return filename

    def load(
        self,
        path:       str,
        model:      torch.nn.Module,
        optimizer:  Optional[torch.optim.Optimizer] = None,
        scheduler:  Any = None,
        device:     str = 'cpu',
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            path:       path to .pt file (or 'best' / 'latest' shortcuts)
            model:      model to load weights into
            optimizer:  optional optimizer to restore
            scheduler:  optional scheduler to restore

        Returns:
            dict with 'epoch' and 'metrics'
        """
        if path in ('best', 'latest'):
            path = os.path.join(self.ckpt_dir, f'{path}.pt')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model'])

        if optimizer and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])

        if scheduler and ckpt.get('scheduler'):
            scheduler.load_state_dict(ckpt['scheduler'])

        logger.info(f"Loaded checkpoint: {path}  (epoch={ckpt['epoch']})")
        return {'epoch': ckpt['epoch'], 'metrics': ckpt.get('metrics', {})}

    def _cleanup(self):
        """Remove old epoch checkpoints beyond keep_last."""
        pattern = os.path.join(self.ckpt_dir, 'epoch_*.pt')
        ckpts = sorted(glob.glob(pattern))
        to_remove = ckpts[:max(0, len(ckpts) - self.keep_last)]
        for f in to_remove:
            os.remove(f)
            logger.debug(f"Removed old checkpoint: {f}")

    @property
    def latest_path(self) -> Optional[str]:
        p = os.path.join(self.ckpt_dir, 'latest.pt')
        return p if os.path.exists(p) else None

    @property
    def best_path(self) -> Optional[str]:
        p = os.path.join(self.ckpt_dir, 'best.pt')
        return p if os.path.exists(p) else None
