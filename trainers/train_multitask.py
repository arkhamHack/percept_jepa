"""
Multitask Training Script
==========================
Main training loop for RearFusionModel with joint detection + tracking + prediction.

Features:
  - Mixed-precision training (torch.cuda.amp)
  - Gradient clipping
  - Cosine LR scheduling with warmup
  - TensorBoard logging
  - Checkpoint saving / resuming
  - Backbone unfreeze after warmup

Usage:
    python trainers/train_multitask.py --config configs/default.yaml

    # Resume from checkpoint:
    python trainers/train_multitask.py --config configs/default.yaml \
        --resume checkpoints/latest.pt
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.nuscenes_dataset import build_dataset
from data.collate import collate_fn
from models.full_model import RearFusionModel
from losses.detection_loss import DetectionLoss
from losses.tracking_loss import TrackingLoss
from losses.prediction_loss import PredictionLoss
from utils.checkpointing import CheckpointManager
from utils.metrics import MeanAveragePrecision, ade_fde

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str):
    """Load YAML config using OmegaConf (preferred) or simple dict fallback."""
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(path)
    except ImportError:
        import yaml
        with open(path) as f:
            cfg_dict = yaml.safe_load(f)
        # Wrap in a simple namespace-like object
        class AttrDict(dict):
            def __getattr__(self, k):
                v = self[k]
                return AttrDict(v) if isinstance(v, dict) else v
        def _convert(d):
            return AttrDict({k: _convert(v) if isinstance(v, dict) else v for k, v in d.items()})
        cfg = _convert(cfg_dict)
    return cfg


# ---------------------------------------------------------------------------
# LR Scheduler
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    """Cosine annealing with linear warmup."""
    total_steps = cfg.training.num_epochs * steps_per_epoch
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        import math
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    batch,
    model,
    det_loss_fn,
    track_loss_fn,
    pred_loss_fn,
    optimizer,
    scaler,
    device,
    cfg,
):
    """Single training iteration. Returns loss dict."""
    # Move to device
    images     = batch['images'].to(device, non_blocking=True)
    radar_pts  = batch['radar_points'].to(device, non_blocking=True)
    radar_mask = batch['radar_mask'].to(device, non_blocking=True)
    ego_motion = batch['ego_motion'].to(device, non_blocking=True)
    gt_boxes   = batch['boxes'].to(device, non_blocking=True)
    gt_labels  = batch['labels'].to(device, non_blocking=True)
    gt_traj    = batch['future_trajectories'].to(device, non_blocking=True)
    fut_mask   = batch['future_mask'].to(device, non_blocking=True)
    ann_mask   = batch['ann_mask'].to(device, non_blocking=True)
    track_ids  = batch['track_ids'].to(device, non_blocking=True)

    optimizer.zero_grad()

    with torch.cuda.amp.autocast(enabled=cfg.hardware.mixed_precision):
        outputs = model(images, radar_pts, radar_mask, ego_motion)

        # ---- Detection loss ----------------------------------------
        det_losses = det_loss_fn(
            outputs['class_logits'],
            outputs['pred_boxes'],
            gt_boxes,
            gt_labels,
            ann_mask,
        )

        # ---- Tracking loss -----------------------------------------
        # For tracking we need query-to-GT assignment from detection head
        # Here we use a simplified version (pass None to skip for now)
        track_losses = track_loss_fn(
            outputs['track_embeds'],
            track_ids,
            ann_mask,
            query_to_gt=None,   # TODO: wire in Hungarian assignments
        )

        # ---- Prediction loss ---------------------------------------
        pred_losses = pred_loss_fn(
            outputs['pred_traj'],
            gt_traj,
            fut_mask,
            ann_mask,
            query_to_gt=torch.full(
                (images.shape[0], cfg.model.detection.num_queries),
                -1, dtype=torch.long, device=device
            ),  # TODO: use Hungarian assignments
        )

        total_loss = det_losses['total'] + track_losses['total'] + pred_losses['total']

    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
    scaler.step(optimizer)
    scaler.update()

    return {
        'total':      total_loss.item(),
        'det_cls':    det_losses['cls'].item(),
        'det_bbox':   det_losses['bbox'].item(),
        'det_giou':   det_losses['giou'].item(),
        'tracking':   track_losses['tracking_contrastive'].item(),
        'prediction': pred_losses['prediction_l1'].item(),
    }


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    val_loader,
    model,
    det_loss_fn,
    pred_loss_fn,
    device,
    cfg,
):
    """Run validation loop, return metric dict."""
    model.eval()
    total_losses = {}
    map_metric = MeanAveragePrecision(
        num_classes=cfg.model.detection.num_classes,
        iou_threshold=0.5,
    )

    for batch in val_loader:
        images     = batch['images'].to(device)
        radar_pts  = batch['radar_points'].to(device)
        radar_mask = batch['radar_mask'].to(device)
        ego_motion = batch['ego_motion'].to(device)
        gt_boxes   = batch['boxes'].to(device)
        gt_labels  = batch['labels'].to(device)
        ann_mask   = batch['ann_mask'].to(device)

        with torch.cuda.amp.autocast(enabled=cfg.hardware.mixed_precision):
            outputs = model(images, radar_pts, radar_mask, ego_motion)

        det_losses = det_loss_fn(
            outputs['class_logits'], outputs['pred_boxes'],
            gt_boxes, gt_labels, ann_mask,
        )
        for k, v in det_losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()

        # mAP update (use first sample in batch)
        from models.detection_head import DetectionHead
        pred_dec = DetectionHead.decode_boxes(outputs['pred_boxes'][0])
        pred_cls = outputs['class_logits'][0].softmax(-1)
        pred_scores, pred_lbl = pred_cls[:, :-1].max(-1)
        conf = outputs['confidence'][0, :, 0]
        valid_q = conf > cfg.eval.score_threshold
        if valid_q.any() and ann_mask[0].any():
            gt_valid = ann_mask[0].bool()
            map_metric.update(
                pred_dec[valid_q], pred_scores[valid_q], pred_lbl[valid_q],
                gt_boxes[0][gt_valid], gt_labels[0][gt_valid],
            )

    n = len(val_loader)
    result = {f'val_{k}': v / n for k, v in total_losses.items()}
    result.update(map_metric.compute())
    model.train()
    return result


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg_path: str, resume: str = None):
    cfg = load_config(cfg_path)

    # Seed
    torch.manual_seed(cfg.hardware.seed)

    # Device
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")

    # ---- Datasets ------------------------------------------------------
    train_ds = build_dataset(cfg, split='train')
    val_ds   = build_dataset(cfg, split='val')

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
    )

    logger.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ---- Model ---------------------------------------------------------
    model = RearFusionModel(cfg).to(device)
    param_counts = model.count_parameters()
    logger.info("Parameter counts:")
    for name, count in param_counts.items():
        logger.info(f"  {name:30s}: {count:,}")

    # ---- Losses --------------------------------------------------------
    det_loss_fn   = DetectionLoss(cfg).to(device)
    track_loss_fn = TrackingLoss(cfg).to(device)
    pred_loss_fn  = PredictionLoss(cfg).to(device)

    # ---- Optimiser -----------------------------------------------------
    param_groups = model.get_parameter_groups()
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.training.weight_decay,
    )

    # ---- Scheduler -----------------------------------------------------
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler    = torch.cuda.amp.GradScaler(enabled=cfg.hardware.mixed_precision)

    # ---- Checkpoint manager -------------------------------------------
    ckpt_mgr  = CheckpointManager(cfg.training.checkpoint_dir)
    writer    = SummaryWriter(cfg.training.log_dir)

    start_epoch = 0
    if resume:
        meta = ckpt_mgr.load(resume, model, optimizer, scheduler, device=str(device))
        start_epoch = meta['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    # ---- Training loop ------------------------------------------------
    global_step = start_epoch * len(train_loader)
    model.train()

    for epoch in range(start_epoch, cfg.training.num_epochs):
        # Unfreeze backbone after warmup
        if epoch == cfg.training.unfreeze_backbone_epoch:
            model.unfreeze_backbone()
            logger.info(f"Epoch {epoch}: backbone unfrozen")

        t0 = time.time()
        epoch_losses = {}

        for step, batch in enumerate(train_loader):
            losses = train_step(
                batch, model, det_loss_fn, track_loss_fn, pred_loss_fn,
                optimizer, scaler, device, cfg,
            )
            scheduler.step()
            global_step += 1

            # Accumulate
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

            # Log to TensorBoard every 50 steps
            if global_step % 50 == 0:
                for k, v in losses.items():
                    writer.add_scalar(f'train/{k}', v, global_step)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)

            if step % 20 == 0:
                lr_now = optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch:3d} | Step {step:4d}/{len(train_loader)} "
                    f"| loss={losses['total']:.4f} "
                    f"| det_cls={losses['det_cls']:.3f} "
                    f"| det_bbox={losses['det_bbox']:.3f} "
                    f"| lr={lr_now:.2e}"
                )

        # Epoch summary
        n = len(train_loader)
        avg_losses = {k: v / n for k, v in epoch_losses.items()}
        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch} done in {elapsed:.1f}s | "
            f"avg total={avg_losses.get('total', 0):.4f}"
        )

        # Validation
        if (epoch + 1) % cfg.training.val_interval == 0:
            val_metrics = validate(val_loader, model, det_loss_fn, pred_loss_fn, device, cfg)
            for k, v in val_metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)
            logger.info(f"  Validation: {val_metrics}")
        else:
            val_metrics = {'val_loss': avg_losses.get('total', float('inf'))}

        # Checkpoint
        if (epoch + 1) % cfg.training.save_interval == 0:
            ckpt_mgr.save(
                epoch, model, optimizer, scheduler,
                metrics=val_metrics,
                cfg_dict=dict(cfg) if hasattr(cfg, 'items') else {},
            )

    writer.close()
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RearFusionModel')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (or "best"/"latest")')
    args = parser.parse_args()
    train(args.config, args.resume)
