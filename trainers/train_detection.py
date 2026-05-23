"""
Detection-Only Training Script
================================
A simplified training script focused purely on detection quality.
Useful for:
  1. Verifying the detection pipeline before adding tracking/prediction.
  2. Pre-training detection before multitask fine-tuning.
  3. Ablation studies isolating detection performance.

Compared to train_multitask.py:
  - Only detection loss is active
  - Tracking + prediction heads still run (for embedding) but their
    losses are zeroed
  - Supports a larger batch size due to reduced memory footprint

Usage:
    python trainers/train_detection.py --config configs/default.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.nuscenes_dataset import build_dataset
from data.collate import collate_fn
from models.full_model import RearFusionModel
from models.detection_head import DetectionHead
from losses.detection_loss import DetectionLoss
from utils.checkpointing import CheckpointManager
from utils.metrics import MeanAveragePrecision

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def load_config(path):
    try:
        from omegaconf import OmegaConf
        return OmegaConf.load(path)
    except ImportError:
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        class A(dict):
            def __getattr__(self, k):
                v = self[k]
                return A(v) if isinstance(v, dict) else v
        def _c(d):
            return A({k: _c(v) if isinstance(v, dict) else v for k,v in d.items()})
        return _c(d)


def train_detection(cfg_path: str):
    cfg = load_config(cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.hardware.seed)

    train_ds = build_dataset(cfg, split='train')
    val_ds   = build_dataset(cfg, split='val')

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=cfg.data.num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=cfg.data.num_workers)

    model        = RearFusionModel(cfg).to(device)
    det_loss_fn  = DetectionLoss(cfg).to(device)
    optimizer    = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )
    scaler       = torch.cuda.amp.GradScaler(enabled=cfg.hardware.mixed_precision)
    ckpt_mgr     = CheckpointManager(os.path.join(cfg.training.checkpoint_dir, 'detection'))
    writer       = SummaryWriter(os.path.join(cfg.training.log_dir, 'detection'))

    global_step = 0
    for epoch in range(cfg.training.num_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            images     = batch['images'].to(device)
            radar_pts  = batch['radar_points'].to(device)
            radar_mask = batch['radar_mask'].to(device)
            ego_motion = batch['ego_motion'].to(device)
            gt_boxes   = batch['boxes'].to(device)
            gt_labels  = batch['labels'].to(device)
            ann_mask   = batch['ann_mask'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg.hardware.mixed_precision):
                out = model(images, radar_pts, radar_mask, ego_motion)
                losses = det_loss_fn(
                    out['class_logits'], out['pred_boxes'],
                    gt_boxes, gt_labels, ann_mask
                )

            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            if step % 10 == 0:
                writer.add_scalar('det/total', losses['total'].item(), global_step)
                logger.info(
                    f"E{epoch} S{step}/{len(train_loader)} "
                    f"total={losses['total']:.4f} cls={losses['cls']:.3f} "
                    f"bbox={losses['bbox']:.3f} giou={losses['giou']:.3f}"
                )

        if (epoch + 1) % cfg.training.save_interval == 0:
            ckpt_mgr.save(epoch, model, optimizer, None, metrics={'epoch': epoch})

    writer.close()
    logger.info("Detection-only training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    train_detection(args.config)
