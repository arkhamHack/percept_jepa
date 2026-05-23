"""
Prediction-Focused Training Script
=====================================
Fine-tunes the prediction head while keeping detection frozen.
Useful when you have a pre-trained detection checkpoint and want
to improve trajectory prediction without disturbing detection quality.

Strategy:
  1. Load a pre-trained detection checkpoint.
  2. Freeze the entire model except PredictionHead.
  3. Train only the prediction loss.

Usage:
    python trainers/train_prediction.py \
        --config configs/default.yaml \
        --det_ckpt checkpoints/detection/best.pt
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
from losses.prediction_loss import PredictionLoss
from utils.checkpointing import CheckpointManager
from utils.metrics import ade_fde

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
                v = self[k]; return A(v) if isinstance(v, dict) else v
        def _c(d): return A({k: _c(v) if isinstance(v, dict) else v for k,v in d.items()})
        return _c(d)


def train_prediction(cfg_path: str, det_ckpt: str = None):
    cfg    = load_config(cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.hardware.seed)

    train_ds = build_dataset(cfg, 'train')
    val_ds   = build_dataset(cfg, 'val')
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=cfg.data.num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=cfg.data.num_workers)

    model = RearFusionModel(cfg).to(device)

    # Load pre-trained detection checkpoint (optional)
    if det_ckpt:
        ckpt = torch.load(det_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        logger.info(f"Loaded detection checkpoint: {det_ckpt}")

    # Freeze everything except prediction head
    for name, p in model.named_parameters():
        if 'prediction_head' not in name:
            p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters (prediction head only): {n_params:,}")

    pred_loss_fn = PredictionLoss(cfg).to(device)
    optimizer    = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    scaler   = torch.cuda.amp.GradScaler(enabled=cfg.hardware.mixed_precision)
    ckpt_mgr = CheckpointManager(os.path.join(cfg.training.checkpoint_dir, 'prediction'))
    writer   = SummaryWriter(os.path.join(cfg.training.log_dir, 'prediction'))

    global_step = 0
    for epoch in range(cfg.training.num_epochs):
        model.train()
        # Keep frozen parts in eval mode
        model.vjepa_encoder.eval()
        model.radar_encoder.eval()
        model.fusion_transformer.eval()
        model.detection_head.eval()
        model.tracking_head.eval()

        for step, batch in enumerate(train_loader):
            images     = batch['images'].to(device)
            radar_pts  = batch['radar_points'].to(device)
            radar_mask = batch['radar_mask'].to(device)
            ego_motion = batch['ego_motion'].to(device)
            gt_traj    = batch['future_trajectories'].to(device)
            fut_mask   = batch['future_mask'].to(device)
            ann_mask   = batch['ann_mask'].to(device)
            B, Q       = images.shape[0], cfg.model.detection.num_queries

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg.hardware.mixed_precision):
                with torch.no_grad():
                    out_frozen = model(images, radar_pts, radar_mask, ego_motion)
                query_embeds = out_frozen['scene_latent'].tokens[:, :Q, :]   # [B, Q, D]
                pred_traj = model.prediction_head(query_embeds)              # [B, Q, K, 2]
                losses = pred_loss_fn(
                    pred_traj, gt_traj, fut_mask, ann_mask,
                    query_to_gt=torch.full((B, Q), -1, dtype=torch.long, device=device)
                )

            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                cfg.training.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            if step % 10 == 0:
                writer.add_scalar('pred/total', losses['total'].item(), global_step)
                logger.info(
                    f"E{epoch} S{step}/{len(train_loader)} "
                    f"pred_loss={losses['total']:.4f}"
                )

        if (epoch + 1) % cfg.training.save_interval == 0:
            ckpt_mgr.save(epoch, model, optimizer, None, metrics={'epoch': epoch})

    writer.close()
    logger.info("Prediction training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default='configs/default.yaml')
    parser.add_argument('--det_ckpt', default=None, help='Pre-trained detection checkpoint')
    args = parser.parse_args()
    train_prediction(args.config, args.det_ckpt)
