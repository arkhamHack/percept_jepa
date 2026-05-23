"""
Collate Functions
==================
Custom collation for variable-length per-sample tensors:
  - boxes, labels, track_ids:    M varies per sample
  - future_trajectories, masks:  M varies per sample
  - radar_points, radar_mask:    already fixed-length N_max  (from dataset)
  - images, ego_motion:          fixed shape — default collation works

All variable-length tensors are stacked into padded tensors with an
accompanying 'valid' mask, so batches are dense.
"""

from typing import Any, Dict, List

import torch
import numpy as np
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate that handles:
      1. Variable-length annotation tensors (boxes, labels, track_ids,
         future_trajectories, future_mask)  → padded to max M in the batch
      2. Fixed-size tensors (images, radar_points, ego_motion, radar_mask)
         → stacked normally via default_collate
      3. Sample tokens → list of strings

    Output shapes (B = batch size, M_max = max annotations in batch):
        images:               [B, T, C, H, W]
        radar_points:         [B, N, 6]
        radar_mask:           [B, N]
        ego_motion:           [B, 6]
        boxes:                [B, M_max, 7]
        labels:               [B, M_max]       long
        track_ids:            [B, M_max]       long
        future_trajectories:  [B, M_max, K, 2]
        future_mask:          [B, M_max, K]
        ann_mask:             [B, M_max]       1 = real annotation, 0 = pad
    """
    # ---- Fixed-shape fields (default collation) -------------------------
    fixed_keys = ['images', 'radar_points', 'radar_mask', 'ego_motion']
    fixed_batch = {k: default_collate([s[k] for s in batch]) for k in fixed_keys}

    # ---- Variable-length annotation fields ------------------------------
    B = len(batch)
    M_vals = [batch[i]['boxes'].shape[0] for i in range(B)]
    M_max = max(M_vals) if M_vals else 0

    # Retrieve shape info from first non-empty sample
    K = batch[0]['future_trajectories'].shape[1] if M_vals[0] > 0 else 12

    boxes_pad      = torch.zeros(B, M_max, 7)
    labels_pad     = torch.zeros(B, M_max, dtype=torch.long)
    track_ids_pad  = torch.zeros(B, M_max, dtype=torch.long)
    fut_traj_pad   = torch.zeros(B, M_max, K, 2)
    fut_mask_pad   = torch.zeros(B, M_max, K)
    ann_mask       = torch.zeros(B, M_max)

    for i, sample in enumerate(batch):
        M = M_vals[i]
        if M == 0:
            continue
        boxes_pad[i, :M]      = sample['boxes']
        labels_pad[i, :M]     = sample['labels']
        track_ids_pad[i, :M]  = sample['track_ids']
        fut_traj_pad[i, :M]   = sample['future_trajectories']
        fut_mask_pad[i, :M]   = sample['future_mask']
        ann_mask[i, :M]       = 1.0

    # ---- Sample tokens --------------------------------------------------
    sample_tokens = [s['sample_token'] for s in batch]

    return {
        **fixed_batch,
        'boxes':                boxes_pad,        # [B, M_max, 7]
        'labels':               labels_pad,        # [B, M_max]
        'track_ids':            track_ids_pad,     # [B, M_max]
        'future_trajectories':  fut_traj_pad,      # [B, M_max, K, 2]
        'future_mask':          fut_mask_pad,       # [B, M_max, K]
        'ann_mask':             ann_mask,           # [B, M_max]
        'sample_tokens':        sample_tokens,      # List[str]
    }
