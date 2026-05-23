"""
Prediction Loss — Future Trajectory Regression
================================================
Computes loss between predicted future trajectories and GT future positions.

Loss: Smooth L1 (Huber) on normalised displacements, weighted by a
validity mask (not all instances have K future annotations).

Options:
  1. Displacement regression  (default): predict Δ(x,y) at each step
  2. Cumulative position regression:     predict absolute (x,y) positions

We use displacement regression because:
  - Target range is smaller and consistent across ego-motion magnitude
  - Relative motions are more transferable across scenes
  - Errors don't accumulate (each step is independently predicted)
"""

import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PredictionLoss(nn.Module):
    """
    Future trajectory regression loss.

    Applies Smooth L1 loss between predicted displacements and GT
    future positions (interpreted as displacements from current centre).

    Args:
        cfg: full config
    """

    def __init__(self, cfg):
        super().__init__()
        self.weight = cfg.training.loss_weights.prediction
        self.future_steps = cfg.model.prediction.future_steps
        # Normalise displacement by rough expected magnitude (m)
        self.pos_norm = 5.0   # expected max single-step displacement in metres

    def forward(
        self,
        pred_traj:  torch.Tensor,    # [B, Q, K, 2]  predicted displacements
        gt_traj:    torch.Tensor,    # [B, M_max, K, 2]  GT future positions (ego frame)
        fut_mask:   torch.Tensor,    # [B, M_max, K]   1 = valid step
        ann_mask:   torch.Tensor,    # [B, M_max]       1 = valid annotation
        query_to_gt: torch.Tensor,   # [B, Q]  -1 = background, ≥0 = matched GT index
    ) -> Dict[str, torch.Tensor]:
        """
        Compute trajectory prediction loss only for matched, valid GT instances.

        Args:
            pred_traj:    [B, Q, K, 2]
            gt_traj:      [B, M_max, K, 2]   absolute (x,y) in ego frame
                          converted to displacements internally
            fut_mask:     [B, M_max, K]
            ann_mask:     [B, M_max]
            query_to_gt:  [B, Q]

        Returns:
            dict with 'total' and 'prediction_l1' keys
        """
        B, Q, K, _ = pred_traj.shape
        total_loss = pred_traj.new_zeros(1)
        n_matched = 0

        for b in range(B):
            for q in range(Q):
                gt_idx = query_to_gt[b, q].item()
                if gt_idx < 0:
                    continue
                if not ann_mask[b, gt_idx]:
                    continue

                # Future mask for this GT instance [K]
                step_mask = fut_mask[b, gt_idx]    # [K]
                if not step_mask.any():
                    continue

                # GT future positions [K, 2] (absolute in ego frame)
                gt_pos = gt_traj[b, gt_idx]       # [K, 2]
                # Predicted displacements [K, 2]
                pred  = pred_traj[b, q]            # [K, 2]

                # We treat GT positions as displacements directly because
                # they're already expressed relative to current ego frame
                # (which is the same as displacement from origin).
                # Normalise both by pos_norm
                pred_n = pred / self.pos_norm
                gt_n   = gt_pos / self.pos_norm

                # Apply step validity mask
                valid_steps = step_mask.bool()  # [K]
                loss_step = F.smooth_l1_loss(
                    pred_n[valid_steps],
                    gt_n[valid_steps],
                    reduction='mean',
                )
                total_loss = total_loss + loss_step
                n_matched += 1

        if n_matched > 0:
            total_loss = total_loss / n_matched

        result = self.weight * total_loss.squeeze()
        return {
            'total':         result,
            'prediction_l1': total_loss.detach().squeeze(),
        }
