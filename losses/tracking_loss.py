"""
Tracking Loss — Contrastive Identity Embedding Loss
=====================================================
Trains the tracking head to produce embeddings such that:
  - Embeddings of the **same instance** in consecutive frames are similar.
  - Embeddings of **different instances** are dissimilar.

Loss: InfoNCE (Noise Contrastive Estimation)
  For each anchor embedding, we treat matched instances (same track_id) as
  positive pairs and all other instances in the batch as negatives.

  L = -log( exp(sim(z_i, z_i+) / τ) / Σ_j exp(sim(z_i, z_j) / τ) )

  where z_i is an anchor, z_i+ is its positive, and τ is temperature.

Alternative: Triplet loss (also implemented).
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TrackingLoss(nn.Module):
    """
    Contrastive tracking embedding loss.

    Uses InfoNCE over matched query embeddings across the batch.

    Args:
        cfg: full config
    """

    def __init__(self, cfg, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.weight = cfg.training.loss_weights.tracking

    def forward(
        self,
        track_embeds: torch.Tensor,    # [B, Q, E]  normalised embeddings
        track_ids:    torch.Tensor,    # [B, M_max]  GT track IDs (long)
        ann_mask:     torch.Tensor,    # [B, M_max]  1 = valid annotation
        query_to_gt:  Optional[torch.Tensor] = None,  # [B, Q] GT assignment
    ) -> Dict[str, torch.Tensor]:
        """
        Compute InfoNCE contrastive loss on matched query embeddings.

        Args:
            track_embeds: [B, Q, E]   L2-normalised from TrackingHead
            track_ids:    [B, M_max]  GT track IDs for valid annotations
            ann_mask:     [B, M_max]  validity mask
            query_to_gt:  [B, Q]     maps each query to a GT index
                                     (-1 = background, ≥0 = matched GT)
                                     If None, skip loss for this sample.

        Returns:
            dict with 'total' and 'tracking_contrastive' keys
        """
        if query_to_gt is None:
            return {'total': track_embeds.new_zeros(1).squeeze(),
                    'tracking_contrastive': track_embeds.new_zeros(1).squeeze()}

        B, Q, E = track_embeds.shape
        total_loss = 0.0
        n_pairs = 0

        # Build a flat list of (embedding, track_id) for all matched queries
        all_embeds = []
        all_labels = []

        for b in range(B):
            for q in range(Q):
                gt_idx = query_to_gt[b, q].item()
                if gt_idx < 0:
                    continue
                valid_m = ann_mask[b].bool()
                ids = track_ids[b]  # [M_max]
                if gt_idx >= valid_m.sum():
                    continue
                # Map gt_idx within valid annotations to absolute track_id
                valid_ids = ids[valid_m]
                if gt_idx >= len(valid_ids):
                    continue
                tid = valid_ids[gt_idx].item()
                all_embeds.append(track_embeds[b, q])   # [E]
                all_labels.append(tid)

        if len(all_embeds) < 2:
            return {'total': track_embeds.new_zeros(1).squeeze(),
                    'tracking_contrastive': track_embeds.new_zeros(1).squeeze()}

        embeds = torch.stack(all_embeds)   # [N, E]
        labels = torch.tensor(all_labels, device=embeds.device)  # [N]

        # Compute pairwise cosine similarities (already normalised)
        sim = (embeds @ embeds.T) / self.temperature   # [N, N]

        # Mask self-similarity
        N = embeds.shape[0]
        mask_diag = torch.eye(N, dtype=torch.bool, device=embeds.device)
        sim = sim.masked_fill(mask_diag, float('-inf'))

        # Positive mask: same track_id, different sample position
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask_diag
        # [N, N]

        # InfoNCE: for each anchor, the positive is the hardest positive
        # Numerator: log(sum of exp of positive similarities)
        # (if no positive exists for this anchor, skip it)
        has_pos = pos_mask.any(dim=1)   # [N]
        if not has_pos.any():
            return {'total': track_embeds.new_zeros(1).squeeze(),
                    'tracking_contrastive': track_embeds.new_zeros(1).squeeze()}

        sim_valid = sim[has_pos]         # [N', N]
        pos_valid = pos_mask[has_pos]    # [N', N]

        log_sum_exp_all = sim_valid.logsumexp(dim=1)   # [N']
        pos_logits = (sim_valid * pos_valid.float()).sum(dim=1) / (pos_valid.float().sum(dim=1) + 1e-6)
        # Approximate InfoNCE: mean of positive log-prob
        loss = (log_sum_exp_all - pos_logits).mean()

        result = self.weight * loss
        return {
            'total':                  result,
            'tracking_contrastive':   loss.detach(),
        }


class TripletTrackingLoss(nn.Module):
    """
    Triplet margin loss alternative for tracking.

    For each matched query, we sample:
      - positive: another query with the same track_id
      - negative: a query with a different track_id

    Trains embeddings with margin separation.
    """

    def __init__(self, cfg, margin: float = 0.5):
        super().__init__()
        self.margin = margin
        self.weight = cfg.training.loss_weights.tracking

    def forward(
        self,
        track_embeds: torch.Tensor,    # [B, Q, E]
        track_ids:    torch.Tensor,    # [B, M_max]
        ann_mask:     torch.Tensor,    # [B, M_max]
        query_to_gt:  Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if query_to_gt is None:
            return {'total': track_embeds.new_zeros(1).squeeze()}

        # (similar construction as InfoNCE — collect matched embeddings)
        B, Q, E = track_embeds.shape
        all_embeds, all_labels = [], []

        for b in range(B):
            for q in range(Q):
                gt_idx = query_to_gt[b, q].item()
                if gt_idx < 0:
                    continue
                valid_ids = track_ids[b][ann_mask[b].bool()]
                if gt_idx >= len(valid_ids):
                    continue
                all_embeds.append(track_embeds[b, q])
                all_labels.append(valid_ids[gt_idx].item())

        if len(all_embeds) < 3:
            return {'total': track_embeds.new_zeros(1).squeeze()}

        embeds = torch.stack(all_embeds)                          # [N, E]
        labels = torch.tensor(all_labels, device=embeds.device)  # [N]

        # Pairwise cosine distances
        dist = 1 - (embeds @ embeds.T)   # [N, N]

        loss = torch.zeros(1, device=embeds.device)
        n_triplets = 0
        for i in range(len(all_labels)):
            pos_mask = (labels == labels[i]) & (torch.arange(len(labels), device=embeds.device) != i)
            neg_mask = labels != labels[i]
            if not pos_mask.any() or not neg_mask.any():
                continue
            d_pos = dist[i][pos_mask].min()
            d_neg = dist[i][neg_mask].min()
            loss += F.relu(d_pos - d_neg + self.margin)
            n_triplets += 1

        if n_triplets > 0:
            loss = loss / n_triplets

        return {'total': self.weight * loss.squeeze()}
