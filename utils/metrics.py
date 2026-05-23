"""
Evaluation Metrics
===================
Standard AV perception metrics:
  - Detection: mAP (2-D BEV IoU)
  - Tracking:  AMOTA / AMOTP (simplified)
  - Prediction: ADE / FDE (Average/Final Displacement Error)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection Metrics
# ---------------------------------------------------------------------------

def compute_bev_iou(
    pred_boxes: torch.Tensor,  # [Q, 7]
    gt_boxes:   torch.Tensor,  # [M, 7]
) -> torch.Tensor:
    """
    Compute pairwise 2-D BEV IoU between predicted and GT boxes.

    Uses axis-aligned approximation (ignores yaw) for efficiency.
    Returns Tensor[Q, M].
    """
    # Extract (cx, cy, dx, dy)
    def _xyxy(b):
        cx, cy = b[:, 0], b[:, 1]
        dx, dy = b[:, 3], b[:, 4]
        return cx - dx/2, cy - dy/2, cx + dx/2, cy + dy/2

    px1, py1, px2, py2 = _xyxy(pred_boxes)
    gx1, gy1, gx2, gy2 = _xyxy(gt_boxes)

    ix1 = torch.max(px1.unsqueeze(1), gx1.unsqueeze(0))
    iy1 = torch.max(py1.unsqueeze(1), gy1.unsqueeze(0))
    ix2 = torch.min(px2.unsqueeze(1), gx2.unsqueeze(0))
    iy2 = torch.min(py2.unsqueeze(1), gy2.unsqueeze(0))

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    area_p = (pred_boxes[:, 3] * pred_boxes[:, 4]).unsqueeze(1)
    area_g = (gt_boxes[:, 3]   * gt_boxes[:, 4]).unsqueeze(0)
    union = area_p + area_g - inter + 1e-6
    return inter / union


class MeanAveragePrecision:
    """
    Compute per-class AP and mAP at a fixed IoU threshold.

    Usage:
        metric = MeanAveragePrecision(num_classes=10, iou_threshold=0.5)
        for batch in val_loader:
            metric.update(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, gt_mask)
        results = metric.compute()
    """

    def __init__(self, num_classes: int = 10, iou_threshold: float = 0.5):
        self.C = num_classes
        self.iou_thresh = iou_threshold
        self.reset()

    def reset(self):
        # Per-class: list of (score, tp) tuples
        self._per_class: Dict[int, List[Tuple[float, int]]] = {c: [] for c in range(self.C)}
        self._n_gt: Dict[int, int] = {c: 0 for c in range(self.C)}

    def update(
        self,
        pred_boxes:   torch.Tensor,   # [Q, 7]  decoded
        pred_scores:  torch.Tensor,   # [Q]
        pred_labels:  torch.Tensor,   # [Q]  long
        gt_boxes:     torch.Tensor,   # [M, 7]
        gt_labels:    torch.Tensor,   # [M]  long
    ):
        if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
            for c in range(self.C):
                self._n_gt[c] += (gt_labels == c).sum().item()
            return

        iou = compute_bev_iou(pred_boxes, gt_boxes)   # [Q, M]

        matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)

        # Sort predictions by score
        order = pred_scores.argsort(descending=True)

        for q in order:
            c = pred_labels[q].item()
            score = pred_scores[q].item()

            # Find GT boxes of same class
            gt_same_cls = (gt_labels == c).nonzero(as_tuple=False).squeeze(1)
            if len(gt_same_cls) == 0:
                self._per_class[c].append((score, 0))
                continue

            iou_q = iou[q, gt_same_cls]  # IoU with GT of same class
            best_m = iou_q.argmax().item()
            best_iou = iou_q[best_m].item()
            gt_idx = gt_same_cls[best_m].item()

            if best_iou >= self.iou_thresh and not matched_gt[gt_idx]:
                matched_gt[gt_idx] = True
                self._per_class[c].append((score, 1))  # TP
            else:
                self._per_class[c].append((score, 0))  # FP

        for c in range(self.C):
            self._n_gt[c] += (gt_labels == c).sum().item()

    def compute(self) -> Dict[str, float]:
        ap_per_class = {}
        for c in range(self.C):
            dets = sorted(self._per_class[c], key=lambda x: -x[0])
            n_gt = self._n_gt[c]
            if n_gt == 0:
                ap_per_class[c] = 0.0
                continue
            tp_cum = 0
            fp_cum = 0
            precision, recall = [], []
            for score, tp in dets:
                if tp:
                    tp_cum += 1
                else:
                    fp_cum += 1
                precision.append(tp_cum / (tp_cum + fp_cum))
                recall.append(tp_cum / n_gt)

            # Interpolated AP (11-point)
            ap = 0.0
            for thresh in np.linspace(0, 1, 11):
                prec_at_thresh = [p for p, r in zip(precision, recall) if r >= thresh]
                ap += max(prec_at_thresh) if prec_at_thresh else 0.0
            ap /= 11.0
            ap_per_class[c] = ap

        mAP = float(np.mean(list(ap_per_class.values())))
        return {'mAP': mAP, **{f'AP_class_{c}': v for c, v in ap_per_class.items()}}


# ---------------------------------------------------------------------------
# Trajectory Metrics
# ---------------------------------------------------------------------------

def ade_fde(
    pred_traj: torch.Tensor,    # [N, K, 2]  predicted future positions
    gt_traj:   torch.Tensor,    # [N, K, 2]  GT future positions
    mask:      torch.Tensor,    # [N, K]     valid step mask
) -> Tuple[float, float]:
    """
    Compute Average Displacement Error (ADE) and Final Displacement Error (FDE).

    ADE = mean L2 displacement over all valid steps
    FDE = L2 displacement at the final valid step

    Args:
        pred_traj, gt_traj: [N, K, 2]
        mask:               [N, K]  1 = valid

    Returns:
        (ADE, FDE)  in metres
    """
    l2 = ((pred_traj - gt_traj) ** 2).sum(-1).sqrt()   # [N, K]

    # ADE
    valid = mask.bool()
    ade = (l2 * valid.float()).sum() / valid.float().sum().clamp(min=1)

    # FDE: last valid step for each trajectory
    fde_vals = []
    for n in range(l2.shape[0]):
        valid_steps = valid[n].nonzero(as_tuple=False)
        if len(valid_steps) == 0:
            continue
        last = valid_steps[-1].item()
        fde_vals.append(l2[n, last].item())
    fde = float(np.mean(fde_vals)) if fde_vals else 0.0

    return ade.item(), fde


# ---------------------------------------------------------------------------
# Tracking Metrics (simplified AMOTA)
# ---------------------------------------------------------------------------

def compute_motp(
    pred_boxes: torch.Tensor,   # [M, 7]
    gt_boxes:   torch.Tensor,   # [M, 7]  already matched
) -> float:
    """
    MOTP: average IoU / distance of matched pairs.
    Higher = more precise localisation.
    """
    if pred_boxes.shape[0] == 0:
        return 0.0
    iou_vals = compute_bev_iou(pred_boxes, gt_boxes)  # [M, M]
    diag_iou = iou_vals.diag()
    return diag_iou.mean().item()
