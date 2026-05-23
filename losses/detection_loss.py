"""
Detection Loss
===============
Computes the combined detection loss using Hungarian (bipartite) matching
between predicted object queries and ground-truth annotations.

Loss components:
  1. Classification loss (focal cross-entropy)
  2. Box regression loss (L1 on normalised coordinates)
  3. Generalised IoU loss (GIoU, 2-D BEV projection)

Hungarian Matching
------------------
DETR uses set prediction: all Q queries must match to M ground truth objects
with no duplicate assignments.  The optimal one-to-one assignment is found
via the Hungarian algorithm (scipy.optimize.linear_sum_assignment).

Matching cost (per query-GT pair):
  C(q, m) = λ_cls * (1 - P_class(m)) + λ_bbox * L1(box) + λ_giou * GIoU

After matching:
  - Matched queries incur detection losses (cls + box + giou)
  - Unmatched queries are assigned the "background" class (cls loss only)

NOTE: In the PoC we use a simplified 2-D (BEV) GIoU since full 3-D IoU
is computationally expensive. For production, use nuScenes official IoU.
"""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available — Hungarian matching will use greedy assignment.")

from models.detection_head import DetectionHead, BG_CLASS_IDX

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hungarian Matcher
# ---------------------------------------------------------------------------

@torch.no_grad()
def hungarian_match(
    pred_logits: torch.Tensor,    # [Q, C+1]
    pred_boxes:  torch.Tensor,    # [Q, 8]
    gt_labels:   torch.Tensor,    # [M]    long
    gt_boxes:    torch.Tensor,    # [M, 7]
    w_cls: float = 1.0,
    w_bbox: float = 5.0,
    w_giou: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find optimal one-to-one assignment between Q predictions and M GT objects.

    Args:
        pred_logits:  [Q, C+1]
        pred_boxes:   [Q, 8]    (encoded: cx,cy,cz,log_dx,log_dy,log_dz,sin_y,cos_y)
        gt_labels:    [M]
        gt_boxes:     [M, 7]    (cx,cy,cz,dx,dy,dz,yaw)

    Returns:
        row_ind: Tensor[K]  indices into predictions
        col_ind: Tensor[K]  indices into GT
        K ≤ min(Q, M) matched pairs
    """
    Q = pred_logits.shape[0]
    M = gt_labels.shape[0]

    if M == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)

    device = pred_logits.device

    # ---- Class cost: -log(P(gt_class)) for each pred-GT pair ---------
    pred_probs = pred_logits.softmax(-1)   # [Q, C+1]
    # For each GT label m, pick the probability of that class from each query
    cls_cost = -pred_probs[:, gt_labels]   # [Q, M]  (lower cost = higher prob)

    # ---- Box L1 cost -------------------------------------------------
    pred_xyzcwh = DetectionHead.decode_boxes(pred_boxes)  # [Q, 7]
    # Normalise by scene range for stable gradients (rough: ±50m, ±3m height)
    norm = torch.tensor([50., 50., 3., 5., 3., 3., 3.14], device=device)
    pred_norm = pred_xyzcwh / norm
    gt_norm   = gt_boxes / norm
    # Pairwise L1: [Q, M]
    bbox_cost = (pred_norm.unsqueeze(1) - gt_norm.unsqueeze(0)).abs().sum(-1)

    # ---- GIoU cost (2-D BEV, using x and y extent only) ---------------
    giou_cost = -bev_giou(pred_xyzcwh, gt_boxes)  # [Q, M]  (higher GIoU = lower cost)

    # ---- Combined cost matrix ----------------------------------------
    C = w_cls * cls_cost + w_bbox * bbox_cost + w_giou * giou_cost
    C_np = C.cpu().numpy()

    if SCIPY_AVAILABLE:
        row_ind, col_ind = linear_sum_assignment(C_np)
    else:
        # Greedy fallback (not optimal but functional)
        row_ind, col_ind = _greedy_match(C_np)

    return (
        torch.as_tensor(row_ind, dtype=torch.long),
        torch.as_tensor(col_ind, dtype=torch.long),
    )


def _greedy_match(C: np.ndarray):
    """Greedy matching fallback if scipy is unavailable."""
    Q, M = C.shape
    used_q, used_m = set(), set()
    rows, cols = [], []
    flat = np.argsort(C.ravel())
    for idx in flat:
        q, m = divmod(int(idx), M)
        if q in used_q or m in used_m:
            continue
        rows.append(q); cols.append(m)
        used_q.add(q); used_m.add(m)
        if len(rows) == min(Q, M):
            break
    return np.array(rows), np.array(cols)


def bev_giou(
    pred_boxes: torch.Tensor,  # [Q, 7]
    gt_boxes:   torch.Tensor,  # [M, 7]
) -> torch.Tensor:
    """
    Compute 2-D BEV GIoU between all pairs of predicted and GT boxes.

    Only uses (cx, cy, dx, dy) — ignores z and yaw for simplicity.
    Returns pairwise GIoU matrix [Q, M].

    GIoU = IoU - |C \ (A ∪ B)| / |C|
    where C is the smallest enclosing box.
    """
    # Extract 2-D corners
    def _box2d(boxes):
        cx, cy = boxes[:, 0], boxes[:, 1]
        dx, dy = boxes[:, 3], boxes[:, 4]
        return cx - dx/2, cy - dy/2, cx + dx/2, cy + dy/2

    px1, py1, px2, py2 = _box2d(pred_boxes)   # each [Q]
    gx1, gy1, gx2, gy2 = _box2d(gt_boxes)     # each [M]

    # Broadcast: [Q, 1] vs [1, M]
    ix1 = torch.max(px1.unsqueeze(1), gx1.unsqueeze(0))  # [Q, M]
    iy1 = torch.max(py1.unsqueeze(1), gy1.unsqueeze(0))
    ix2 = torch.min(px2.unsqueeze(1), gx2.unsqueeze(0))
    iy2 = torch.min(py2.unsqueeze(1), gy2.unsqueeze(0))

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)   # [Q, M]
    area_p = (pred_boxes[:, 3] * pred_boxes[:, 4]).unsqueeze(1)  # [Q, 1]
    area_g = (gt_boxes[:, 3]   * gt_boxes[:, 4]).unsqueeze(0)    # [1, M]
    union = area_p + area_g - inter + 1e-6

    iou = inter / union  # [Q, M]

    # Enclosing box
    ex1 = torch.min(px1.unsqueeze(1), gx1.unsqueeze(0))
    ey1 = torch.min(py1.unsqueeze(1), gy1.unsqueeze(0))
    ex2 = torch.max(px2.unsqueeze(1), gx2.unsqueeze(0))
    ey2 = torch.max(py2.unsqueeze(1), gy2.unsqueeze(0))
    enc = (ex2 - ex1).clamp(0) * (ey2 - ey1).clamp(0) + 1e-6

    giou = iou - (enc - union) / enc  # [Q, M]
    return giou


# ---------------------------------------------------------------------------
# Focal Loss helper
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(
    inputs: torch.Tensor,   # [N, C]  raw logits
    targets: torch.Tensor,  # [N]     long class labels (includes BG_CLASS)
    num_classes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Sigmoid-based focal loss for multi-class detection.

    Each class is treated as a binary classification problem (one-vs-rest).
    Returns mean loss over N predictions.
    """
    N, C = inputs.shape
    # One-hot encode targets
    target_onehot = torch.zeros(N, C, device=inputs.device)
    fg_mask = targets < num_classes  # exclude background from positive labels
    target_onehot[fg_mask, targets[fg_mask]] = 1.0

    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, target_onehot, reduction='none')
    p_t = p * target_onehot + (1 - p) * (1 - target_onehot)
    loss = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * target_onehot + (1 - alpha) * (1 - target_onehot)
    loss = alpha_t * loss
    return loss.mean()


# ---------------------------------------------------------------------------
# Detection Loss Module
# ---------------------------------------------------------------------------

class DetectionLoss(nn.Module):
    """
    Multitask detection loss with Hungarian matching.

    Args:
        cfg: full config
    """

    def __init__(self, cfg):
        super().__init__()
        wt = cfg.training.loss_weights
        self.w_cls  = wt.detection_cls
        self.w_bbox = wt.detection_bbox
        self.w_giou = wt.detection_giou
        self.num_classes = cfg.model.detection.num_classes

    def forward(
        self,
        pred_logits: torch.Tensor,   # [B, Q, C+1]
        pred_boxes:  torch.Tensor,   # [B, Q, 8]
        gt_boxes:    torch.Tensor,   # [B, M_max, 7]
        gt_labels:   torch.Tensor,   # [B, M_max]   long
        ann_mask:    torch.Tensor,   # [B, M_max]   1=real, 0=pad
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Hungarian-matched detection loss for a batch.

        Returns dict with individual loss components and total.
        """
        B = pred_logits.shape[0]
        Q = pred_logits.shape[1]
        C = self.num_classes

        total_cls  = torch.zeros(1, device=pred_logits.device)
        total_bbox = torch.zeros(1, device=pred_logits.device)
        total_giou = torch.zeros(1, device=pred_logits.device)
        n_valid = 0

        for b in range(B):
            # Get valid GT annotations
            valid_m = ann_mask[b].bool()
            gt_b    = gt_boxes[b][valid_m]   # [M, 7]
            lab_b   = gt_labels[b][valid_m]  # [M]
            M = gt_b.shape[0]

            # Hungarian matching
            row_ind, col_ind = hungarian_match(
                pred_logits[b], pred_boxes[b],
                lab_b, gt_b,
                w_cls=self.w_cls, w_bbox=self.w_bbox, w_giou=self.w_giou,
            )

            # ---- Classification loss --------------------------------
            # Default target: all queries → background
            cls_targets = torch.full((Q,), BG_CLASS_IDX, dtype=torch.long,
                                     device=pred_logits.device)
            if len(row_ind) > 0:
                cls_targets[row_ind] = lab_b[col_ind]

            loss_cls = sigmoid_focal_loss(
                pred_logits[b], cls_targets, C
            )

            # ---- Box regression loss --------------------------------
            if len(row_ind) > 0:
                pred_dec = DetectionHead.decode_boxes(pred_boxes[b][row_ind])  # [K, 7]
                norm = torch.tensor([50., 50., 3., 5., 3., 3., 3.14],
                                    device=pred_dec.device)
                loss_bbox = F.l1_loss(
                    pred_dec / norm,
                    gt_b[col_ind] / norm,
                )
                giou_vals = bev_giou(pred_dec, gt_b[col_ind])
                diag_giou = giou_vals[torch.arange(len(row_ind)),
                                      torch.arange(len(row_ind))]
                loss_giou = (1 - diag_giou).mean()
                n_valid += 1
            else:
                loss_bbox = pred_boxes.new_zeros(1)[0]
                loss_giou = pred_boxes.new_zeros(1)[0]

            total_cls  += loss_cls
            total_bbox += loss_bbox
            total_giou += loss_giou

        denom = max(B, 1)
        loss_dict = {
            'cls':  total_cls  / denom,
            'bbox': total_bbox / denom,
            'giou': total_giou / denom,
        }
        loss_dict['total'] = (
            self.w_cls  * loss_dict['cls']  +
            self.w_bbox * loss_dict['bbox'] +
            self.w_giou * loss_dict['giou']
        )
        return loss_dict
