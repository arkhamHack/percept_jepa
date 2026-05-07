"""nuScenes-aligned evaluation metrics for 2D detection and velocity.

Implements simplified but faithful versions of the metrics used in the
nuScenes detection benchmark: mAP (2D IoU), ATE, ASE, AVE, and NDS.
"""

from __future__ import annotations

import torch
import numpy as np


# ──────────────────────────────────────────────────────────────
#  IoU helpers
# ──────────────────────────────────────────────────────────────

def compute_iou_matrix(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
) -> torch.Tensor:
    """Pairwise IoU between two sets of ``[x1, y1, x2, y2]`` boxes.

    Args:
        boxes_a: ``(M, 4)``
        boxes_b: ``(N, 4)``

    Returns:
        ``(M, N)`` IoU matrix.
    """
    x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-8)


# ──────────────────────────────────────────────────────────────
#  Average Precision (per-threshold)
# ──────────────────────────────────────────────────────────────

def compute_ap(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> float:
    """Compute Average Precision at a single IoU threshold.

    Uses the 11-point interpolation method (PASCAL VOC style).

    Args:
        pred_boxes: ``(P, 4)`` predicted boxes.
        pred_scores: ``(P,)`` confidence scores.
        gt_boxes: ``(G, 4)`` ground-truth boxes.
        iou_threshold: IoU threshold for a true positive.

    Returns:
        AP as a float in ``[0, 1]``.
    """
    if gt_boxes.numel() == 0:
        return 1.0 if pred_boxes.numel() == 0 else 0.0
    if pred_boxes.numel() == 0:
        return 0.0

    order = pred_scores.argsort(descending=True)
    pred_boxes = pred_boxes[order]

    iou = compute_iou_matrix(pred_boxes, gt_boxes)  # (P, G)
    matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device)

    tp = torch.zeros(len(pred_boxes))
    fp = torch.zeros(len(pred_boxes))

    for i in range(len(pred_boxes)):
        ious_i = iou[i]
        ious_i[matched_gt] = 0.0
        best_j = ious_i.argmax().item()
        if ious_i[best_j] >= iou_threshold:
            tp[i] = 1
            matched_gt[best_j] = True
        else:
            fp[i] = 1

    tp_cum = tp.cumsum(0)
    fp_cum = fp.cumsum(0)
    recall = tp_cum / gt_boxes.shape[0]
    precision = tp_cum / (tp_cum + fp_cum).clamp(min=1e-8)

    # 11-point interpolation
    ap = 0.0
    for r_thr in np.linspace(0, 1, 11):
        mask = recall >= r_thr
        if mask.any():
            ap += precision[mask].max().item()
    return ap / 11.0


def compute_map(
    all_pred_boxes: list[torch.Tensor],
    all_pred_scores: list[torch.Tensor],
    all_gt_boxes: list[torch.Tensor],
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Mean Average Precision across IoU thresholds.

    Args:
        all_pred_boxes: List of ``(P_i, 4)`` tensors per sample.
        all_pred_scores: List of ``(P_i,)`` tensors per sample.
        all_gt_boxes: List of ``(G_i, 4)`` tensors per sample.
        iou_thresholds: IoU thresholds to average over.

    Returns:
        Dict with ``'mAP'`` and per-threshold ``'AP@{thr}'`` keys.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.3, 0.5, 0.7]

    results: dict[str, float] = {}
    aps_per_thr: list[float] = []

    for thr in iou_thresholds:
        sample_aps = []
        for preds, scores, gts in zip(all_pred_boxes, all_pred_scores, all_gt_boxes):
            sample_aps.append(compute_ap(preds, scores, gts, thr))
        mean_ap = float(np.mean(sample_aps)) if sample_aps else 0.0
        results[f"AP@{thr:.1f}"] = mean_ap
        aps_per_thr.append(mean_ap)

    results["mAP"] = float(np.mean(aps_per_thr))
    return results


# ──────────────────────────────────────────────────────────────
#  Velocity Error
# ──────────────────────────────────────────────────────────────

def compute_velocity_error(
    pred_vel: torch.Tensor,
    gt_vel: torch.Tensor,
    num_objects: torch.Tensor,
) -> dict[str, float]:
    """Mean absolute velocity error over valid objects.

    Args:
        pred_vel: ``(B, M, 2)`` predicted ``(vx, vy)``.
        gt_vel: ``(B, M, 2)`` ground-truth ``(vx, vy)``.
        num_objects: ``(B,)`` number of valid objects per sample.

    Returns:
        Dict with ``'mAVE'`` (mean abs velocity error) and ``'medAVE'``.
    """
    B, M, _ = pred_vel.shape
    device = pred_vel.device

    idx = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
    valid = idx < num_objects.unsqueeze(1)

    diff = (pred_vel - gt_vel).abs()  # (B, M, 2)
    per_obj_err = diff.norm(dim=-1)   # (B, M) — L2 velocity error

    valid_errors = per_obj_err[valid]
    if valid_errors.numel() == 0:
        return {"mAVE": 0.0, "medAVE": 0.0}

    return {
        "mAVE": valid_errors.mean().item(),
        "medAVE": valid_errors.median().item(),
    }


# ──────────────────────────────────────────────────────────────
#  nuScenes True Positive metrics (simplified 2D versions)
# ──────────────────────────────────────────────────────────────

def compute_ate(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    matched_pairs: torch.Tensor,
) -> float:
    """Average Translation Error — mean L2 centre distance for matched pairs.

    Args:
        pred_boxes: ``(P, 4)`` in ``[x1, y1, x2, y2]``.
        gt_boxes: ``(G, 4)``.
        matched_pairs: ``(K, 2)`` int tensor of ``(pred_idx, gt_idx)`` pairs.

    Returns:
        Scalar ATE.
    """
    if matched_pairs.numel() == 0:
        return 1.0

    pred_cx = (pred_boxes[matched_pairs[:, 0], 0] + pred_boxes[matched_pairs[:, 0], 2]) / 2
    pred_cy = (pred_boxes[matched_pairs[:, 0], 1] + pred_boxes[matched_pairs[:, 0], 3]) / 2
    gt_cx = (gt_boxes[matched_pairs[:, 1], 0] + gt_boxes[matched_pairs[:, 1], 2]) / 2
    gt_cy = (gt_boxes[matched_pairs[:, 1], 1] + gt_boxes[matched_pairs[:, 1], 3]) / 2

    dist = ((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2).sqrt()
    return dist.mean().item()


def compute_ase(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    matched_pairs: torch.Tensor,
) -> float:
    """Average Scale Error — ``1 − IoU`` for matched pred/gt pairs.

    Args:
        pred_boxes, gt_boxes, matched_pairs: same as :func:`compute_ate`.

    Returns:
        Scalar ASE (lower is better).
    """
    if matched_pairs.numel() == 0:
        return 1.0

    preds = pred_boxes[matched_pairs[:, 0]]
    gts = gt_boxes[matched_pairs[:, 1]]

    iou = compute_iou_matrix(preds, gts)
    diag_iou = iou.diag()
    return (1.0 - diag_iou).mean().item()


def compute_ave(
    pred_vel: torch.Tensor,
    gt_vel: torch.Tensor,
    matched_pairs: torch.Tensor,
) -> float:
    """Average Velocity Error for matched pairs.

    Args:
        pred_vel: ``(P, 2)`` per-object velocities.
        gt_vel: ``(G, 2)``.
        matched_pairs: ``(K, 2)`` int matched indices.

    Returns:
        Scalar AVE.
    """
    if matched_pairs.numel() == 0:
        return 1.0

    pv = pred_vel[matched_pairs[:, 0]]
    gv = gt_vel[matched_pairs[:, 1]]
    return (pv - gv).norm(dim=-1).mean().item()


# ──────────────────────────────────────────────────────────────
#  nuScenes Detection Score (NDS)
# ──────────────────────────────────────────────────────────────

def compute_nds(
    map_score: float,
    ate: float,
    ase: float,
    ave: float,
    aoe: float = 0.0,
    aae: float = 0.0,
) -> float:
    """nuScenes Detection Score — weighted combination of mAP and TP errors.

    ``NDS = (1/10) * [5 * mAP + sum(max(1 − TP_err, 0))]``

    where TP errors are ATE, ASE, AOE, AVE, AAE. For our 2D setup AOE and
    AAE default to 0 (perfect).

    Returns:
        Scalar NDS in ``[0, 1]``.
    """
    tp_scores = [
        max(1.0 - ate, 0.0),
        max(1.0 - ase, 0.0),
        max(1.0 - aoe, 0.0),
        max(1.0 - ave, 0.0),
        max(1.0 - aae, 0.0),
    ]
    return (5.0 * map_score + sum(tp_scores)) / 10.0


# ──────────────────────────────────────────────────────────────
#  Trajectory Prediction metrics (nuScenes prediction challenge)
# ──────────────────────────────────────────────────────────────

def compute_min_ade(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    num_objects: torch.Tensor,
) -> float:
    """Minimum Average Displacement Error across modes.

    For each agent, computes ADE for every predicted mode and takes the
    minimum.  Returns the mean over all valid agents.

    Args:
        pred_traj: ``(B, M, K, T, 2)`` — K modes of T waypoints.
        gt_traj: ``(B, M, T, 2)`` — ground-truth trajectory.
        num_objects: ``(B,)`` valid agents per sample.
    """
    B, M, K, T, _ = pred_traj.shape
    device = pred_traj.device

    gt_exp = gt_traj.unsqueeze(2).expand_as(pred_traj)  # (B, M, K, T, 2)
    per_mode_ade = (pred_traj - gt_exp).norm(dim=-1).mean(dim=-1)  # (B, M, K)
    min_ade = per_mode_ade.min(dim=-1).values  # (B, M)

    idx = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
    valid = idx < num_objects.unsqueeze(1)
    valid_errors = min_ade[valid]

    return valid_errors.mean().item() if valid_errors.numel() > 0 else 0.0


def compute_min_fde(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    num_objects: torch.Tensor,
) -> float:
    """Minimum Final Displacement Error across modes.

    Same as minADE but uses only the last timestep.
    """
    B, M, K, T, _ = pred_traj.shape
    device = pred_traj.device

    pred_final = pred_traj[:, :, :, -1, :]  # (B, M, K, 2)
    gt_final = gt_traj[:, :, -1, :]         # (B, M, 2)
    gt_final_exp = gt_final.unsqueeze(2).expand_as(pred_final)

    per_mode_fde = (pred_final - gt_final_exp).norm(dim=-1)  # (B, M, K)
    min_fde = per_mode_fde.min(dim=-1).values  # (B, M)

    idx = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
    valid = idx < num_objects.unsqueeze(1)
    valid_errors = min_fde[valid]

    return valid_errors.mean().item() if valid_errors.numel() > 0 else 0.0


def compute_miss_rate(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    num_objects: torch.Tensor,
    threshold: float = 2.0,
) -> float:
    """Miss rate — fraction of agents where best-mode FDE exceeds threshold.

    Args:
        threshold: Distance in metres for a prediction to count as a "miss".
    """
    B, M, K, T, _ = pred_traj.shape
    device = pred_traj.device

    pred_final = pred_traj[:, :, :, -1, :]
    gt_final = gt_traj[:, :, -1, :].unsqueeze(2).expand_as(pred_final)

    per_mode_fde = (pred_final - gt_final).norm(dim=-1)  # (B, M, K)
    min_fde = per_mode_fde.min(dim=-1).values  # (B, M)

    idx = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
    valid = idx < num_objects.unsqueeze(1)
    valid_fde = min_fde[valid]

    if valid_fde.numel() == 0:
        return 0.0
    return (valid_fde > threshold).float().mean().item()


def compute_prediction_metrics(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    num_objects: torch.Tensor,
    miss_threshold: float = 2.0,
) -> dict[str, float]:
    """Compute all prediction metrics in one call.

    Returns:
        Dict with ``minADE``, ``minFDE``, ``MissRate``.
    """
    return {
        "minADE": compute_min_ade(pred_traj, gt_traj, num_objects),
        "minFDE": compute_min_fde(pred_traj, gt_traj, num_objects),
        "MissRate": compute_miss_rate(pred_traj, gt_traj, num_objects, miss_threshold),
    }


def match_predictions(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Greedy matching of predictions to ground-truth by IoU.

    Returns:
        ``(K, 2)`` int64 tensor of ``(pred_idx, gt_idx)`` matched pairs.
    """
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.int64)

    order = pred_scores.argsort(descending=True)
    pred_boxes = pred_boxes[order]

    iou = compute_iou_matrix(pred_boxes, gt_boxes)
    matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device)
    pairs: list[tuple[int, int]] = []

    for i in range(len(pred_boxes)):
        ious_i = iou[i].clone()
        ious_i[matched_gt] = 0.0
        best_j = ious_i.argmax().item()
        if ious_i[best_j] >= iou_threshold:
            pairs.append((order[i].item(), best_j))
            matched_gt[best_j] = True

    if not pairs:
        return torch.zeros((0, 2), dtype=torch.int64)
    return torch.tensor(pairs, dtype=torch.int64)
