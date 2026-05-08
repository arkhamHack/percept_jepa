"""Loss functions for multimodal perception training.

Covers anchor-free detection (heatmap + box regression), spatial velocity,
tracking embedding, and legacy per-object losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
#  Anchor-free spatial losses (for MultimodalPerceptionModel)
# ──────────────────────────────────────────────────────────────

def heatmap_focal_loss(
    pred_heatmap: torch.Tensor,
    gt_heatmap: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """CenterNet-style focal loss for objectness heatmaps.

    Args:
        pred_heatmap: ``(B, C, H, W)`` predicted logits.
        gt_heatmap: ``(B, C, H, W)`` ground-truth Gaussian heatmap in [0, 1].
    """
    pred = pred_heatmap.sigmoid().clamp(1e-6, 1.0 - 1e-6)

    pos_mask = gt_heatmap.eq(1.0).float()
    neg_mask = gt_heatmap.lt(1.0).float()

    pos_loss = -((1.0 - pred) ** alpha) * pred.log() * pos_mask
    neg_loss = -((1.0 - gt_heatmap) ** beta) * (pred ** alpha) * (1.0 - pred).log() * neg_mask

    n_pos = pos_mask.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def box_regression_loss(
    pred_box: torch.Tensor,
    gt_box: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 loss on box regression at positive heatmap locations.

    Args:
        pred_box: ``(B, 4, H, W)``
        gt_box: ``(B, 4, H, W)``
        mask: ``(B, 1, H, W)`` positive cell mask.
    """
    n_pos = mask.sum().clamp(min=1.0)
    loss = F.smooth_l1_loss(pred_box, gt_box, reduction="none")
    return (loss * mask).sum() / n_pos


def spatial_velocity_loss(
    pred_vel: torch.Tensor,
    gt_vel: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 loss on per-cell velocity at positive locations.

    Args:
        pred_vel: ``(B, 2, H, W)``
        gt_vel: ``(B, 2, H, W)``
        mask: ``(B, 1, H, W)``
    """
    n_pos = mask.sum().clamp(min=1.0)
    loss = F.smooth_l1_loss(pred_vel, gt_vel, reduction="none")
    return (loss * mask).sum() / n_pos


def tracking_contrastive_loss(
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Simple contrastive regularisation on tracking embeddings.

    Encourages embeddings at positive cells to be diverse (not collapsed).

    Args:
        embeddings: ``(B, D, H, W)`` L2-normalised.
        mask: ``(B, 1, H, W)`` positive cells.
    """
    B, D, H, W = embeddings.shape
    emb_flat = embeddings.view(B, D, -1).permute(0, 2, 1)  # (B, HW, D)
    mask_flat = mask.view(B, -1)  # (B, HW)

    total_loss = torch.tensor(0.0, device=embeddings.device)
    count = 0
    for b in range(B):
        pos_idx = mask_flat[b].nonzero(as_tuple=True)[0]
        if pos_idx.shape[0] < 2:
            continue
        pos_emb = emb_flat[b, pos_idx]  # (K, D)
        sim = pos_emb @ pos_emb.T / temperature  # (K, K)
        # Each embedding should be similar to itself, dissimilar to others
        labels = torch.arange(pos_emb.shape[0], device=sim.device)
        total_loss = total_loss + F.cross_entropy(sim, labels)
        count += 1

    return total_loss / max(count, 1)


def multimodal_combined_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Combined loss for MultimodalPerceptionModel.

    Expected targets:
        gt_heatmap: ``(B, C, H, W)``
        gt_box_reg: ``(B, 4, H, W)``
        gt_velocity_map: ``(B, 2, H, W)``
        pos_mask: ``(B, 1, H, W)``
    """
    w = {"detection": 1.0, "velocity": 0.5, "tracking": 0.1, **(weights or {})}
    device = outputs["heatmap"].device
    zero = torch.tensor(0.0, device=device)

    det_heatmap = heatmap_focal_loss(outputs["heatmap"], targets["gt_heatmap"])
    det_box = box_regression_loss(outputs["box_reg"], targets["gt_box_reg"], targets["pos_mask"])
    det_loss = det_heatmap + det_box

    vel_loss = zero
    if "gt_velocity_map" in targets:
        vel_loss = spatial_velocity_loss(outputs["velocity"], targets["gt_velocity_map"], targets["pos_mask"])

    track_loss = zero
    if outputs.get("tracking_embed") is not None:
        track_loss = tracking_contrastive_loss(outputs["tracking_embed"], targets["pos_mask"])

    total = w["detection"] * det_loss + w["velocity"] * vel_loss + w["tracking"] * track_loss

    return {
        "total": total,
        "detection": det_loss,
        "detection_heatmap": det_heatmap,
        "detection_box": det_box,
        "velocity": vel_loss,
        "tracking": track_loss,
    }


# ──────────────────────────────────────────────────────────────
#  Legacy per-object losses (kept for JEPAModel baseline)
# ──────────────────────────────────────────────────────────────

def jepa_latent_loss(z_predicted: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    z_predicted = F.normalize(z_predicted, dim=-1)
    z_target = F.normalize(z_target, dim=-1)
    return F.mse_loss(z_predicted, z_target)


def detection_loss(pred_boxes, gt_boxes, num_objects):
    B, M, _ = pred_boxes.shape
    device = pred_boxes.device
    idx = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
    valid = idx < num_objects.unsqueeze(1)
    valid_f = valid.float()
    n_valid = valid_f.sum().clamp(min=1.0)
    box_loss = F.smooth_l1_loss(pred_boxes[:, :, :4], gt_boxes[:, :, :4], reduction="none")
    box_loss = (box_loss * valid_f.unsqueeze(-1)).sum() / n_valid
    conf_loss = F.binary_cross_entropy_with_logits(pred_boxes[:, :, 4], gt_boxes[:, :, 4], reduction="mean")
    return box_loss + conf_loss


def velocity_loss(pred_vel, gt_vel, num_objects):
    B, M, _ = pred_vel.shape
    device = pred_vel.device
    idx = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
    valid = idx < num_objects.unsqueeze(1)
    valid_f = valid.float()
    n_valid = valid_f.sum().clamp(min=1.0)
    loss = F.smooth_l1_loss(pred_vel, gt_vel, reduction="none")
    return (loss * valid_f.unsqueeze(-1)).sum() / n_valid


def trajectory_loss(pred_traj, pred_logits, gt_traj, num_objects):
    B, M, K, T, _ = pred_traj.shape
    device = pred_traj.device
    idx = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
    valid = idx < num_objects.unsqueeze(1)
    valid_f = valid.float()
    n_valid = valid_f.sum().clamp(min=1.0)
    gt_expanded = gt_traj.unsqueeze(2).expand_as(pred_traj)
    per_mode_ade = (pred_traj - gt_expanded).norm(dim=-1).mean(dim=-1)
    best_mode = per_mode_ade.argmin(dim=-1)
    best_idx = best_mode.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, T, 2)
    best_traj = pred_traj.gather(2, best_idx).squeeze(2)
    reg_loss = F.smooth_l1_loss(best_traj, gt_traj, reduction="none").mean(dim=(-1, -2))
    reg_loss = (reg_loss * valid_f).sum() / n_valid
    cls_loss = F.cross_entropy(pred_logits.view(B * M, K), best_mode.view(B * M), reduction="none").view(B, M)
    cls_loss = (cls_loss * valid_f).sum() / n_valid
    return reg_loss + 0.5 * cls_loss


_DEFAULT_WEIGHTS = {"jepa": 1.0, "detection": 1.0, "velocity": 0.5, "trajectory": 1.0}


def combined_loss(outputs, targets, weights=None):
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}
    device = outputs["boxes"].device
    zero = torch.tensor(0.0, device=device)
    det = detection_loss(outputs["boxes"], targets["boxes"], targets["num_objects"])
    vel = velocity_loss(outputs["velocity"], targets["velocity"], targets["num_objects"])
    jepa_l = zero
    if outputs.get("z_target") is not None:
        jepa_l = jepa_latent_loss(outputs["z_predicted"], outputs["z_target"])
    traj_l = zero
    if "trajectories" in outputs and "gt_trajectories" in targets:
        traj_l = trajectory_loss(outputs["trajectories"], outputs["traj_logits"], targets["gt_trajectories"], targets["num_objects"])
    total = w["detection"] * det + w["velocity"] * vel + w["jepa"] * jepa_l + w["trajectory"] * traj_l
    return {"total": total, "jepa": jepa_l, "detection": det, "velocity": vel, "trajectory": traj_l}
