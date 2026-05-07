"""Loss functions for radar-camera fusion training.

Covers detection, velocity, trajectory prediction, and JEPA latent losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def jepa_latent_loss(
    z_predicted: torch.Tensor,
    z_target: torch.Tensor,
) -> torch.Tensor:
    """MSE loss between L2-normalised predicted and target latent vectors."""
    z_predicted = F.normalize(z_predicted, dim=-1)
    z_target = F.normalize(z_target, dim=-1)
    return F.mse_loss(z_predicted, z_target)


def detection_loss(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    num_objects: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 (coordinates) + BCE (confidence) detection loss."""
    B, M, _ = pred_boxes.shape
    device = pred_boxes.device

    idx = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
    valid = idx < num_objects.unsqueeze(1)
    valid_f = valid.float()
    n_valid = valid_f.sum().clamp(min=1.0)

    coord_pred = pred_boxes[:, :, :4]
    coord_gt = gt_boxes[:, :, :4]
    box_loss = F.smooth_l1_loss(coord_pred, coord_gt, reduction="none")
    box_loss = (box_loss * valid_f.unsqueeze(-1)).sum() / n_valid

    conf_pred = pred_boxes[:, :, 4]
    conf_gt = gt_boxes[:, :, 4]
    conf_loss = F.binary_cross_entropy_with_logits(
        conf_pred, conf_gt, reduction="mean",
    )

    return box_loss + conf_loss


def velocity_loss(
    pred_vel: torch.Tensor,
    gt_vel: torch.Tensor,
    num_objects: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 loss on velocity predictions for valid objects."""
    B, M, _ = pred_vel.shape
    device = pred_vel.device

    idx = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
    valid = idx < num_objects.unsqueeze(1)
    valid_f = valid.float()
    n_valid = valid_f.sum().clamp(min=1.0)

    loss = F.smooth_l1_loss(pred_vel, gt_vel, reduction="none")
    return (loss * valid_f.unsqueeze(-1)).sum() / n_valid


def trajectory_loss(
    pred_traj: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_traj: torch.Tensor,
    num_objects: torch.Tensor,
) -> torch.Tensor:
    """Winner-take-all trajectory prediction loss.

    For each agent, finds the best-matching mode (lowest ADE) and trains
    only that mode's trajectory regression.  A cross-entropy term encourages
    the mode probabilities to assign high weight to the best mode.

    Args:
        pred_traj: ``(B, M, K, T, 2)`` — K modes, T timesteps.
        pred_logits: ``(B, M, K)`` — unnormalised mode logits.
        gt_traj: ``(B, M, T, 2)`` — ground-truth future trajectory.
        num_objects: ``(B,)`` — valid objects per sample.

    Returns:
        Scalar loss (regression + classification).
    """
    B, M, K, T, _ = pred_traj.shape
    device = pred_traj.device

    idx = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
    valid = idx < num_objects.unsqueeze(1)  # (B, M)
    valid_f = valid.float()
    n_valid = valid_f.sum().clamp(min=1.0)

    gt_expanded = gt_traj.unsqueeze(2).expand_as(pred_traj)  # (B, M, K, T, 2)
    per_mode_ade = (pred_traj - gt_expanded).norm(dim=-1).mean(dim=-1)  # (B, M, K)

    best_mode = per_mode_ade.argmin(dim=-1)  # (B, M)

    best_idx = best_mode.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    best_idx = best_idx.expand(-1, -1, 1, T, 2)
    best_traj = pred_traj.gather(2, best_idx).squeeze(2)  # (B, M, T, 2)

    reg_loss = F.smooth_l1_loss(best_traj, gt_traj, reduction="none")  # (B, M, T, 2)
    reg_loss = reg_loss.mean(dim=(-1, -2))  # (B, M)
    reg_loss = (reg_loss * valid_f).sum() / n_valid

    cls_loss = F.cross_entropy(
        pred_logits.view(B * M, K),
        best_mode.view(B * M),
        reduction="none",
    ).view(B, M)
    cls_loss = (cls_loss * valid_f).sum() / n_valid

    return reg_loss + 0.5 * cls_loss


# -----------------------------------------------------------------------
# Default loss weights
# -----------------------------------------------------------------------
_DEFAULT_WEIGHTS = {
    "jepa": 1.0,
    "detection": 1.0,
    "velocity": 0.5,
    "trajectory": 1.0,
}


def combined_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate all JEPA loss terms: detection + velocity + JEPA latent + trajectory.

    Returns:
        Dict with ``'total'``, ``'jepa'``, ``'detection'``, ``'velocity'``,
        ``'trajectory'`` scalar tensors.
    """
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
        traj_l = trajectory_loss(
            outputs["trajectories"],
            outputs["traj_logits"],
            targets["gt_trajectories"],
            targets["num_objects"],
        )

    total = (
        w["detection"] * det
        + w["velocity"] * vel
        + w["jepa"] * jepa_l
        + w["trajectory"] * traj_l
    )

    return {
        "total": total,
        "jepa": jepa_l,
        "detection": det,
        "velocity": vel,
        "trajectory": traj_l,
    }
