"""
Geometry Utilities
===================
Coordinate transforms, box conversions, and projection helpers.
"""

import math
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Box conversions
# ---------------------------------------------------------------------------

def box7_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert 7-D boxes to 8 corner points (3-D).

    Args:
        boxes: Tensor[..., 7]  (cx, cy, cz, dx, dy, dz, yaw)

    Returns:
        Tensor[..., 8, 3]  eight corner points in world frame
    """
    cx, cy, cz = boxes[..., 0], boxes[..., 1], boxes[..., 2]
    dx, dy, dz = boxes[..., 3], boxes[..., 4], boxes[..., 5]
    yaw         = boxes[..., 6]

    # Local corner offsets [8, 3] for a unit box
    # Ordering: front-top-right, ..., back-bottom-left
    offsets = torch.tensor([
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
    ], device=boxes.device, dtype=boxes.dtype)  # [8, 3]

    # Scale by box dimensions
    scale = torch.stack([dx, dy, dz], dim=-1)  # [..., 3]
    corners = offsets * scale.unsqueeze(-2)     # [..., 8, 3]

    # Rotate around z-axis by yaw
    cos_y = yaw.cos().unsqueeze(-1)   # [..., 1]
    sin_y = yaw.sin().unsqueeze(-1)

    x_rot = corners[..., 0] * cos_y - corners[..., 1] * sin_y
    y_rot = corners[..., 0] * sin_y + corners[..., 1] * cos_y
    z_rot = corners[..., 2:3].expand_as(corners[..., 2:3])

    corners_rot = torch.stack([x_rot, y_rot, corners[..., 2]], dim=-1)  # [..., 8, 3]

    # Translate to box centre
    center = torch.stack([cx, cy, cz], dim=-1).unsqueeze(-2)  # [..., 1, 3]
    return corners_rot + center  # [..., 8, 3]


def boxes_to_bev(boxes: torch.Tensor) -> torch.Tensor:
    """
    Project 3-D boxes to 2-D BEV (top-down) rectangles.

    Args:
        boxes: Tensor[..., 7]

    Returns:
        Tensor[..., 4, 2]  four BEV corner points (x, y)
    """
    corners = box7_to_corners(boxes)   # [..., 8, 3]
    return corners[..., :4, :2]        # take top 4 corners, xy only


def ego_to_pixel(
    xyz_ego: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3-D points in ego frame to pixel coordinates.

    Args:
        xyz_ego:    Tensor[N, 3]    points in ego frame
        intrinsics: Tensor[3, 3]    camera K matrix
        extrinsics: Tensor[4, 4]    ego → camera transform

    Returns:
        uv:     Tensor[N, 2]  pixel coordinates (u=col, v=row)
        valid:  Tensor[N]     bool, True if in front of camera
    """
    N = xyz_ego.shape[0]
    ones = torch.ones(N, 1, device=xyz_ego.device, dtype=xyz_ego.dtype)
    xyz_h = torch.cat([xyz_ego, ones], dim=-1)   # [N, 4]

    # Transform to camera frame
    xyz_cam = (extrinsics @ xyz_h.T).T           # [N, 4]
    xyz_cam = xyz_cam[:, :3]                      # [N, 3]

    # Points behind camera
    valid = xyz_cam[:, 2] > 0.1                  # [N]

    # Project
    xyz_norm = xyz_cam / xyz_cam[:, 2:3].clamp(min=1e-6)   # [N, 3]
    uv_h = (intrinsics @ xyz_norm.T).T                       # [N, 3]
    uv = uv_h[:, :2]                                         # [N, 2]
    return uv, valid


# ---------------------------------------------------------------------------
# Pose / transform helpers
# ---------------------------------------------------------------------------

def make_rotation_z(yaw: float) -> np.ndarray:
    """Build a 4×4 rotation matrix around the z-axis."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([
        [ c, -s, 0, 0],
        [ s,  c, 0, 0],
        [ 0,  0, 1, 0],
        [ 0,  0, 0, 1],
    ], dtype=np.float64)


def interpolate_poses(
    pose0: np.ndarray,  # [4, 4] transform at t=0
    pose1: np.ndarray,  # [4, 4] transform at t=1
    t: float,           # interpolation factor [0, 1]
) -> np.ndarray:
    """
    Linearly interpolate between two SE(3) poses.

    Approximation: lerp on translation, slerp on rotation would be ideal
    but linear rotation interp is fine for small angles.
    """
    return pose0 * (1 - t) + pose1 * t


# ---------------------------------------------------------------------------
# BEV utilities
# ---------------------------------------------------------------------------

def world_to_bev_pixel(
    xy: torch.Tensor,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    bev_h: int,
    bev_w: int,
) -> torch.Tensor:
    """
    Convert world (x, y) coordinates to integer BEV pixel indices.

    Args:
        xy:       Tensor[..., 2]  (x, y) in world/ego metres
        x_range:  (min_x, max_x)
        y_range:  (min_y, max_y)
        bev_h, bev_w: BEV image size in pixels

    Returns:
        Tensor[..., 2]  (col, row) pixel indices (long)
    """
    col = (xy[..., 0] - x_range[0]) / (x_range[1] - x_range[0]) * bev_w
    row = (xy[..., 1] - y_range[0]) / (y_range[1] - y_range[0]) * bev_h
    return torch.stack([col, row], dim=-1).long()


def rotate_bev_boxes(
    boxes_bev: torch.Tensor,  # [N, 4, 2]  BEV corners
    angle_deg: float,
) -> torch.Tensor:
    """Rotate BEV corners around the origin by angle_deg degrees."""
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    R = torch.tensor([[c, -s], [s, c]], dtype=boxes_bev.dtype, device=boxes_bev.device)
    return (boxes_bev @ R.T)
