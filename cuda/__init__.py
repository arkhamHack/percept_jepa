"""
CUDA-accelerated radar utilities.

Provides three operations implemented as custom CUDA kernels with
pure-PyTorch fallbacks:
  1. radar_project   — 3D→2D projection via intrinsics/extrinsics
  2. bev_voxelize    — scatter radar features into a BEV grid (atomicAdd)
  3. radar_rasterize — paint projected radar features onto a 2D image canvas
"""

from pathlib import Path
from typing import Tuple

import torch

_ext = None
_CSRC = Path(__file__).resolve().parent / "csrc"


def _try_load_extension():
    global _ext
    if _ext is not None:
        return
    try:
        from torch.utils.cpp_extension import load

        _ext = load(
            name="radar_jepa_cuda",
            sources=[
                str(_CSRC / "bindings.cpp"),
                str(_CSRC / "radar_projection.cu"),
                str(_CSRC / "bev_voxelize.cu"),
                str(_CSRC / "radar_rasterize.cu"),
            ],
            verbose=False,
        )
    except Exception:
        _ext = None


# ---------------------------------------------------------------------------
# Pure-PyTorch fallbacks
# ---------------------------------------------------------------------------

def _radar_project_torch(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xyz = points[:, :3]  # (N, 3)
    R = extrinsics[:3, :3]  # (3, 3)
    t = extrinsics[:3, 3]   # (3,)

    cam = (R @ xyz.T).T + t  # (N, 3)
    valid = cam[:, 2] > 0

    img = (intrinsics @ cam.T).T  # (N, 3)
    z = img[:, 2:3].clamp(min=1e-8)
    coords_2d = img[:, :2] / z  # (N, 2)

    coords_2d[~valid] = 0.0
    return coords_2d, valid


def _bev_voxelize_torch(
    points: torch.Tensor,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    grid_h: int, grid_w: int,
) -> torch.Tensor:
    N, C = points.shape
    bev = torch.zeros(grid_h, grid_w, C, dtype=points.dtype, device=points.device)

    px = points[:, 0]
    py = points[:, 1]

    in_bounds = (px >= x_min) & (px < x_max) & (py >= y_min) & (py < y_max)
    pts = points[in_bounds]

    if pts.numel() == 0:
        return bev

    cell_w = (x_max - x_min) / grid_w
    cell_h = (y_max - y_min) / grid_h

    col = ((pts[:, 0] - x_min) / cell_w).long().clamp(0, grid_w - 1)
    row = ((pts[:, 1] - y_min) / cell_h).long().clamp(0, grid_h - 1)

    linear = row * grid_w + col
    bev_flat = bev.view(-1, C)
    bev_flat.index_add_(0, linear, pts)

    return bev


def _radar_rasterize_torch(
    coords_2d: torch.Tensor,
    features: torch.Tensor,
    valid: torch.Tensor,
    height: int,
    width: int,
    radius: int,
) -> torch.Tensor:
    N, C = features.shape
    canvas = torch.zeros(height, width, C, dtype=features.dtype, device=features.device)

    for i in range(N):
        if not valid[i]:
            continue
        cu = int(coords_2d[i, 0].round().item())
        cv = int(coords_2d[i, 1].round().item())

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                px, py = cu + dx, cv + dy
                if 0 <= px < width and 0 <= py < height and dx * dx + dy * dy <= radius * radius:
                    canvas[py, px] += features[i]

    return canvas


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def radar_project(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project radar 3D points to 2D image coordinates.

    Args:
        points: (N, 3+) radar points with at least x, y, z.
        intrinsics: (3, 3) camera intrinsic matrix K.
        extrinsics: (4, 4) camera extrinsic matrix [R|t].

    Returns:
        coords_2d: (N, 2) projected pixel coordinates (u, v).
        valid_mask: (N,) boolean mask for points in front of camera.
    """
    _try_load_extension()

    if _ext is not None and points.is_cuda:
        pts = points[:, :3].contiguous().float()
        K = intrinsics.contiguous().float()
        T = extrinsics.contiguous().float()
        return _ext.radar_project(pts, K, T)

    return _radar_project_torch(points, intrinsics, extrinsics)


def bev_voxelize(
    points: torch.Tensor,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Voxelize radar points into a Bird's Eye View grid.

    Args:
        points: (N, C) radar points; columns 0, 1 are x, y.
        x_bounds: (x_min, x_max) extent in metres.
        y_bounds: (y_min, y_max) extent in metres.
        grid_h: Number of rows in the BEV grid.
        grid_w: Number of columns in the BEV grid.

    Returns:
        bev: (H, W, C) accumulated feature grid.
    """
    _try_load_extension()

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    if _ext is not None and points.is_cuda:
        pts = points.contiguous().float()
        return _ext.bev_voxelize(pts, x_min, x_max, y_min, y_max, grid_h, grid_w)

    return _bev_voxelize_torch(points, x_min, x_max, y_min, y_max, grid_h, grid_w)


def radar_rasterize(
    coords_2d: torch.Tensor,
    features: torch.Tensor,
    valid: torch.Tensor,
    height: int,
    width: int,
    radius: int = 3,
) -> torch.Tensor:
    """Rasterize projected radar points onto a 2D feature canvas.

    Splats each valid projected point's feature vector onto a circular
    region of the canvas using atomicAdd (CUDA) or loop (CPU fallback).

    Args:
        coords_2d: (N, 2) pixel coordinates from :func:`radar_project`.
        features: (N, C) per-point feature vectors.
        valid: (N,) boolean mask for valid projections.
        height: Canvas height in pixels.
        width: Canvas width in pixels.
        radius: Splat radius in pixels.

    Returns:
        canvas: (H, W, C) rasterized feature map.
    """
    _try_load_extension()

    if _ext is not None and features.is_cuda:
        c = coords_2d.contiguous().float()
        f = features.contiguous().float()
        v = valid.contiguous()
        return _ext.radar_rasterize(c, f, v, height, width, radius)

    return _radar_rasterize_torch(coords_2d, features, valid, height, width, radius)
