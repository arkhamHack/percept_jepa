"""
Radar Visualisation
====================
BEV radar point cloud plots overlaid on a top-down grid.
Also supports camera-radar overlay (projection).
"""

import os
from typing import Optional, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from utils.geometry import box7_to_corners


def plot_radar_bev(
    radar_pts:  np.ndarray,              # [N, 6] (x,y,z,rcs,vx_comp,vy_comp)
    radar_mask: Optional[np.ndarray] = None,  # [N] bool
    boxes:      Optional[np.ndarray] = None,  # [M, 7]
    labels:     Optional[np.ndarray] = None,  # [M] int
    pred_boxes: Optional[np.ndarray] = None,  # [P, 7]
    x_range:    Tuple[float,float] = (-50, 50),
    y_range:    Tuple[float,float] = (-50, 50),
    title:      str = "Radar BEV",
    save_path:  Optional[str] = None,
) -> np.ndarray:
    """
    Plot radar points in BEV with optional GT and predicted boxes.

    Returns:
        np.ndarray [H, W, 3]  BGR image
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    ax.set_title(title, color='white', fontsize=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Draw ego vehicle marker
    ax.plot(0, 0, 's', color='cyan', markersize=10, zorder=5, label='Ego')
    ax.annotate('↑', (0, 0), textcoords='offset points', xytext=(0, 8),
                ha='center', fontsize=14, color='cyan')

    # Plot radar points
    if radar_mask is not None:
        valid = radar_mask.astype(bool)
        pts = radar_pts[valid]
    else:
        pts = radar_pts

    if len(pts) > 0:
        rcs = pts[:, 3]
        vr  = np.sqrt(pts[:, 4]**2 + pts[:, 5]**2)

        sc = ax.scatter(
            pts[:, 0], pts[:, 1],
            c=rcs, cmap='plasma',
            s=np.clip(vr * 3 + 5, 5, 50),
            alpha=0.85,
            zorder=4,
        )
        plt.colorbar(sc, ax=ax, label='RCS (dBsm)', fraction=0.03)

        # Draw velocity arrows
        arrow_scale = 0.3
        for p in pts[::max(1, len(pts)//50)]:   # subsample for clarity
            ax.annotate('', xy=(p[0] + p[4]*arrow_scale, p[1] + p[5]*arrow_scale),
                       xytext=(p[0], p[1]),
                       arrowprops=dict(arrowstyle='->', color='#00ff88', lw=0.8))

    # Draw GT boxes
    if boxes is not None:
        _draw_boxes_bev(ax, boxes, color='lime', label='GT', linestyle='-')

    # Draw predicted boxes
    if pred_boxes is not None:
        _draw_boxes_bev(ax, pred_boxes, color='red', label='Pred', linestyle='--')

    ax.legend(facecolor='#2a2a3e', labelcolor='white', loc='upper left')
    ax.grid(True, color='#333355', linestyle=':', alpha=0.5)

    fig.patch.set_facecolor('#0d0d1a')
    plt.tight_layout()

    # Render to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.close(fig)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        cv2.imwrite(save_path, img_bgr)

    return img_bgr


def _draw_boxes_bev(ax, boxes, color, label, linestyle='-'):
    """Draw BEV box footprints on a matplotlib axis."""
    added_label = False
    for box in boxes:
        cx, cy, _, dx, dy, _, yaw = box[:7]
        corners = _box_corners_2d(cx, cy, dx, dy, yaw)  # [4, 2]
        poly = plt.Polygon(
            corners, fill=False,
            edgecolor=color, linestyle=linestyle, linewidth=1.5,
            label=label if not added_label else None,
        )
        ax.add_patch(poly)
        added_label = True


def _box_corners_2d(cx, cy, dx, dy, yaw):
    """Compute 4 corners of a 2-D rotated box."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    local = np.array([
        [ dx/2,  dy/2],
        [-dx/2,  dy/2],
        [-dx/2, -dy/2],
        [ dx/2, -dy/2],
    ])
    return local @ R.T + np.array([[cx, cy]])


def overlay_radar_on_image(
    image:      np.ndarray,        # [H, W, 3]  BGR
    radar_pts:  np.ndarray,        # [N, 6]
    intrinsics: np.ndarray,        # [3, 3]
    extrinsics: np.ndarray,        # [4, 4]  ego → camera
    radar_mask: Optional[np.ndarray] = None,
    point_size: int = 6,
    cmap_name:  str = 'plasma',
) -> np.ndarray:
    """
    Project radar points onto the camera image.

    Args:
        image:      [H, W, 3] BGR
        radar_pts:  [N, 6] in ego frame
        intrinsics: camera K matrix
        extrinsics: camera extrinsics (ego→cam)

    Returns:
        np.ndarray [H, W, 3] with radar overlaid
    """
    H, W = image.shape[:2]
    img = image.copy()

    if radar_mask is not None:
        pts = radar_pts[radar_mask.astype(bool)]
    else:
        pts = radar_pts

    if len(pts) == 0:
        return img

    # Build homogeneous coordinates
    xyz = pts[:, :3].T    # [3, N]
    ones = np.ones((1, xyz.shape[1]))
    xyz_h = np.vstack([xyz, ones])    # [4, N]

    # Transform to camera frame
    xyz_cam = (extrinsics @ xyz_h)[:3, :]   # [3, N]

    # Only keep points in front of camera
    valid = xyz_cam[2, :] > 0.5
    xyz_cam = xyz_cam[:, valid]
    rcs = pts[valid, 3]
    depth = xyz_cam[2, :]

    # Project to pixel
    uv = (intrinsics @ xyz_cam)[:2, :] / xyz_cam[2:3, :]  # [2, N]
    u, v = uv[0, :], uv[1, :]

    # Clip to image bounds
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, rcs, depth = u[in_bounds], v[in_bounds], rcs[in_bounds], depth[in_bounds]

    # Colour by depth
    d_norm = np.clip((depth - depth.min()) / (depth.max() - depth.min() + 1e-6), 0, 1)
    cmap = plt.get_cmap(cmap_name)

    for i in range(len(u)):
        colour = cmap(d_norm[i])[:3]
        colour_bgr = (int(colour[2]*255), int(colour[1]*255), int(colour[0]*255))
        cv2.circle(img, (int(u[i]), int(v[i])), point_size, colour_bgr, -1)

    return img
