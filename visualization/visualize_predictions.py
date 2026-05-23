"""
Trajectory Prediction Visualisation
=====================================
Draws predicted future trajectories on BEV and camera views.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def draw_predictions_bev(
    boxes:       np.ndarray,       # [M, 7]  current-frame boxes
    track_ids:   np.ndarray,       # [M]
    pred_traj:   np.ndarray,       # [M, K, 2]  predicted future (x,y)
    gt_traj:     Optional[np.ndarray] = None,  # [M, K, 2]  GT (for comparison)
    scores:      Optional[np.ndarray] = None,
    x_range:     Tuple[float,float] = (-50, 50),
    y_range:     Tuple[float,float] = (-50, 50),
    bev_size:    int = 800,
    score_thresh: float = 0.3,
    save_path:   Optional[str] = None,
) -> np.ndarray:
    """
    Visualise predicted trajectories on BEV.

    Draws:
      - Current 3-D box footprints
      - Predicted trajectory (solid coloured line with dots)
      - GT trajectory if provided (dashed white line)

    Returns:
        np.ndarray [bev_size, bev_size, 3] BGR
    """
    canvas = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    canvas[:] = 25  # dark background

    def w2p(x, y):
        px = int((x - x_range[0]) / (x_range[1] - x_range[0]) * bev_size)
        py = int((y - y_range[0]) / (y_range[1] - y_range[0]) * bev_size)
        return px, bev_size - 1 - py

    # Grid
    for v in range(-40, 50, 10):
        _, py = w2p(0, v)
        cv2.line(canvas, (0, py), (bev_size, py), (45, 45, 45), 1)
    for u in range(-40, 50, 10):
        px, _ = w2p(u, 0)
        cv2.line(canvas, (px, 0), (px, bev_size), (45, 45, 45), 1)

    # Ego marker
    ex, ey = w2p(0, 0)
    cv2.rectangle(canvas, (ex-6, ey-12), (ex+6, ey+12), (0, 255, 255), -1)

    # Per-object colour cycle
    cmap = cm.get_cmap('tab10')

    for i, (box, tid, traj) in enumerate(zip(boxes, track_ids, pred_traj)):
        if scores is not None and scores[i] < score_thresh:
            continue

        # Box colour
        colour_f = cmap(i % 10)[:3]
        colour = (int(colour_f[2]*255), int(colour_f[1]*255), int(colour_f[0]*255))

        # Draw box
        cx, cy, _, dx, dy, _, yaw = box[:7]
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c,-s],[s,c]])
        local = np.array([[ dx/2, dy/2],[-dx/2, dy/2],[-dx/2,-dy/2],[ dx/2,-dy/2]])
        corners = local @ R.T + np.array([[cx, cy]])
        pts = np.array([w2p(p[0], p[1]) for p in corners], dtype=np.int32)
        cv2.polylines(canvas, [pts], True, colour, 2)

        # Draw predicted trajectory
        K = traj.shape[0]
        waypoints = [(cx, cy)] + [(traj[k, 0], traj[k, 1]) for k in range(K)]
        for k in range(1, len(waypoints)):
            alpha = k / len(waypoints)
            c_fade = tuple(int(v * (0.3 + 0.7 * alpha)) for v in colour)
            p1 = w2p(*waypoints[k-1])
            p2 = w2p(*waypoints[k])
            cv2.line(canvas, p1, p2, c_fade, 2)
            if k < len(waypoints) - 1:
                cv2.circle(canvas, p2, 3, c_fade, -1)
        # Terminal dot
        cv2.circle(canvas, w2p(*waypoints[-1]), 5, colour, -1)

        # Draw GT trajectory (white dashed)
        if gt_traj is not None:
            gt = gt_traj[i]
            gt_pts = [(cx, cy)] + [(gt[k, 0], gt[k, 1]) for k in range(K)]
            for k in range(1, len(gt_pts)):
                p1 = w2p(*gt_pts[k-1])
                p2 = w2p(*gt_pts[k])
                # Dashed: draw segments at intervals
                if k % 2 == 0:
                    cv2.line(canvas, p1, p2, (220, 220, 220), 1)

        # Label
        px, py = w2p(cx, cy)
        cv2.putText(canvas, f'T{int(tid)}', (px+5, py-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)

    # Legend
    cv2.putText(canvas, 'Predicted  Trajectory', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)
    cv2.putText(canvas, 'GT Trajectory (white)', (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        cv2.imwrite(save_path, canvas)

    return canvas


def make_prediction_summary_figure(
    image:        np.ndarray,        # [H, W, 3] BGR
    bev_canvas:   np.ndarray,        # [S, S, 3] BGR
    attn_overlay: Optional[np.ndarray] = None,  # [H, W, 3] BGR
    title:        str = '',
    save_path:    Optional[str] = None,
) -> np.ndarray:
    """
    Combine camera image, BEV, and optional attention map into a single figure.

    Returns np.ndarray [H_out, W_out, 3] BGR
    """
    H_img, W_img = image.shape[:2]
    S = bev_canvas.shape[0]

    # Resize BEV to match image height
    bev_r = cv2.resize(bev_canvas, (int(S * H_img / S), H_img))

    panels = [image, bev_r]
    if attn_overlay is not None:
        panels.append(cv2.resize(attn_overlay, (W_img, H_img)))

    composite = np.concatenate(panels, axis=1)

    if title:
        cv2.putText(composite, title, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        cv2.imwrite(save_path, composite)

    return composite
