"""
Tracking Visualisation
=======================
Draw tracked objects with persistent IDs and motion trails on BEV and camera images.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Deterministic colour per track ID (hash-based)
_TRACK_PALETTE = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())


def _track_colour(track_id: int) -> Tuple[int, int, int]:
    """Return a consistent BGR colour for a given track ID."""
    hex_c = _TRACK_PALETTE[track_id % len(_TRACK_PALETTE)].lstrip('#')
    r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
    return (b, g, r)  # BGR


def draw_tracked_boxes_bev(
    boxes:     np.ndarray,        # [M, 7]
    track_ids: np.ndarray,        # [M]
    scores:    Optional[np.ndarray] = None,  # [M]
    labels:    Optional[np.ndarray] = None,  # [M]
    label_names: Optional[List[str]] = None,
    trail:     Optional[Dict[int, List[Tuple[float, float]]]] = None,
    x_range:   Tuple[float,float] = (-50, 50),
    y_range:   Tuple[float,float] = (-50, 50),
    bev_size:  int = 800,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Draw tracked 3-D boxes on a BEV canvas.

    Args:
        boxes:      [M, 7] decoded boxes
        track_ids:  [M] integer IDs
        trail:      dict mapping track_id → list of past (x,y) positions

    Returns:
        np.ndarray [bev_size, bev_size, 3]  BGR image
    """
    canvas = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    canvas[:] = 30   # dark background

    def world2pix(x, y):
        px = int((x - x_range[0]) / (x_range[1] - x_range[0]) * bev_size)
        py = int((y - y_range[0]) / (y_range[1] - y_range[0]) * bev_size)
        py = bev_size - 1 - py   # flip y (image y=0 at top)
        return px, py

    # Draw grid lines
    for v in range(-40, 50, 10):
        x1, y1 = world2pix(x_range[0], v)
        x2, y2 = world2pix(x_range[1], v)
        cv2.line(canvas, (0, y1), (bev_size, y1), (50, 50, 50), 1)
    for u in range(-40, 50, 10):
        x1, y1 = world2pix(u, y_range[0])
        cv2.line(canvas, (x1, 0), (x1, bev_size), (50, 50, 50), 1)

    # Ego vehicle marker
    ex, ey = world2pix(0, 0)
    cv2.rectangle(canvas, (ex-6, ey-12), (ex+6, ey+12), (0, 255, 255), -1)
    cv2.arrowedLine(canvas, (ex, ey), (ex, ey-20), (0, 255, 255), 2)

    # Draw motion trails
    if trail:
        for tid, positions in trail.items():
            if len(positions) < 2:
                continue
            colour = _track_colour(int(tid))
            for i in range(1, len(positions)):
                p1 = world2pix(*positions[i-1])
                p2 = world2pix(*positions[i])
                alpha = 0.3 + 0.7 * (i / len(positions))
                c = tuple(int(v * alpha) for v in colour)
                cv2.line(canvas, p1, p2, c, 2)

    # Draw boxes
    for i, (box, tid) in enumerate(zip(boxes, track_ids)):
        cx, cy, _, dx, dy, _, yaw = box[:7]
        colour = _track_colour(int(tid))

        # Compute BEV corners
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        local = np.array([[ dx/2,  dy/2], [-dx/2,  dy/2],
                           [-dx/2, -dy/2], [ dx/2, -dy/2]])
        corners = local @ R.T + np.array([[cx, cy]])  # [4, 2]

        pts = np.array([world2pix(p[0], p[1]) for p in corners], dtype=np.int32)
        cv2.polylines(canvas, [pts], True, colour, 2)

        # Heading arrow
        head_local = np.array([[dx*0.6, 0]])
        head_world = head_local @ R.T + np.array([[cx, cy]])
        cv2.arrowedLine(canvas, world2pix(cx, cy), world2pix(*head_world[0]),
                        colour, 2, tipLength=0.4)

        # Track ID label
        px, py = world2pix(cx, cy)
        score_txt = f'{scores[i]:.2f}' if scores is not None else ''
        label_txt = label_names[labels[i]] if (labels is not None and label_names) else ''
        cv2.putText(canvas, f'T{int(tid)} {label_txt} {score_txt}',
                    (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1,
                    cv2.LINE_AA)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        cv2.imwrite(save_path, canvas)

    return canvas


def draw_tracked_boxes_image(
    image:      np.ndarray,       # [H, W, 3] BGR
    boxes_2d:   np.ndarray,       # [M, 4]  (x1, y1, x2, y2) pixel coords
    track_ids:  np.ndarray,       # [M]
    scores:     Optional[np.ndarray] = None,
    labels:     Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    save_path:  Optional[str] = None,
) -> np.ndarray:
    """
    Draw 2-D tracking boxes on a camera image.
    """
    img = image.copy()
    for i, (box2d, tid) in enumerate(zip(boxes_2d, track_ids)):
        x1, y1, x2, y2 = box2d.astype(int)
        colour = _track_colour(int(tid))
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        txt = f'T{int(tid)}'
        if labels is not None and label_names:
            txt += f' {label_names[labels[i]]}'
        if scores is not None:
            txt += f' {scores[i]:.2f}'
        cv2.putText(img, txt, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, colour, 1, cv2.LINE_AA)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        cv2.imwrite(save_path, img)

    return img
