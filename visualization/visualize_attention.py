"""
Attention Map Visualisation
============================
Visualise cross-attention weights from the detection head decoder.

Renders a heatmap per object query showing which scene tokens
(visual patches / radar tokens) each object query attends to.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2


def visualize_detection_attention(
    image:        np.ndarray,          # [H, W, 3] BGR camera image
    attn_weights: torch.Tensor,        # [Q, N_total]  last-layer cross-attn weights
    n_visual:     int,                 # number of visual tokens T*P
    n_patches_per_frame: int,          # P = (img_h/patch)*(img_w/patch)
    patch_grid:   Tuple[int, int],     # (h_patches, w_patches)
    query_idx:    int = 0,             # which object query to visualise
    img_size:     Tuple[int, int] = (224, 224),
    save_path:    Optional[str] = None,
    alpha:        float = 0.5,
) -> np.ndarray:
    """
    Overlay the cross-attention heatmap of a specific object query on the image.

    For temporal sequences, we average attention across time steps to get
    a spatial-only heatmap per frame.

    Args:
        image:         camera frame (BGR)
        attn_weights:  [Q, N_total]
        n_visual:      T * P
        n_patches_per_frame:  P (patches per single frame)
        patch_grid:    (h_p, w_p)
        query_idx:     which query to visualise

    Returns:
        np.ndarray [H, W, 3] BGR overlay
    """
    H_img, W_img = image.shape[:2]
    h_p, w_p = patch_grid

    # Extract visual attention for this query
    vis_attn = attn_weights[query_idx, :n_visual].detach().cpu().float()
    # vis_attn: [T*P]

    T = n_visual // n_patches_per_frame
    # Reshape to [T, P] and average over time
    vis_attn_tp = vis_attn.view(T, n_patches_per_frame)  # [T, P]
    vis_attn_avg = vis_attn_tp.mean(dim=0)               # [P]

    # Reshape to patch grid
    heatmap_small = vis_attn_avg.view(h_p, w_p).numpy()  # [h_p, w_p]

    # Upsample to image size
    heatmap = cv2.resize(heatmap_small, (W_img, H_img), interpolation=cv2.INTER_CUBIC)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_colour = cm.jet(heatmap)[:, :, :3]  # [H, W, 3] RGB float
    heatmap_bgr = (heatmap_colour[:, :, ::-1] * 255).astype(np.uint8)  # BGR

    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_bgr, alpha, 0)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        cv2.imwrite(save_path, overlay)

    return overlay


def visualize_all_query_attentions(
    image:        np.ndarray,
    attn_weights: torch.Tensor,        # [Q, N_total]
    n_visual:     int,
    n_patches_per_frame: int,
    patch_grid:   Tuple[int, int],
    confidence:   torch.Tensor,        # [Q]  objectness scores
    top_k:        int = 6,
    img_size:     Tuple[int, int] = (224, 224),
    save_path:    Optional[str] = None,
) -> np.ndarray:
    """
    Grid plot of attention heatmaps for the top-K highest-confidence queries.

    Returns:
        np.ndarray [H_grid, W_grid, 3] BGR grid image
    """
    # Select top-K queries by confidence
    topk = confidence.topk(min(top_k, confidence.shape[0])).indices   # [k]

    ncols = 3
    nrows = (len(topk) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.array(axes).ravel()

    for i, q_idx in enumerate(topk.tolist()):
        overlay = visualize_detection_attention(
            image, attn_weights, n_visual,
            n_patches_per_frame, patch_grid,
            query_idx=q_idx,
        )
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        axes[i].imshow(overlay_rgb)
        axes[i].set_title(f'Query {q_idx}  conf={confidence[q_idx]:.2f}', fontsize=8)
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Object Query Attention Heatmaps', fontsize=12)
    plt.tight_layout()

    fig.canvas.draw()
    img_grid = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_grid = img_grid.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img_grid_bgr = cv2.cvtColor(img_grid, cv2.COLOR_RGB2BGR)
    plt.close(fig)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        cv2.imwrite(save_path, img_grid_bgr)

    return img_grid_bgr
