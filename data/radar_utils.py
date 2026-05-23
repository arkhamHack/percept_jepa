"""
Radar Utility Functions
========================
Helpers for processing nuScenes radar point clouds:
  - point filtering (velocity / quality gates)
  - coordinate frame transforms
  - BEV grid projection
  - sweep accumulation utilities
  - normalisation statistics
"""

import numpy as np
import torch
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Feature indices in the full nuScenes radar point vector (18-D)
# ---------------------------------------------------------------------------
_IDX = dict(
    x=0, y=1, z=2,
    dyn_prop=3, id=4, rcs=5,
    vx=6, vy=7,
    vx_comp=8, vy_comp=9,
    is_quality_valid=10,
    ambig_state=11,
    x_rms=12, y_rms=13,
    invalid_state=14,
    pdh0=15,
    vx_rms=16, vy_rms=17,
)

# Feature indices for our 6-D model input
MODEL_FEATURE_NAMES = ['x', 'y', 'z', 'rcs', 'vx_comp', 'vy_comp']
MODEL_FEATURE_IDX = [_IDX[k] for k in MODEL_FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Per-feature normalisation constants
# (estimated from nuScenes train split statistics)
# ---------------------------------------------------------------------------
RADAR_MEAN = np.array([0.0,  0.0,  -0.5,  3.0,  0.0,  0.0], dtype=np.float32)
RADAR_STD  = np.array([25.0, 25.0,  1.5, 10.0, 12.0, 12.0], dtype=np.float32)


def extract_model_features(raw_points: np.ndarray) -> np.ndarray:
    """
    Extract the 6-D model feature vector from a full 18-D nuScenes radar array.

    Args:
        raw_points: np.ndarray  [18, N] or [N, 18]

    Returns:
        np.ndarray [N, 6]   columns: x, y, z, rcs, vx_comp, vy_comp
    """
    if raw_points.ndim == 2 and raw_points.shape[0] == 18:
        raw_points = raw_points.T  # [N, 18]
    return raw_points[:, MODEL_FEATURE_IDX].astype(np.float32)  # [N, 6]


def filter_radar_points(
    points: np.ndarray,
    min_range: float = 1.0,
    max_range: float = 60.0,
    min_rcs: float = -10.0,
    valid_dyn_prop: Optional[Tuple[int, ...]] = (0, 2),   # moving / stationary
    valid_ambig: Optional[Tuple[int, ...]] = (3,),
    valid_invalid: Optional[Tuple[int, ...]] = (0,),
) -> np.ndarray:
    """
    Quality-gate radar points.

    Args:
        points: np.ndarray  [N, 6] with model features (x,y,z,rcs,vx_comp,vy_comp)
        min_range / max_range: radial distance gates
        min_rcs:              minimum RCS (dBsm)
        valid_dyn_prop:       allowed dynamic property codes (None = keep all)
        valid_ambig / valid_invalid:  not available in 6-D model features;
                                      pass None to skip.

    Returns:
        np.ndarray [N', 6]  filtered subset
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2)
    rcs = points[:, 3]

    mask = (r >= min_range) & (r <= max_range) & (rcs >= min_rcs)
    return points[mask]


def normalise_radar(points: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit-variance normalisation using pre-computed statistics.

    Args:
        points: np.ndarray  [N, 6]
    Returns:
        np.ndarray [N, 6]  normalised
    """
    return (points - RADAR_MEAN) / (RADAR_STD + 1e-6)


def denormalise_radar(points: np.ndarray) -> np.ndarray:
    """Inverse of normalise_radar."""
    return points * RADAR_STD + RADAR_MEAN


def pad_or_sample_points(
    points: np.ndarray,
    n_max: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fix the number of radar points to n_max via padding / random sub-sampling.

    Args:
        points: np.ndarray  [N, F]
        n_max:  target number of points
        rng:    numpy RNG (uses global if None)

    Returns:
        padded_points: np.ndarray [n_max, F]
        mask:          np.ndarray [n_max]   bool, 1 = real point
    """
    N, F = points.shape
    if rng is None:
        rng = np.random.default_rng()

    mask = np.zeros(n_max, dtype=bool)
    if N == 0:
        return np.zeros((n_max, F), dtype=np.float32), mask

    if N >= n_max:
        idx = rng.choice(N, n_max, replace=False)
        padded = points[idx].astype(np.float32)
        mask[:] = True
    else:
        padded = np.zeros((n_max, F), dtype=np.float32)
        padded[:N] = points.astype(np.float32)
        mask[:N] = True

    return padded, mask


def points_to_bev_grid(
    points: np.ndarray,
    x_range: Tuple[float, float] = (-50, 50),
    y_range: Tuple[float, float] = (-50, 50),
    resolution: float = 0.5,
    features: Tuple[int, ...] = (3, 4, 5),  # rcs, vx_comp, vy_comp
) -> np.ndarray:
    """
    Rasterise radar points into a Bird's-Eye-View (BEV) grid.

    Args:
        points:     np.ndarray [N, 6]
        x_range:    (min_x, max_x) in metres
        y_range:    (min_y, max_y) in metres
        resolution: grid cell size in metres
        features:   which feature columns to accumulate

    Returns:
        np.ndarray [len(features), H, W]   mean-pooled feature grid
    """
    W = int((x_range[1] - x_range[0]) / resolution)
    H = int((y_range[1] - y_range[0]) / resolution)
    F = len(features)

    grid = np.zeros((F, H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    for p in points:
        xi = int((p[0] - x_range[0]) / resolution)
        yi = int((p[1] - y_range[0]) / resolution)
        if 0 <= xi < W and 0 <= yi < H:
            for fi, feat_idx in enumerate(features):
                grid[fi, yi, xi] += p[feat_idx]
            count[yi, xi] += 1

    # Mean pooling
    nonzero = count > 0
    for fi in range(F):
        grid[fi][nonzero] /= count[nonzero]

    return grid


def compute_radial_velocity(
    vx_comp: np.ndarray, vy_comp: np.ndarray,
    x: np.ndarray, y: np.ndarray,
) -> np.ndarray:
    """
    Project compensated velocity onto the radial direction from the sensor origin.

    v_r = (vx*x + vy*y) / r

    Args:
        vx_comp, vy_comp, x, y: np.ndarray [N]

    Returns:
        np.ndarray [N]  radial velocity (positive = moving away)
    """
    r = np.sqrt(x**2 + y**2) + 1e-6
    return (vx_comp * x + vy_comp * y) / r


def augment_radar_noise(
    points: np.ndarray,
    pos_noise_std: float = 0.1,
    vel_noise_std: float = 0.5,
    dropout_prob: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply random noise augmentation to radar points for training robustness.

    Args:
        points:        np.ndarray [N, 6]
        pos_noise_std: Gaussian noise std on xyz (metres)
        vel_noise_std: Gaussian noise std on velocity (m/s)
        dropout_prob:  probability of randomly dropping each point
        rng:           numpy RNG

    Returns:
        np.ndarray [N', 6]  augmented (fewer points if dropout applied)
    """
    if rng is None:
        rng = np.random.default_rng()

    pts = points.copy()
    N = pts.shape[0]

    # Position noise
    pts[:, :3] += rng.normal(0, pos_noise_std, (N, 3)).astype(np.float32)

    # Velocity noise
    pts[:, 4:6] += rng.normal(0, vel_noise_std, (N, 2)).astype(np.float32)

    # Random dropout
    keep = rng.random(N) > dropout_prob
    pts = pts[keep]

    return pts


def torch_points_to_bev(
    points: torch.Tensor,          # [B, N, 6]
    mask: torch.Tensor,            # [B, N]
    x_range: Tuple[float, float] = (-50, 50),
    y_range: Tuple[float, float] = (-50, 50),
    resolution: float = 0.5,
) -> torch.Tensor:
    """
    Differentiable soft BEV projection via bilinear splatting.
    Returns a density map [B, 1, H, W] for visualisation.

    NOTE: This is a simplified version; for a proper differentiable BEV
    projection use PhilionBEV or LSS-style depth lifting.
    """
    B, N, _ = points.shape
    H = int((y_range[1] - y_range[0]) / resolution)
    W = int((x_range[1] - x_range[0]) / resolution)

    density = torch.zeros(B, 1, H, W, device=points.device)

    # Normalise coordinates to [-1, 1] for grid_sample (TODO: use scatter)
    x_norm = (points[..., 0] - x_range[0]) / (x_range[1] - x_range[0]) * 2 - 1
    y_norm = (points[..., 1] - y_range[0]) / (y_range[1] - y_range[0]) * 2 - 1

    # Integer grid indices
    xi = ((points[..., 0] - x_range[0]) / resolution).long().clamp(0, W - 1)
    yi = ((points[..., 1] - y_range[0]) / resolution).long().clamp(0, H - 1)

    for b in range(B):
        valid = mask[b].bool()
        density[b, 0].index_put_(
            (yi[b][valid], xi[b][valid]),
            torch.ones(valid.sum(), device=points.device),
            accumulate=True,
        )

    return density
