"""
Data Transforms
================
Image and radar augmentation transforms for training and validation.

Design philosophy:
  - All image transforms are torchvision-compatible.
  - Radar transforms are numpy-based; applied before conversion to tensors.
  - Augmentations are disabled at inference time via the `train` flag.
"""

from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_image_transform(
    img_size: Tuple[int, int] = (224, 224),
    train: bool = False,
    color_jitter_strength: float = 0.4,
) -> T.Compose:
    """
    Build a torchvision transform pipeline.

    Training augmentations:
      - RandomResizedCrop (preserves some scale variation)
      - ColorJitter (brightness / contrast / saturation / hue)
      - RandomHorizontalFlip  (NOTE: disabled for radar alignment; set to 0.0)
      - Normalise to ImageNet statistics

    Validation / inference:
      - Resize + CenterCrop
      - Normalise

    Args:
        img_size: target (H, W)
        train:    if True, apply augmentations

    Returns:
        transform callable: PIL.Image → Tensor[3, H, W]
    """
    H, W = img_size

    if train:
        return T.Compose([
            T.Resize((int(H * 1.15), int(W * 1.15))),
            T.RandomCrop((H, W)),
            T.ColorJitter(
                brightness=0.4 * color_jitter_strength,
                contrast=0.4 * color_jitter_strength,
                saturation=0.4 * color_jitter_strength,
                hue=0.1 * color_jitter_strength,
            ),
            # NOTE: Horizontal flip disabled because it would misalign
            # the camera and radar coordinate systems.
            # T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((H, W)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def denormalise_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalisation for visualisation.

    Args:
        tensor: Tensor[..., 3, H, W]  normalised
    Returns:
        Tensor[..., 3, H, W]  values in [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)


class RadarAugmentation:
    """
    Augmentation pipeline for radar point clouds.

    Applied in the Dataset's __getitem__ (numpy domain, before Tensor conversion).
    """

    def __init__(
        self,
        train: bool = False,
        pos_noise_std: float = 0.1,
        vel_noise_std: float = 0.5,
        dropout_prob: float = 0.1,
        global_rotation_range: float = 0.0,   # set > 0 for BEV rotation aug
        seed: int = None,
    ):
        self.train = train
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.dropout_prob = dropout_prob
        self.global_rotation_range = global_rotation_range
        self._rng = np.random.default_rng(seed)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Args:
            points: np.ndarray [N, 6]  (x, y, z, rcs, vx_comp, vy_comp)
        Returns:
            np.ndarray [N', 6]  augmented
        """
        if not self.train or points.shape[0] == 0:
            return points

        pts = points.copy()

        # 1. Gaussian noise on positions
        pts[:, :3] += self._rng.normal(
            0, self.pos_noise_std, (pts.shape[0], 3)
        ).astype(np.float32)

        # 2. Gaussian noise on velocity
        pts[:, 4:6] += self._rng.normal(
            0, self.vel_noise_std, (pts.shape[0], 2)
        ).astype(np.float32)

        # 3. Random point dropout
        keep = self._rng.random(pts.shape[0]) > self.dropout_prob
        pts = pts[keep]

        # 4. Optional global rotation (around z-axis, BEV plane)
        if self.global_rotation_range > 0 and pts.shape[0] > 0:
            angle = self._rng.uniform(
                -self.global_rotation_range, self.global_rotation_range
            )
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            pts[:, :2] = pts[:, :2] @ R.T      # rotate xy positions
            pts[:, 4:6] = pts[:, 4:6] @ R.T    # rotate velocity components

        return pts
