"""Data augmentations and preprocessing for radar-camera fusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def normalize(
    image: torch.Tensor,
    mean: torch.Tensor = IMAGENET_MEAN,
    std: torch.Tensor = IMAGENET_STD,
) -> torch.Tensor:
    """Normalize a (3, H, W) float image with channel-wise mean/std."""
    mean = mean.to(image.device).view(3, 1, 1)
    std = std.to(image.device).view(3, 1, 1)
    return (image - mean) / std


def denormalize(
    image: torch.Tensor,
    mean: torch.Tensor = IMAGENET_MEAN,
    std: torch.Tensor = IMAGENET_STD,
) -> torch.Tensor:
    """Inverse of :func:`normalize`."""
    mean = mean.to(image.device).view(3, 1, 1)
    std = std.to(image.device).view(3, 1, 1)
    return image * std + mean


def resize(
    image: torch.Tensor,
    size: tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize a (3, H, W) image tensor to ``(H_new, W_new)``."""
    return F.interpolate(
        image.unsqueeze(0), size=size, mode=mode, align_corners=False
    ).squeeze(0)


# ------------------------------------------------------------------
# Stress-test augmentations (operate on (3, H, W) float tensors)
# ------------------------------------------------------------------

def low_light(image: torch.Tensor, factor: float = 0.3) -> torch.Tensor:
    """Reduce brightness by a multiplicative *factor* in [0, 1]."""
    return (image * factor).clamp(0.0, 1.0)


def fog(
    image: torch.Tensor,
    severity: float = 0.5,
    kernel_size: int = 15,
) -> torch.Tensor:
    """Simulate fog with Gaussian blur + additive white noise.

    *severity* ∈ [0, 1] controls the blend ratio.  ``kernel_size`` must be odd.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    sigma = 2.0 + severity * 6.0
    coords = torch.arange(kernel_size, dtype=image.dtype, device=image.device) - kernel_size // 2
    gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]
    kernel_2d = kernel_2d.expand(3, 1, -1, -1)

    padding = kernel_size // 2
    blurred = F.conv2d(
        image.unsqueeze(0), kernel_2d, padding=padding, groups=3
    ).squeeze(0)

    noise = torch.randn_like(image) * severity * 0.08
    fogged = (1.0 - severity) * image + severity * blurred + noise
    return fogged.clamp(0.0, 1.0)


def occlusion(
    image: torch.Tensor,
    num_patches: int = 3,
    patch_size: int = 64,
) -> torch.Tensor:
    """Apply random rectangular masks (cutout) to the image."""
    _, h, w = image.shape
    out = image.clone()
    for _ in range(num_patches):
        cy = torch.randint(0, h, (1,)).item()
        cx = torch.randint(0, w, (1,)).item()
        y1 = max(cy - patch_size // 2, 0)
        y2 = min(cy + patch_size // 2, h)
        x1 = max(cx - patch_size // 2, 0)
        x2 = min(cx + patch_size // 2, w)
        out[:, y1:y2, x1:x2] = 0.0
    return out
