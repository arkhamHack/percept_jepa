"""Backbone networks for radar-camera fusion JEPA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ImageEncoder(nn.Module):
    """ResNet18 backbone that outputs a spatial feature map.

    Args:
        pretrained: Load ImageNet weights.
        light: Use only the first 3 ResNet blocks (lower VRAM).
        out_channels: Number of output channels. If *None*, uses the native
            channel count (256 for light, 512 for full).
    """

    def __init__(
        self,
        pretrained: bool = True,
        light: bool = False,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = tv_models.resnet18(weights=weights)

        if light:
            self.backbone = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3,
            )
            native_channels = 256
        else:
            self.backbone = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            )
            native_channels = 512

        self.out_channels = out_channels or native_channels
        self.proj: nn.Module | None = None
        if self.out_channels != native_channels:
            self.proj = nn.Conv2d(native_channels, self.out_channels, 1, bias=False)
            nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if self.proj is not None:
            feat = self.proj(feat)
        return feat

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class RadarEncoder(nn.Module):
    """MLP encoder for sparse radar point clouds with masked mean pooling.

    Args:
        input_dim: Per-point feature dimension (default 6: x, y, z, vx, vy, rcs).
        radar_dim: Output feature dimension.
    """

    def __init__(self, input_dim: int = 6, radar_dim: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, radar_dim),
        )
        self.radar_dim = radar_dim
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(
        self, radar_points: torch.Tensor, radar_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with masked mean pooling.

        Args:
            radar_points: ``(B, N, input_dim)`` radar point features.
            radar_mask: ``(B, N)`` bool/float mask — 1 for valid, 0 for padding.

        Returns:
            Pooled radar feature ``(B, radar_dim)``.
        """
        point_features = self.mlp(radar_points)

        mask = radar_mask.unsqueeze(-1).float()  # (B, N, 1)
        masked_sum = (point_features * mask).sum(dim=1)  # (B, radar_dim)
        counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        return masked_sum / counts

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
