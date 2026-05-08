"""Backbone networks: pretrained V-JEPA visual encoder + radar BEV encoder.

The V-JEPA encoder uses a ViT backbone (via timm) pretrained with the V-JEPA
objective.  It preserves spatial feature maps by reshaping patch tokens back to
(H', W') grids.  The radar BEV encoder uses CUDA-accelerated voxelization to
build a BEV tensor and processes it through a lightweight CNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None  # type: ignore[assignment]

from cuda import bev_voxelize


# ──────────────────────────────────────────────────────────────
#  V-JEPA visual encoder (pretrained ViT backbone)
# ──────────────────────────────────────────────────────────────

class VJEPAEncoder(nn.Module):
    """Pretrained V-JEPA visual encoder that preserves spatial feature maps.

    Uses a timm ViT model.  Extracts patch tokens from the last transformer
    block and reshapes them to ``(B, feat_dim, H', W')``.

    Args:
        model_name: timm model identifier (e.g. ``"vit_base_patch16_224"``).
        pretrained_path: Optional local ``.pth`` weights.  If *None*, uses
            timm's default pretrained weights.
        freeze: If *True*, all parameters are frozen (stage 1).
        unfreeze_last_n: Number of trailing ViT blocks to unfreeze (stage 2).
        feat_dim: Expected output feature dimension (for verification).
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained_path: str | None = None,
        freeze: bool = True,
        unfreeze_last_n: int = 0,
        feat_dim: int = 768,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for V-JEPA encoder: pip install timm")

        self.feat_dim = feat_dim
        self.vit = timm.create_model(
            model_name,
            pretrained=(pretrained_path is None),
            num_classes=0,       # remove classification head
            global_pool="",      # keep all patch tokens
        )

        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            if "model" in state:
                state = state["model"]
            if "encoder" in state:
                state = state["encoder"]
            missing, unexpected = self.vit.load_state_dict(state, strict=False)
            if missing:
                print(f"[VJEPAEncoder] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")

        # Determine spatial dims from patch embed
        patch_size = self.vit.patch_embed.patch_size
        if isinstance(patch_size, (tuple, list)):
            self.patch_h, self.patch_w = patch_size
        else:
            self.patch_h = self.patch_w = patch_size

        # Freeze/unfreeze
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

        if unfreeze_last_n > 0 and hasattr(self.vit, "blocks"):
            for block in self.vit.blocks[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map from image.

        Args:
            x: ``(B, 3, H, W)`` input image (224×224 expected).

        Returns:
            ``(B, feat_dim, H', W')`` spatial feature map where
            ``H' = H / patch_size``, ``W' = W / patch_size``.
        """
        B, _, H, W = x.shape
        tokens = self.vit.forward_features(x)  # (B, num_patches+1, feat_dim) or (B, num_patches, feat_dim)

        # Remove CLS token if present
        if hasattr(self.vit, "num_prefix_tokens") and self.vit.num_prefix_tokens > 0:
            tokens = tokens[:, self.vit.num_prefix_tokens:, :]

        H_out = H // self.patch_h
        W_out = W // self.patch_w
        # Reshape to spatial: (B, H'*W', C) -> (B, C, H', W')
        feat_map = tokens.transpose(1, 2).reshape(B, self.feat_dim, H_out, W_out)
        return feat_map

    def set_stage(self, stage: int, unfreeze_last_n: int = 4) -> None:
        """Switch training stage.

        Stage 1: freeze all.  Stage 2: unfreeze last N blocks with lower LR.
        """
        if stage == 1:
            for p in self.vit.parameters():
                p.requires_grad = False
        elif stage >= 2 and hasattr(self.vit, "blocks"):
            # Keep early blocks frozen, unfreeze last N
            for p in self.vit.parameters():
                p.requires_grad = False
            for block in self.vit.blocks[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────
#  Radar BEV encoder (CUDA voxelization + lightweight CNN)
# ──────────────────────────────────────────────────────────────

class BEVRadarEncoder(nn.Module):
    """Encodes radar point clouds via BEV voxelization + CNN.

    Pipeline:
        raw radar points → CUDA ``bev_voxelize`` → BEV tensor (H, W, C)
        → Conv2D layers → BEV feature map ``(B, bev_channels, H', W')``

    Args:
        input_dim: Per-point feature dimension (default 6).
        bev_channels: Output feature channels.
        x_bounds: ``(x_min, x_max)`` BEV extent in metres.
        y_bounds: ``(y_min, y_max)`` BEV extent in metres.
        grid_h: BEV grid rows.
        grid_w: BEV grid columns.
    """

    def __init__(
        self,
        input_dim: int = 6,
        bev_channels: int = 64,
        x_bounds: tuple[float, float] = (-50.0, 50.0),
        y_bounds: tuple[float, float] = (-50.0, 50.0),
        grid_h: int = 200,
        grid_w: int = 200,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.bev_channels = bev_channels
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Lightweight CNN on BEV grid
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, bev_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.cnn:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _voxelize_batch(self, radar_points: torch.Tensor, radar_mask: torch.Tensor) -> torch.Tensor:
        """Voxelize a batch of radar point clouds into BEV grids.

        Args:
            radar_points: ``(B, N, input_dim)``
            radar_mask: ``(B, N)``

        Returns:
            ``(B, input_dim, grid_h, grid_w)`` BEV tensor.
        """
        B = radar_points.shape[0]
        bev_list = []
        for i in range(B):
            mask_i = radar_mask[i].bool()
            pts_i = radar_points[i][mask_i]  # (K, input_dim)
            if pts_i.shape[0] == 0:
                bev_i = torch.zeros(
                    self.grid_h, self.grid_w, self.input_dim,
                    dtype=radar_points.dtype, device=radar_points.device,
                )
            else:
                bev_i = bev_voxelize(
                    pts_i, self.x_bounds, self.y_bounds,
                    self.grid_h, self.grid_w,
                )  # (H, W, C)
            bev_list.append(bev_i)
        bev = torch.stack(bev_list)  # (B, H, W, C)
        return bev.permute(0, 3, 1, 2)  # (B, C, H, W)

    def forward(
        self, radar_points: torch.Tensor, radar_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode radar to BEV feature map.

        Args:
            radar_points: ``(B, N, input_dim)``
            radar_mask: ``(B, N)``

        Returns:
            ``(B, bev_channels, H', W')`` BEV feature map.
        """
        bev = self._voxelize_batch(radar_points, radar_mask)  # (B, C, H, W)
        return self.cnn(bev)

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────
#  Legacy: keep original encoders for baselines
# ──────────────────────────────────────────────────────────────

class ImageEncoder(nn.Module):
    """ResNet18 backbone (kept for camera-only baseline)."""

    def __init__(
        self,
        pretrained: bool = True,
        light: bool = False,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        import torchvision.models as tv_models
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
    """MLP encoder for sparse radar (kept for legacy/baseline)."""

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
