#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion

from cuda import bev_voxelize, radar_project, radar_rasterize


# Match the dataset feature selection in dataset/nuscenes_dataset.py
RADAR_XYZ = [0, 1, 2]
RADAR_VEL = [8, 9]
RADAR_RCS = [5]
RADAR_FEAT_IDX = RADAR_XYZ + RADAR_VEL + RADAR_RCS  # x, y, z, vx, vy, rcs

FRONT_CAM = "CAM_FRONT"
FRONT_RADAR = "RADAR_FRONT"


def get_camera_matrices(nusc: NuScenes, cam_data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    intrinsic = np.array(calib["camera_intrinsic"], dtype=np.float32)

    rotation = np.array(calib["rotation"], dtype=np.float64)
    translation = np.array(calib["translation"], dtype=np.float64)
    rot_mat = Quaternion(rotation).rotation_matrix.astype(np.float32)

    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = rot_mat
    extrinsic[:3, 3] = translation

    return torch.from_numpy(intrinsic), torch.from_numpy(extrinsic)


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    denom = max(x.max(), 1e-6)
    x = x / denom
    return (255.0 * x).clip(0, 255).astype(np.uint8)


def make_radar_rgb_features(points6: torch.Tensor) -> torch.Tensor:
    """
    points6: (N, 6) = [x, y, z, vx, vy, rcs]
    Returns (N, 3) features for image rasterization.
    """
    vx = points6[:, 3]
    vy = points6[:, 4]
    rcs = points6[:, 5]

    speed = torch.sqrt(vx * vx + vy * vy)
    speed = speed / max(speed.max().item(), 1e-6)

    rcs_norm = rcs - rcs.min()
    rcs_norm = rcs_norm / max(rcs_norm.max().item(), 1e-6)

    vx_norm = (vx.abs() / max(vx.abs().max().item(), 1e-6)).clamp(0, 1)

    rgb = torch.stack([speed, rcs_norm, vx_norm], dim=-1)
    return rgb


def main() -> None:
    # Change these for your machine.
    nuscenes_root = "/data/nuscenes"
    version = "v1.0-mini"
    output_dir = Path("tmp/kernel_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    # Optional: if you want the extension compiled ahead of time instead of JIT-loading:
    # python cuda/setup_cuda.py build_ext --inplace

    nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=False)
    sample = nusc.sample[0]

    cam_token = sample["data"][FRONT_CAM]
    radar_token = sample["data"][FRONT_RADAR]

    cam_data = nusc.get("sample_data", cam_token)
    radar_data = nusc.get("sample_data", radar_token)

    image_path = Path(nuscenes_root) / cam_data["filename"]
    radar_path = Path(nuscenes_root) / radar_data["filename"]

    image = np.array(Image.open(image_path).convert("RGB"))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]

    pc = RadarPointCloud.from_file(str(radar_path))
    raw_points = pc.points.T.astype(np.float32)  # (N, 18)
    points6 = raw_points[:, RADAR_FEAT_IDX]      # (N, 6)

    points6_t = torch.from_numpy(points6).to(device)
    intrinsics, extrinsics = get_camera_matrices(nusc, cam_data)
    intrinsics = intrinsics.to(device)
    extrinsics = extrinsics.to(device)

    print(f"[info] image={image.shape}, radar_points={points6_t.shape}")

    # ------------------------------------------------------------------
    # 1. Test BEV voxelization kernel
    # ------------------------------------------------------------------
    bev = bev_voxelize(
        points6_t,
        x_bounds=(-50.0, 50.0),
        y_bounds=(-50.0, 50.0),
        grid_h=200,
        grid_w=200,
    )  # (H, W, C)

    bev_cpu = bev.detach().cpu().numpy()
    bev_mag = np.linalg.norm(bev_cpu, axis=-1)
    bev_img = normalize_to_uint8(bev_mag)
    bev_img = cv2.applyColorMap(bev_img, cv2.COLORMAP_TURBO)

    bev_out = output_dir / "bev_voxelize.png"
    cv2.imwrite(str(bev_out), bev_img)
    print(f"[ok] saved {bev_out}")

    # ------------------------------------------------------------------
    # 2. Test radar projection kernel
    # ------------------------------------------------------------------
    coords_2d, valid = radar_project(points6_t[:, :3], intrinsics, extrinsics)

    coords_2d_cpu = coords_2d.detach().cpu().numpy()
    valid_cpu = valid.detach().cpu().numpy()

    proj_vis = image_bgr.copy()
    for i in range(coords_2d_cpu.shape[0]):
        if not valid_cpu[i]:
            continue
        u = int(round(coords_2d_cpu[i, 0]))
        v = int(round(coords_2d_cpu[i, 1]))
        if 0 <= u < width and 0 <= v < height:
            cv2.circle(proj_vis, (u, v), 2, (0, 255, 0), -1)

    proj_out = output_dir / "radar_project_points.png"
    cv2.imwrite(str(proj_out), proj_vis)
    print(f"[ok] saved {proj_out}")

    # ------------------------------------------------------------------
    # 3. Test radar rasterize kernel
    # ------------------------------------------------------------------
    raster_features = make_radar_rgb_features(points6_t).to(device)  # (N, 3)
    raster = radar_rasterize(
        coords_2d=coords_2d,
        features=raster_features,
        valid=valid,
        height=height,
        width=width,
        radius=3,
    )  # (H, W, 3)

    raster_cpu = raster.detach().cpu().numpy()
    raster_img = normalize_to_uint8(raster_cpu)
    raster_bgr = cv2.cvtColor(raster_img, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(image_bgr, 0.75, raster_bgr, 0.75, 0.0)

    raster_out = output_dir / "radar_rasterize.png"
    overlay_out = output_dir / "radar_overlay.png"
    cv2.imwrite(str(raster_out), raster_bgr)
    cv2.imwrite(str(overlay_out), overlay)

    print(f"[ok] saved {raster_out}")
    print(f"[ok] saved {overlay_out}")

    # ------------------------------------------------------------------
    # 4. Quick sanity stats
    # ------------------------------------------------------------------
    in_frame = 0
    for i in range(coords_2d_cpu.shape[0]):
        if not valid_cpu[i]:
            continue
        u = coords_2d_cpu[i, 0]
        v = coords_2d_cpu[i, 1]
        if 0 <= u < width and 0 <= v < height:
            in_frame += 1

    print(f"[stats] valid_projected={int(valid_cpu.sum())}")
    print(f"[stats] projected_in_frame={in_frame}")
    print(f"[stats] bev_shape={tuple(bev.shape)}")
    print(f"[stats] raster_shape={tuple(raster.shape)}")


if __name__ == "__main__":
    main()