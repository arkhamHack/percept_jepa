"""NuScenes radar-camera fusion dataset for JEPA pre-training and evaluation.

Supports both **detection** and **prediction** tasks:
  - Detection: 2D boxes + velocity labels (per frame)
  - Prediction: per-agent future trajectories + agent state vectors,
    compatible with the nuScenes prediction challenge format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from .transforms import normalize, resize

logger = logging.getLogger(__name__)

_RADAR_XYZ = [0, 1, 2]
_RADAR_VEL = [8, 9]
_RADAR_RCS = [5]
_RADAR_FEAT_IDX = _RADAR_XYZ + _RADAR_VEL + _RADAR_RCS  # length 6
_RADAR_FEAT_DIM = len(_RADAR_FEAT_IDX)

_FRONT_CAM = "CAM_FRONT"
_FRONT_RADAR = "RADAR_FRONT"

_AGENT_STATE_DIM = 3  # velocity, acceleration, heading_change_rate


class NuScenesRadarCameraDataset(Dataset):
    """Paired radar + camera dataset for detection and trajectory prediction.

    Each item contains a front-camera image, the corresponding front-radar
    point cloud, projected 2D bounding boxes, velocity labels, camera
    matrices, per-agent future trajectories, agent state vectors, and
    (optionally) the next temporal frame for JEPA targets.

    Args:
        nuscenes_root: Path to the nuScenes dataset root.
        version: Dataset version string (``v1.0-mini``, ``v1.0-trainval``).
        split: ``train`` or ``val``.
        image_size: ``(H, W)`` for resizing camera images.
        max_radar_points: Max radar points per sample (pad/clip).
        use_future_frame: Load next-frame image for JEPA target encoder.
        prediction_seconds: Prediction horizon in seconds (nuScenes default 6).
        prediction_hz: Trajectory sampling rate (nuScenes default 2).
    """

    def __init__(
        self,
        nuscenes_root: str | Path,
        version: str = "v1.0-mini",
        split: str = "train",
        image_size: tuple[int, int] = (448, 800),
        max_radar_points: int = 256,
        use_future_frame: bool = True,
        prediction_seconds: float = 6.0,
        prediction_hz: int = 2,
    ) -> None:
        super().__init__()
        self.nuscenes_root = Path(nuscenes_root)
        self.version = version
        self.split = split
        self.image_size = image_size
        self.max_radar_points = max_radar_points
        self.use_future_frame = use_future_frame
        self.prediction_seconds = prediction_seconds
        self.prediction_hz = prediction_hz
        self.pred_steps = int(prediction_seconds * prediction_hz)  # 12

        self.nusc = NuScenes(version=version, dataroot=str(self.nuscenes_root), verbose=False)
        self.predict_helper = PredictHelper(self.nusc)
        self.samples = self._split_samples()
        logger.info("Loaded %d samples for split='%s'", len(self.samples), split)

    # ------------------------------------------------------------------
    # Split logic
    # ------------------------------------------------------------------

    def _split_samples(self) -> list[dict]:
        """Return sample records for the requested split.

        nuScenes doesn't ship an official train/val split list for mini, so
        we use a 80/20 scene-level split deterministically.
        """
        scene_names: list[str] = [s["name"] for s in self.nusc.scene]
        scene_names.sort()
        n_train = max(1, int(0.8 * len(scene_names)))
        train_scenes = set(scene_names[:n_train])
        val_scenes = set(scene_names[n_train:])
        target_scenes = train_scenes if self.split == "train" else val_scenes

        scene_token_set = {
            s["token"] for s in self.nusc.scene if s["name"] in target_scenes
        }
        samples = [
            s for s in self.nusc.sample if s["scene_token"] in scene_token_set
        ]
        return samples

    # ------------------------------------------------------------------
    # Core loading helpers
    # ------------------------------------------------------------------

    def _load_image(self, cam_data: dict) -> tuple[torch.Tensor, float, float]:
        """Load and resize the camera image.

        Returns ``(image_tensor, scale_y, scale_x)`` where scale factors map
        from original to resized coordinates.
        """
        img_path = self.nuscenes_root / cam_data["filename"]
        try:
            img = Image.open(str(img_path)).convert("RGB")
        except Exception:
            logger.warning("Failed to load image %s -- returning zeros", img_path)
            h, w = self.image_size
            return torch.zeros(3, h, w, dtype=torch.float32), 1.0, 1.0

        orig_w, orig_h = img.size
        h, w = self.image_size
        img_np = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)   # (3, H, W)
        img_t = resize(img_t, self.image_size)
        img_t = normalize(img_t)
        return img_t, h / orig_h, w / orig_w

    def _load_radar(self, radar_data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Load front radar pointcloud and return padded features + mask."""
        pc_path = self.nuscenes_root / radar_data["filename"]
        try:
            pc = RadarPointCloud.from_file(str(pc_path))
            points = pc.points.T  # (N, 18)
        except Exception:
            logger.warning("Failed to load radar %s -- returning zeros", pc_path)
            points = np.zeros((0, 18), dtype=np.float32)

        feats = points[:, _RADAR_FEAT_IDX].astype(np.float32) if len(points) > 0 else np.zeros((0, _RADAR_FEAT_DIM), dtype=np.float32)

        n = feats.shape[0]
        max_pts = self.max_radar_points
        padded = np.zeros((max_pts, _RADAR_FEAT_DIM), dtype=np.float32)
        mask = np.zeros(max_pts, dtype=bool)

        if n > 0:
            keep = min(n, max_pts)
            padded[:keep] = feats[:keep]
            mask[:keep] = True

        return torch.from_numpy(padded), torch.from_numpy(mask)

    def _get_camera_matrices(
        self, cam_data: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return intrinsics (3x3) and camera-to-ego extrinsics (4x4)."""
        calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        intrinsic = np.array(calib["camera_intrinsic"], dtype=np.float32)

        rotation = np.array(calib["rotation"], dtype=np.float64)
        translation = np.array(calib["translation"], dtype=np.float64)
        rot_mat = Quaternion(rotation).rotation_matrix.astype(np.float32)

        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = rot_mat
        extrinsic[:3, 3] = translation

        return torch.from_numpy(intrinsic), torch.from_numpy(extrinsic)

    def _load_future_image(self, sample: dict) -> torch.Tensor | None:
        """Load the camera image from the next temporal sample, if available."""
        next_token = sample["next"]
        if not next_token:
            return None
        next_sample = self.nusc.get("sample", next_token)
        cam_token = next_sample["data"][_FRONT_CAM]
        cam_data = self.nusc.get("sample_data", cam_token)
        img_t, _, _ = self._load_image(cam_data)
        return img_t

    # ------------------------------------------------------------------
    # Prediction task helpers
    # ------------------------------------------------------------------

    def _get_agent_prediction_data(
        self,
        sample: dict,
        ann_tokens_used: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Extract per-agent future trajectories and state vectors.

        Args:
            sample: nuScenes sample record.
            ann_tokens_used: Annotation tokens for objects that passed the
                camera-visibility filter (same order as boxes_2d).

        Returns:
            gt_trajectories: ``(N_agents, pred_steps, 2)`` future (x, y)
                in the agent frame. Padded with zeros for agents without futures.
            agent_states: ``(N_agents, 3)`` -- velocity, acceleration,
                heading change rate.
            instance_tokens: List of instance token strings.
        """
        sample_token = sample["token"]
        n = len(ann_tokens_used)
        gt_traj = np.zeros((n, self.pred_steps, 2), dtype=np.float32)
        agent_states = np.zeros((n, _AGENT_STATE_DIM), dtype=np.float32)
        instance_tokens: list[str] = []

        for i, ann_token in enumerate(ann_tokens_used):
            ann = self.nusc.get("sample_annotation", ann_token)
            inst_token = ann["instance_token"]
            instance_tokens.append(inst_token)

            try:
                future = self.predict_helper.get_future_for_agent(
                    inst_token, sample_token,
                    seconds=self.prediction_seconds,
                    in_agent_frame=True,
                )
                steps = min(future.shape[0], self.pred_steps)
                if steps > 0:
                    gt_traj[i, :steps] = future[:steps]
            except Exception:
                pass

            try:
                vel = self.predict_helper.get_velocity_for_agent(
                    inst_token, sample_token,
                )
                acc = self.predict_helper.get_acceleration_for_agent(
                    inst_token, sample_token,
                )
                hdg = self.predict_helper.get_heading_change_rate_for_agent(
                    inst_token, sample_token,
                )
                agent_states[i] = [
                    vel if not np.isnan(vel) else 0.0,
                    acc if not np.isnan(acc) else 0.0,
                    hdg if not np.isnan(hdg) else 0.0,
                ]
            except Exception:
                pass

        return (
            torch.from_numpy(gt_traj),
            torch.from_numpy(agent_states),
            instance_tokens,
        )

    # ------------------------------------------------------------------
    # Annotation extraction (tracks which ann_tokens pass camera filter)
    # ------------------------------------------------------------------

    def _get_annotations_with_tokens(
        self,
        sample: dict,
        cam_data: dict,
        intrinsic: np.ndarray,
        scale_y: float,
        scale_x: float,
    ) -> tuple[torch.Tensor, torch.Tensor, int, list[str]]:
        """Project 3D annotations to 2D camera boxes and extract velocities.

        Returns:
            ``(boxes_2d, velocities, num_objects, ann_tokens_used)``
        """
        h, w = self.image_size
        boxes_2d_list: list[np.ndarray] = []
        vel_list: list[np.ndarray] = []
        ann_tokens_used: list[str] = []

        cam_calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        cam_intrinsic = np.array(cam_calib["camera_intrinsic"], dtype=np.float64)

        ego_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
        ego_translation = np.array(ego_pose["translation"])
        ego_rotation_inv = Quaternion(ego_pose["rotation"]).inverse

        cam_translation = np.array(cam_calib["translation"])
        cam_rotation_inv = Quaternion(cam_calib["rotation"]).inverse

        for ann_token in sample["anns"]:
            box = self.nusc.get_box(ann_token)

            box.translate(-ego_translation)
            box.rotate(ego_rotation_inv)

            box.translate(-cam_translation)
            box.rotate(cam_rotation_inv)

            if box.center[2] <= 0:
                continue

            corners_3d = box.corners()
            corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2]
            x1 = float(np.min(corners_2d[0])) * scale_x
            y1 = float(np.min(corners_2d[1])) * scale_y
            x2 = float(np.max(corners_2d[0])) * scale_x
            y2 = float(np.max(corners_2d[1])) * scale_y

            x1 = np.clip(x1, 0, w) / w
            y1 = np.clip(y1, 0, h) / h
            x2 = np.clip(x2, 0, w) / w
            y2 = np.clip(y2, 0, h) / h

            if (x2 - x1) < 1e-4 or (y2 - y1) < 1e-4:
                continue

            boxes_2d_list.append(np.array([x1, y1, x2, y2], dtype=np.float32))

            vel_3d = self.nusc.box_velocity(ann_token)
            if np.any(np.isnan(vel_3d)):
                vel = np.zeros(2, dtype=np.float32)
            else:
                vel = vel_3d[:2].astype(np.float32)
            vel_list.append(vel)
            ann_tokens_used.append(ann_token)

        if len(boxes_2d_list) == 0:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0, 2), dtype=torch.float32),
                0,
                [],
            )

        boxes_2d = torch.from_numpy(np.stack(boxes_2d_list))
        velocities = torch.from_numpy(np.stack(vel_list))
        return boxes_2d, velocities, len(boxes_2d_list), ann_tokens_used

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        cam_token = sample["data"][_FRONT_CAM]
        radar_token = sample["data"][_FRONT_RADAR]

        cam_data = self.nusc.get("sample_data", cam_token)
        radar_data = self.nusc.get("sample_data", radar_token)

        image, scale_y, scale_x = self._load_image(cam_data)
        radar_points, radar_mask = self._load_radar(radar_data)
        intrinsics, extrinsics = self._get_camera_matrices(cam_data)

        cam_calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        cam_intrinsic_np = np.array(cam_calib["camera_intrinsic"], dtype=np.float64)
        boxes_2d, velocity, num_objects, ann_tokens_used = (
            self._get_annotations_with_tokens(
                sample, cam_data, cam_intrinsic_np, scale_y, scale_x
            )
        )

        gt_traj, agent_states, instance_tokens = (
            self._get_agent_prediction_data(sample, ann_tokens_used)
        )

        out: dict[str, Any] = {
            "image": image,
            "radar_points": radar_points,
            "radar_mask": radar_mask,
            "boxes_2d": boxes_2d,
            "velocity": velocity,
            "num_objects": num_objects,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "gt_trajectories": gt_traj,
            "agent_states": agent_states,
            "instance_tokens": instance_tokens,
            "sample_token": sample["token"],
        }

        if self.use_future_frame:
            future = self._load_future_image(sample)
            if future is not None:
                out["future_image"] = future
            else:
                out["future_image"] = torch.zeros_like(image)

        return out


# ------------------------------------------------------------------
# Collate function
# ------------------------------------------------------------------

def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collation that pads variable-length boxes/velocity/trajectories
    to the batch maximum and stacks fixed-size tensors normally."""
    max_obj = max(item["num_objects"] for item in batch)
    max_obj = max(max_obj, 1)

    batch_size = len(batch)
    first = batch[0]

    out: dict[str, Any] = {
        "image": torch.stack([item["image"] for item in batch]),
        "radar_points": torch.stack([item["radar_points"] for item in batch]),
        "radar_mask": torch.stack([item["radar_mask"] for item in batch]),
        "intrinsics": torch.stack([item["intrinsics"] for item in batch]),
        "extrinsics": torch.stack([item["extrinsics"] for item in batch]),
        "num_objects": torch.tensor(
            [item["num_objects"] for item in batch], dtype=torch.int64
        ),
    }

    padded_boxes = torch.zeros(batch_size, max_obj, 5, dtype=torch.float32)
    padded_vel = torch.zeros(batch_size, max_obj, 2, dtype=torch.float32)
    for i, item in enumerate(batch):
        n = item["num_objects"]
        if n > 0:
            padded_boxes[i, :n, :4] = item["boxes_2d"][:n]
            padded_boxes[i, :n, 4] = 1.0
            padded_vel[i, :n] = item["velocity"][:n]
    out["boxes"] = padded_boxes
    out["velocity"] = padded_vel

    if "gt_trajectories" in first:
        pred_steps = first["gt_trajectories"].shape[-2] if first["gt_trajectories"].numel() > 0 else 12
        padded_traj = torch.zeros(batch_size, max_obj, pred_steps, 2, dtype=torch.float32)
        padded_states = torch.zeros(batch_size, max_obj, _AGENT_STATE_DIM, dtype=torch.float32)
        for i, item in enumerate(batch):
            n = item["num_objects"]
            if n > 0:
                padded_traj[i, :n] = item["gt_trajectories"][:n]
                padded_states[i, :n] = item["agent_states"][:n]
        out["gt_trajectories"] = padded_traj
        out["agent_states"] = padded_states

    if "instance_tokens" in first:
        out["instance_tokens"] = [item["instance_tokens"] for item in batch]
        out["sample_tokens"] = [item["sample_token"] for item in batch]

    if "future_image" in first:
        out["future_image"] = torch.stack(
            [item["future_image"] for item in batch]
        )

    return out
