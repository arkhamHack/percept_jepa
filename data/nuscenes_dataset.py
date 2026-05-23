"""
NuScenes Rear Camera + Radar Dataset
======================================
Loads synchronized temporal sequences of:
  - CAM_BACK (rear RGB camera)
  - RADAR_BACK_LEFT + RADAR_BACK_RIGHT (rear radar point clouds)
  - Ego-vehicle motion
  - 3-D object annotations with track IDs
  - Future trajectories for each annotated instance

Output per sample:
  {
    "images":              Tensor[T, C, H, W]       - temporal visual frames
    "radar_points":        Tensor[N, 6]             - accumulated radar (x,y,z,rcs,vx,vy)
    "radar_mask":          Tensor[N]                - valid point mask (for padding)
    "ego_motion":          Tensor[6]                - (vx,vy,ax,ay,yaw_rate,speed)
    "boxes":               Tensor[M, 7]             - (x,y,z,dx,dy,dz,yaw) in ego frame
    "labels":              Tensor[M]                - class indices
    "track_ids":           Tensor[M]                - unique instance id per box
    "future_trajectories": Tensor[M, K, 2]          - future (x,y) positions in ego frame
    "future_mask":         Tensor[M, K]             - valid future step mask
  }
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix

from data.transforms import get_image_transform, get_radar_transform

logger = logging.getLogger(__name__)

# nuScenes class → integer label mapping (10 foreground classes)
NUSCENES_CLASSES = {
    'car': 0,
    'truck': 1,
    'bus': 2,
    'trailer': 3,
    'construction_vehicle': 4,
    'pedestrian': 5,
    'motorcycle': 6,
    'bicycle': 7,
    'traffic_cone': 8,
    'barrier': 9,
}


class NuScenesRearDataset(Dataset):
    """
    Temporal dataset for rear-facing sensor fusion on nuScenes.

    Each item is a sliding window of `sequence_length` consecutive samples
    from a single scene.  The **last** frame is the 'current' timestep for
    which we produce detections/predictions; the preceding frames supply
    temporal context to the V-JEPA encoder.
    """

    def __init__(self, cfg, split: str = 'train', transforms=None):
        """
        Args:
            cfg:        OmegaConf / dict config (data sub-section)
            split:      'train' or 'val'
            transforms: optional augmentation callable applied to images
        """
        self.cfg = cfg
        self.split = split
        self.seq_len = cfg.data.sequence_length
        self.future_steps = cfg.data.future_steps
        self.future_dt = cfg.data.future_dt
        self.cameras = cfg.data.cameras          # e.g. ['CAM_BACK']
        self.radars = cfg.data.radars            # e.g. ['RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        self.radar_max_pts = cfg.data.radar_max_points
        self.radar_num_sweeps = cfg.data.radar_num_sweeps
        self.img_size = tuple(cfg.data.image_size)   # (H, W)

        # Image transforms (resize, normalise)
        self.img_transform = transforms if transforms else get_image_transform(self.img_size)

        # Initialise NuScenes SDK
        logger.info(f"Loading nuScenes {cfg.data.nuscenes_version} ...")
        self.nusc = NuScenes(
            version=cfg.data.nuscenes_version,
            dataroot=cfg.data.nuscenes_root,
            verbose=False,
        )

        # Build list of (scene, [sample_tokens]) sequences
        self.sequences: List[Tuple[str, List[str]]] = []
        self._build_sequences(split)
        logger.info(f"[{split}] {len(self.sequences)} temporal sequences loaded.")

        # Build instance → unique integer id map (for track IDs)
        self._instance_id_map: Dict[str, int] = {}
        self._next_instance_id = 0

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene_token, sample_tokens = self.sequences[idx]

        # ---- Images: load T frames from rear camera  -----------------
        images = []
        for tok in sample_tokens:
            img = self._load_camera_image(tok, self.cameras[0])
            images.append(img)
        # images: List[Tensor[C,H,W]]  →  Tensor[T,C,H,W]
        images = torch.stack(images, dim=0)

        # ---- Radar: accumulate across sweeps for LAST sample ---------
        current_token = sample_tokens[-1]
        radar_points, radar_mask = self._load_radar_points(
            current_token, self.radars, self.radar_num_sweeps
        )
        # radar_points: Tensor[N, 6],  radar_mask: Tensor[N]

        # ---- Ego motion for current sample ---------------------------
        ego_motion = self._load_ego_motion(current_token)
        # ego_motion: Tensor[6]

        # ---- Annotations for current sample --------------------------
        boxes, labels, instance_tokens = self._load_annotations(current_token)
        # boxes: Tensor[M,7], labels: Tensor[M], instance_tokens: List[str]

        # ---- Future trajectories for each annotated instance ---------
        future_traj, future_mask = self._load_future_trajectories(
            current_token, instance_tokens
        )
        # future_traj: Tensor[M, K, 2],  future_mask: Tensor[M, K]

        # ---- Track IDs (persistent integers per instance) ------------
        track_ids = torch.tensor(
            [self._get_instance_id(t) for t in instance_tokens],
            dtype=torch.long,
        )  # Tensor[M]

        return {
            'images': images,                          # [T, C, H, W]
            'radar_points': radar_points,              # [N, 6]
            'radar_mask': radar_mask,                  # [N]
            'ego_motion': ego_motion,                  # [6]
            'boxes': boxes,                            # [M, 7]
            'labels': labels,                          # [M]
            'track_ids': track_ids,                    # [M]
            'future_trajectories': future_traj,        # [M, K, 2]
            'future_mask': future_mask,                # [M, K]
            'sample_token': current_token,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sequences(self, split: str):
        """
        Walk every scene and extract sliding-window sequences.

        For nuScenes mini, all 10 scenes are used for training;
        for v1.0-trainval, the standard splits are respected.
        """
        # Determine which scenes belong to this split
        split_file = os.path.join(
            os.path.dirname(__file__), f'../configs/{split}_scenes.txt'
        )
        if os.path.exists(split_file):
            with open(split_file) as f:
                allowed_names = set(f.read().splitlines())
        else:
            # For mini, use all scenes; rough 80/20 split by index
            scenes = self.nusc.scene
            n = len(scenes)
            if split == 'train':
                scenes = scenes[:int(0.8 * n)]
            else:
                scenes = scenes[int(0.8 * n):]
            allowed_names = {s['name'] for s in scenes}

        for scene in self.nusc.scene:
            if scene['name'] not in allowed_names:
                continue

            # Collect ordered sample tokens for this scene
            sample_tokens: List[str] = []
            sample = self.nusc.get('sample', scene['first_sample_token'])
            while True:
                sample_tokens.append(sample['token'])
                if sample['next'] == '':
                    break
                sample = self.nusc.get('sample', sample['next'])

            # Sliding window sequences of length seq_len
            for start in range(len(sample_tokens) - self.seq_len + 1):
                window = sample_tokens[start: start + self.seq_len]
                self.sequences.append((scene['token'], window))

    def _load_camera_image(
        self, sample_token: str, camera_name: str
    ) -> torch.Tensor:
        """
        Load and transform a single camera frame.

        Returns:
            Tensor[C, H, W]  (normalised float32)
        """
        sample = self.nusc.get('sample', sample_token)
        cam_data = self.nusc.get('sample_data', sample['data'][camera_name])
        img_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        img = Image.open(img_path).convert('RGB')
        return self.img_transform(img)  # Tensor[C, H, W]

    def _load_radar_points(
        self,
        sample_token: str,
        radar_names: List[str],
        num_sweeps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and accumulate radar point clouds from multiple sensors and sweeps.

        NuScenes RadarPointCloud feature order:
          [x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp,
           is_quality_valid, ambig_state, x_rms, y_rms, invalid_state,
           pdh0, vx_rms, vy_rms]

        We extract: x(0), y(1), z(2), rcs(5), vx_comp(8), vy_comp(9)

        All points are transformed into the **ego vehicle frame** of the
        current sample.

        Returns:
            radar_points: Tensor[N_max, 6]   (padded / truncated)
            radar_mask:   Tensor[N_max]       (1 = real point, 0 = padding)
        """
        sample = self.nusc.get('sample', sample_token)
        ego_pose = self.nusc.get('ego_pose',
            self.nusc.get('sample_data',
                sample['data'][radar_names[0]])['ego_pose_token'])
        ego_from_global = np.linalg.inv(
            transform_matrix(
                ego_pose['translation'],
                Quaternion(ego_pose['rotation']),
                inverse=False,
            )
        )

        all_points: List[np.ndarray] = []

        for radar_name in radar_names:
            sd_token = sample['data'][radar_name]
            pc, _ = RadarPointCloud.from_file_multisweep(
                self.nusc,
                sample,
                chan=radar_name,
                ref_chan='LIDAR_TOP',
                nsweeps=num_sweeps,
            )
            # pc.points: [18, N]
            pts = pc.points  # [18, N]

            # Transform from LIDAR_TOP frame to ego frame
            # (multisweep already returns in ref_chan = LIDAR_TOP frame;
            #  we still need to bring it to ego frame)
            lidar_sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            cs = self.nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
            lidar_from_ego = transform_matrix(
                cs['translation'], Quaternion(cs['rotation']), inverse=True
            )
            # Bring XYZ from LIDAR → ego
            xyz = pts[:3, :]  # [3, N]
            ones = np.ones((1, xyz.shape[1]))
            xyz_h = np.vstack([xyz, ones])          # [4, N]
            xyz_ego = (np.linalg.inv(lidar_from_ego) @ xyz_h)[:3, :]  # [3, N]

            # Compose our 6-D feature vector
            feat = np.vstack([
                xyz_ego,          # rows 0-2: x, y, z in ego frame
                pts[5:6, :],      # row 3:   rcs
                pts[8:10, :],     # rows 4-5: vx_comp, vy_comp
            ])  # [6, N]

            all_points.append(feat.T)  # [N, 6]

        if len(all_points) == 0:
            combined = np.zeros((0, 6), dtype=np.float32)
        else:
            combined = np.concatenate(all_points, axis=0)  # [N_total, 6]

        N = combined.shape[0]
        N_max = self.radar_max_pts

        # Pad or truncate to fixed size
        if N >= N_max:
            idx = np.random.choice(N, N_max, replace=False)
            combined = combined[idx]
            mask = np.ones(N_max, dtype=np.float32)
        else:
            pad = np.zeros((N_max - N, 6), dtype=np.float32)
            combined = np.vstack([combined, pad])
            mask = np.array([1.0] * N + [0.0] * (N_max - N), dtype=np.float32)

        return (
            torch.from_numpy(combined.astype(np.float32)),  # [N_max, 6]
            torch.from_numpy(mask),                         # [N_max]
        )

    def _load_ego_motion(self, sample_token: str) -> torch.Tensor:
        """
        Extract ego vehicle kinematics from CAN bus metadata (if available)
        or approximate from consecutive ego poses.

        Returns:
            Tensor[6]  →  (vx, vy, ax, ay, yaw_rate, speed)
        """
        try:
            # Attempt CAN bus data (only available in full trainval)
            from nuscenes.can_bus.can_bus_api import NuScenesCanBus
            nusc_can = NuScenesCanBus(dataroot=self.cfg.data.nuscenes_root)
            sample = self.nusc.get('sample', sample_token)
            scene = self.nusc.get('scene', sample['scene_token'])
            pose_list = nusc_can.get_messages(scene['name'], 'pose')
            # Find closest message by utime
            utime = sample['timestamp']
            closest = min(pose_list, key=lambda p: abs(p['utime'] - utime))
            vx, vy = closest['vel'][0], closest['vel'][1]
            ax, ay = closest['accel'][0], closest['accel'][1]
            yaw_rate = closest['rotation_rate'][2]
            speed = np.sqrt(vx**2 + vy**2)
            return torch.tensor([vx, vy, ax, ay, yaw_rate, speed], dtype=torch.float32)
        except Exception:
            pass

        # Fallback: approximate from consecutive ego poses
        sample = self.nusc.get('sample', sample_token)
        sd = self.nusc.get('sample_data', sample['data'][self.cameras[0]])
        pose_curr = self.nusc.get('ego_pose', sd['ego_pose_token'])

        ego_motion = np.zeros(6, dtype=np.float32)
        if sample['prev'] != '':
            sd_prev = self.nusc.get('sample_data',
                self.nusc.get('sample', sample['prev'])['data'][self.cameras[0]])
            pose_prev = self.nusc.get('ego_pose', sd_prev['ego_pose_token'])
            dt = (pose_curr['timestamp'] - pose_prev['timestamp']) * 1e-6  # µs → s
            if dt > 0:
                dx = pose_curr['translation'][0] - pose_prev['translation'][0]
                dy = pose_curr['translation'][1] - pose_prev['translation'][1]
                ego_motion[0] = dx / dt   # vx
                ego_motion[1] = dy / dt   # vy
                ego_motion[5] = np.sqrt(ego_motion[0]**2 + ego_motion[1]**2)

        return torch.from_numpy(ego_motion)  # [6]

    def _load_annotations(
        self, sample_token: str
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Load 3-D bounding boxes for the current sample in the ego frame.

        Returns:
            boxes:           Tensor[M, 7]   (cx, cy, cz, dx, dy, dz, yaw)
            labels:          Tensor[M]       long  class indices
            instance_tokens: List[str]       length M
        """
        sample = self.nusc.get('sample', sample_token)

        # Ego pose at current timestep
        sd = self.nusc.get('sample_data', sample['data'][self.cameras[0]])
        ego_pose = self.nusc.get('ego_pose', sd['ego_pose_token'])
        ego_from_global = np.linalg.inv(
            transform_matrix(
                ego_pose['translation'],
                Quaternion(ego_pose['rotation']),
            )
        )

        boxes_list, labels_list, tokens_list = [], [], []

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            category = ann['category_name'].split('.')[1] \
                if '.' in ann['category_name'] else ann['category_name']

            if category not in NUSCENES_CLASSES:
                continue

            # Global → ego frame
            center_g = np.array(ann['translation'] + [1.0])  # [4]
            center_e = (ego_from_global @ center_g)[:3]       # [3]

            # Rotation: global quaternion → ego yaw
            q_global = Quaternion(ann['rotation'])
            q_ego_inv = Quaternion(ego_pose['rotation']).inverse
            q_ego = q_ego_inv * q_global
            yaw = q_ego.yaw_pitch_roll[0]  # scalar

            w, l, h = ann['size']   # nuScenes stores [w, l, h]
            box = [center_e[0], center_e[1], center_e[2], l, w, h, yaw]

            boxes_list.append(box)
            labels_list.append(NUSCENES_CLASSES[category])
            tokens_list.append(ann['instance_token'])

        if len(boxes_list) == 0:
            return (
                torch.zeros(0, 7),
                torch.zeros(0, dtype=torch.long),
                [],
            )

        return (
            torch.tensor(boxes_list, dtype=torch.float32),   # [M, 7]
            torch.tensor(labels_list, dtype=torch.long),      # [M]
            tokens_list,                                       # List[M]
        )

    def _load_future_trajectories(
        self,
        sample_token: str,
        instance_tokens: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trace future positions of each instance for `future_steps` timesteps.

        Positions are expressed as (x, y) displacements in the **current**
        ego frame (i.e. relative to the ego vehicle's current pose).

        Returns:
            future_traj: Tensor[M, K, 2]   (x, y) per future step
            future_mask: Tensor[M, K]       1 if annotation exists
        """
        M = len(instance_tokens)
        K = self.future_steps

        future_traj = np.zeros((M, K, 2), dtype=np.float32)
        future_mask = np.zeros((M, K), dtype=np.float32)

        if M == 0:
            return torch.from_numpy(future_traj), torch.from_numpy(future_mask)

        # Current ego pose (reference frame)
        sample = self.nusc.get('sample', sample_token)
        sd = self.nusc.get('sample_data', sample['data'][self.cameras[0]])
        ego_pose_curr = self.nusc.get('ego_pose', sd['ego_pose_token'])
        ego_from_global = np.linalg.inv(
            transform_matrix(
                ego_pose_curr['translation'],
                Quaternion(ego_pose_curr['rotation']),
            )
        )

        for m, inst_token in enumerate(instance_tokens):
            # Find the annotation for this instance in the current sample
            ann_token = None
            for a in sample['anns']:
                ann = self.nusc.get('sample_annotation', a)
                if ann['instance_token'] == inst_token:
                    ann_token = a
                    break
            if ann_token is None:
                continue

            ann = self.nusc.get('sample_annotation', ann_token)
            for k in range(K):
                if ann['next'] == '':
                    break
                ann = self.nusc.get('sample_annotation', ann['next'])
                center_g = np.array(ann['translation'] + [1.0])
                center_e = (ego_from_global @ center_g)[:2]  # (x, y)
                future_traj[m, k] = center_e
                future_mask[m, k] = 1.0

        return torch.from_numpy(future_traj), torch.from_numpy(future_mask)

    def _get_instance_id(self, instance_token: str) -> int:
        """Assign a consistent integer ID to each instance token."""
        if instance_token not in self._instance_id_map:
            self._instance_id_map[instance_token] = self._next_instance_id
            self._next_instance_id += 1
        return self._instance_id_map[instance_token]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_dataset(cfg, split: str = 'train') -> NuScenesRearDataset:
    """Create dataset for the given split."""
    return NuScenesRearDataset(cfg, split=split)
