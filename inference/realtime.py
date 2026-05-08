"""Real-time inference loop with OpenCV visualisation."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from dataset.transforms import normalize, resize


class RealtimeInference:
    """Run live or video-file inference with bounding-box, velocity, and trajectory overlay.

    Args:
        model: Trained JEPAModel (single-GPU, no DataParallel).
        image_size: ``(H, W)`` model input size.
        confidence_threshold: Min sigmoid confidence to draw a box.
        show_velocity: Draw velocity arrows.
        device: Torch device string (``'cuda'`` / ``'cpu'``).
    """

    _COLORS = [
        (0, 255, 0),
        (255, 128, 0),
        (0, 128, 255),
        (255, 0, 128),
        (128, 255, 0),
    ]

    def __init__(
        self,
        model: nn.Module,
        image_size: tuple[int, int] = (224, 224),
        confidence_threshold: float = 0.3,
        show_velocity: bool = True,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.image_size = image_size
        self.conf_thr = confidence_threshold
        self.show_velocity = show_velocity
        self.is_multimodal = hasattr(model, "set_training_stage")

    # ──────────────────────────────────────────────────────────
    #  Video loop
    # ──────────────────────────────────────────────────────────

    def run_video(
        self,
        source: int | str = 0,
        save_path: str | None = None,
        radar_points: torch.Tensor | None = None,
        radar_mask: torch.Tensor | None = None,
    ) -> None:
        """Open a video source and run inference frame-by-frame.

        Args:
            source: Webcam index or path to a video file.
            save_path: If given, write annotated video to this file.
            radar_points: Optional static radar tensor ``(1, N, 6)``
                to reuse every frame (for demo without live radar).
            radar_mask: Corresponding mask ``(1, N)``.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps, (w_orig, h_orig))

        prev_frame_tensor: torch.Tensor | None = None

        print(f"[Inference] Running on {source} — press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            frame_tensor = self._preprocess(frame)

            outputs = self._infer(
                frame_tensor, radar_points, radar_mask, prev_frame_tensor,
            )
            prev_frame_tensor = frame_tensor

            dt = time.perf_counter() - t0
            annotated = self._draw(frame, outputs, dt)

            if writer is not None:
                writer.write(annotated)

            cv2.imshow("Radar-Camera JEPA", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    # ──────────────────────────────────────────────────────────
    #  Single-frame inference on a nuScenes sample dict
    # ──────────────────────────────────────────────────────────

    def infer_sample(
        self, sample: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch = {
            k: v.unsqueeze(0).to(self.device) if isinstance(v, torch.Tensor) and v.dim() > 0 else v
            for k, v in sample.items()
        }
        return self._forward(batch)

    # ──────────────────────────────────────────────────────────
    #  Internals
    # ──────────────────────────────────────────────────────────

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        tensor = resize(tensor, self.image_size)
        tensor = normalize(tensor)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _infer(
        self,
        image: torch.Tensor,
        radar_points: torch.Tensor | None,
        radar_mask: torch.Tensor | None,
        future_image: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        dummy_radar = torch.zeros(1, 256, 6, device=self.device)
        dummy_mask = torch.zeros(1, 256, dtype=torch.bool, device=self.device)

        rp = radar_points.to(self.device) if radar_points is not None else dummy_radar
        rm = radar_mask.to(self.device) if radar_mask is not None else dummy_mask
        fi = future_image if future_image is not None else image

        return self._forward({
            "image": image,
            "radar_points": rp,
            "radar_mask": rm,
            "future_image": fi,
        })

    def _forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            if self.is_multimodal:
                outputs = self.model(
                    image=batch["image"],
                    radar_points=batch["radar_points"],
                    radar_mask=batch["radar_mask"],
                )
                # Convert spatial outputs to per-object format for drawing
                return self._spatial_to_objects(outputs, batch["image"].shape)
            else:
                return self.model(
                    image=batch["image"],
                    radar_points=batch["radar_points"],
                    radar_mask=batch["radar_mask"],
                    future_image=batch.get("future_image"),
                    agent_states=batch.get("agent_states"),
                )

    def _spatial_to_objects(
        self, outputs: dict[str, torch.Tensor], img_shape: tuple,
    ) -> dict[str, torch.Tensor]:
        """Convert anchor-free spatial outputs to per-object boxes + velocity."""
        heatmap = outputs["heatmap"].sigmoid()  # (B, 1, H, W)
        box_reg = outputs["box_reg"]            # (B, 4, H, W)
        vel = outputs["velocity"]               # (B, 2, H, W)

        B, _, H, W = heatmap.shape

        # Flatten and get top-K detections
        scores_flat = heatmap.view(B, -1)  # (B, H*W)
        K = min(50, scores_flat.shape[1])
        topk_scores, topk_idx = scores_flat.topk(K, dim=1)  # (B, K)

        gy = (topk_idx // W).float()
        gx = (topk_idx % W).float()

        # Extract box regression at top-K positions
        box_flat = box_reg.view(B, 4, -1)  # (B, 4, H*W)
        topk_box = box_flat.gather(2, topk_idx.unsqueeze(1).expand(-1, 4, -1))  # (B, 4, K)
        topk_box = topk_box.permute(0, 2, 1)  # (B, K, 4)

        # Decode boxes to normalised coords
        cx = (gx / W + topk_box[:, :, 0]).clamp(0, 1)
        cy = (gy / H + topk_box[:, :, 1]).clamp(0, 1)
        bw = topk_box[:, :, 2].abs()
        bh = topk_box[:, :, 3].abs()

        x1 = (cx - bw / 2).clamp(0, 1)
        y1 = (cy - bh / 2).clamp(0, 1)
        x2 = (cx + bw / 2).clamp(0, 1)
        y2 = (cy + bh / 2).clamp(0, 1)

        # Use raw scores as confidence (already sigmoid)
        boxes = torch.stack([x1, y1, x2, y2, topk_scores], dim=-1)  # (B, K, 5)

        # Extract velocity at top-K positions
        vel_flat = vel.view(B, 2, -1)
        topk_vel = vel_flat.gather(2, topk_idx.unsqueeze(1).expand(-1, 2, -1))
        topk_vel = topk_vel.permute(0, 2, 1)  # (B, K, 2)

        return {"boxes": boxes, "velocity": topk_vel}

    def _draw(
        self,
        frame: np.ndarray,
        outputs: dict[str, torch.Tensor],
        dt: float,
    ) -> np.ndarray:
        """Annotate a frame with detections, velocity arrows, and trajectories."""
        h, w = frame.shape[:2]
        vis = frame.copy()

        boxes = outputs["boxes"][0].cpu()
        vel = outputs["velocity"][0].cpu()

        # For multimodal model, scores are already sigmoided
        if self.is_multimodal:
            scores = boxes[:, 4]
        else:
            scores = boxes[:, 4].sigmoid()
        keep = scores > self.conf_thr

        boxes = boxes[keep]
        vel = vel[keep]
        scores = scores[keep]

        has_traj = "trajectories" in outputs and "traj_logits" in outputs
        if has_traj:
            traj = outputs["trajectories"][0].cpu()
            logits = outputs["traj_logits"][0].cpu()
            traj = traj[keep]
            logits = logits[keep]
            best_mode = logits.argmax(dim=-1)

        for i in range(len(boxes)):
            x1 = int(boxes[i, 0].item() * w)
            y1 = int(boxes[i, 1].item() * h)
            x2 = int(boxes[i, 2].item() * w)
            y2 = int(boxes[i, 3].item() * h)
            score = scores[i].item()
            color = self._COLORS[i % len(self._COLORS)]

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = f"{score:.2f}"
            cv2.putText(vis, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if self.show_velocity:
                vx = vel[i, 0].item()
                vy = vel[i, 1].item()
                scale = 20.0
                ex = int(cx + vx * scale)
                ey = int(cy + vy * scale)
                cv2.arrowedLine(vis, (cx, cy), (ex, ey), (0, 0, 255), 2, tipLength=0.3)

            if has_traj:
                mode_i = best_mode[i].item()
                waypoints = traj[i, mode_i]
                traj_scale = 8.0
                pts = []
                for t in range(waypoints.shape[0]):
                    tx = int(cx + waypoints[t, 0].item() * traj_scale)
                    ty = int(cy + waypoints[t, 1].item() * traj_scale)
                    pts.append((tx, ty))
                for t in range(1, len(pts)):
                    alpha = t / len(pts)
                    tc = (
                        int(color[0] * (1 - alpha) + 255 * alpha),
                        int(color[1] * (1 - alpha)),
                        int(color[2] * (1 - alpha)),
                    )
                    cv2.line(vis, pts[t - 1], pts[t], tc, 2)
                if pts:
                    cv2.circle(vis, pts[-1], 4, (0, 0, 255), -1)

        fps_text = f"FPS: {1.0 / max(dt, 1e-8):.1f}  |  Latency: {dt * 1000:.1f}ms"
        cv2.putText(vis, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        n_det = int(keep.sum().item())
        cv2.putText(vis, f"Detections: {n_det}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return vis
