"""Full-pipeline evaluator for nuScenes detection + prediction benchmarking."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .metrics import (
    compute_map,
    compute_velocity_error,
    compute_ate,
    compute_ase,
    compute_ave,
    compute_nds,
    compute_prediction_metrics,
    match_predictions,
)


class Evaluator:
    """Run JEPA model evaluation and compute nuScenes-aligned metrics.

    Supports multi-GPU via ``DataParallel`` for inference-only
    benchmarking, and computes both **detection** and **prediction** metrics.

    Args:
        model: Trained JEPAModel.
        dataloader: Validation / test ``DataLoader``.
        iou_thresholds: IoU thresholds for mAP computation.
        multi_gpu: Wrap model in ``DataParallel`` for inference.
        amp_enabled: Use AMP for faster inference.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        iou_thresholds: list[float] | None = None,
        multi_gpu: bool = True,
        amp_enabled: bool = True,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        if multi_gpu and torch.cuda.device_count() > 1:
            print(f"[Evaluator] Using DataParallel on {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        self.dataloader = dataloader
        self.iou_thresholds = iou_thresholds or [0.3, 0.5, 0.7]
        self.amp_enabled = amp_enabled

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        """Run evaluation and return all metrics.

        Returns:
            Dict with detection metrics (mAP, NDS, ATE, ASE, AVE, etc.)
            and prediction metrics (minADE, minFDE, MissRate) when the
            model produces trajectory outputs.
        """
        self.model.eval()

        all_pred_boxes: list[torch.Tensor] = []
        all_pred_scores: list[torch.Tensor] = []
        all_gt_boxes: list[torch.Tensor] = []
        all_pred_vel: list[torch.Tensor] = []
        all_gt_vel: list[torch.Tensor] = []
        all_num_objects: list[torch.Tensor] = []

        all_pred_traj: list[torch.Tensor] = []
        all_gt_traj: list[torch.Tensor] = []
        all_traj_num: list[torch.Tensor] = []

        all_submission_entries: list[dict] = []

        total_time = 0.0
        total_samples = 0

        for batch in self.dataloader:
            batch = _to_device(batch, self.device)
            B = batch["image"].shape[0]

            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                outputs = self._forward(batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            total_time += elapsed
            total_samples += B

            pred_b = outputs["boxes"]
            pred_v = outputs["velocity"]
            gt_b = batch["boxes"]
            gt_v = batch["velocity"]
            num_obj = batch["num_objects"]

            for i in range(B):
                n_gt = num_obj[i].item()
                n_pred = (pred_b[i, :, 4].sigmoid() > 0.3).sum().item()
                n_pred = max(n_pred, 1)

                scores = pred_b[i, :n_pred, 4].sigmoid()
                pboxes = pred_b[i, :n_pred, :4]
                gboxes = gt_b[i, :n_gt, :4]

                all_pred_boxes.append(pboxes.cpu())
                all_pred_scores.append(scores.cpu())
                all_gt_boxes.append(gboxes.cpu())
                all_pred_vel.append(pred_v[i, :n_pred].cpu())
                all_gt_vel.append(gt_v[i, :n_gt].cpu())
                all_num_objects.append(num_obj[i:i+1].cpu())

            if "trajectories" in outputs and "gt_trajectories" in batch:
                all_pred_traj.append(outputs["trajectories"].cpu())
                all_gt_traj.append(batch["gt_trajectories"].cpu())
                all_traj_num.append(num_obj.cpu())

                if "instance_tokens" in batch and "sample_tokens" in batch:
                    traj_out = outputs["trajectories"].cpu()
                    logits = outputs["traj_logits"].cpu()
                    probs = torch.softmax(logits, dim=-1)

                    for i in range(B):
                        inst_tokens = batch["instance_tokens"][i]
                        sample_tok = batch["sample_tokens"][i]
                        n = min(num_obj[i].item(), len(inst_tokens))
                        for j in range(n):
                            traj_np = traj_out[i, j].numpy()
                            prob_np = probs[i, j].numpy()
                            all_submission_entries.append({
                                "instance": inst_tokens[j],
                                "sample": sample_tok,
                                "prediction": traj_np.tolist(),
                                "probabilities": prob_np.tolist(),
                            })

        # Detection metrics
        map_results = compute_map(
            all_pred_boxes, all_pred_scores, all_gt_boxes,
            iou_thresholds=self.iou_thresholds,
        )

        if all_pred_vel and all_gt_vel:
            max_p = max(v.shape[0] for v in all_pred_vel)
            max_g = max(v.shape[0] for v in all_gt_vel)
            max_m = max(max_p, max_g, 1)

            padded_pv = torch.zeros(len(all_pred_vel), max_m, 2)
            padded_gv = torch.zeros(len(all_gt_vel), max_m, 2)
            num_obj_t = torch.cat(all_num_objects)

            for i, (pv, gv) in enumerate(zip(all_pred_vel, all_gt_vel)):
                padded_pv[i, :pv.shape[0]] = pv
                padded_gv[i, :gv.shape[0]] = gv

            vel_results = compute_velocity_error(padded_pv, padded_gv, num_obj_t)
        else:
            vel_results = {"mAVE": 0.0, "medAVE": 0.0}

        all_ate, all_ase, all_ave_tp = [], [], []
        for pb, ps, gb, pv, gv in zip(
            all_pred_boxes, all_pred_scores, all_gt_boxes,
            all_pred_vel, all_gt_vel,
        ):
            pairs = match_predictions(pb, ps, gb, iou_threshold=0.5)
            all_ate.append(compute_ate(pb, gb, pairs))
            all_ase.append(compute_ase(pb, gb, pairs))
            if pairs.numel() > 0 and pv.shape[0] > 0 and gv.shape[0] > 0:
                all_ave_tp.append(compute_ave(pv, gv, pairs))
            else:
                all_ave_tp.append(1.0)

        mean_ate = float(sum(all_ate) / max(len(all_ate), 1))
        mean_ase = float(sum(all_ase) / max(len(all_ase), 1))
        mean_ave = float(sum(all_ave_tp) / max(len(all_ave_tp), 1))

        nds = compute_nds(map_results["mAP"], mean_ate, mean_ase, mean_ave)

        # Prediction metrics
        pred_metrics: dict[str, float] = {}
        if all_pred_traj:
            cat_pred = torch.cat(all_pred_traj, dim=0)
            cat_gt = torch.cat(all_gt_traj, dim=0)
            cat_num = torch.cat(all_traj_num, dim=0)
            pred_metrics = compute_prediction_metrics(cat_pred, cat_gt, cat_num)

        latency_ms = (total_time / max(total_samples, 1)) * 1000
        throughput = total_samples / max(total_time, 1e-8)

        results = {
            **map_results,
            **vel_results,
            "ATE": mean_ate,
            "ASE": mean_ase,
            "AVE": mean_ave,
            "NDS": nds,
            **pred_metrics,
            "latency_ms": latency_ms,
            "throughput_fps": throughput,
            "total_samples": total_samples,
        }

        self._print_results(results)
        return results

    def save_prediction_submission(
        self,
        output_path: str = "prediction_submission.json",
    ) -> str:
        """Run inference and save predictions in nuScenes submission format."""
        self.model.eval()
        entries: list[dict] = []

        with torch.no_grad():
            for batch in self.dataloader:
                batch = _to_device(batch, self.device)
                B = batch["image"].shape[0]

                with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                    outputs = self._forward(batch)

                if "trajectories" not in outputs:
                    continue
                if "instance_tokens" not in batch:
                    continue

                traj = outputs["trajectories"].cpu()
                logits = outputs["traj_logits"].cpu()
                probs = torch.softmax(logits, dim=-1)
                num_obj = batch["num_objects"]

                for i in range(B):
                    inst_tokens = batch["instance_tokens"][i]
                    sample_tok = batch["sample_tokens"][i]
                    n = min(num_obj[i].item(), len(inst_tokens))

                    for j in range(n):
                        entries.append({
                            "instance": inst_tokens[j],
                            "sample": sample_tok,
                            "prediction": traj[i, j].numpy().tolist(),
                            "probabilities": probs[i, j].numpy().tolist(),
                        })

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(entries, f)

        print(f"[Evaluator] Saved {len(entries)} predictions to {output_path}")
        return output_path

    def _forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model(
            image=batch["image"],
            radar_points=batch["radar_points"],
            radar_mask=batch["radar_mask"],
            future_image=batch.get("future_image"),
            agent_states=batch.get("agent_states"),
        )

    @staticmethod
    def _print_results(results: dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("  nuScenes Evaluation Results (Detection + Prediction)")
        print("=" * 60)
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k:20s}: {v:.4f}")
            else:
                print(f"  {k:20s}: {v}")
        print("=" * 60 + "\n")


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out
